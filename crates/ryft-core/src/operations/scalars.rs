use std::fmt::{Debug, Display};
use std::sync::Arc;

use crate::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use crate::operations::constants::{
    ONE_LIKE_OPERATION_NAME, OneLikeOperation, OneOperation, SupportsOne, SupportsOneLike, SupportsZero,
    SupportsZeroLike, ZERO_LIKE_OPERATION_NAME, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::{Operation, OperationFormatter};
use crate::parameters::Parameter;
use crate::tracing::{Traceable, TracingError};
use crate::tracing_v2::operations::{
    CosOperation, CustomPrimitive, LinearCustomPrimitive, SinOperation, SupportsCos, SupportsCustom,
    SupportsLinearCustom, SupportsSin,
};
use crate::types::{DataType, TypeError};

/// Closed scalar operation carrier for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar
/// [`DataType`] metadata. Array-only primitives such as reshaping and matrix multiplication remain
/// available as standalone operations and through array backend carriers, but they are not variants
/// of this enum.
#[derive(Clone, Debug)]
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

    /// Scalar negation.
    Neg,

    /// Scalar addition.
    Add,

    /// Scalar subtraction.
    Sub,

    /// Scalar scaling by a captured factor.
    Scale { factor: V },

    /// Scalar multiplication.
    Mul,

    /// Scalar division.
    Div,

    /// Scalar sine.
    Sin,

    /// Scalar cosine.
    Cos,

    /// Escape hatch for user- or crate-defined scalar operations.
    Custom(Arc<CustomPrimitive<DataType, V>>),
}

/// Closed scalar operation carrier for staged linear scalar programs.
#[derive(Clone, Debug)]
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

    /// Scalar negation.
    Neg,

    /// Scalar addition.
    Add,

    /// Scalar subtraction.
    Sub,

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

impl<V> SupportsSub<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
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

impl<V> SupportsDiv<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn div_operation() -> Self {
        Self::Div
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
    fn as_zero_operation(&self) -> Option<&ZeroOperation<DataType>> {
        match self {
            Self::Zero(zero) => Some(zero),
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

impl<V> SupportsSub<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
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
    fn as_zero_operation(&self) -> Option<&ZeroOperation<DataType>> {
        match self {
            Self::Zero(zero) => Some(zero),
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

impl<V> ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
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
            Self::Scale { .. } => SCALE_OPERATION_NAME,
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
            Self::ZeroLike => ZERO_LIKE_OPERATION_NAME,
            Self::OneLike => ONE_LIKE_OPERATION_NAME,
            Self::Add => ADD_OPERATION_NAME,
            Self::Sub => SUB_OPERATION_NAME,
            Self::Neg => NEG_OPERATION_NAME,
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::Custom(op) => op.name(),
        }
    }
}

impl<V> Display for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V> Display for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for ScalarOperation<V> {
    #[inline]
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
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Div => DivOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
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
    #[inline]
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
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
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
