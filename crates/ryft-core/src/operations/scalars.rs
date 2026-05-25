use std::fmt::{Debug, Display};

use crate::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use crate::operations::constants::{
    ONE_LIKE_OPERATION_NAME, OneLikeOperation, OneOperation, SupportsOne, SupportsOneLike, SupportsZero,
    SupportsZeroLike, ZERO_LIKE_OPERATION_NAME, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::trigonometric::{
    COS_OPERATION_NAME, CosOperation, SIN_OPERATION_NAME, SinOperation, SupportsCos, SupportsSin,
};
use crate::operations::{Operation, OperationFormatter};
use crate::tracing::Traceable;
use crate::types::{DataType, TypeError};

// TODO(eaplatanios): This file needs a careful review.

/// Closed scalar operation type for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar [`DataType`] metadata.
/// Array-only primitives such as reshaping and matrix multiplication remain available as standalone operations and
/// through array-based backends, but they are not variants of this enum.
#[derive(Clone, Debug)]
pub enum ScalarOperation<V: Traceable<DataType>> {
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
}

/// Closed scalar operation type for staged linear scalar programs.
#[derive(Clone, Debug)]
pub enum LinearScalarOperation<V: Traceable<DataType>> {
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
}

impl<V: Traceable<DataType>> SupportsAdd<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<V: Traceable<DataType>> SupportsSub<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
    }
}

impl<V: Traceable<DataType>> SupportsMul<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn mul_operation() -> Self {
        Self::Mul
    }
}

impl<V: Traceable<DataType>> SupportsDiv<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn div_operation() -> Self {
        Self::Div
    }
}

impl<V: Traceable<DataType>> SupportsNeg<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<V: Traceable<DataType>> SupportsSin<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn sin_operation() -> Self {
        Self::Sin
    }
}

impl<V: Traceable<DataType>> SupportsCos<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn cos_operation() -> Self {
        Self::Cos
    }
}

impl<V: Traceable<DataType>> SupportsZero<DataType, V> for ScalarOperation<V> {
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

impl<V: Traceable<DataType>> SupportsOne<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<V: Traceable<DataType>> SupportsZeroLike<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<V: Traceable<DataType>> SupportsOneLike<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<V: Traceable<DataType>> SupportsScale<DataType, V, V> for ScalarOperation<V> {
    #[inline]
    fn scale_operation(factor: V) -> Self {
        Self::Scale { factor }
    }
}

impl<V: Traceable<DataType>> SupportsAdd<DataType, V> for LinearScalarOperation<V> {
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<V: Traceable<DataType>> SupportsSub<DataType, V> for LinearScalarOperation<V> {
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
    }
}

impl<V: Traceable<DataType>> SupportsZero<DataType, V> for LinearScalarOperation<V> {
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

impl<V: Traceable<DataType>> SupportsOne<DataType, V> for LinearScalarOperation<V> {
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<V: Traceable<DataType>> SupportsZeroLike<DataType, V> for LinearScalarOperation<V> {
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<V: Traceable<DataType>> SupportsOneLike<DataType, V> for LinearScalarOperation<V> {
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<V: Traceable<DataType>> SupportsNeg<DataType, V> for LinearScalarOperation<V> {
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<V: Traceable<DataType>> SupportsScale<DataType, V, V> for LinearScalarOperation<V> {
    #[inline]
    fn scale_operation(factor: V) -> Self {
        Self::Scale { factor }
    }
}

impl<V: Traceable<DataType>> ScalarOperation<V> {
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
            Self::Sin => SIN_OPERATION_NAME,
            Self::Cos => COS_OPERATION_NAME,
            Self::Scale { .. } => SCALE_OPERATION_NAME,
        }
    }
}

impl<V: Traceable<DataType>> LinearScalarOperation<V> {
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
        }
    }
}

impl<V: Traceable<DataType>> Display for ScalarOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: Traceable<DataType>> Display for LinearScalarOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: Traceable<DataType>> Operation<DataType> for ScalarOperation<V> {
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
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType>> Operation<DataType> for LinearScalarOperation<V> {
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
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            _ => Display::fmt(self, formatter),
        }
    }
}
