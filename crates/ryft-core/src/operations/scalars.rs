use std::fmt::{Debug, Display};

use crate::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use crate::operations::constants::{
    ConstantOperation, ONE_LIKE_OPERATION_NAME, OneLikeOperation, OneOperation, SupportsConstant, SupportsOne,
    SupportsOneLike, SupportsZero, SupportsZeroLike, ZERO_LIKE_OPERATION_NAME, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::trigonometric::{
    COS_OPERATION_NAME, CosOperation, SIN_OPERATION_NAME, SinOperation, SupportsCos, SupportsSin,
};
use crate::operations::{Operation, OperationFormatter};
use crate::programs::Value;
use crate::types::{DataType, TypeError};

// TODO(eaplatanios): This file needs a careful review.

/// Closed scalar operation type for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar [`DataType`] metadata.
/// Array-only primitives such as reshaping and matrix multiplication remain available as standalone operations and
/// through array-based backends, but they are not variants of this enum.
#[derive(Clone, Debug)]
pub enum ScalarOperation<V: Value<DataType>> {
    /// Typed scalar zero with no inputs and one output.
    Zero(ZeroOperation<DataType>),

    /// Scalar exemplar-derived zero.
    ZeroLike,

    /// Typed scalar one with no inputs and one output.
    One(OneOperation<DataType>),

    /// Scalar exemplar-derived one.
    OneLike,

    /// Typed scalar constant with no inputs and one output.
    Constant(ConstantOperation<DataType, V>),

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
pub enum LinearScalarOperation<F: Value<DataType>> {
    /// Typed scalar zero with no inputs and one output.
    Zero(ZeroOperation<DataType>),

    /// Scalar exemplar-derived zero map.
    ZeroLike,

    /// Typed scalar one with no inputs and one output.
    One(OneOperation<DataType>),

    /// Scalar exemplar-derived one map.
    OneLike,

    /// Typed scalar constant with no inputs and one output.
    Constant(ConstantOperation<DataType, F>),

    /// Scalar negation.
    Neg,

    /// Scalar addition.
    Add,

    /// Scalar subtraction.
    Sub,

    /// Scalar scaling by a captured factor.
    Scale { factor: F },
}

impl<V: Value<DataType>> SupportsAdd<DataType> for ScalarOperation<V> {
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<V: Value<DataType>> SupportsSub<DataType> for ScalarOperation<V> {
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
    }
}

impl<V: Value<DataType>> SupportsMul<DataType> for ScalarOperation<V> {
    #[inline]
    fn mul_operation() -> Self {
        Self::Mul
    }
}

impl<V: Value<DataType>> SupportsDiv<DataType> for ScalarOperation<V> {
    #[inline]
    fn div_operation() -> Self {
        Self::Div
    }
}

impl<V: Value<DataType>> SupportsNeg<DataType> for ScalarOperation<V> {
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<V: Value<DataType>> SupportsSin<DataType> for ScalarOperation<V> {
    #[inline]
    fn sin_operation() -> Self {
        Self::Sin
    }
}

impl<V: Value<DataType>> SupportsCos<DataType> for ScalarOperation<V> {
    #[inline]
    fn cos_operation() -> Self {
        Self::Cos
    }
}

impl<V: Value<DataType>> SupportsZero<DataType> for ScalarOperation<V> {
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

impl<V: Value<DataType>> SupportsOne<DataType> for ScalarOperation<V> {
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<V: Value<DataType>> SupportsZeroLike<DataType> for ScalarOperation<V> {
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<V: Value<DataType>> SupportsOneLike<DataType> for ScalarOperation<V> {
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<V: Value<DataType>> SupportsScale<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn scale_operation(factor: V) -> Self {
        Self::Scale { factor }
    }
}

impl<V: Value<DataType>> SupportsConstant<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn constant_operation(value: V) -> Self {
        Self::Constant(ConstantOperation::new(value))
    }

    #[inline]
    fn as_constant_operation(&self) -> Option<&ConstantOperation<DataType, V>> {
        match self {
            Self::Constant(constant) => Some(constant),
            _ => None,
        }
    }
}

impl<F: Value<DataType>> SupportsAdd<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<F: Value<DataType>> SupportsSub<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
    }
}

impl<F: Value<DataType>> SupportsZero<DataType> for LinearScalarOperation<F> {
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

impl<F: Value<DataType>> SupportsOne<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<F: Value<DataType>> SupportsZeroLike<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<F: Value<DataType>> SupportsOneLike<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<F: Value<DataType>> SupportsNeg<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<F: Value<DataType>> SupportsScale<DataType, F> for LinearScalarOperation<F> {
    #[inline]
    fn scale_operation(factor: F) -> Self {
        Self::Scale { factor }
    }
}

impl<F: Value<DataType>> SupportsConstant<DataType, F> for LinearScalarOperation<F> {
    #[inline]
    fn constant_operation(value: F) -> Self {
        Self::Constant(ConstantOperation::new(value))
    }

    #[inline]
    fn as_constant_operation(&self) -> Option<&ConstantOperation<DataType, F>> {
        match self {
            Self::Constant(constant) => Some(constant),
            _ => None,
        }
    }
}

impl<V: Value<DataType>> ScalarOperation<V> {
    #[inline]
    fn operation_name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::ZeroLike => ZERO_LIKE_OPERATION_NAME,
            Self::One(one) => one.name(),
            Self::OneLike => ONE_LIKE_OPERATION_NAME,
            Self::Constant(constant) => constant.name(),
            Self::Neg => NEG_OPERATION_NAME,
            Self::Add => ADD_OPERATION_NAME,
            Self::Sub => SUB_OPERATION_NAME,
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::Mul => MUL_OPERATION_NAME,
            Self::Div => DIV_OPERATION_NAME,
            Self::Sin => SIN_OPERATION_NAME,
            Self::Cos => COS_OPERATION_NAME,
        }
    }
}

impl<F: Value<DataType>> LinearScalarOperation<F> {
    #[inline]
    fn operation_name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::ZeroLike => ZERO_LIKE_OPERATION_NAME,
            Self::One(one) => one.name(),
            Self::OneLike => ONE_LIKE_OPERATION_NAME,
            Self::Constant(constant) => constant.name(),
            Self::Neg => NEG_OPERATION_NAME,
            Self::Add => ADD_OPERATION_NAME,
            Self::Sub => SUB_OPERATION_NAME,
            Self::Scale { .. } => SCALE_OPERATION_NAME,
        }
    }
}

impl<V: Value<DataType>> Display for ScalarOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<F: Value<DataType>> Display for LinearScalarOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: Value<DataType>> Operation<DataType> for ScalarOperation<V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Constant(constant) => constant.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Div => DivOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<F: Value<DataType>> Operation<DataType> for LinearScalarOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Constant(constant) => constant.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            _ => Display::fmt(self, formatter),
        }
    }
}
