use std::fmt::{Debug, Display};

use crate::macros::check_count;
use crate::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use crate::operations::compare::{COMPARE_OPERATION_NAME, CompareOperation, ComparisonDirection, SupportsCompare};
use crate::operations::constants::{
    ConstantOperation, ONE_LIKE_OPERATION_NAME, OneLikeOperation, OneOperation, SupportsConstant, SupportsOne,
    SupportsOneLike, SupportsZero, SupportsZeroLike, ZERO_LIKE_OPERATION_NAME, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{SELECT_OPERATION_NAME, SelectOperation, SupportsSelect};
use crate::operations::differentiation::{STOP_GRADIENT_OPERATION_NAME, StopGradientOperation, SupportsStopGradient};
use crate::operations::trigonometric::{
    COS_OPERATION_NAME, CosOperation, SIN_OPERATION_NAME, SinOperation, SupportsCos, SupportsSin,
};
use crate::operations::{Operation, OperationFormatter};
use crate::programs::{Program, Value};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation, SupportsCustomJvp, SupportsCustomVjp,
    SupportsCustomVjpCall,
};
use crate::tracing_v2::operations::select::SupportsLinearSelect;
use crate::tracing_v2::rematerialization::{
    MaybeRematerializationName, REMATERIALIZATION_NAME_OPERATION_NAME, RematerializationNameOperation,
    SupportsRematerializationName,
};
use crate::types::{DataType, TypeError};

// TODO(eaplatanios): Review this file.

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

    /// Scalar pairwise comparison with an in-band Boolean result.
    Compare { direction: ComparisonDirection },

    /// Scalar selection between two values driven by a Boolean condition.
    Select,

    /// Gradient-severing identity.
    StopGradient,

    /// Rematerialize-policy name-tagging identity.
    RematerializationName(RematerializationNameOperation),

    /// Higher-order call pairing a primal program with a user-supplied JVP program.
    CustomJvp(Box<CustomJvpOperation<V, ScalarOperation<V>, DataType>>),

    /// Higher-order call pairing a primal program with user-supplied forward/backward (VJP) programs.
    CustomVjp(Box<CustomVjpOperation<V, ScalarOperation<V>, DataType>>),
}

/// Closed scalar operation type for staged linear scalar programs.
///
/// The `C` parameter is the constant type of the
/// [`DifferentiationContext`](crate::tracing_v2::DifferentiationContext) that stages the linear program: every
/// context pins `C` to its [`Domain::Constant`](crate::domains::Domain) in its `LinearOperation` associated-type
/// definition. It types the user-supplied backward programs captured by [`CustomVjpCall`](Self::CustomVjpCall),
/// which are written over context constants rather than over the factor type `F`.
#[derive(Clone, Debug)]
pub enum LinearScalarOperation<C: Value<DataType>, F: Value<DataType> = C> {
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

    /// Captured-condition select: linear map `(t, f) ↦ select(condition, t, f)`. Scalar counterpart of the `Select`
    /// variant of [`LinearArrayOperation`](crate::tracing_v2::ArrayOperation) emitted by the JVP of
    /// [`ScalarOperation::Select`]: the Boolean primal condition is captured as a factor (it has no tangent space, so
    /// the map is linear in the two branch operands). Its transpose routes the output cotangent into the selected
    /// branch: the `on_true` cotangent is `select(condition, cotangent, 0)` and the `on_false` cotangent is
    /// `select(condition, 0, cotangent)`.
    Select {
        /// Captured Boolean condition that drives the selection.
        condition: F,
    },

    /// Opaque linear call staged by a `custom_vjp` linearization; its transpose replays the user's backward program.
    CustomVjpCall(Box<CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType>>),
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

impl<V: Value<DataType>> SupportsStopGradient<DataType> for ScalarOperation<V> {
    #[inline]
    fn stop_gradient_operation() -> Self {
        Self::StopGradient
    }
}

impl<V: Value<DataType>> SupportsCos<DataType> for ScalarOperation<V> {
    #[inline]
    fn cos_operation() -> Self {
        Self::Cos
    }
}

impl<V: Value<DataType>> SupportsCompare<DataType> for ScalarOperation<V> {
    #[inline]
    fn compare_operation(direction: ComparisonDirection) -> Self {
        Self::Compare { direction }
    }
}

impl<V: Value<DataType>> SupportsSelect<DataType> for ScalarOperation<V> {
    #[inline]
    fn select_operation() -> Self {
        Self::Select
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

impl<C: Value<DataType>, F: Value<DataType>> SupportsAdd<DataType> for LinearScalarOperation<C, F> {
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsSub<DataType> for LinearScalarOperation<C, F> {
    #[inline]
    fn sub_operation() -> Self {
        Self::Sub
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsZero<DataType> for LinearScalarOperation<C, F> {
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

impl<C: Value<DataType>, F: Value<DataType>> SupportsOne<DataType> for LinearScalarOperation<C, F> {
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsZeroLike<DataType> for LinearScalarOperation<C, F> {
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsOneLike<DataType> for LinearScalarOperation<C, F> {
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsNeg<DataType> for LinearScalarOperation<C, F> {
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsScale<DataType, F> for LinearScalarOperation<C, F> {
    #[inline]
    fn scale_operation(factor: F) -> Self {
        Self::Scale { factor }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsLinearSelect<DataType, F> for LinearScalarOperation<C, F> {
    #[inline]
    fn linear_select_operation(condition: F) -> Self {
        Self::Select { condition }
    }
}

impl<V: Value<DataType>> SupportsRematerializationName<DataType> for ScalarOperation<V> {
    #[inline]
    fn rematerialization_name_operation(name: String) -> Self {
        Self::RematerializationName(RematerializationNameOperation::new(name))
    }
}

impl<V: Value<DataType>> MaybeRematerializationName for ScalarOperation<V> {
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        match self {
            Self::RematerializationName(operation) => Some(operation.tag()),
            _ => None,
        }
    }
}

impl<V: Value<DataType>> crate::tracing_v2::operations::dot::MaybeDot for ScalarOperation<V> {
    #[inline]
    fn dot_dimensions(&self) -> Option<&crate::tracing_v2::operations::dot::DotDimensionNumbers> {
        None
    }
}

impl<V: Value<DataType>> SupportsCustomJvp<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn custom_jvp_operation(operation: CustomJvpOperation<V, Self, DataType>) -> Self {
        Self::CustomJvp(Box::new(operation))
    }
}

impl<V: Value<DataType>> SupportsCustomVjp<DataType, V> for ScalarOperation<V> {
    #[inline]
    fn custom_vjp_operation(operation: CustomVjpOperation<V, Self, DataType>) -> Self {
        Self::CustomVjp(Box::new(operation))
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsCustomVjpCall<DataType, C, ScalarOperation<C>, F>
    for LinearScalarOperation<C, F>
{
    #[inline]
    fn custom_vjp_call_operation(
        backward: Program<DataType, C, ScalarOperation<C>, Vec<C>, Vec<C>>,
        tangent: Option<Program<DataType, C, ScalarOperation<C>, Vec<C>, Vec<C>>>,
        residuals: Vec<F>,
        transposed: bool,
        prevent_cse: bool,
    ) -> Self {
        Self::CustomVjpCall(Box::new(CustomVjpCallOperation::new(
            backward,
            tangent,
            residuals,
            transposed,
            prevent_cse,
        )))
    }
}

impl<C: Value<DataType>, F: Value<DataType>> SupportsConstant<DataType, F> for LinearScalarOperation<C, F> {
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
            Self::Compare { .. } => COMPARE_OPERATION_NAME,
            Self::Select => SELECT_OPERATION_NAME,
            Self::StopGradient => STOP_GRADIENT_OPERATION_NAME,
            Self::RematerializationName(_) => REMATERIALIZATION_NAME_OPERATION_NAME,
            Self::CustomJvp(_) => "custom_jvp",
            Self::CustomVjp(_) => "custom_vjp",
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> LinearScalarOperation<C, F> {
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
            Self::Select { .. } => SELECT_OPERATION_NAME,
            Self::CustomVjpCall(call) => {
                if call.transposed() {
                    "custom_vjp_backward"
                } else {
                    "custom_vjp_tangent"
                }
            }
        }
    }
}

impl<V: Value<DataType>> Display for ScalarOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<C: Value<DataType>, F: Value<DataType>> Display for LinearScalarOperation<C, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
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
            Self::Compare { direction } => CompareOperation::new(*direction).infer_output_types(input_types),
            Self::Select => SelectOperation.infer_output_types(input_types),
            Self::StopGradient => StopGradientOperation.infer_output_types(input_types),
            Self::RematerializationName(operation) => operation.infer_output_types(input_types),
            Self::CustomJvp(operation) => operation.infer_output_types(input_types),
            Self::CustomVjp(operation) => operation.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Compare { direction } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("direction", direction)),
            _ => formatter.write_str(self.operation_name()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> Operation<DataType> for LinearScalarOperation<C, F> {
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
            // The captured-condition select is linear in its two branch tangents, which share a type that is the
            // output type. The Boolean primal condition is captured in band as an `f64` residual, so it is not
            // revalidated here (the primal trace already typed it); only the branch tangents are checked.
            Self::Select { .. } => {
                check_count!("input", input_types, 2, TypeError);
                if input_types[0] != input_types[1] {
                    return Err(TypeError {
                        message: format!(
                            "select on_true data type {} differs from on_false data type {}",
                            input_types[0], input_types[1],
                        ),
                    });
                }
                Ok(vec![input_types[0]])
            }
            Self::CustomVjpCall(call) => call.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Select { condition } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("condition", condition)),
            _ => formatter.write_str(self.operation_name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::compare::Compare;
    use crate::operations::control_flow::Select;
    use crate::scalars::ScalarDomain;
    use crate::tracing::trace;

    use super::*;

    #[test]
    fn test_scalar_compare_and_select_program() {
        // `f(x, y) = select(x > y, x + x, y)` staged through `ScalarOperation` tracers.
        let domain = ScalarDomain::<f64>::new();
        let (output_type, program) = trace(
            &domain,
            |(x, y)| {
                let mask = x.clone().greater_than(&y);
                Select::select(&mask, &(x.clone() + x), &y)
            },
            (DataType::F64, DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:bool = compare [direction=GreaterThan] %0 %1
                    %3:f64 = add %0 %0
                    %4:f64 = select %2 %3 %1
                in (%4)
            "}
            .trim_end(),
        );

        // Interpreting the staged program exercises the in-band Boolean condition encoding of scalar values.
        assert_eq!(program.interpret((3.0, 2.0)), Ok(6.0));
        assert_eq!(program.interpret((1.0, 2.0)), Ok(2.0));
    }
}
