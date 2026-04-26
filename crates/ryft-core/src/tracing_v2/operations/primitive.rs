use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

use crate::{
    parameters::{Parameter, Parameterized},
    tracing::{AtomId, OperationFormatter, Traceable, TracingError, Value},
    tracing_v2::{
        Cos, MatrixOps, Sin,
        engines::DifferentiableEngine,
        forward::{Differentiable, JvpContext, JvpTracer},
        jit::Tracer,
        operations::{
            AddOperation, CosOperation, LeftMatMulOperation, MatMulOperation, MatrixTransposeOperation, MulOperation,
            NegOperation, ReshapeOperation, RightMatMulOperation, ScaleOperation, SinOperation,
            constants::{OneLike, ZeroLike, ZeroOperation},
            control_flow::{ConditionOperation, ControlFlowValue, WhileOperation},
            left_matmul::left_matmul_abstract_eval,
            right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, TypeError, Typed},
};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation,
    add::SupportsAdd,
    constants::SupportsZero,
    cos::SupportsCos,
    custom::{CustomPrimitive, LinearCustomPrimitive, SupportsCustom, SupportsLinearCustom},
    left_matmul::SupportsLeftMatMul,
    matmul::SupportsMatMul,
    matrix_transpose::SupportsMatrixTranspose,
    mul::SupportsMul,
    neg::SupportsNeg,
    rematerialize::{SupportsLinearRematerialize, SupportsRematerialize},
    reshape::SupportsReshape,
    right_matmul::SupportsRightMatMul,
    scale::SupportsScale,
    sin::SupportsSin,
};

/// Closed set of built-in staged operations.
///
/// [`PrimitiveOperation`] is the default ordinary-program carrier for `ryft-core`. Each variant is a
/// thin tag around one semantic primitive defined elsewhere in [`super`], and the carrier exists so
/// tracing entry points can store "one of the built-in operations" without resorting to trait
/// objects for the common case.
#[derive(Clone)]
pub enum PrimitiveOperation<V: Traceable<ArrayType> + Parameter> {
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
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(
        Box<
            crate::tracing_v2::operations::RematerializeOperation<
                ArrayType,
                V,
                PrimitiveOperation<V>,
                LinearPrimitiveOperation<V>,
            >,
        >,
    ),

    /// Higher-order conditional carrying true and false branch programs.
    Condition(Box<ConditionOperation<V, PrimitiveOperation<V>>>),

    /// Higher-order while loop carrying condition and body programs.
    While(Box<WhileOperation<V, PrimitiveOperation<V>>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<ArrayType, V>>),
}

/// Closed set of operations that may appear in staged linear programs.
///
/// [`LinearPrimitiveOperation`] is the linear-program sibling of [`PrimitiveOperation`]. It contains only the
/// operations that make sense in tangent and cotangent programs plus the linearized higher-order
/// ops needed by rematerialization.
#[derive(Clone)]
pub enum LinearPrimitiveOperation<V: Traceable<ArrayType> + Parameter> {
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
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Typed zero with no inputs and one output, carrying a [`ZeroOperation`].
    ///
    /// Emitted by the transpose pass at the boundary of pullbacks for primal inputs that
    /// receive no cotangent contribution from any output. Interpreting it requires
    /// [`Zero<ArrayType>`](crate::tracing_v2::operations::constants::Zero) on the value type;
    /// staged tracer programs must materialize these ops away (via the outer-trace builder)
    /// before being interpreted.
    Zero(ZeroOperation),

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(
        Box<crate::tracing_v2::operations::LinearRematerializeOperation<ArrayType, V, LinearPrimitiveOperation<V>>>,
    ),

    /// Higher-order conditional restricted to linear branch programs.
    Condition(Box<ConditionOperation<V, LinearPrimitiveOperation<V>>>),

    /// Higher-order while loop restricted to linear condition and body programs.
    While(Box<WhileOperation<V, LinearPrimitiveOperation<V>>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<ArrayType, V>>),
}

impl<V: Traceable<ArrayType> + 'static> LinearPrimitiveOperation<V> {
    /// Wraps one custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<ArrayType, V>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<ArrayType, V>>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<V: Traceable<ArrayType>> SupportsAdd<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn add_operation() -> Self {
        PrimitiveOperation::Add
    }
}

impl<V: Traceable<ArrayType>> SupportsMul<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn mul_operation() -> Self {
        PrimitiveOperation::Mul
    }
}

impl<V: Traceable<ArrayType>> SupportsNeg<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn neg_operation() -> Self {
        PrimitiveOperation::Neg
    }
}

impl<V: Traceable<ArrayType>> SupportsSin<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn sin_operation() -> Self {
        PrimitiveOperation::Sin
    }
}

impl<V: Traceable<ArrayType>> SupportsCos<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn cos_operation() -> Self {
        PrimitiveOperation::Cos
    }
}

impl<V: Traceable<ArrayType>> SupportsMatMul<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn matmul_operation() -> Self {
        PrimitiveOperation::MatMul
    }
}

impl<V: Traceable<ArrayType>> SupportsMatrixTranspose<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn matrix_transpose_operation() -> Self {
        PrimitiveOperation::MatrixTranspose
    }
}

impl<V: Traceable<ArrayType>> SupportsScale<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn scale_operation(factor: V) -> Self {
        PrimitiveOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType>> SupportsLeftMatMul<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn left_matmul_operation(factor: V) -> Self {
        PrimitiveOperation::LeftMatMul { factor }
    }
}

impl<V: Traceable<ArrayType>> SupportsRightMatMul<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn right_matmul_operation(factor: V) -> Self {
        PrimitiveOperation::RightMatMul { factor }
    }
}

impl<V: Traceable<ArrayType>> SupportsReshape<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn reshape_operation(input_type: ArrayType, output_type: ArrayType) -> Self {
        PrimitiveOperation::Reshape { input_type, output_type }
    }
}

impl<V: Traceable<ArrayType>> SupportsRematerialize<ArrayType, V, LinearPrimitiveOperation<V>>
    for PrimitiveOperation<V>
{
    #[inline]
    fn rematerialize_operation(
        op: crate::tracing_v2::operations::RematerializeOperation<ArrayType, V, Self, LinearPrimitiveOperation<V>>,
    ) -> Self {
        PrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl<V: Traceable<ArrayType>> SupportsCustom<ArrayType, V> for PrimitiveOperation<V> {
    #[inline]
    fn custom_operation(primitive: Arc<CustomPrimitive<ArrayType, V>>) -> Self {
        PrimitiveOperation::Custom(primitive)
    }
}

impl<V: Traceable<ArrayType>> SupportsAdd<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn add_operation() -> Self {
        LinearPrimitiveOperation::Add
    }
}

impl<V: Traceable<ArrayType>> SupportsZero<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn zero_operation(r#type: ArrayType) -> Self {
        LinearPrimitiveOperation::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero(&self) -> Option<&ArrayType> {
        match self {
            Self::Zero(zero) => Some(zero.output_type()),
            _ => None,
        }
    }
}

impl<V: Traceable<ArrayType>> SupportsNeg<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn neg_operation() -> Self {
        LinearPrimitiveOperation::Neg
    }
}

impl<V: Traceable<ArrayType>> SupportsMatrixTranspose<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn matrix_transpose_operation() -> Self {
        LinearPrimitiveOperation::MatrixTranspose
    }
}

impl<V: Traceable<ArrayType>> SupportsScale<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn scale_operation(factor: V) -> Self {
        LinearPrimitiveOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType>> SupportsLeftMatMul<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn left_matmul_operation(factor: V) -> Self {
        LinearPrimitiveOperation::LeftMatMul { factor }
    }
}

impl<V: Traceable<ArrayType>> SupportsRightMatMul<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn right_matmul_operation(factor: V) -> Self {
        LinearPrimitiveOperation::RightMatMul { factor }
    }
}

impl<V: Traceable<ArrayType>> SupportsReshape<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn reshape_operation(input_type: ArrayType, output_type: ArrayType) -> Self {
        LinearPrimitiveOperation::Reshape { input_type, output_type }
    }
}

impl<V: Traceable<ArrayType>> SupportsLinearRematerialize<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn rematerialize_operation(
        op: crate::tracing_v2::operations::LinearRematerializeOperation<ArrayType, V, Self>,
    ) -> Self {
        LinearPrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl<V: Traceable<ArrayType>> From<ConditionOperation<V, LinearPrimitiveOperation<V>>> for LinearPrimitiveOperation<V> {
    #[inline]
    fn from(op: ConditionOperation<V, LinearPrimitiveOperation<V>>) -> Self {
        LinearPrimitiveOperation::Condition(Box::new(op))
    }
}

impl<V: Traceable<ArrayType> + 'static> SupportsLinearCustom<ArrayType, V> for LinearPrimitiveOperation<V> {
    #[inline]
    fn custom_operation(primitive: CustomPrimitive<ArrayType, V>) -> Result<Self, TracingError> {
        Ok(LinearPrimitiveOperation::Custom(Arc::new(primitive.into_linear()?)))
    }

    #[inline]
    fn custom_arc_operation(primitive: Arc<CustomPrimitive<ArrayType, V>>) -> Result<Self, TracingError> {
        Ok(LinearPrimitiveOperation::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<V: Traceable<ArrayType>> Debug for PrimitiveOperation<V> {
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
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Condition(condition) => Debug::fmt(condition, formatter),
            Self::While(while_operation) => Debug::fmt(while_operation, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V: Traceable<ArrayType>> Display for PrimitiveOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl<V: Traceable<ArrayType>> Debug for LinearPrimitiveOperation<V> {
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
            Self::Zero(zero) => Debug::fmt(zero, formatter),
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Condition(condition) => Debug::fmt(condition, formatter),
            Self::While(while_operation) => Debug::fmt(while_operation, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V: Traceable<ArrayType>> Display for LinearPrimitiveOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

/// [`Operation`] for [`PrimitiveOperation`] relies only on the [`Traceable`] value contract; shape validation works for
/// any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Operation<ArrayType> for PrimitiveOperation<V> {
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
            Self::Rematerialize(remat) => remat.name(),
            Self::Condition(condition) => condition.name(),
            Self::While(while_operation) => while_operation.name(),
            Self::Custom(op) => op.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::MatMul => MatMulOperation.infer_output_types(input_types),
            Self::MatrixTranspose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<ArrayType, V>::abstract_eval_static(input_types),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::Reshape { input_type, output_type } => {
                <ReshapeOperation as Operation<ArrayType>>::infer_output_types(
                    &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                    input_types,
                )
            }
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::LeftMatMul { factor } | Self::RightMatMul { factor } => {
                OperationFormatter::new(formatter, indentation, self.name())?
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

/// [`Operation`] for [`LinearPrimitiveOperation`] relies only on the [`Traceable`] value contract; shape validation
/// works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Operation<ArrayType> for LinearPrimitiveOperation<V> {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Neg => "neg",
            Self::MatrixTranspose => "matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::Zero(zero) => zero.name(),
            Self::Rematerialize(remat) => remat.name(),
            Self::Condition(condition) => condition.name(),
            Self::While(while_operation) => while_operation.name(),
            Self::Custom(op) => op.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::MatrixTranspose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<ArrayType, V>::abstract_eval_static(input_types),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::Reshape { input_type, output_type } => {
                <ReshapeOperation as Operation<ArrayType>>::infer_output_types(
                    &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                    input_types,
                )
            }
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::LeftMatMul { factor } | Self::RightMatMul { factor } => {
                OperationFormatter::new(formatter, indentation, self.name())?
                    .bracketed(|operation| operation.field("factor", factor))
            }
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::Rematerialize(remat) => remat.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
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
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
> InterpretableOperation<ArrayType, V> for PrimitiveOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
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
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
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
        + crate::tracing_v2::operations::constants::Zero<ArrayType>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
> InterpretableOperation<ArrayType, V> for LinearPrimitiveOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
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
            Self::Zero(zero) => zero.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
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
        + crate::tracing_v2::operations::constants::Zero<ArrayType>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
> LinearOperation<ArrayType, V> for LinearPrimitiveOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<
            '_,
            ArrayType,
            V,
            LinearPrimitiveOperation<V>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Add => AddOperation.transpose(context, output_cotangents),
            Self::Neg => NegOperation.transpose(context, output_cotangents),
            Self::MatrixTranspose => MatrixTransposeOperation.transpose(context, output_cotangents),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(context, output_cotangents),
            Self::LeftMatMul { factor } => {
                LeftMatMulOperation::new(factor.clone()).transpose(context, output_cotangents)
            }
            Self::RightMatMul { factor } => {
                RightMatMulOperation::new(factor.clone()).transpose(context, output_cotangents)
            }
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).transpose(context, output_cotangents)
            }
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::Rematerialize(remat) => remat.transpose(context, output_cotangents),
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
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
        + crate::tracing_v2::operations::constants::Zero<ArrayType>
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    E: DifferentiableEngine<
            Type = ArrayType,
            Value = V,
            Operation = PrimitiveOperation<V>,
            LinearOperation = LinearPrimitiveOperation<V>,
        > + 'static,
> DifferentiableOperation<E> for PrimitiveOperation<V>
where
    V: Differentiable<ArrayType, Tangent = V>,
    V::ParameterStructure: Clone + std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    LinearPrimitiveOperation<V>: super::SupportsAdd<ArrayType, V>
        + super::SupportsNeg<ArrayType, V>
        + super::SupportsScale<ArrayType, V>
        + super::SupportsLeftMatMul<ArrayType, V>
        + super::SupportsRightMatMul<ArrayType, V>
        + super::SupportsMatrixTranspose<ArrayType, V>
        + super::SupportsReshape<ArrayType, V>,
{
    fn jvp(
        &self,
        engine: &E,
        context: &mut JvpContext<'_, V, E::LinearOperation>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
        match self {
            Self::Add => AddOperation.jvp(engine, context, inputs),
            Self::Mul => MulOperation.jvp(engine, context, inputs),
            Self::Neg => NegOperation.jvp(engine, context, inputs),
            Self::Sin => SinOperation.jvp(engine, context, inputs),
            Self::Cos => CosOperation.jvp(engine, context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(engine, context, inputs),
            Self::MatMul => MatMulOperation.jvp(engine, context, inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.jvp(engine, context, inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).jvp(engine, context, inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).jvp(engine, context, inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).jvp(engine, context, inputs)
            }
            Self::Rematerialize(remat) => remat.as_ref().jvp(engine, context, inputs),
            Self::Condition(condition) => condition.as_ref().jvp(engine, context, inputs),
            Self::While(while_operation) => while_operation.as_ref().jvp(engine, context, inputs),
            Self::Custom(op) => op.jvp(engine, context, inputs),
        }
    }
}

/// Linearization-engine dispatcher for [`PrimitiveOperation`] under the traced-linearization path.
///
/// Forwards each variant to the per-op JVP rule, picking up the
/// [`LinearizationEngine`](crate::tracing_v2::LinearizationEngine)-keyed impls for V-capturing
/// variants ([`Scale`](Self::Scale), [`LeftMatMul`](Self::LeftMatMul),
/// [`RightMatMul`](Self::RightMatMul)), the [`Rematerialize`](Self::Rematerialize) impl that
/// recurses via [`linearize_traced_program`](crate::tracing_v2::linear::linearize_traced_program),
/// the [`Condition`](Self::Condition) / [`While`](Self::While) stub impls (predicate extraction
/// does not work at trace time), and the [`Custom`](Self::Custom) bridge to the registered traced
/// linearization rule.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing_v2::LinearizationEngine<'engine, EInner>>
    for PrimitiveOperation<V>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::constants::Zero<ArrayType>
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    EInner: crate::tracing_v2::TracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>>
        + ?Sized
        + 'static,
    V::ParameterStructure: Clone + std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    LinearPrimitiveOperation<V>: super::SupportsAdd<ArrayType, V>
        + super::SupportsNeg<ArrayType, V>
        + super::SupportsScale<ArrayType, V>
        + super::SupportsLeftMatMul<ArrayType, V>
        + super::SupportsRightMatMul<ArrayType, V>
        + super::SupportsMatrixTranspose<ArrayType, V>
        + super::SupportsReshape<ArrayType, V>
        + Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, LinearPrimitiveOperation<V>>,
    Tracer<'engine, EInner>: Add<Output = Tracer<'engine, EInner>>
        + Mul<Output = Tracer<'engine, EInner>>
        + Neg<Output = Tracer<'engine, EInner>>
        + Sin
        + Cos
        + MatrixOps,
    for<'a> LinearPrimitiveOperation<Tracer<'a, EInner>>: Clone
        + InterpretableOperation<ArrayType, Tracer<'a, EInner>>
        + LinearOperation<ArrayType, Tracer<'a, EInner>, LinearPrimitiveOperation<Tracer<'a, EInner>>>,
{
    fn jvp(
        &self,
        engine: &crate::tracing_v2::LinearizationEngine<'engine, EInner>,
        context: &mut JvpContext<'_, Tracer<'engine, EInner>, LinearPrimitiveOperation<Tracer<'engine, EInner>>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
        match self {
            Self::Add => AddOperation.jvp(engine, context, inputs),
            Self::Mul => MulOperation.jvp(engine, context, inputs),
            Self::Neg => NegOperation.jvp(engine, context, inputs),
            Self::Sin => SinOperation.jvp(engine, context, inputs),
            Self::Cos => CosOperation.jvp(engine, context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(engine, context, inputs),
            Self::MatMul => MatMulOperation.jvp(engine, context, inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.jvp(engine, context, inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).jvp(engine, context, inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).jvp(engine, context, inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).jvp(engine, context, inputs)
            }
            Self::Rematerialize(remat) => remat.as_ref().jvp(engine, context, inputs),
            Self::Condition(condition) => condition.as_ref().jvp(engine, context, inputs),
            Self::While(while_operation) => while_operation.as_ref().jvp(engine, context, inputs),
            Self::Custom(op) => op.jvp(engine, context, inputs),
        }
    }
}

impl<'engine, V, EInner> crate::tracing_v2::linear::TracedLinearizableOperation<'engine, EInner>
    for PrimitiveOperation<V>
where
    V: Traceable<ArrayType> + Differentiable<ArrayType, Tangent = V> + Parameter + 'static,
    EInner: crate::tracing_v2::TracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>>
        + ?Sized
        + 'engine,
    PrimitiveOperation<V>: DifferentiableOperation<crate::tracing_v2::LinearizationEngine<'engine, EInner>>,
{
    fn jvp_traced_linearization(
        &self,
        engine: &crate::tracing_v2::LinearizationEngine<'engine, EInner>,
        context: &mut JvpContext<'_, Tracer<'engine, EInner>, LinearPrimitiveOperation<Tracer<'engine, EInner>>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
        <Self as DifferentiableOperation<crate::tracing_v2::LinearizationEngine<'engine, EInner>>>::jvp(
            self, engine, context, inputs,
        )
    }
}

#[cfg(test)]
mod tests {
    // Primitive-operation behavior is exercised through the per-operation modules and transform tests.
}
