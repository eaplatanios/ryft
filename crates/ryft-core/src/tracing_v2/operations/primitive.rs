use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::macros::check_count;
use crate::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, Scale, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use crate::operations::constants::{
    ONE_LIKE_OPERATION_NAME, One, OneLike, OneLikeOperation, OneOperation, SupportsOne, SupportsOneLike, SupportsZero,
    SupportsZeroLike, ZERO_LIKE_OPERATION_NAME, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::operations::trigonometric::{
    COS_OPERATION_NAME, Cos, CosOperation, SIN_OPERATION_NAME, Sin, SinOperation, SupportsCos, SupportsSin,
};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::engines::Tracer;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer, Tangent};
use crate::tracing_v2::operations::control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, WhileOperation,
};
use crate::tracing_v2::operations::left_matmul::left_matmul_abstract_eval;
use crate::tracing_v2::operations::right_matmul::right_matmul_abstract_eval;
use crate::tracing_v2::operations::{
    LeftMatMulOperation, MatMulOperation, MatrixTransposeOperation, ReshapeOperation, RightMatMulOperation,
};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine, MatrixOps};
use crate::types::{ArrayType, DataType, Shape, Type, TypeError, Typed};

use super::custom::{CustomPrimitive, LinearCustomPrimitive, SupportsCustom, SupportsLinearCustom};
use super::left_matmul::SupportsLeftMatMul;
use super::matmul::SupportsMatMul;
use super::matrix_transpose::SupportsMatrixTranspose;
use super::reshape::SupportsReshape;
use super::right_matmul::SupportsRightMatMul;

type ZeroScalarTangent = Tangent<DataType, Infallible>;
type ZeroArrayTangent = Tangent<ArrayType, Infallible>;

/// Default closed carrier for ordinary staged programs.
///
/// [`ArrayOperation`] is the reusable array operation enum for core tests and backend crates that do not need a fully
/// custom carrier. Most variants are thin tags around one semantic primitive defined elsewhere in [`super`]. The
/// [`Custom`](Self::Custom) variant is the explicit escape hatch for operations outside that default set, so the
/// carrier remains closed for normal dispatch while still allowing user- or backend-defined extensions.
#[derive(Clone, Debug)]
pub enum ArrayOperation<V, T>
where
    T: PartialEq + Type,
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

    /// Elementwise subtraction.
    Sub,

    /// Elementwise multiplication.
    Mul,

    /// Elementwise division.
    Div,

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
#[derive(Clone, Debug)]
pub enum LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    /// Typed zero with no inputs and one output, carrying a [`ZeroOperation`].
    ///
    /// Emitted by the transpose pass at the boundary of pullbacks for primal inputs that receive
    /// no cotangent contribution from any output. Interpreting it requires
    /// [`Zero<ArrayType>`](crate::operations::constants::Zero) on the value type;
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

    /// Elementwise subtraction.
    Sub,

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

    /// Higher-order conditional restricted to linear branch programs.
    Condition(Box<ConditionOperation<V, LinearArrayOperation<V, T>, T>>),

    /// Higher-order while loop restricted to linear condition and body programs.
    While(Box<WhileOperation<V, LinearArrayOperation<V, T>, T>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<T, V>>),
}

impl<T, V> LinearArrayOperation<V, T>
where
    T: PartialEq + Type + 'static,
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

impl<T, V> SupportsAdd<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        ArrayOperation::Add
    }
}

impl<T, V> SupportsSub<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sub_operation() -> Self {
        ArrayOperation::Sub
    }
}

impl<T, V> SupportsMul<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn mul_operation() -> Self {
        ArrayOperation::Mul
    }
}

impl<T, V> SupportsDiv<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn div_operation() -> Self {
        ArrayOperation::Div
    }
}

impl<T, V> SupportsNeg<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        ArrayOperation::Neg
    }
}

impl<T, V> SupportsSin<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sin_operation() -> Self {
        ArrayOperation::Sin
    }
}

impl<T, V> SupportsCos<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn cos_operation() -> Self {
        ArrayOperation::Cos
    }
}

impl<T, V> SupportsZero<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_operation(r#type: T) -> Self {
        ArrayOperation::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero_operation(&self) -> Option<&ZeroOperation<T>> {
        match self {
            Self::Zero(zero) => Some(zero),
            _ => None,
        }
    }
}

impl<T, V> SupportsOne<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        ArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V> SupportsZeroLike<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        ArrayOperation::ZeroLike
    }
}

impl<T, V> SupportsOneLike<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        ArrayOperation::OneLike
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsMatMul<ArrayType, V> for ArrayOperation<V, ArrayType> {
    #[inline]
    fn matmul_operation() -> Self {
        ArrayOperation::MatrixMultiply
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsMatrixTranspose<ArrayType, V> for ArrayOperation<V, ArrayType> {
    #[inline]
    fn matrix_transpose_operation() -> Self {
        ArrayOperation::Transpose
    }
}

impl<T, V> SupportsScale<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        ArrayOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsReshape<ArrayType, V> for ArrayOperation<V, ArrayType> {
    #[inline]
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        ArrayOperation::Reshape { input_shape, output_shape }
    }
}

impl<T, V> SupportsCustom<T, V> for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn custom_operation(primitive: Arc<CustomPrimitive<T, V>>) -> Self {
        ArrayOperation::Custom(primitive)
    }
}

impl<T, V> SupportsAdd<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        LinearArrayOperation::Add
    }
}

impl<T, V> SupportsSub<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sub_operation() -> Self {
        LinearArrayOperation::Sub
    }
}

impl<T, V> SupportsZero<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_operation(r#type: T) -> Self {
        LinearArrayOperation::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero_operation(&self) -> Option<&ZeroOperation<T>> {
        match self {
            Self::Zero(zero) => Some(zero),
            _ => None,
        }
    }
}

impl<T, V> SupportsOne<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        LinearArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V> SupportsZeroLike<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        LinearArrayOperation::ZeroLike
    }
}

impl<T, V> SupportsOneLike<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        LinearArrayOperation::OneLike
    }
}

impl<T, V> SupportsNeg<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        LinearArrayOperation::Neg
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsMatrixTranspose<ArrayType, V> for LinearArrayOperation<V, ArrayType> {
    #[inline]
    fn matrix_transpose_operation() -> Self {
        LinearArrayOperation::Transpose
    }
}

impl<T, V> SupportsScale<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        LinearArrayOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsLeftMatMul<ArrayType, V> for LinearArrayOperation<V, ArrayType> {
    #[inline]
    fn left_matmul_operation(factor: V) -> Self {
        LinearArrayOperation::LeftMatMul { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsRightMatMul<ArrayType, V> for LinearArrayOperation<V, ArrayType> {
    #[inline]
    fn right_matmul_operation(factor: V) -> Self {
        LinearArrayOperation::RightMatMul { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsReshape<ArrayType, V> for LinearArrayOperation<V, ArrayType> {
    #[inline]
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        LinearArrayOperation::Reshape { input_shape, output_shape }
    }
}

impl<V: Traceable<ArrayType> + Parameter> From<ConditionOperation<V, LinearArrayOperation<V, ArrayType>, ArrayType>>
    for LinearArrayOperation<V, ArrayType>
{
    #[inline]
    fn from(op: ConditionOperation<V, LinearArrayOperation<V, ArrayType>, ArrayType>) -> Self {
        LinearArrayOperation::Condition(Box::new(op))
    }
}

impl<T, V> SupportsLinearCustom<T, V> for LinearArrayOperation<V, T>
where
    T: PartialEq + Type + 'static,
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
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
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
            Self::Sin => SIN_OPERATION_NAME,
            Self::Cos => COS_OPERATION_NAME,
            Self::MatrixMultiply => "matmul",
            Self::Transpose => "matrix_transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::Reshape { .. } => "reshape",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<T, V> LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
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
            Self::Transpose => "matrix_transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<T, V> Display for ArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "{}{output_shape}", self.operation_name()),
            _ => write!(formatter, "{}", self.operation_name()),
        }
    }
}

impl<T, V> Display for LinearArrayOperation<V, T>
where
    T: PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "{}{output_shape}", self.operation_name()),
            _ => write!(formatter, "{}", self.operation_name()),
        }
    }
}

fn unsupported_scalar_metadata_operation(operation_name: &'static str) -> TypeError {
    TypeError { message: format!("{operation_name} is not supported for scalar data type metadata") }
}

fn unsupported_symbolic_zero_custom_interpretation(operation_name: &'static str) -> TypeError {
    TypeError { message: format!("symbolic-zero custom interpretation is not implemented for {operation_name}") }
}

fn unsupported_tangent_value_custom_interpretation(operation_name: &'static str) -> TypeError {
    TypeError { message: format!("mixed symbolic-zero custom interpretation is not implemented for {operation_name}") }
}

fn symbolic_zero_one_error<T: Type>(r#type: &T) -> TypeError {
    TypeError { message: format!("zero tangent space has no one value for {type}", type = r#type) }
}

fn infer_zero_only_tangent_output_types<T, O>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
) -> Result<Vec<T>, TracingError>
where
    T: Type,
    O: Operation<T>,
{
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    Ok(operation.infer_output_types(input_types.as_slice())?)
}

fn interpret_zero_only_tangent_operation<T, O>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
) -> Result<Vec<Tangent<T, Infallible>>, TracingError>
where
    T: Type,
    O: Operation<T>,
{
    Ok(infer_zero_only_tangent_output_types(operation, inputs)?.into_iter().map(Tangent::zero).collect())
}

fn reject_zero_only_tangent_one_operation<T, O>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
) -> Result<Vec<Tangent<T, Infallible>>, TracingError>
where
    T: Type,
    O: Operation<T>,
{
    let output_types = infer_zero_only_tangent_output_types(operation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    Err(symbolic_zero_one_error(&output_types[0]).into())
}

fn infer_tangent_value_output_types<T, V, O>(operation: &O, inputs: &[Tangent<T, V>]) -> Result<Vec<T>, TracingError>
where
    T: Type,
    V: Traceable<T>,
    O: Operation<T>,
{
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    Ok(operation.infer_output_types(input_types.as_slice())?)
}

fn symbolic_zero_tangent_value_outputs<T, V>(output_types: Vec<T>) -> Vec<Tangent<T, V>>
where
    T: Type,
    V: Traceable<T>,
{
    output_types.into_iter().map(Tangent::Zero).collect()
}

fn materialize_tangent_value<T, V>(value: &Tangent<T, V>) -> Result<V, TracingError>
where
    T: Type,
    V: Traceable<T> + Zero<T>,
{
    match value {
        Tangent::Zero(r#type) => V::zero(r#type),
        Tangent::NonZero(value) => Ok(value.clone()),
    }
}

fn materialize_tangent_value_inputs<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<V>, TracingError>
where
    T: Type,
    V: Traceable<T> + Zero<T>,
{
    inputs.iter().map(materialize_tangent_value).collect()
}

fn interpret_materialized_tangent_value_operation<T, V, O>(
    operation: &O,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Type,
    V: Traceable<T> + Zero<T>,
    O: InterpretableOperation<T, V>,
{
    Ok(operation
        .interpret(materialize_tangent_value_inputs(inputs)?.as_slice())?
        .into_iter()
        .map(Tangent::NonZero)
        .collect())
}

fn tangent_value_non_zero_type_matches<T, V>(value: &V, output_type: &T) -> bool
where
    T: PartialEq + Type,
    V: Traceable<T>,
{
    value.r#type().as_ref() == output_type
}

fn interpret_tangent_value_add<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: PartialEq + Type,
    V: Traceable<T> + Add<Output = V> + Zero<T>,
    AddOperation: Operation<T> + InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&AddOperation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    if inputs.iter().all(Tangent::is_zero) {
        return Ok(symbolic_zero_tangent_value_outputs(output_types));
    }
    let output_type = &output_types[0];
    match inputs {
        [Tangent::NonZero(value), Tangent::Zero(_)] if tangent_value_non_zero_type_matches(value, output_type) => {
            Ok(vec![Tangent::NonZero(value.clone())])
        }
        [Tangent::Zero(_), Tangent::NonZero(value)] if tangent_value_non_zero_type_matches(value, output_type) => {
            Ok(vec![Tangent::NonZero(value.clone())])
        }
        _ => interpret_materialized_tangent_value_operation(&AddOperation, inputs),
    }
}

fn interpret_tangent_value_sub<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: PartialEq + Type,
    V: Traceable<T> + Neg<Output = V> + Sub<Output = V> + Zero<T>,
    SubOperation: Operation<T> + InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&SubOperation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    if inputs.iter().all(Tangent::is_zero) {
        return Ok(symbolic_zero_tangent_value_outputs(output_types));
    }
    let output_type = &output_types[0];
    match inputs {
        [Tangent::NonZero(value), Tangent::Zero(_)] if tangent_value_non_zero_type_matches(value, output_type) => {
            Ok(vec![Tangent::NonZero(value.clone())])
        }
        [Tangent::Zero(_), Tangent::NonZero(value)] if tangent_value_non_zero_type_matches(value, output_type) => {
            Ok(vec![Tangent::NonZero(-value.clone())])
        }
        _ => interpret_materialized_tangent_value_operation(&SubOperation, inputs),
    }
}

fn interpret_tangent_value_neg<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Type,
    V: Traceable<T> + Neg<Output = V>,
    NegOperation: Operation<T> + InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&NegOperation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    match inputs {
        [Tangent::Zero(_)] => Ok(symbolic_zero_tangent_value_outputs(output_types)),
        [Tangent::NonZero(value)] => {
            Ok(NegOperation.interpret(std::slice::from_ref(value))?.into_iter().map(Tangent::NonZero).collect())
        }
        _ => unreachable!("neg output type inference validates the input count"),
    }
}

fn interpret_tangent_value_zero_like<T, V, O>(
    operation: &O,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Type,
    V: Traceable<T>,
    O: Operation<T>,
{
    Ok(symbolic_zero_tangent_value_outputs(infer_tangent_value_output_types(operation, inputs)?))
}

fn interpret_tangent_value_one_like<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Type,
    V: Traceable<T> + OneLike,
    OneLikeOperation: Operation<T>,
{
    let output_types = infer_tangent_value_output_types(&OneLikeOperation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    match inputs {
        [Tangent::Zero(r#type)] => Err(symbolic_zero_one_error(r#type).into()),
        [Tangent::NonZero(value)] => Ok(vec![Tangent::NonZero(value.one_like())]),
        _ => unreachable!("one_like output type inference validates the input count"),
    }
}

fn interpret_tangent_value_scale<T, V, O>(
    operation: &O,
    factor: &Tangent<T, V>,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Type,
    V: Traceable<T> + Scale<Output = V>,
    O: Operation<T>,
    ScaleOperation<T, V>: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(operation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    match inputs {
        [input] if factor.is_zero() || input.is_zero() => Ok(symbolic_zero_tangent_value_outputs(output_types)),
        [Tangent::NonZero(input)] => {
            let Tangent::NonZero(factor) = factor else {
                unreachable!("zero factors are handled before concrete scale interpretation")
            };
            Ok(ScaleOperation::new(factor.clone())
                .interpret(std::slice::from_ref(input))?
                .into_iter()
                .map(Tangent::NonZero)
                .collect())
        }
        _ => unreachable!("scale output type inference validates the input count"),
    }
}

fn interpret_tangent_value_unary_non_zero_or_zero<T, V, MetadataOperation, ConcreteOperation>(
    metadata_operation: &MetadataOperation,
    concrete_operation: &ConcreteOperation,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Type,
    V: Traceable<T>,
    MetadataOperation: Operation<T>,
    ConcreteOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(metadata_operation, inputs)?;
    check_count!("output", output_types, 1, TracingError);
    match inputs {
        [Tangent::Zero(_)] => Ok(symbolic_zero_tangent_value_outputs(output_types)),
        [Tangent::NonZero(input)] => Ok(concrete_operation
            .interpret(std::slice::from_ref(input))?
            .into_iter()
            .map(Tangent::NonZero)
            .collect()),
        _ => unreachable!("unary output type inference validates the input count"),
    }
}

impl<V: Traceable<ArrayType> + Parameter> Operation<ArrayType> for ArrayOperation<V, ArrayType> {
    #[inline]
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
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Div => DivOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::MatrixMultiply => MatMulOperation.infer_output_types(input_types),
            Self::Transpose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
            }
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
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for ArrayOperation<V, DataType> {
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
            Self::MatrixMultiply | Self::Transpose | Self::Reshape { .. } | Self::Condition(_) | Self::While(_) => {
                Err(unsupported_scalar_metadata_operation(self.operation_name()))
            }
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
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<ArrayType> + Parameter> Operation<ArrayType> for LinearArrayOperation<V, ArrayType> {
    #[inline]
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
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Transpose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
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
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for LinearArrayOperation<V, DataType> {
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
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
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
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType>> InterpretableOperation<DataType, V> for ScalarOperation<V>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Sin
        + Cos
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
    Vec<V>: Parameterized<
            V,
            Family: crate::parameters::ParameterizedFamily<V>,
            To<V> = Vec<V>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Div => <DivOperation as InterpretableOperation<DataType, V>>::interpret(&DivOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Sin => <SinOperation as InterpretableOperation<DataType, V>>::interpret(&SinOperation, inputs),
            Self::Cos => <CosOperation as InterpretableOperation<DataType, V>>::interpret(&CosOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl InterpretableOperation<DataType, ZeroScalarTangent> for LinearScalarOperation<ZeroScalarTangent> {
    fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, TracingError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Custom(op) => Err(unsupported_symbolic_zero_custom_interpretation(op.name()).into()),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Traceable<DataType>> InterpretableOperation<DataType, Tangent<DataType, V>>
    for LinearScalarOperation<Tangent<DataType, V>>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<DataType>
        + One<DataType>
        + OneLike,
{
    fn interpret(&self, inputs: &[Tangent<DataType, V>]) -> Result<Vec<Tangent<DataType, V>>, TracingError> {
        match self {
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type)]),
            Self::One(one) => Ok(vec![Tangent::NonZero(V::one(&one.r#type)?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Scale { factor } => interpret_tangent_value_scale(self, factor, inputs),
            Self::Custom(op) => Err(unsupported_tangent_value_custom_interpretation(op.name()).into()),
        }
    }
}

impl<V: Traceable<DataType>> InterpretableOperation<DataType, V> for LinearScalarOperation<V>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<'engine, E> InterpretableOperation<DataType, Tracer<'engine, E>> for LinearScalarOperation<Tracer<'engine, E>>
where
    E: DifferentiableTracingEngine<Type = DataType> + 'static,
    Tracer<'engine, E>: Add<Output = Tracer<'engine, E>>
        + Sub<Output = Tracer<'engine, E>>
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
                    &zero.r#type
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    &one.r#type
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&SubOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
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
impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, V> for ArrayOperation<V, ArrayType>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Sin
        + Cos
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Sub => SubOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Div => DivOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatrixMultiply => MatMulOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<V: Traceable<DataType>> InterpretableOperation<DataType, V> for ArrayOperation<V, DataType>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Sin
        + Cos
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Div => <DivOperation as InterpretableOperation<DataType, V>>::interpret(&DivOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Sin => <SinOperation as InterpretableOperation<DataType, V>>::interpret(&SinOperation, inputs),
            Self::Cos => <CosOperation as InterpretableOperation<DataType, V>>::interpret(&CosOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
            Self::MatrixMultiply | Self::Transpose | Self::Reshape { .. } | Self::Condition(_) | Self::While(_) => {
                Err(unsupported_scalar_metadata_operation(self.operation_name()).into())
            }
        }
    }
}

impl InterpretableOperation<ArrayType, ZeroArrayTangent> for LinearArrayOperation<ZeroArrayTangent, ArrayType> {
    fn interpret(&self, inputs: &[ZeroArrayTangent]) -> Result<Vec<ZeroArrayTangent>, TracingError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Condition(condition) => {
                let output_types = infer_zero_only_tangent_output_types(self, inputs)?;
                let branch = match condition.predicate {
                    ConditionPredicate::Captured(predicate) => {
                        if predicate {
                            &condition.true_branch
                        } else {
                            &condition.false_branch
                        }
                    }
                    ConditionPredicate::RuntimeInput(_) => {
                        return Err(ControlFlowError::MissingTransformRule {
                            transform: "runtime-predicate symbolic-zero condition interpretation",
                        }
                        .into());
                    }
                };
                let outputs = branch.interpret(inputs.to_vec())?;
                check_count!("output", outputs, output_types.len(), TracingError);
                Ok(outputs)
            }
            Self::While(while_operation) => {
                let output_types = infer_zero_only_tangent_output_types(self, inputs)?;
                let condition_outputs = while_operation.condition.interpret(inputs.to_vec())?;
                check_count!("output", condition_outputs, 1, TracingError);
                let outputs = while_operation.body.interpret(inputs.to_vec())?;
                check_count!("output", outputs, output_types.len(), TracingError);
                Ok(outputs)
            }
            Self::Custom(op) => Err(unsupported_symbolic_zero_custom_interpretation(op.name()).into()),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, Tangent<ArrayType, V>>
    for LinearArrayOperation<Tangent<ArrayType, V>, ArrayType>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
{
    fn interpret(&self, inputs: &[Tangent<ArrayType, V>]) -> Result<Vec<Tangent<ArrayType, V>>, TracingError> {
        match self {
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type.clone())]),
            Self::One(one) => Ok(vec![Tangent::NonZero(V::one(&one.r#type)?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Transpose => interpret_tangent_value_unary_non_zero_or_zero(
                &MatrixTransposeOperation,
                &MatrixTransposeOperation,
                inputs,
            ),
            Self::Scale { factor } => interpret_tangent_value_scale(self, factor, inputs),
            Self::LeftMatMul { factor } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, TracingError);
                match inputs {
                    [input] if factor.is_zero() || input.is_zero() => {
                        Ok(symbolic_zero_tangent_value_outputs(output_types))
                    }
                    [Tangent::NonZero(input)] => {
                        let Tangent::NonZero(factor) = factor else {
                            unreachable!("zero factors are handled before concrete left_matmul interpretation")
                        };
                        Ok(LeftMatMulOperation::new(factor.clone())
                            .interpret(std::slice::from_ref(input))?
                            .into_iter()
                            .map(Tangent::NonZero)
                            .collect())
                    }
                    _ => unreachable!("left_matmul output type inference validates the input count"),
                }
            }
            Self::RightMatMul { factor } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, TracingError);
                match inputs {
                    [input] if factor.is_zero() || input.is_zero() => {
                        Ok(symbolic_zero_tangent_value_outputs(output_types))
                    }
                    [Tangent::NonZero(input)] => {
                        let Tangent::NonZero(factor) = factor else {
                            unreachable!("zero factors are handled before concrete right_matmul interpretation")
                        };
                        Ok(RightMatMulOperation::new(factor.clone())
                            .interpret(std::slice::from_ref(input))?
                            .into_iter()
                            .map(Tangent::NonZero)
                            .collect())
                    }
                    _ => unreachable!("right_matmul output type inference validates the input count"),
                }
            }
            Self::Reshape { input_shape, output_shape } => interpret_tangent_value_unary_non_zero_or_zero(
                &ReshapeOperation::new(input_shape.clone(), output_shape.clone()),
                &ReshapeOperation::new(input_shape.clone(), output_shape.clone()),
                inputs,
            ),
            Self::Condition(condition) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let (predicate, operands) = match condition.predicate {
                    ConditionPredicate::RuntimeInput(_) => {
                        let predicate = match &inputs[0] {
                            Tangent::Zero(_) => {
                                return Err(ControlFlowError::MissingTransformRule {
                                    transform: "runtime-predicate mixed symbolic-zero condition interpretation",
                                }
                                .into());
                            }
                            Tangent::NonZero(predicate) => predicate.control_flow_predicate()?,
                        };
                        (predicate, &inputs[1..])
                    }
                    ConditionPredicate::Captured(predicate) => (predicate, inputs),
                };
                let branch = if predicate { &condition.true_branch } else { &condition.false_branch };
                let outputs = branch.interpret(operands.to_vec())?;
                check_count!("output", outputs, output_types.len(), TracingError);
                Ok(outputs)
            }
            Self::While(while_operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let mut state = inputs.to_vec();
                loop {
                    let condition_outputs = while_operation.condition.interpret(state.clone())?;
                    check_count!("output", condition_outputs, 1, TracingError);
                    let predicate = match &condition_outputs[0] {
                        Tangent::Zero(_) => {
                            return Err(ControlFlowError::MissingTransformRule {
                                transform: "mixed symbolic-zero while predicate interpretation",
                            }
                            .into());
                        }
                        Tangent::NonZero(predicate) => predicate.control_flow_predicate()?,
                    };
                    if !predicate {
                        check_count!("output", state, output_types.len(), TracingError);
                        return Ok(state);
                    }
                    state = while_operation.body.interpret(state)?;
                    check_count!("output", state, while_operation.state_types().len(), TracingError);
                }
            }
            Self::Custom(op) => Err(unsupported_tangent_value_custom_interpretation(op.name()).into()),
        }
    }
}

impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, V> for LinearArrayOperation<V, ArrayType>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Sub => SubOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl InterpretableOperation<DataType, ZeroScalarTangent> for LinearArrayOperation<ZeroScalarTangent, DataType> {
    fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, TracingError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Custom(op) => Err(unsupported_symbolic_zero_custom_interpretation(op.name()).into()),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Traceable<DataType>> InterpretableOperation<DataType, Tangent<DataType, V>>
    for LinearArrayOperation<Tangent<DataType, V>, DataType>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<DataType>
        + One<DataType>
        + OneLike,
{
    fn interpret(&self, inputs: &[Tangent<DataType, V>]) -> Result<Vec<Tangent<DataType, V>>, TracingError> {
        match self {
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type)]),
            Self::One(one) => Ok(vec![Tangent::NonZero(V::one(&one.r#type)?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Scale { factor } => interpret_tangent_value_scale(self, factor, inputs),
            Self::Custom(op) => Err(unsupported_tangent_value_custom_interpretation(op.name()).into()),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<V: Traceable<DataType>> InterpretableOperation<DataType, V> for LinearArrayOperation<V, DataType>
where
    V: Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<'engine, E> InterpretableOperation<ArrayType, Tracer<'engine, E>>
    for LinearArrayOperation<Tracer<'engine, E>, ArrayType>
where
    E: DifferentiableTracingEngine<Type = ArrayType>,
    Tracer<'engine, E>: Add<Output = Tracer<'engine, E>>
        + Sub<Output = Tracer<'engine, E>>
        + Neg<Output = Tracer<'engine, E>>
        + Mul<Output = Tracer<'engine, E>>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Vec<Tracer<'engine, E>>: Parameterized<
            Tracer<'engine, E>,
            To<Tracer<'engine, E>> = Vec<Tracer<'engine, E>>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    &zero.r#type
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    &one.r#type
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<ArrayType, Tracer<'engine, E>>>::interpret(
                &AddOperation,
                inputs,
            ),
            Self::Sub => <SubOperation as InterpretableOperation<ArrayType, Tracer<'engine, E>>>::interpret(
                &SubOperation,
                inputs,
            ),
            Self::Neg => <NegOperation as InterpretableOperation<ArrayType, Tracer<'engine, E>>>::interpret(
                &NegOperation,
                inputs,
            ),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<'engine, E> InterpretableOperation<DataType, Tracer<'engine, E>>
    for LinearArrayOperation<Tracer<'engine, E>, DataType>
where
    E: DifferentiableTracingEngine<Type = DataType> + 'static,
    Tracer<'engine, E>: Add<Output = Tracer<'engine, E>>
        + Sub<Output = Tracer<'engine, E>>
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
                    &zero.r#type
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    &one.r#type
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&SubOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::Custom(op) => op.interpret(inputs),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

fn transpose_array_type_metadata(r#type: &ArrayType) -> Result<ArrayType, TracingError> {
    let output_types = MatrixTransposeOperation.infer_output_types(&[r#type.clone()])?;
    check_count!("output", output_types, 1, TracingError);
    Ok(output_types[0].clone())
}

fn transpose_zero_only_tangent_array_metadata(factor: &ZeroArrayTangent) -> Result<ZeroArrayTangent, TracingError> {
    let factor_type = factor.r#type();
    Ok(Tangent::zero(transpose_array_type_metadata(factor_type.as_ref())?))
}

fn transpose_tangent_value_array_factor<V>(
    factor: &Tangent<ArrayType, V>,
) -> Result<Tangent<ArrayType, V>, TracingError>
where
    V: Traceable<ArrayType> + MatrixOps,
{
    match factor {
        Tangent::Zero(r#type) => Ok(Tangent::Zero(transpose_array_type_metadata(r#type)?)),
        Tangent::NonZero(value) => Ok(Tangent::NonZero(value.clone().transpose_matrix())),
    }
}

impl<V: Traceable<DataType>>
    LinearOperation<DataType, Tangent<DataType, V>, LinearScalarOperation<Tangent<DataType, V>>>
    for LinearScalarOperation<Tangent<DataType, V>>
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            DataType,
            Tangent<DataType, V>,
            LinearScalarOperation<Tangent<DataType, V>>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0], output_cotangents[0]])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let negated_outputs = context.stage(Self::Neg, &[atom])?;
                        check_count!("output", negated_outputs, 1, TracingError);
                        Ok(vec![Some(atom), Some(negated_outputs[0])])
                    }
                    None => Ok(vec![None, None]),
                }
            }
            Self::Neg | Self::Scale { .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(self.clone(), &[atom])?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Custom(_) => Err(TypeError {
                message: "custom scalar linear transpose requires a carrier-specific transpose rule".to_string(),
            }
            .into()),
        }
    }
}

impl LinearOperation<ArrayType, ZeroArrayTangent, LinearArrayOperation<ZeroArrayTangent, ArrayType>>
    for LinearArrayOperation<ZeroArrayTangent, ArrayType>
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            ZeroArrayTangent,
            LinearArrayOperation<ZeroArrayTangent, ArrayType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add | Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0], output_cotangents[0]])
            }
            Self::Neg | Self::Scale { .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0]])
            }
            Self::Transpose => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(Self::Transpose, &[atom])?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::LeftMatMul { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(
                            Self::LeftMatMul { factor: transpose_zero_only_tangent_array_metadata(factor)? },
                            &[atom],
                        )?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::RightMatMul { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(
                            Self::RightMatMul { factor: transpose_zero_only_tangent_array_metadata(factor)? },
                            &[atom],
                        )?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Reshape { input_shape, output_shape } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(
                            Self::Reshape { input_shape: output_shape.clone(), output_shape: input_shape.clone() },
                            &[atom],
                        )?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<ArrayType>>
    LinearOperation<ArrayType, Tangent<ArrayType, V>, LinearArrayOperation<Tangent<ArrayType, V>, ArrayType>>
    for LinearArrayOperation<Tangent<ArrayType, V>, ArrayType>
where
    V: MatrixOps,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, ArrayType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0], output_cotangents[0]])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let negated_outputs = context.stage(Self::Neg, &[atom])?;
                        check_count!("output", negated_outputs, 1, TracingError);
                        Ok(vec![Some(atom), Some(negated_outputs[0])])
                    }
                    None => Ok(vec![None, None]),
                }
            }
            Self::Neg | Self::Transpose | Self::Scale { .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(self.clone(), &[atom])?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::LeftMatMul { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(
                            Self::LeftMatMul { factor: transpose_tangent_value_array_factor(factor)? },
                            &[atom],
                        )?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::RightMatMul { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(
                            Self::RightMatMul { factor: transpose_tangent_value_array_factor(factor)? },
                            &[atom],
                        )?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Reshape { input_shape, output_shape } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(
                            Self::Reshape { input_shape: output_shape.clone(), output_shape: input_shape.clone() },
                            &[atom],
                        )?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<DataType>>
    LinearOperation<DataType, Tangent<DataType, V>, LinearArrayOperation<Tangent<DataType, V>, DataType>>
    for LinearArrayOperation<Tangent<DataType, V>, DataType>
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            DataType,
            Tangent<DataType, V>,
            LinearArrayOperation<Tangent<DataType, V>, DataType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0], output_cotangents[0]])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let negated_outputs = context.stage(Self::Neg, &[atom])?;
                        check_count!("output", negated_outputs, 1, TracingError);
                        Ok(vec![Some(atom), Some(negated_outputs[0])])
                    }
                    None => Ok(vec![None, None]),
                }
            }
            Self::Neg | Self::Scale { .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(self.clone(), &[atom])?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Custom(op) => op.transpose(context, output_cotangents),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<V: Traceable<DataType>> LinearOperation<DataType, V, LinearScalarOperation<V>> for LinearScalarOperation<V>
where
    V: Parameter + Add<Output = V> + Neg<Output = V> + ZeroLike + OneLike,
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
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0], output_cotangents[0]])
            }
            Self::Sub => SubOperation.transpose(context, output_cotangents),
            Self::Neg => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(Self::Neg, &[atom])?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Scale { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match output_cotangents[0] {
                    Some(atom) => {
                        let outputs = context.stage(Self::Scale { factor: factor.clone() }, &[atom])?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(vec![Some(outputs[0])])
                    }
                    None => Ok(vec![None]),
                }
            }
            Self::Custom(_) => Err(TypeError {
                message: "custom scalar linear transpose requires a carrier-specific transpose rule".to_string(),
            }
            .into()),
        }
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>>
    for LinearArrayOperation<V, ArrayType>
where
    V: Parameter
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            V,
            LinearArrayOperation<V, ArrayType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => AddOperation.transpose(context, output_cotangents),
            Self::Sub => SubOperation.transpose(context, output_cotangents),
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
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<DataType>> LinearOperation<DataType, V, LinearArrayOperation<V, DataType>>
    for LinearArrayOperation<V, DataType>
where
    V: Parameter + Add<Output = V> + Neg<Output = V> + Mul<Output = V> + ZeroLike + OneLike,
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
            Self::Sub => SubOperation.transpose(context, output_cotangents),
            Self::Neg => NegOperation.transpose(context, output_cotangents),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<F, E> DifferentiableOperation<E> for ScalarOperation<F>
where
    F: Traceable<DataType> + Parameter + Clone,
    E: DifferentiableEngine<Type = DataType>,
    E::Value: Add<Output = E::Value>
        + Sub<Output = E::Value>
        + Mul<Output = E::Value>
        + Div<Output = E::Value>
        + Neg<Output = E::Value>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Parameterized<E::Value>
        + Differentiable<DataType, Tangent = E::Tangent>,
    <E::Value as Parameterized<E::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<E::Value>: Parameterized<E::Value, ParameterStructure: std::fmt::Debug + PartialEq>,
    ScaleOperation<DataType, F>: DifferentiableOperation<E>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier: SupportsZeroLike<DataType, E::Tangent>
        + SupportsNeg<DataType, E::Tangent>
        + SupportsSub<DataType, E::Tangent>
        + SupportsScale<DataType, E::Tangent, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
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

impl<V: Value<ArrayType>, E> DifferentiableOperation<E> for ArrayOperation<V, ArrayType>
where
    V: Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
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
        + Differentiable<ArrayType, Tangent = E::Tangent>
        + 'static,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<
            V,
            Family: crate::parameters::ParameterizedFamily<E::Tangent>,
            To<E::Tangent> = Vec<E::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier: SupportsZeroLike<ArrayType, E::Tangent>
        + SupportsNeg<ArrayType, E::Tangent>
        + SupportsSub<ArrayType, E::Tangent>
        + SupportsScale<ArrayType, E::Tangent, V>
        + super::SupportsLeftMatMul<ArrayType, E::Tangent, V>
        + super::SupportsRightMatMul<ArrayType, E::Tangent, V>
        + super::SupportsMatrixTranspose<ArrayType, E::Tangent>
        + super::SupportsReshape<ArrayType, E::Tangent>,
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
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Condition(_) | Self::While(_) => {
                Err(TypeError { message: format!("{} does not support generic array jvp dispatch", self.name()) }
                    .into())
            }
            Self::Custom(op) => op.jvp(context, inputs),
        }
    }
}

impl<V: Value<DataType>, E> DifferentiableOperation<E> for ArrayOperation<V, DataType>
where
    V: Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + Differentiable<DataType, Tangent = E::Tangent>
        + 'static,
    E: DifferentiableEngine<Type = DataType, Value = V> + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier: SupportsZeroLike<DataType, E::Tangent>
        + SupportsNeg<DataType, E::Tangent>
        + SupportsSub<DataType, E::Tangent>
        + SupportsScale<DataType, E::Tangent, V>,
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
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply
            | Self::Transpose
            | Self::Reshape { .. }
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
/// [`Scale`](Self::Scale), the [`Condition`](Self::Condition) / [`While`](Self::While) stub impls
/// (predicate extraction does not work at trace time), and the [`Custom`](Self::Custom) bridge to
/// the registered traced linearization rule.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for ArrayOperation<V, ArrayType>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
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
        + Differentiable<ArrayType>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
        + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V, ArrayType>: Clone
        + SupportsZero<ArrayType, V>
        + SupportsZeroLike<ArrayType, V>
        + SupportsNeg<ArrayType, V>
        + SupportsAdd<ArrayType, V>
        + SupportsSub<ArrayType, V>
        + SupportsScale<ArrayType, V>
        + super::SupportsLeftMatMul<ArrayType, V>
        + super::SupportsRightMatMul<ArrayType, V>
        + super::SupportsMatrixTranspose<ArrayType, V>
        + super::SupportsReshape<ArrayType, V>
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
    Tracer<'engine, EInner>: Add<Output = Tracer<'engine, EInner>>
        + Sub<Output = Tracer<'engine, EInner>>
        + Mul<Output = Tracer<'engine, EInner>>
        + Div<Output = Tracer<'engine, EInner>>
        + Neg<Output = Tracer<'engine, EInner>>
        + Sin
        + Cos
        + MatrixOps
        + ZeroLike
        + OneLike,
    EInner::LinearOperationCarrier<'engine>: SupportsZeroLike<ArrayType, Tracer<'engine, EInner>>
        + SupportsSub<ArrayType, Tracer<'engine, EInner>>
        + SupportsLeftMatMul<ArrayType, Tracer<'engine, EInner>>
        + SupportsRightMatMul<ArrayType, Tracer<'engine, EInner>>
        + SupportsMatrixTranspose<ArrayType, Tracer<'engine, EInner>>
        + SupportsReshape<ArrayType, Tracer<'engine, EInner>>,
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
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
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
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + Differentiable<DataType>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = DataType, Value = V, OperationCarrier = ArrayOperation<V, DataType>>
        + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V, DataType>: Clone
        + SupportsZero<DataType, V>
        + SupportsZeroLike<DataType, V>
        + SupportsNeg<DataType, V>
        + SupportsAdd<DataType, V>
        + SupportsSub<DataType, V>
        + SupportsScale<DataType, V>
        + InterpretableOperation<DataType, V>
        + LinearOperation<DataType, V, LinearArrayOperation<V, DataType>>,
    Tracer<'engine, EInner>: Add<Output = Tracer<'engine, EInner>>
        + Sub<Output = Tracer<'engine, EInner>>
        + Mul<Output = Tracer<'engine, EInner>>
        + Div<Output = Tracer<'engine, EInner>>
        + Neg<Output = Tracer<'engine, EInner>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike,
    EInner::LinearOperationCarrier<'engine>:
        SupportsZeroLike<DataType, Tracer<'engine, EInner>> + SupportsSub<DataType, Tracer<'engine, EInner>>,
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
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply
            | Self::Transpose
            | Self::Reshape { .. }
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
    use pretty_assertions::assert_eq;

    use crate::operations::InterpretableOperation as _;
    use crate::parameters::Placeholder;
    use crate::tracing::transposition::{LinearOperation, TranspositionContext};
    use crate::tracing::{Program, ProgramBuilder};
    use crate::tracing_v2::test_util::TestArray;
    use crate::types::Size;

    use super::*;

    type ZeroArrayOperation = LinearArrayOperation<ZeroArrayTangent, ArrayType>;
    type ZeroArrayProgram =
        Program<ArrayType, ZeroArrayTangent, ZeroArrayOperation, Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>;
    type MixedScalar = Tangent<DataType, f64>;
    type MixedScalarOperation = LinearScalarOperation<MixedScalar>;
    type MixedArray = Tangent<ArrayType, TestArray>;
    type MixedArrayOperation = LinearArrayOperation<MixedArray, ArrayType>;

    fn array_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(dimensions.iter().copied().map(Size::Static).collect()), None, None)
            .unwrap()
    }

    fn f64_array_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Size::Static).collect()), None, None)
            .unwrap()
    }

    fn identity_zero_array_program(input_type: ArrayType) -> ZeroArrayProgram {
        let mut builder = ProgramBuilder::<ArrayType, ZeroArrayTangent, ZeroArrayOperation>::new();
        let input = builder.add_input(input_type);
        builder
            .build::<Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    fn one_zero_array_program(input_type: ArrayType, output_type: ArrayType) -> ZeroArrayProgram {
        let mut builder = ProgramBuilder::<ArrayType, ZeroArrayTangent, ZeroArrayOperation>::new();
        builder.add_input(input_type);
        let output =
            builder.add_instruction(ZeroArrayOperation::One(OneOperation::new(output_type)), vec![]).unwrap()[0];
        builder
            .build::<Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    fn zero_bool_condition_program(state_type: ArrayType) -> ZeroArrayProgram {
        let mut builder = ProgramBuilder::<ArrayType, ZeroArrayTangent, ZeroArrayOperation>::new();
        builder.add_input(state_type);
        let output = builder
            .add_instruction(ZeroArrayOperation::Zero(ZeroOperation::new(ArrayType::scalar(DataType::Boolean))), vec![])
            .unwrap()[0];
        builder
            .build::<Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    #[test]
    fn test_linear_scalar_zero_only_tangent_interpretation_uses_inferred_metadata() {
        let tangent = Tangent::zero(DataType::F32);
        let add = LinearScalarOperation::<ZeroScalarTangent>::Add;
        let neg = LinearScalarOperation::<ZeroScalarTangent>::Neg;
        let zero = LinearScalarOperation::<ZeroScalarTangent>::Zero(ZeroOperation::new(DataType::F32));
        let one = LinearScalarOperation::<ZeroScalarTangent>::One(OneOperation::new(DataType::F32));
        let one_like = LinearScalarOperation::<ZeroScalarTangent>::OneLike;

        assert_eq!(add.interpret(&[tangent.clone(), tangent.clone()]), Ok(vec![tangent.clone()]));
        assert_eq!(neg.interpret(std::slice::from_ref(&tangent)), Ok(vec![tangent.clone()]));
        assert_eq!(zero.interpret(&[]), Ok(vec![tangent.clone()]));
        assert_eq!(one.interpret(&[]).unwrap_err().to_string(), "zero tangent space has no one value for f32");
        assert_eq!(
            one_like.interpret(std::slice::from_ref(&tangent)).unwrap_err().to_string(),
            "zero tangent space has no one value for f32"
        );
    }

    #[test]
    fn test_linear_array_zero_only_tangent_program_propagates_metadata() {
        let input_type = array_type(&[2, 3]);
        let reshaped_type = array_type(&[3, 2]);
        let mut builder = ProgramBuilder::<ArrayType, ZeroArrayTangent, ZeroArrayOperation>::new();
        let input = builder.add_input(input_type.clone());
        let reshaped = builder
            .add_instruction(
                ZeroArrayOperation::Reshape {
                    input_shape: input_type.shape.clone(),
                    output_shape: reshaped_type.shape.clone(),
                },
                vec![input],
            )
            .unwrap()[0];
        let transposed = builder.add_instruction(ZeroArrayOperation::Transpose, vec![reshaped]).unwrap()[0];
        let negated = builder.add_instruction(ZeroArrayOperation::Neg, vec![transposed]).unwrap()[0];
        let output = builder.add_instruction(ZeroArrayOperation::Add, vec![negated, input]).unwrap()[0];
        let program = builder
            .build::<Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(program.interpret(vec![Tangent::zero(input_type.clone())]), Ok(vec![Tangent::zero(input_type)]));
    }

    #[test]
    fn test_linear_array_zero_only_tangent_matmul_metadata() {
        let input_type = array_type(&[2, 3]);
        let right_factor_type = array_type(&[3, 4]);
        let right_matmul = ZeroArrayOperation::RightMatMul { factor: Tangent::zero(right_factor_type) };

        assert_eq!(
            right_matmul.interpret(&[Tangent::zero(input_type.clone())]),
            Ok(vec![Tangent::zero(array_type(&[2, 4]))])
        );

        let left_factor_type = array_type(&[4, 2]);
        let left_matmul = ZeroArrayOperation::LeftMatMul { factor: Tangent::zero(left_factor_type) };

        assert_eq!(left_matmul.interpret(&[Tangent::zero(input_type)]), Ok(vec![Tangent::zero(array_type(&[4, 3]))]));
    }

    #[test]
    fn test_linear_array_zero_only_tangent_control_flow_interprets_nested_programs() {
        let state_type = array_type(&[2, 3]);
        let true_branch = identity_zero_array_program(state_type.clone());
        let false_branch = one_zero_array_program(state_type.clone(), state_type.clone());
        let condition = ZeroArrayOperation::Condition(Box::new(
            ConditionOperation::with_captured_predicate(true, true_branch.clone(), false_branch.clone()).unwrap(),
        ));

        assert_eq!(
            condition.interpret(&[Tangent::zero(state_type.clone())]),
            Ok(vec![Tangent::zero(state_type.clone())])
        );

        let condition = ZeroArrayOperation::Condition(Box::new(
            ConditionOperation::with_captured_predicate(false, true_branch, false_branch).unwrap(),
        ));
        assert_eq!(
            condition.interpret(&[Tangent::zero(state_type.clone())]).unwrap_err().to_string(),
            format!("zero tangent space has no one value for {state_type}")
        );

        let while_operation = ZeroArrayOperation::While(Box::new(
            WhileOperation::new(
                zero_bool_condition_program(state_type.clone()),
                identity_zero_array_program(state_type.clone()),
            )
            .unwrap(),
        ));

        assert_eq!(
            while_operation.interpret(&[Tangent::zero(state_type.clone())]),
            Ok(vec![Tangent::zero(state_type)])
        );
    }

    #[test]
    fn test_linear_scalar_tangent_value_interpretation_mixes_non_zero_and_zero() {
        let non_zero = MixedScalar::non_zero(3.0);
        let zero = MixedScalar::zero(DataType::F64);

        assert_eq!(MixedScalarOperation::Add.interpret(&[non_zero.clone(), zero.clone()]), Ok(vec![non_zero.clone()]));
        assert_eq!(MixedScalarOperation::Add.interpret(&[zero.clone(), non_zero.clone()]), Ok(vec![non_zero.clone()]));
        assert_eq!(
            MixedScalarOperation::Sub.interpret(&[zero.clone(), non_zero.clone()]),
            Ok(vec![MixedScalar::non_zero(-3.0)])
        );
        assert_eq!(
            (MixedScalarOperation::Scale { factor: MixedScalar::zero(DataType::F64) })
                .interpret(std::slice::from_ref(&non_zero)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            (MixedScalarOperation::Scale { factor: MixedScalar::non_zero(2.0) }).interpret(std::slice::from_ref(&zero)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            (MixedScalarOperation::Scale { factor: MixedScalar::non_zero(2.0) })
                .interpret(std::slice::from_ref(&non_zero)),
            Ok(vec![MixedScalar::non_zero(6.0)])
        );
        assert_eq!(MixedScalarOperation::ZeroLike.interpret(std::slice::from_ref(&non_zero)), Ok(vec![zero.clone()]));
        assert_eq!(
            MixedScalarOperation::One(OneOperation::new(DataType::F64)).interpret(&[]),
            Ok(vec![MixedScalar::non_zero(1.0)])
        );
        assert_eq!(
            MixedScalarOperation::OneLike.interpret(std::slice::from_ref(&zero)).unwrap_err().to_string(),
            "zero tangent space has no one value for f64"
        );
    }

    #[test]
    fn test_linear_array_tangent_value_interpretation_preserves_symbolic_zero_metadata() {
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input_zero = MixedArray::zero(input.r#type.clone());

        assert_eq!(
            MixedArrayOperation::Add.interpret(&[MixedArray::non_zero(input.clone()), input_zero.clone()]),
            Ok(vec![MixedArray::non_zero(input.clone())])
        );
        assert_eq!(MixedArrayOperation::Neg.interpret(std::slice::from_ref(&input_zero)), Ok(vec![input_zero.clone()]));

        let reshaped_type = f64_array_type(&[3, 2]);
        assert_eq!(
            (MixedArrayOperation::Reshape {
                input_shape: input.r#type.shape.clone(),
                output_shape: reshaped_type.shape.clone(),
            })
            .interpret(std::slice::from_ref(&input_zero)),
            Ok(vec![MixedArray::zero(reshaped_type.clone())])
        );

        let left_factor_type = f64_array_type(&[4, 2]);
        assert_eq!(
            (MixedArrayOperation::LeftMatMul { factor: MixedArray::zero(left_factor_type) })
                .interpret(&[MixedArray::non_zero(input.clone())]),
            Ok(vec![MixedArray::zero(f64_array_type(&[4, 3]))])
        );

        let right_factor = TestArray::matrix(3, 4, vec![0.0; 12]);
        assert_eq!(
            (MixedArrayOperation::RightMatMul { factor: MixedArray::non_zero(right_factor) })
                .interpret(std::slice::from_ref(&input_zero)),
            Ok(vec![MixedArray::zero(f64_array_type(&[2, 4]))])
        );
    }

    #[test]
    fn test_linear_scalar_tangent_value_program_supports_nested_structured_parameters() {
        let mut builder = ProgramBuilder::<DataType, MixedScalar, MixedScalarOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let sum = builder.add_instruction(MixedScalarOperation::Add, vec![left, right]).unwrap()[0];
        let difference = builder.add_instruction(MixedScalarOperation::Sub, vec![right, left]).unwrap()[0];
        let scaled = builder
            .add_instruction(MixedScalarOperation::Scale { factor: MixedScalar::zero(DataType::F64) }, vec![sum])
            .unwrap()[0];
        let program = builder
            .build::<(MixedScalar, MixedScalar), (MixedScalar, (MixedScalar, MixedScalar))>(
                vec![sum, difference, scaled],
                (Placeholder, Placeholder),
                (Placeholder, (Placeholder, Placeholder)),
            )
            .unwrap();

        assert_eq!(
            program.interpret((MixedScalar::non_zero(2.0), MixedScalar::zero(DataType::F64))),
            Ok((MixedScalar::non_zero(2.0), (MixedScalar::non_zero(-2.0), MixedScalar::zero(DataType::F64))))
        );
    }

    #[derive(Clone, Debug)]
    struct TestCustomOperation;

    impl Operation<DataType> for TestCustomOperation {
        fn name(&self) -> &'static str {
            "test_custom_zero"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 0, TypeError);
            Ok(vec![DataType::F32])
        }
    }

    impl InterpretableOperation<DataType, ZeroScalarTangent> for TestCustomOperation {
        fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, TracingError> {
            interpret_zero_only_tangent_operation(self, inputs)
        }
    }

    impl InterpretableOperation<DataType, Tangent<DataType, f32>> for TestCustomOperation {
        fn interpret(&self, inputs: &[Tangent<DataType, f32>]) -> Result<Vec<Tangent<DataType, f32>>, TracingError> {
            Ok(symbolic_zero_tangent_value_outputs(infer_tangent_value_output_types(self, inputs)?))
        }
    }

    impl LinearOperation<DataType, Tangent<DataType, f32>, LinearArrayOperation<Tangent<DataType, f32>, DataType>>
        for TestCustomOperation
    {
        fn transpose(
            &self,
            _context: &mut TranspositionContext<
                DataType,
                Tangent<DataType, f32>,
                LinearArrayOperation<Tangent<DataType, f32>, DataType>,
            >,
            _output_cotangents: &[Option<AtomId>],
        ) -> Result<Vec<Option<AtomId>>, TracingError> {
            Ok(Vec::new())
        }
    }

    impl LinearOperation<DataType, ZeroScalarTangent, LinearArrayOperation<ZeroScalarTangent, DataType>>
        for TestCustomOperation
    {
        fn transpose(
            &self,
            _context: &mut TranspositionContext<
                DataType,
                ZeroScalarTangent,
                LinearArrayOperation<ZeroScalarTangent, DataType>,
            >,
            _output_cotangents: &[Option<AtomId>],
        ) -> Result<Vec<Option<AtomId>>, TracingError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn test_linear_scalar_zero_only_tangent_custom_interpretation_is_explicitly_unsupported() {
        let custom = LinearScalarOperation::<ZeroScalarTangent>::custom(
            CustomPrimitive::new(TestCustomOperation).with_transpose_rule(TestCustomOperation),
        )
        .unwrap();

        assert_eq!(
            custom.interpret(&[]).unwrap_err().to_string(),
            "symbolic-zero custom interpretation is not implemented for test_custom_zero"
        );
    }

    #[test]
    fn test_linear_scalar_tangent_value_custom_interpretation_is_explicitly_unsupported() {
        let primitive: CustomPrimitive<DataType, Tangent<DataType, f32>> =
            CustomPrimitive::new(TestCustomOperation).with_transpose_rule(TestCustomOperation);
        let custom = LinearScalarOperation::<Tangent<DataType, f32>>::custom(primitive).unwrap();

        assert_eq!(
            custom.interpret(&[]).unwrap_err().to_string(),
            "mixed symbolic-zero custom interpretation is not implemented for test_custom_zero"
        );
    }
}
