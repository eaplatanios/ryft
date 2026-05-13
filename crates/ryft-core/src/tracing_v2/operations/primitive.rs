//! Reusable staged-operation carriers for built-in primitives and backend extensions.
//!
//! [`ArrayOperation`] and [`LinearArrayOperation`] contain the core operations implemented by `ryft-core` plus an
//! optional statically typed backend extension slot. A backend that needs additional operations should define an
//! ordinary extension enum, define a linear extension enum when it has linear-only operations, implement the standard
//! operation traits for those enums, and select `ArrayOperation<Value, Type, Extension>` and
//! `LinearArrayOperation<Tangent, Type, LinearExtension>` as its tracing carriers.
//!
//! `ryft-core` intentionally does not expose a universal dynamic custom-operation primitive. Backend-specific or
//! user-defined operations should be represented by a backend extension variant, so transform, interpretation, and
//! lowering rules remain statically typed and owned by the backend that understands the operation.

use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::differentiation::{Cotangent, LinearOperation, Tangent};
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
use crate::tracing::domains::{RuntimeDomain, Tracer, TracingContext, TracingDomain};
use crate::tracing::{ProgramTracingContext, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::operations::control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, WhileOperation,
};
use crate::tracing_v2::operations::dot::{LeftDot, RightDot, SupportsLeftDot, SupportsRightDot};
use crate::tracing_v2::operations::select::{Select, SelectOperation};
use crate::tracing_v2::operations::transpose::Transpose;
use crate::tracing_v2::operations::{
    DotDimensionNumbers, DotOperation, ReshapeOperation, SupportsDot, SupportsTranspose, TransposeOperation,
};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation, DifferentiableTracingDomain};
use crate::types::{ArrayType, DataType, Shape, Type, TypeError, Typed};

use super::reshape::{Reshape, SupportsReshape};

type ZeroScalarTangent = Tangent<DataType, Infallible>;
type ZeroArrayTangent = Tangent<ArrayType, Infallible>;

/// Concrete value types whose ordinary programs can be replayed with traced leaves.
///
/// This marker is used for carrier compositions such as `ArrayOperation<ConcreteValue, Type, Extension>` when a
/// transform needs to reinterpret the same staged operation with [`Tracer`] leaves from a domain whose concrete value
/// type is `ConcreteValue`. Core primitives are replayed by staging the same operation in the traced context, while the
/// backend extension remains responsible for interpreting its own extension variants over traced leaves.
pub trait TracerReplayValue<T: Type>: Traceable<T> + Parameter {}

/// Uninhabited operation-extension type for carriers that only contain the built-in operation set.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NoOperationExtension {}

/// Reusable carrier for ordinary staged programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates. Most variants are thin tags
/// around one semantic primitive defined elsewhere in [`super`]. The [`Extension`](Self::Extension) variant lets
/// backends statically compose their own operation enum into the same carrier without dynamic custom-operation
/// registries. Backends that only need built-in operations can omit the `Extension` parameter and use the
/// [`NoOperationExtension`] default.
#[derive(Clone, Debug)]
pub enum ArrayOperation<V, T, Extension = NoOperationExtension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
    Extension: Clone,
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

    /// Generalized dot product (tensor contraction).
    ///
    /// Lowers to StableHLO's `dot_general` op in the XLA backend. The dimension numbers
    /// describe contracting and batching axes for the two operands. See
    /// [`DotDimensionNumbers`] for the convention.
    Dot {
        /// Contracting and batching dimensions for the two operands.
        dimensions: DotDimensionNumbers,
    },

    /// N-dimensional axis permutation.
    ///
    /// Reorders the operand's axes according to `permutation`, which must be a permutation of
    /// `0..rank(input)`. Lowers to StableHLO's `transpose` op in the XLA backend.
    Transpose {
        /// Permutation of input axes.
        permutation: Vec<usize>,
    },

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Reshape from one shape to another.
    Reshape { input_shape: Shape, output_shape: Shape },

    /// Per-element select between two values driven by a predicate.
    ///
    /// Inputs are `(predicate, on_true, on_false)`, each with the same shape. The output's `i`-th
    /// element is `on_true`'s `i`-th element when the predicate's `i`-th element is logically
    /// true, and `on_false`'s otherwise. Lowers to StableHLO's `select` op in the XLA backend.
    Select,

    /// Higher-order conditional carrying true and false branch programs.
    Condition(Box<ConditionOperation<V, ArrayOperation<V, T, Extension>, T>>),

    /// Higher-order while loop carrying condition and body programs.
    While(Box<WhileOperation<V, ArrayOperation<V, T, Extension>, T>>),

    /// Backend-owned extension operation.
    Extension(Extension),
}

/// Reusable carrier for staged linear programs.
///
/// [`LinearArrayOperation`] is the linear-program sibling of [`ArrayOperation`]. It contains
/// operations that can appear in tangent and cotangent programs, including captured-factor linear
/// maps such as [`LeftDot`](Self::LeftDot) and [`RightDot`](Self::RightDot), and the
/// linearized higher-order operations needed by rematerialization and control flow. The
/// [`Extension`](Self::Extension) variant lets backends statically compose linear backend operations into the same
/// carrier. Backends that only need built-in linear operations can omit the `Extension` parameter and use the
/// [`NoOperationExtension`] default.
#[derive(Clone, Debug)]
pub enum LinearArrayOperation<V, T, Extension = NoOperationExtension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
    Extension: Clone,
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

    /// N-dimensional axis permutation; linear-side analogue of [`ArrayOperation::Transpose`].
    Transpose {
        /// Permutation of input axes.
        permutation: Vec<usize>,
    },

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Captured-factor left dot: linear map `t ↦ dot(factor, t; dimensions)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::Dot`] when the LHS primal is held
    /// constant.
    LeftDot {
        /// Captured constant factor (LHS of the underlying dot).
        factor: V,

        /// Dimension numbers of the underlying dot.
        dimensions: DotDimensionNumbers,
    },

    /// Captured-factor right dot: linear map `t ↦ dot(t, factor; dimensions)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::Dot`] when the RHS primal is held
    /// constant.
    RightDot {
        /// Captured constant factor (RHS of the underlying dot).
        factor: V,

        /// Dimension numbers of the underlying dot.
        dimensions: DotDimensionNumbers,
    },

    /// Reshape from one shape to another.
    Reshape { input_shape: Shape, output_shape: Shape },

    /// Higher-order conditional restricted to linear branch programs.
    Condition(Box<ConditionOperation<V, LinearArrayOperation<V, T, Extension>, T>>),

    /// Higher-order while loop restricted to linear condition and body programs.
    While(Box<WhileOperation<V, LinearArrayOperation<V, T, Extension>, T>>),

    /// Backend-owned linear extension operation.
    Extension(Extension),
}

impl Display for NoOperationExtension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = formatter;
        match *self {}
    }
}

impl<T: Type> Operation<T> for NoOperationExtension {
    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        match *self {}
    }
}

impl<T: Type, V: Typed<T>> InterpretableOperation<T, V> for NoOperationExtension {
    fn interpret(&self, _inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match *self {}
    }
}

impl<T, V, O> LinearOperation<T, V, O> for NoOperationExtension
where
    T: Parameter + Type,
    V: Traceable<T>,
    O: Operation<T>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError> {
        match *self {}
    }
}

impl<D: DifferentiableDomain> DifferentiableOperation<D> for NoOperationExtension {
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        _inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        match *self {}
    }
}

impl<T, V, Extension: Clone> SupportsAdd<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        ArrayOperation::Add
    }
}

impl<T, V, Extension: Clone> SupportsSub<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sub_operation() -> Self {
        ArrayOperation::Sub
    }
}

impl<T, V, Extension: Clone> SupportsMul<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn mul_operation() -> Self {
        ArrayOperation::Mul
    }
}

impl<T, V, Extension: Clone> SupportsDiv<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn div_operation() -> Self {
        ArrayOperation::Div
    }
}

impl<T, V, Extension: Clone> SupportsNeg<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        ArrayOperation::Neg
    }
}

impl<T, V, Extension: Clone> SupportsSin<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sin_operation() -> Self {
        ArrayOperation::Sin
    }
}

impl<T, V, Extension: Clone> SupportsCos<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn cos_operation() -> Self {
        ArrayOperation::Cos
    }
}

impl<T, V, Extension: Clone> SupportsZero<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
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

impl<T, V, Extension: Clone> SupportsOne<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        ArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V, Extension: Clone> SupportsZeroLike<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        ArrayOperation::ZeroLike
    }
}

impl<T, V, Extension: Clone> SupportsOneLike<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        ArrayOperation::OneLike
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> SupportsDot<ArrayType, V>
    for ArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn dot_operation(dimensions: DotDimensionNumbers) -> Self {
        ArrayOperation::Dot { dimensions }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> SupportsTranspose<ArrayType, V>
    for ArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn transpose_operation(permutation: Vec<usize>) -> Self {
        ArrayOperation::Transpose { permutation }
    }
}

impl<T, V, Extension: Clone> SupportsScale<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        ArrayOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> SupportsReshape<ArrayType, V>
    for ArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        ArrayOperation::Reshape { input_shape, output_shape }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone>
    crate::tracing_v2::operations::select::SupportsSelect<ArrayType, V> for ArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn select_operation() -> Self {
        ArrayOperation::Select
    }
}

impl<T, V, Extension: Clone> SupportsAdd<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        LinearArrayOperation::Add
    }
}

impl<T, V, Extension: Clone> SupportsSub<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sub_operation() -> Self {
        LinearArrayOperation::Sub
    }
}

impl<T, V, Extension: Clone> SupportsZero<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
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

impl<T, V, Extension: Clone> SupportsOne<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        LinearArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V, Extension: Clone> SupportsZeroLike<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        LinearArrayOperation::ZeroLike
    }
}

impl<T, V, Extension: Clone> SupportsOneLike<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        LinearArrayOperation::OneLike
    }
}

impl<T, V, Extension: Clone> SupportsNeg<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        LinearArrayOperation::Neg
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> SupportsTranspose<ArrayType, V>
    for LinearArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn transpose_operation(permutation: Vec<usize>) -> Self {
        LinearArrayOperation::Transpose { permutation }
    }
}

impl<T, V, Extension: Clone> SupportsScale<T, V> for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        LinearArrayOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> super::dot::SupportsLeftDot<ArrayType, V, V>
    for LinearArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn left_dot_operation(factor: V, dimensions: DotDimensionNumbers) -> Self {
        LinearArrayOperation::LeftDot { factor, dimensions }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> super::dot::SupportsRightDot<ArrayType, V, V>
    for LinearArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn right_dot_operation(factor: V, dimensions: DotDimensionNumbers) -> Self {
        LinearArrayOperation::RightDot { factor, dimensions }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone> SupportsReshape<ArrayType, V>
    for LinearArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        LinearArrayOperation::Reshape { input_shape, output_shape }
    }
}

impl<V: Traceable<ArrayType> + Parameter, Extension: Clone>
    From<ConditionOperation<V, LinearArrayOperation<V, ArrayType, Extension>, ArrayType>>
    for LinearArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn from(op: ConditionOperation<V, LinearArrayOperation<V, ArrayType, Extension>, ArrayType>) -> Self {
        LinearArrayOperation::Condition(Box::new(op))
    }
}

impl<T, V, Extension: Clone> ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
    Extension: Operation<T>,
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
            Self::Dot { .. } => "dot",
            Self::Transpose { .. } => "transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::Reshape { .. } => "reshape",
            Self::Select => "select",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::Extension(extension) => extension.name(),
        }
    }
}

impl<T, V, Extension: Clone> LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
    Extension: Operation<T>,
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
            Self::Transpose { .. } => "transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::LeftDot { .. } => "left_dot",
            Self::RightDot { .. } => "right_dot",
            Self::Reshape { .. } => "reshape",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::Extension(extension) => extension.name(),
        }
    }
}

impl<T, V, Extension: Clone> Display for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
    Extension: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "{}{output_shape}", self.operation_name()),
            _ => write!(formatter, "{}", self.operation_name()),
        }
    }
}

impl<T, V, Extension: Clone> Display for LinearArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T> + Parameter,
    Extension: Operation<T>,
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
        Tangent::Value(value) => Ok(value.clone()),
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
        .map(Tangent::Value)
        .collect())
}

fn tangent_value_type_matches<T, V>(value: &V, output_type: &T) -> bool
where
    T: Parameter + PartialEq + Type,
    V: Traceable<T>,
{
    value.r#type().as_ref() == output_type
}

fn interpret_tangent_value_add<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Parameter + PartialEq + Type,
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
        [Tangent::Value(value), Tangent::Zero(_)] if tangent_value_type_matches(value, output_type) => {
            Ok(vec![Tangent::Value(value.clone())])
        }
        [Tangent::Zero(_), Tangent::Value(value)] if tangent_value_type_matches(value, output_type) => {
            Ok(vec![Tangent::Value(value.clone())])
        }
        _ => interpret_materialized_tangent_value_operation(&AddOperation, inputs),
    }
}

fn interpret_tangent_value_sub<T, V>(inputs: &[Tangent<T, V>]) -> Result<Vec<Tangent<T, V>>, TracingError>
where
    T: Parameter + PartialEq + Type,
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
        [Tangent::Value(value), Tangent::Zero(_)] if tangent_value_type_matches(value, output_type) => {
            Ok(vec![Tangent::Value(value.clone())])
        }
        [Tangent::Zero(_), Tangent::Value(value)] if tangent_value_type_matches(value, output_type) => {
            Ok(vec![Tangent::Value(-value.clone())])
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
        [Tangent::Value(value)] => {
            Ok(NegOperation.interpret(std::slice::from_ref(value))?.into_iter().map(Tangent::Value).collect())
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
        [Tangent::Value(value)] => Ok(vec![Tangent::Value(value.one_like())]),
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
        [Tangent::Value(input)] => {
            let Tangent::Value(factor) = factor else {
                unreachable!("zero factors are handled before concrete scale interpretation")
            };
            Ok(ScaleOperation::new(factor.clone())
                .interpret(std::slice::from_ref(input))?
                .into_iter()
                .map(Tangent::Value)
                .collect())
        }
        _ => unreachable!("scale output type inference validates the input count"),
    }
}

fn interpret_tangent_value_unary_value_or_zero<T, V, MetadataOperation, ConcreteOperation>(
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
        [Tangent::Value(input)] => {
            Ok(concrete_operation.interpret(std::slice::from_ref(input))?.into_iter().map(Tangent::Value).collect())
        }
        _ => unreachable!("unary output type inference validates the input count"),
    }
}

impl<V, Extension> Operation<ArrayType> for ArrayOperation<V, ArrayType, Extension>
where
    V: Traceable<ArrayType> + Parameter,
    Extension: Clone + Operation<ArrayType>,
{
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
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).infer_output_types(input_types),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).infer_output_types(input_types)
            }
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
            }
            Self::Select => SelectOperation.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::Extension(extension) => extension.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).render(formatter, indentation),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).render(formatter, indentation)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V, Extension> Operation<DataType> for ArrayOperation<V, DataType, Extension>
where
    V: Traceable<DataType> + Parameter,
    Extension: Clone + Operation<DataType>,
{
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
            Self::Extension(extension) => extension.infer_output_types(input_types),
            Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Select
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
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V, Extension> Operation<ArrayType> for LinearArrayOperation<V, ArrayType, Extension>
where
    V: Traceable<ArrayType> + Parameter,
    Extension: Clone + Operation<ArrayType>,
{
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
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).infer_output_types(input_types)
            }
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::LeftDot { factor, dimensions } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone()).infer_output_types(input_types)
            }
            Self::RightDot { factor, dimensions } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone()).infer_output_types(input_types)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
            }
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::Extension(extension) => extension.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).render(formatter, indentation)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::LeftDot { factor, dimensions } | Self::RightDot { factor, dimensions } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("factor", factor)?;
                    operation.field("dimensions", dimensions)
                })
            }
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V, Extension> Operation<DataType> for LinearArrayOperation<V, DataType, Extension>
where
    V: Traceable<DataType> + Parameter,
    Extension: Clone + Operation<DataType>,
{
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
            Self::Extension(extension) => extension.infer_output_types(input_types),
            Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
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
            Self::LeftDot { factor, dimensions } | Self::RightDot { factor, dimensions } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("factor", factor)?;
                    operation.field("dimensions", dimensions)
                })
            }
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
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
        }
    }
}

impl InterpretableOperation<DataType, ZeroScalarTangent> for LinearScalarOperation<ZeroScalarTangent> {
    fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, TracingError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
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
            Self::One(one) => Ok(vec![Tangent::Value(V::one(&one.r#type)?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Scale { factor } => interpret_tangent_value_scale(self, factor, inputs),
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
        }
    }
}

impl<'domain, D> InterpretableOperation<DataType, Tracer<'domain, D>> for LinearScalarOperation<Tracer<'domain, D>>
where
    D: TracingDomain<Type = DataType>,
    Tracer<'domain, D>: Add<Output = Tracer<'domain, D>>
        + Sub<Output = Tracer<'domain, D>>
        + Neg<Output = Tracer<'domain, D>>
        + Mul<Output = Tracer<'domain, D>>
        + ZeroLike
        + OneLike,
    Vec<Tracer<'domain, D>>: Parameterized<Tracer<'domain, D>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[Tracer<'domain, D>]) -> Result<Vec<Tracer<'domain, D>>, TracingError> {
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
                <AddOperation as InterpretableOperation<DataType, Tracer<'domain, D>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<DataType, Tracer<'domain, D>>>::interpret(&SubOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<'domain, D>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
        }
    }
}

/// [`InterpretableOperation`] for [`ArrayOperation`] requires the full union of value capabilities used by
/// the closed default ordinary-op carrier.
///
/// That broad union is local to [`ArrayOperation`] itself. The higher-level tracing APIs avoid
/// exposing it as one public value-bundle trait and instead express their requirements through the
/// specific staged op carrier bounds they actually exercise.
impl<V, Extension> InterpretableOperation<ArrayType, V> for ArrayOperation<V, ArrayType, Extension>
where
    V: Traceable<ArrayType>
        + Parameter
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
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + Select
        + ControlFlowValue,
    Extension: Clone + InterpretableOperation<ArrayType, V>,
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
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).interpret(inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Select => SelectOperation.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<'domain, D, V, Extension> InterpretableOperation<ArrayType, Tracer<'domain, D>>
    for ArrayOperation<V, ArrayType, Extension>
where
    D: TracingDomain<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType, Extension>>,
    V: TracerReplayValue<ArrayType>,
    Extension: Clone + InterpretableOperation<ArrayType, Tracer<'domain, D>>,
{
    fn interpret(&self, inputs: &[Tracer<'domain, D>]) -> Result<Vec<Tracer<'domain, D>>, TracingError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "typed zero operation over tracer values was not materialized before interpretation for {}",
                    &zero.r#type
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "typed one operation over tracer values was not materialized before interpretation for {}",
                    &one.r#type
                ),
            }
            .into()),
            Self::Extension(extension) => extension.interpret(inputs),
            _ => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                exemplar.context.stage(self.clone(), input_refs.as_slice())
            }
        }
    }
}

impl<V, Extension> InterpretableOperation<DataType, V> for ArrayOperation<V, DataType, Extension>
where
    V: Traceable<DataType>
        + Parameter
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
    Extension: Clone + InterpretableOperation<DataType, V>,
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
            Self::Extension(extension) => extension.interpret(inputs),
            Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<Extension> InterpretableOperation<ArrayType, ZeroArrayTangent>
    for LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>
where
    Extension: Clone + InterpretableOperation<ArrayType, ZeroArrayTangent>,
{
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
            Self::Extension(extension) => extension.interpret(inputs),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V, Extension> InterpretableOperation<ArrayType, Tangent<ArrayType, V>>
    for LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>
where
    V: Traceable<ArrayType>
        + Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Extension: Clone + InterpretableOperation<ArrayType, Tangent<ArrayType, V>>,
{
    fn interpret(&self, inputs: &[Tangent<ArrayType, V>]) -> Result<Vec<Tangent<ArrayType, V>>, TracingError> {
        match self {
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type.clone())]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(&one.r#type)?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Transpose { permutation } => {
                let op = TransposeOperation::new(permutation.clone());
                interpret_tangent_value_unary_value_or_zero(&op, &op, inputs)
            }
            Self::Scale { factor } => interpret_tangent_value_scale(self, factor, inputs),
            Self::LeftDot { factor, dimensions } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, TracingError);
                match inputs {
                    [input] if factor.is_zero() || input.is_zero() => {
                        Ok(symbolic_zero_tangent_value_outputs(output_types))
                    }
                    [Tangent::Value(input)] => {
                        let Tangent::Value(factor) = factor else {
                            unreachable!("zero factors are handled before concrete left_dot interpretation")
                        };
                        Ok(super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone())
                            .interpret(std::slice::from_ref(input))?
                            .into_iter()
                            .map(Tangent::Value)
                            .collect())
                    }
                    _ => unreachable!("left_dot output type inference validates the input count"),
                }
            }
            Self::RightDot { factor, dimensions } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, TracingError);
                match inputs {
                    [input] if factor.is_zero() || input.is_zero() => {
                        Ok(symbolic_zero_tangent_value_outputs(output_types))
                    }
                    [Tangent::Value(input)] => {
                        let Tangent::Value(factor) = factor else {
                            unreachable!("zero factors are handled before concrete right_dot interpretation")
                        };
                        Ok(super::dot::RightDotOperation::new(factor.clone(), dimensions.clone())
                            .interpret(std::slice::from_ref(input))?
                            .into_iter()
                            .map(Tangent::Value)
                            .collect())
                    }
                    _ => unreachable!("right_dot output type inference validates the input count"),
                }
            }
            Self::Reshape { input_shape, output_shape } => interpret_tangent_value_unary_value_or_zero(
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
                            Tangent::Value(predicate) => predicate.control_flow_predicate()?,
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
                        Tangent::Value(predicate) => predicate.control_flow_predicate()?,
                    };
                    if !predicate {
                        check_count!("output", state, output_types.len(), TracingError);
                        return Ok(state);
                    }
                    state = while_operation.body.interpret(state)?;
                    check_count!("output", state, while_operation.state_types().len(), TracingError);
                }
            }
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<V, Extension> InterpretableOperation<ArrayType, V> for LinearArrayOperation<V, ArrayType, Extension>
where
    V: Traceable<ArrayType>
        + Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Extension: Clone + InterpretableOperation<ArrayType, V>,
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
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftDot { factor, dimensions } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::RightDot { factor, dimensions } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<Extension> InterpretableOperation<DataType, ZeroScalarTangent>
    for LinearArrayOperation<ZeroScalarTangent, DataType, Extension>
where
    Extension: Clone + InterpretableOperation<DataType, ZeroScalarTangent>,
{
    fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, TracingError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Extension(extension) => extension.interpret(inputs),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Traceable<DataType>, Extension> InterpretableOperation<DataType, Tangent<DataType, V>>
    for LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>
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
    Extension: Clone + InterpretableOperation<DataType, Tangent<DataType, V>>,
{
    fn interpret(&self, inputs: &[Tangent<DataType, V>]) -> Result<Vec<Tangent<DataType, V>>, TracingError> {
        match self {
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type)]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(&one.r#type)?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Scale { factor } => interpret_tangent_value_scale(self, factor, inputs),
            Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<V: Traceable<DataType>, Extension> InterpretableOperation<DataType, V>
    for LinearArrayOperation<V, DataType, Extension>
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
    Extension: Clone + InterpretableOperation<DataType, V>,
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
            Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<'domain, D, Extension> InterpretableOperation<ArrayType, Tracer<'domain, D>>
    for LinearArrayOperation<Tracer<'domain, D>, ArrayType, Extension>
where
    D: TracingDomain<Type = ArrayType>,
    Extension: Clone + InterpretableOperation<ArrayType, Tracer<'domain, D>>,
    Tracer<'domain, D>: Add<Output = Tracer<'domain, D>>
        + Sub<Output = Tracer<'domain, D>>
        + Neg<Output = Tracer<'domain, D>>
        + Mul<Output = Tracer<'domain, D>>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Vec<Tracer<'domain, D>>: Parameterized<
            Tracer<'domain, D>,
            To<Tracer<'domain, D>> = Vec<Tracer<'domain, D>>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn interpret(&self, inputs: &[Tracer<'domain, D>]) -> Result<Vec<Tracer<'domain, D>>, TracingError> {
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
            Self::Add => <AddOperation as InterpretableOperation<ArrayType, Tracer<'domain, D>>>::interpret(
                &AddOperation,
                inputs,
            ),
            Self::Sub => <SubOperation as InterpretableOperation<ArrayType, Tracer<'domain, D>>>::interpret(
                &SubOperation,
                inputs,
            ),
            Self::Neg => <NegOperation as InterpretableOperation<ArrayType, Tracer<'domain, D>>>::interpret(
                &NegOperation,
                inputs,
            ),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::LeftDot { factor, dimensions } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::RightDot { factor, dimensions } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<'domain, D, Extension> InterpretableOperation<DataType, Tracer<'domain, D>>
    for LinearArrayOperation<Tracer<'domain, D>, DataType, Extension>
where
    D: TracingDomain<Type = DataType>,
    Extension: Clone + InterpretableOperation<DataType, Tracer<'domain, D>>,
    Tracer<'domain, D>: Add<Output = Tracer<'domain, D>>
        + Sub<Output = Tracer<'domain, D>>
        + Neg<Output = Tracer<'domain, D>>
        + Mul<Output = Tracer<'domain, D>>
        + ZeroLike
        + OneLike,
    Vec<Tracer<'domain, D>>: Parameterized<Tracer<'domain, D>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[Tracer<'domain, D>]) -> Result<Vec<Tracer<'domain, D>>, TracingError> {
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
                <AddOperation as InterpretableOperation<DataType, Tracer<'domain, D>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<DataType, Tracer<'domain, D>>>::interpret(&SubOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<'domain, D>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<V: Traceable<DataType>>
    LinearOperation<DataType, Tangent<DataType, V>, LinearScalarOperation<Tangent<DataType, V>>>
    for LinearScalarOperation<Tangent<DataType, V>>
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearScalarOperation<Tangent<DataType, V>>,
        >,
        output_cotangents: &[Cotangent<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearScalarOperation<Tangent<DataType, V>>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, DataType, Tangent<DataType, V>, LinearScalarOperation<Tangent<DataType, V>>>>,
        TracingError,
    > {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Neg => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
        }
    }
}

impl<Extension>
    LinearOperation<ArrayType, ZeroArrayTangent, LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>>
    for LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>
where
    Extension: Clone
        + LinearOperation<ArrayType, ZeroArrayTangent, LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<
            'transpose,
            ArrayType,
            ZeroArrayTangent,
            LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>,
        >,
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            ZeroArrayTangent,
            LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                ArrayType,
                ZeroArrayTangent,
                LinearArrayOperation<ZeroArrayTangent, ArrayType, Extension>,
            >,
        >,
        TracingError,
    > {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add | Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Neg | Self::Scale { .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0].clone()])
            }
            Self::Transpose { permutation } => {
                check_count!("output", output_cotangents, 1, TracingError);
                let inverse = crate::tracing_v2::operations::transpose::inverse_permutation(permutation);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().transpose(inverse))]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::LeftDot { .. } | Self::RightDot { .. } => {
                // Factor for ZeroArrayTangent is always symbolic zero, so dot(zero, t) is zero
                // and the cotangent for `t` is symbolic zero as well.
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![Cotangent::Zero])
            }
            Self::Reshape { input_shape, .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().reshape(input_shape.clone())?)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Extension(extension) => extension.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<ArrayType>, Extension>
    LinearOperation<ArrayType, Tangent<ArrayType, V>, LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>>
    for LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>
where
    V: crate::tracing_v2::operations::matrix::DotOps,
    Extension: Clone
        + LinearOperation<
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>,
        >,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<
            'transpose,
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>,
        >,
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                ArrayType,
                Tangent<ArrayType, V>,
                LinearArrayOperation<Tangent<ArrayType, V>, ArrayType, Extension>,
            >,
        >,
        TracingError,
    > {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Neg => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Transpose { permutation } => {
                check_count!("output", output_cotangents, 1, TracingError);
                let inverse = crate::tracing_v2::operations::transpose::inverse_permutation(permutation);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().transpose(inverse))]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::LeftDot { factor, dimensions } => {
                check_count!("output", output_cotangents, 1, TracingError);
                let Tangent::Value(_) = factor else {
                    return Ok(vec![Cotangent::Zero]);
                };
                let factor_rank = factor.r#type().as_ref().rank();
                let adjoint =
                    crate::tracing_v2::operations::dot::adjoint_dimensions_for_left_dot(dimensions, factor_rank);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().left_dot(factor.clone(), &adjoint))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::RightDot { factor, dimensions } => {
                check_count!("output", output_cotangents, 1, TracingError);
                let Tangent::Value(_) = factor else {
                    return Ok(vec![Cotangent::Zero]);
                };
                let factor_rank = factor.r#type().as_ref().rank();
                let cotangent_rank = match &output_cotangents[0] {
                    Cotangent::Staged(value) => value.r#type().as_ref().rank(),
                    Cotangent::Zero => return Ok(vec![Cotangent::Zero]),
                };
                let t_rank = cotangent_rank + factor_rank
                    - 2 * dimensions.rhs_contracting_dimensions.len()
                    - dimensions.rhs_batching_dimensions.len();
                let adjoint = crate::tracing_v2::operations::dot::adjoint_dimensions_for_right_dot(
                    dimensions,
                    factor_rank,
                    t_rank,
                );
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().right_dot(factor.clone(), &adjoint))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Reshape { input_shape, .. } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().reshape(input_shape.clone())?)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Extension(extension) => extension.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<DataType>, Extension>
    LinearOperation<DataType, Tangent<DataType, V>, LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>>
    for LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>
where
    Extension: Clone
        + LinearOperation<DataType, Tangent<DataType, V>, LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>,
        >,
        output_cotangents: &[Cotangent<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                DataType,
                Tangent<DataType, V>,
                LinearArrayOperation<Tangent<DataType, V>, DataType, Extension>,
            >,
        >,
        TracingError,
    > {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Neg => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<DataType>> LinearOperation<DataType, V, LinearScalarOperation<V>> for LinearScalarOperation<V>
where
    V: Parameter + Add<Output = V> + Neg<Output = V> + ZeroLike + OneLike,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, DataType, V, LinearScalarOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, DataType, V, LinearScalarOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, DataType, V, LinearScalarOperation<V>>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, TracingError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => SubOperation.transpose(context, output_cotangents),
            Self::Neg => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor } => {
                check_count!("output", output_cotangents, 1, TracingError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
        }
    }
}

impl<V: Traceable<ArrayType>, Extension> LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType, Extension>>
    for LinearArrayOperation<V, ArrayType, Extension>
where
    V: Parameter
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Extension: Clone + LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType, Extension>>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearArrayOperation<V, ArrayType, Extension>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearArrayOperation<V, ArrayType, Extension>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearArrayOperation<V, ArrayType, Extension>>>, TracingError>
    {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => AddOperation.transpose(context, output_cotangents),
            Self::Sub => SubOperation.transpose(context, output_cotangents),
            Self::Neg => NegOperation.transpose(context, output_cotangents),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).transpose(context, output_cotangents)
            }
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(context, output_cotangents),
            Self::LeftDot { factor, dimensions } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone())
                    .transpose(context, output_cotangents)
            }
            Self::RightDot { factor, dimensions } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone())
                    .transpose(context, output_cotangents)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).transpose(context, output_cotangents)
            }
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Extension(extension) => extension.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<DataType>, Extension> LinearOperation<DataType, V, LinearArrayOperation<V, DataType, Extension>>
    for LinearArrayOperation<V, DataType, Extension>
where
    V: Parameter + Add<Output = V> + Neg<Output = V> + Mul<Output = V> + ZeroLike + OneLike,
    Extension: Clone + LinearOperation<DataType, V, LinearArrayOperation<V, DataType, Extension>>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, DataType, V, LinearArrayOperation<V, DataType, Extension>>,
        output_cotangents: &[Cotangent<'transpose, DataType, V, LinearArrayOperation<V, DataType, Extension>>],
    ) -> Result<Vec<Cotangent<'transpose, DataType, V, LinearArrayOperation<V, DataType, Extension>>>, TracingError>
    {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => AddOperation.transpose(context, output_cotangents),
            Self::Sub => SubOperation.transpose(context, output_cotangents),
            Self::Neg => NegOperation.transpose(context, output_cotangents),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(context, output_cotangents),
            Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.transpose(context, output_cotangents),
        }
    }
}

impl<F, D> DifferentiableOperation<D> for ScalarOperation<F>
where
    F: Traceable<DataType> + Parameter + Clone,
    D: DifferentiableDomain<Type = DataType>,
    D::Value: Add<Output = D::Value>
        + Sub<Output = D::Value>
        + Mul<Output = D::Value>
        + Div<Output = D::Value>
        + Neg<Output = D::Value>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Parameterized<D::Value>,
    <D::Value as Parameterized<D::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<D::Value>: Parameterized<D::Value, ParameterStructure: std::fmt::Debug + PartialEq>,
    ScaleOperation<DataType, F>: DifferentiableOperation<D>,
    D::LinearOperationCarrier: SupportsZeroLike<DataType, D::Tangent>
        + SupportsNeg<DataType, D::Tangent>
        + SupportsSub<DataType, D::Tangent>
        + SupportsScale<DataType, D::Tangent, D::Value>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
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
        }
    }
}

impl<V: Value<ArrayType>, D, Extension> DifferentiableOperation<D> for ArrayOperation<V, ArrayType, Extension>
where
    V: Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + Scale<Output = V>
        + ZeroLike
        + OneLike
        + Zero<ArrayType>
        + One<ArrayType>
        + Parameterized<V>
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue
        + 'static,
    D: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
    D::Tangent: crate::tracing_v2::operations::transpose::Transpose,
    Extension: Clone + DifferentiableOperation<D>,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<
            V,
            Family: crate::parameters::ParameterizedFamily<D::Tangent>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    D::LinearOperationCarrier: SupportsZeroLike<ArrayType, D::Tangent>
        + SupportsNeg<ArrayType, D::Tangent>
        + SupportsSub<ArrayType, D::Tangent>
        + SupportsScale<ArrayType, D::Tangent, V>
        + SupportsLeftDot<ArrayType, D::Tangent, V>
        + SupportsRightDot<ArrayType, D::Tangent, V>
        + crate::tracing_v2::operations::SupportsTranspose<ArrayType, D::Tangent>
        + super::SupportsReshape<ArrayType, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
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
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).jvp(context, inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Select | Self::Condition(_) | Self::While(_) => {
                Err(TypeError { message: format!("{} does not support generic array jvp dispatch", self.name()) }
                    .into())
            }
            Self::Extension(extension) => extension.jvp(context, inputs),
        }
    }
}

impl<V: Value<DataType>, D, Extension> DifferentiableOperation<D> for ArrayOperation<V, DataType, Extension>
where
    V: Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + Scale<Output = V>
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + 'static,
    D: DifferentiableDomain<Type = DataType, Value = V> + 'static,
    Extension: Clone + DifferentiableOperation<D>,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    D::LinearOperationCarrier: SupportsZeroLike<DataType, D::Tangent>
        + SupportsNeg<DataType, D::Tangent>
        + SupportsSub<DataType, D::Tangent>
        + SupportsScale<DataType, D::Tangent, V>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
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
            Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_) => {
                Err(TypeError { message: format!("{} is not supported for scalar data type metadata", self.name()) }
                    .into())
            }
            Self::Extension(extension) => extension.jvp(context, inputs),
        }
    }
}

/// Linearization-domain dispatcher for [`ArrayOperation`] under the traced-linearization path.
///
/// Forwards each variant to the per-op JVP rule, picking up the
/// [`TracingContext`]-keyed impl for captured
/// [`Scale`](Self::Scale), and the [`Condition`](Self::Condition) / [`While`](Self::While) stub impls
/// (predicate extraction does not work at trace time).
impl<'domain, D, V, Extension> DifferentiableOperation<TracingContext<'domain, D>>
    for ArrayOperation<V, ArrayType, Extension>
where
    D: DifferentiableTracingDomain<
            Type = ArrayType,
            Value = V,
            OperationCarrier = ArrayOperation<V, ArrayType, Extension>,
        > + RuntimeDomain
        + 'domain + 'static,
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
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue
        + Parameter
        + 'static,
    Extension: Clone + DifferentiableOperation<TracingContext<'domain, D>> + 'domain,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    Tracer<'domain, D>: Add<Output = Tracer<'domain, D>>
        + Sub<Output = Tracer<'domain, D>>
        + Mul<Output = Tracer<'domain, D>>
        + Div<Output = Tracer<'domain, D>>
        + Neg<Output = Tracer<'domain, D>>
        + Sin
        + Cos
        + crate::tracing_v2::operations::matrix::DotOps
        + ZeroLike
        + OneLike,
    <TracingContext<'domain, D> as DifferentiableDomain>::LinearOperationCarrier:
        SupportsZeroLike<ArrayType, Tracer<'domain, D>>
            + SupportsSub<ArrayType, Tracer<'domain, D>>
            + SupportsLeftDot<ArrayType, Tracer<'domain, D>, Tracer<'domain, D>>
            + SupportsRightDot<ArrayType, Tracer<'domain, D>, Tracer<'domain, D>>
            + crate::tracing_v2::operations::SupportsTranspose<ArrayType, Tracer<'domain, D>>
            + SupportsReshape<ArrayType, Tracer<'domain, D>>,
    AddOperation: InterpretableOperation<ArrayType, Tracer<'domain, D>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, D>>,
        inputs: &[JvpTracer<Tracer<'domain, D>, D::Type, Tracer<'jvp, TracingContext<'domain, D>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'domain, D>, D::Type, Tracer<'jvp, TracingContext<'domain, D>>>>, TracingError>
    where
        TracingContext<'domain, D>: 'jvp,
    {
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
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).jvp(context, inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Select => {
                Err(TypeError { message: format!("{} does not support generic array jvp dispatch", self.name()) }
                    .into())
            }
            Self::Condition(condition) => condition.as_ref().jvp(context, inputs),
            Self::While(while_operation) => while_operation.as_ref().jvp(context, inputs),
            Self::Extension(extension) => extension.jvp(context, inputs),
        }
    }
}

impl<'domain, D, V, Extension> DifferentiableOperation<TracingContext<'domain, D>>
    for ArrayOperation<V, DataType, Extension>
where
    D: DifferentiableTracingDomain<
            Type = DataType,
            Value = V,
            OperationCarrier = ArrayOperation<V, DataType, Extension>,
        > + RuntimeDomain
        + 'domain,
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
        + Parameter
        + 'static,
    Extension: Clone + DifferentiableOperation<TracingContext<'domain, D>> + 'domain,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    Tracer<'domain, D>: Add<Output = Tracer<'domain, D>>
        + Sub<Output = Tracer<'domain, D>>
        + Mul<Output = Tracer<'domain, D>>
        + Div<Output = Tracer<'domain, D>>
        + Neg<Output = Tracer<'domain, D>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike,
    <TracingContext<'domain, D> as DifferentiableDomain>::LinearOperationCarrier:
        SupportsZeroLike<DataType, Tracer<'domain, D>> + SupportsSub<DataType, Tracer<'domain, D>>,
    AddOperation: InterpretableOperation<DataType, Tracer<'domain, D>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, D>>,
        inputs: &[JvpTracer<Tracer<'domain, D>, D::Type, Tracer<'jvp, TracingContext<'domain, D>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'domain, D>, D::Type, Tracer<'jvp, TracingContext<'domain, D>>>>, TracingError>
    where
        TracingContext<'domain, D>: 'jvp,
    {
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
            Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_) => {
                Err(TypeError { message: format!("{} is not supported for scalar data type metadata", self.name()) }
                    .into())
            }
            Self::Extension(extension) => extension.jvp(context, inputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::InterpretableOperation as _;
    use crate::parameters::Placeholder;
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
        let transposed = builder
            .add_instruction(ZeroArrayOperation::Transpose { permutation: vec![1, 0] }, vec![reshaped])
            .unwrap()[0];
        let negated = builder.add_instruction(ZeroArrayOperation::Neg, vec![transposed]).unwrap()[0];
        let output = builder.add_instruction(ZeroArrayOperation::Add, vec![negated, input]).unwrap()[0];
        let program = builder
            .build::<Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(program.interpret(vec![Tangent::zero(input_type.clone())]), Ok(vec![Tangent::zero(input_type)]));
    }

    #[test]
    fn test_linear_array_zero_only_tangent_dot_metadata() {
        use crate::tracing_v2::operations::dot::DotDimensionNumbers;

        let input_type = array_type(&[2, 3]);
        let right_factor_type = array_type(&[3, 4]);
        let right_dot = ZeroArrayOperation::RightDot {
            factor: Tangent::zero(right_factor_type),
            dimensions: DotDimensionNumbers::matmul(),
        };

        assert_eq!(
            right_dot.interpret(&[Tangent::zero(input_type.clone())]),
            Ok(vec![Tangent::zero(array_type(&[2, 4]))])
        );

        let left_factor_type = array_type(&[4, 2]);
        let left_dot = ZeroArrayOperation::LeftDot {
            factor: Tangent::zero(left_factor_type),
            dimensions: DotDimensionNumbers::matmul(),
        };

        assert_eq!(left_dot.interpret(&[Tangent::zero(input_type)]), Ok(vec![Tangent::zero(array_type(&[4, 3]))]));
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
    fn test_linear_scalar_tangent_value_interpretation_mixes_value_and_zero() {
        let value = MixedScalar::value(3.0);
        let zero = MixedScalar::zero(DataType::F64);

        assert_eq!(MixedScalarOperation::Add.interpret(&[value.clone(), zero.clone()]), Ok(vec![value.clone()]));
        assert_eq!(MixedScalarOperation::Add.interpret(&[zero.clone(), value.clone()]), Ok(vec![value.clone()]));
        assert_eq!(
            MixedScalarOperation::Sub.interpret(&[zero.clone(), value.clone()]),
            Ok(vec![MixedScalar::value(-3.0)])
        );
        assert_eq!(
            (MixedScalarOperation::Scale { factor: MixedScalar::zero(DataType::F64) })
                .interpret(std::slice::from_ref(&value)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            (MixedScalarOperation::Scale { factor: MixedScalar::value(2.0) }).interpret(std::slice::from_ref(&zero)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            (MixedScalarOperation::Scale { factor: MixedScalar::value(2.0) }).interpret(std::slice::from_ref(&value)),
            Ok(vec![MixedScalar::value(6.0)])
        );
        assert_eq!(MixedScalarOperation::ZeroLike.interpret(std::slice::from_ref(&value)), Ok(vec![zero.clone()]));
        assert_eq!(
            MixedScalarOperation::One(OneOperation::new(DataType::F64)).interpret(&[]),
            Ok(vec![MixedScalar::value(1.0)])
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
            MixedArrayOperation::Add.interpret(&[MixedArray::value(input.clone()), input_zero.clone()]),
            Ok(vec![MixedArray::value(input.clone())])
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

        use crate::tracing_v2::operations::dot::DotDimensionNumbers;

        let left_factor_type = f64_array_type(&[4, 2]);
        assert_eq!(
            (MixedArrayOperation::LeftDot {
                factor: MixedArray::zero(left_factor_type),
                dimensions: DotDimensionNumbers::matmul(),
            })
            .interpret(&[MixedArray::value(input.clone())]),
            Ok(vec![MixedArray::zero(f64_array_type(&[4, 3]))])
        );

        let right_factor = TestArray::matrix(3, 4, vec![0.0; 12]);
        assert_eq!(
            (MixedArrayOperation::RightDot {
                factor: MixedArray::value(right_factor),
                dimensions: DotDimensionNumbers::matmul(),
            })
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
            program.interpret((MixedScalar::value(2.0), MixedScalar::zero(DataType::F64))),
            Ok((MixedScalar::value(2.0), (MixedScalar::value(-2.0), MixedScalar::zero(DataType::F64))))
        );
    }
}
