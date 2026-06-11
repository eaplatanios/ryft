//! Reusable staged operation enums for built-in primitives and backend extensions.
//!
//! [`ArrayOperation`] and [`LinearArrayOperation`] contain the core operations implemented by `ryft-core` plus an
//! optional statically typed backend extension slot. A backend that needs additional operations should define an
//! ordinary extension enum, define a linear extension enum when it has linear-only operations, implement the standard
//! operation traits for those enums, and select `ArrayOperation<Value, Type, Extension>` and
//! `LinearArrayOperation<Tangent, Type, LinearExtension>` as its tracing operation types.
//!
//! `ryft-core` intentionally does not expose a universal dynamic custom-operation primitive. Backend-specific or
//! user-defined operations should be represented by a backend extension variant, so transform, interpretation, and
//! lowering rules remain statically typed and owned by the backend that understands the operation.

use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::batching::BatchingError;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::BooleanLike;
use crate::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, Scale, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection, SupportsCompare};
use crate::operations::constants::{
    ConstantOperation, Fill, FillOperation, ONE_LIKE_OPERATION_NAME, One, OneLike, OneLikeOperation, OneOperation,
    SupportsConstant, SupportsFill, SupportsOne, SupportsOneLike, SupportsZero, SupportsZeroLike,
    ZERO_LIKE_OPERATION_NAME, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{ConditionOperation, ConditionPredicate, ControlFlowError, WhileOperation};
use crate::operations::control_flow::{SELECT_OPERATION_NAME, Select, SelectOperation, SupportsSelect};
use crate::operations::differentiation::{STOP_GRADIENT_OPERATION_NAME, StopGradientOperation, SupportsStopGradient};
use crate::operations::logical::{
    AND_OPERATION_NAME, AndOperation, NOT_OPERATION_NAME, NotOperation, OR_OPERATION_NAME, OrOperation, SupportsAnd,
    SupportsNot, SupportsOr, SupportsXor, XOR_OPERATION_NAME, XorOperation,
};
use crate::operations::manipulation::ReshapeOperation;
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, SupportsBroadcast, SupportsTranspose, Transpose, TransposeOperation,
    inverse_permutation,
};
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::operations::trigonometric::{
    COS_OPERATION_NAME, CosOperation, SIN_OPERATION_NAME, SinOperation, SupportsCos, SupportsSin,
};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameter, Parameterized};
use crate::programs::{ProgramError, Value};
use crate::tracing::{AbstractTracingContext, Tracer, TracingContext};
use crate::tracing_v2::DifferentiableOperation;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext, ProgramBatchingContext};
use crate::tracing_v2::differentiation::{
    DifferentiationContext, FactorParameterizedOperation, JvpTracer, LinearOperationOf, ResidualFactor, TangentContext,
};
use crate::tracing_v2::operations::collective::{CollectiveKind, CollectiveOperation, SupportsCollective};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation, CustomVjpResidual, SupportsCustomJvp,
    SupportsCustomVjp, SupportsCustomVjpCall, custom_jvp_rule, custom_vjp_rule,
};
use crate::tracing_v2::operations::dot::{LeftDot, LeftDotOperation, MaybeDot, RightDot, RightDotOperation};
use crate::tracing_v2::operations::memory::{
    SupportsTransferToMemory, TRANSFER_TO_MEMORY_OPERATION_NAME, TransferToMemory, TransferToMemoryOperation,
};
use crate::tracing_v2::operations::reduce::{ReduceOperation, ReductionKind, SupportsReduce};
use crate::tracing_v2::operations::select::SupportsLinearSelect;
use crate::tracing_v2::operations::{DotDimensionNumbers, DotOperation, SupportsDot};
use crate::tracing_v2::rematerialization::{
    MaybeRematerializationName, RematerializationNameOperation, SupportsRematerializationName,
};
use crate::types::{ArrayType, DataType, Memory, Shape, Type, TypeError, Typed};

use super::bounds::{
    SupportsArithmeticOperations, SupportsComparisonOperations, SupportsConstantOperations,
    SupportsLinearAlgebraOperations, SupportsLinearArithmeticOperations, SupportsLinearArrayOperation,
    SupportsLinearScalarOperation, SupportsManipulationOperations, SupportsTrigonometricOperations,
};
use super::matrix::DotOps;
use crate::operations::manipulation::{Reshape, SupportsReshape};

type ZeroScalarTangent = Tangent<DataType, Infallible>;
type ZeroArrayTangent = Tangent<ArrayType, Infallible>;

/// Reusable operation enum for ordinary staged programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates. Most variants are thin tags
/// around one semantic primitive defined elsewhere in [`super`]. The [`Extension`](Self::Extension) variant lets
/// backends statically compose their own operation enum into the same operation type without dynamic custom-operation
/// registries. Backends that only need built-in operations can omit the `Extension` parameter and use the uninhabited
/// [`Infallible`] default.
#[derive(Clone, Debug)]
pub enum ArrayOperation<V, T, Extension = Infallible>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    /// Typed zero with no inputs and one output, carrying a [`ZeroOperation`].
    Zero(ZeroOperation<T>),

    /// Exemplar-derived zero.
    ZeroLike,

    /// Typed one with no inputs and one output, carrying a [`OneOperation`].
    One(OneOperation<T>),

    /// Exemplar-derived one.
    OneLike,

    /// Typed literal constant with no inputs and one output, carrying a [`ConstantOperation`] that holds a fully typed
    /// value of the value type `V`. Its output type is the value's type and interpreting it clones the captured value.
    Constant(ConstantOperation<T, V>),

    /// Scalar fill with no inputs and one output, carrying a [`FillOperation`] that fills the held [`Type`] with the
    /// captured `f64` scalar. Used by transform rules that need to materialize a numeric factor without being
    /// parameterized over the underlying element type (for example, `Mean`'s transpose rule constructs the `1/N`
    /// factor this way before binary multiplication, relying on implicit rank-0 broadcasting).
    Fill(FillOperation<T, f64>),

    /// Elementwise negation.
    Neg,

    /// Elementwise addition.
    Add,

    /// Elementwise subtraction.
    Sub,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Elementwise multiplication.
    Mul,

    /// Elementwise division.
    Div,

    /// Elementwise sine.
    Sin,

    /// Elementwise cosine.
    Cos,

    /// Gradient-severing identity.
    StopGradient,

    /// Rematerialize-policy name-tagging identity.
    RematerializationName(RematerializationNameOperation),

    /// Memory-space transfer moving the operand into a destination [`Memory`].
    TransferToMemory(TransferToMemoryOperation),

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

    /// Reshape from one shape to another.
    Reshape { output_shape: Shape },

    /// N-dimensional broadcast to a target shape.
    ///
    /// Maps each input axis `i` to output axis `output_axes[i]`, replicating along the
    /// remaining axes of `output_type`. Lowers to StableHLO's `broadcast_in_dim` in the XLA
    /// backend.
    Broadcast {
        /// Target output [`ArrayType`].
        output_type: T,

        /// For each input axis, the output axis it maps to.
        output_axes: Vec<usize>,
    },

    /// Axis-collapsing reduction.
    ///
    /// Reduces the input along `axes` using the operator/identity pair selected by `kind`. The
    /// output rank is the input rank minus the number of reduced axes; non-reduced axes keep
    /// their relative order. Lowers to StableHLO's `stablehlo.reduce` op in the XLA backend.
    /// The `input_shape` is recorded so the linear transpose rule can broadcast the cotangent
    /// back to the input rank.
    Reduce {
        /// Axes reduced by this operation.
        axes: Vec<usize>,

        /// Kind of reduction.
        kind: ReductionKind,
    },

    /// Elementwise pairwise comparison.
    ///
    /// Compares two broadcast-compatible operands using the predicate described by `kind` and
    /// returns a Boolean array of the broadcasted shape. Lowers to StableHLO's `stablehlo.compare`
    /// op in the XLA backend.
    Compare {
        /// Kind of comparison.
        direction: ComparisonDirection,
    },

    /// Elementwise logical negation on one Boolean array. Lowers to StableHLO's `stablehlo.not`
    /// op in the XLA backend.
    Not,

    /// Elementwise logical conjunction of two broadcast-compatible Boolean arrays. Lowers to
    /// StableHLO's `stablehlo.and` op in the XLA backend.
    And,

    /// Elementwise logical disjunction of two broadcast-compatible Boolean arrays. Lowers to
    /// StableHLO's `stablehlo.or` op in the XLA backend.
    Or,

    /// Elementwise logical exclusive disjunction of two broadcast-compatible Boolean arrays.
    /// Lowers to StableHLO's `stablehlo.xor` op in the XLA backend.
    Xor,

    /// Named-axis collective operation (`psum`, `pmean`, `pmax`).
    ///
    /// Collectives reference a named axis introduced by [`BatchingContext::with_axis_name`](
    /// crate::tracing_v2::batching::BatchingContext::with_axis_name) on the enclosing `batch`.
    /// Inside that batching domain, the operation collapses the mapped axis; outside it the
    /// operation acts as identity (per-lane semantics has no named axis to reduce).
    Collective {
        /// Axis name referenced by this collective.
        axis_name: String,

        /// Kind of collective.
        kind: CollectiveKind,
    },

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

    /// Higher-order call with a user-supplied JVP (forward-derivative) program.
    CustomJvp(Box<CustomJvpOperation<V, ArrayOperation<V, T, Extension>, T>>),

    /// Higher-order call with user-supplied forward/backward (VJP) programs.
    CustomVjp(Box<CustomVjpOperation<V, ArrayOperation<V, T, Extension>, T>>),

    /// Backend-owned extension operation.
    Extension(Extension),
}

/// Reusable operation enum for staged linear programs.
///
/// [`LinearArrayOperation`] is the linear-program sibling of [`ArrayOperation`]. It contains
/// operations that can appear in tangent and cotangent programs, including captured-factor linear
/// maps such as [`LeftDot`](Self::LeftDot) and [`RightDot`](Self::RightDot), and the
/// linearized higher-order operations needed by rematerialization and control flow. The
/// [`Extension`](Self::Extension) variant lets backends statically compose linear backend operations into the same
/// operation type. Backends that only need built-in linear operations can omit the `Extension` parameter and use the
/// uninhabited [`Infallible`] default.
///
/// The `C` parameter is the constant type of the [`DifferentiationContext`]
/// that stages the linear program: every context pins `C` to its [`Domain::Constant`](crate::domains::Domain) in its
/// `LinearOperation` associated-type definition. It types the user-supplied programs captured by
/// [`CustomVjpCall`](Self::CustomVjpCall), which are written over context constants rather than over the linear value
/// type `V` (`V` instantiates to tracers inside transform contexts, while captured programs always hold concrete
/// constants).
#[derive(Clone, Debug)]
pub enum LinearArrayOperation<V, C, T, Extension = Infallible, F = V, O = ArrayOperation<C, T, Extension>>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    /// Typed zero with no inputs and one output, carrying a [`ZeroOperation`].
    ///
    /// Emitted by the transpose pass at the boundary of pullbacks for primal inputs that receive
    /// no cotangent contribution from any output. Interpreting it requires
    /// [`Zero<ArrayType>`](crate::operations::constants::Zero) on the value type;
    /// staged tracer programs must materialize these ops away before being interpreted.
    Zero(ZeroOperation<T>),

    /// Exemplar-derived zero map.
    ZeroLike,

    /// Typed one with no inputs and one output, carrying a [`OneOperation`].
    One(OneOperation<T>),

    /// Exemplar-derived one map.
    OneLike,

    /// Typed literal constant with no inputs and one output, carrying a [`ConstantOperation`] that holds a fully typed
    /// value of the value type `V`. Its output type is the value's type and interpreting it clones the captured value.
    Constant(ConstantOperation<T, V>),

    /// Scalar fill with no inputs and one output, carrying a [`FillOperation`]; linear-side counterpart of
    /// [`ArrayOperation::Fill`]. Emitted by transform rules that need to scale a tangent or cotangent by a numeric
    /// factor without being parameterized over the value's element type (for example, the `Mean` reduction's transpose
    /// rule multiplies broadcast-back cotangent by `1 / N`, relying on implicit rank-0 broadcasting).
    Fill(FillOperation<T, f64>),

    /// Elementwise negation.
    Neg,

    /// Elementwise addition.
    Add,

    /// Elementwise subtraction.
    Sub,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: F },

    /// Elementwise multiplication of two tangent/cotangent values. Linear-side counterpart of
    /// [`ArrayOperation::Mul`]: although general bilinear multiplication is not linear, this
    /// variant is emitted in the linear operation enum when one operand is itself the staged output of
    /// a constant-producing op (such as [`Self::Fill`]) so that the overall map remains
    /// linear in the original primal input.
    Mul,

    /// Memory-space transfer moving the tangent or cotangent into a destination [`Memory`]; linear-side
    /// counterpart of [`ArrayOperation::TransferToMemory`]. Transposition moves the cotangent back to the
    /// operand's source memory, which is read off the forward operand type during the transpose pass rather
    /// than stored in the operation.
    TransferToMemory {
        /// Destination memory that the tangent or cotangent is moved into.
        destination: Memory,
    },

    /// N-dimensional axis permutation; linear-side analogue of [`ArrayOperation::Transpose`].
    Transpose {
        /// Permutation of input axes.
        permutation: Vec<usize>,
    },

    /// Captured-factor left dot: linear map `t ↦ dot(factor, t; dimensions)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::Dot`] when the LHS primal is held
    /// constant.
    LeftDot {
        /// Captured constant factor (LHS of the underlying dot).
        factor: F,

        /// Dimension numbers of the underlying dot.
        dimensions: DotDimensionNumbers,
    },

    /// Captured-factor right dot: linear map `t ↦ dot(t, factor; dimensions)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::Dot`] when the RHS primal is held
    /// constant.
    RightDot {
        /// Captured constant factor (RHS of the underlying dot).
        factor: F,

        /// Dimension numbers of the underlying dot.
        dimensions: DotDimensionNumbers,
    },

    /// Reshape from one shape to another.
    Reshape { output_shape: Shape },

    /// N-dimensional broadcast to a target shape; linear-side analogue of
    /// [`ArrayOperation::Broadcast`].
    Broadcast {
        /// Target output [`ArrayType`].
        output_type: T,

        /// For each input axis, the output axis it maps to.
        output_axes: Vec<usize>,
    },

    /// Axis-collapsing reduction; linear-side analogue of [`ArrayOperation::Reduce`].
    ///
    /// Only the kinds whose linearization is itself linear are useful here in practice:
    /// [`ReductionKind::Sum`] and [`ReductionKind::Mean`]. Other variants (`Max`/`Min`/`Any`/`All`)
    /// are not linear and should not be emitted by JVP rules; they are accepted in the variant
    /// for uniform enum coverage but cause the transpose rule to error.
    Reduce {
        /// Axes reduced by this operation.
        axes: Vec<usize>,

        /// Kind of reduction.
        kind: ReductionKind,
    },

    /// Captured-condition per-element select: linear map `(t, f) ↦ select(condition, t, f)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::Select`], with the Boolean primal condition captured as
    /// a factor (the condition itself has no tangent space, so the map is linear in the two branch operands). Its
    /// transpose routes the output cotangent into the selected branch: the `on_true` cotangent is
    /// `select(condition, cotangent, 0)` and the `on_false` cotangent is `select(condition, 0, cotangent)`.
    Select {
        /// Captured Boolean condition that drives the selection.
        condition: F,
    },

    /// Higher-order conditional restricted to linear branch programs.
    Condition(Box<ConditionOperation<V, LinearArrayOperation<V, C, T, Extension, F, O>, T>>),

    /// Higher-order while loop restricted to linear condition and body programs.
    While(Box<WhileOperation<V, LinearArrayOperation<V, C, T, Extension, F, O>, T>>),

    /// Opaque linear call staged by a `custom_vjp` linearization; its transpose replays the user's backward program.
    /// The backward program is valued at the context's constant type `C` rather than the linear value type `V`, so
    /// the call can be staged by any [`DifferentiationContext`] whose
    /// constants match the captured program, including transform contexts whose tangents are tracers.
    CustomVjpCall(Box<CustomVjpCallOperation<C, O, F, T>>),

    /// Backend-owned linear extension operation.
    Extension(Extension),
}

impl<T: Type> Operation<T> for Infallible {
    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        match *self {}
    }
}

impl<T: Type, V: Typed<T>> InterpretableOperation<T, V> for Infallible {
    fn interpret(&self, _inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match *self {}
    }
}

impl<T, V, O> TransposableOperation<T, V, O> for Infallible
where
    T: Parameter + Type,
    V: Value<T>,
    O: Operation<T>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        match *self {}
    }
}

impl<D: DifferentiationContext> DifferentiableOperation<D> for Infallible {
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        _inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        match *self {}
    }
}

impl<T, V, Extension> SupportsAdd<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn add_operation() -> Self {
        ArrayOperation::Add
    }
}

impl<T, V, Extension> SupportsSub<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn sub_operation() -> Self {
        ArrayOperation::Sub
    }
}

impl<T, V, Extension> SupportsMul<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn mul_operation() -> Self {
        ArrayOperation::Mul
    }
}

impl<T, V, Extension> SupportsDiv<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn div_operation() -> Self {
        ArrayOperation::Div
    }
}

impl<T, V, Extension> SupportsNeg<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn neg_operation() -> Self {
        ArrayOperation::Neg
    }
}

impl<T, V, Extension> SupportsSin<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn sin_operation() -> Self {
        ArrayOperation::Sin
    }
}

impl<T, V, Extension> SupportsCos<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn cos_operation() -> Self {
        ArrayOperation::Cos
    }
}

impl<T, V, Extension> SupportsRematerializationName<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn rematerialization_name_operation(name: String) -> Self {
        ArrayOperation::RematerializationName(RematerializationNameOperation::new(name))
    }
}

impl<T, V, Extension> MaybeRematerializationName for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    Extension: MaybeRematerializationName,
{
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        match self {
            Self::RematerializationName(operation) => Some(operation.tag()),
            Self::Extension(extension) => extension.rematerialization_name(),
            _ => None,
        }
    }
}

impl MaybeRematerializationName for Infallible {
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        match *self {}
    }
}

impl<T, V, Extension> SupportsStopGradient<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn stop_gradient_operation() -> Self {
        ArrayOperation::StopGradient
    }
}

impl<V: Value<ArrayType>, Extension> SupportsTransferToMemory<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn transfer_to_memory_operation(destination: Memory) -> Self {
        ArrayOperation::TransferToMemory(TransferToMemoryOperation::new(destination))
    }
}

impl<T, V, Extension> SupportsZero<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
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

impl<T, V, Extension> SupportsOne<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        ArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V, Extension> SupportsFill<T, f64> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn fill_operation(r#type: T, value: f64) -> Self {
        ArrayOperation::Fill(FillOperation::new(r#type, value))
    }

    #[inline]
    fn as_fill_operation(&self) -> Option<&FillOperation<T, f64>> {
        match self {
            Self::Fill(fill) => Some(fill),
            _ => None,
        }
    }
}

impl<T, V, Extension> SupportsConstant<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn constant_operation(value: V) -> Self {
        ArrayOperation::Constant(ConstantOperation::new(value))
    }

    #[inline]
    fn as_constant_operation(&self) -> Option<&ConstantOperation<T, V>> {
        match self {
            Self::Constant(constant) => Some(constant),
            _ => None,
        }
    }
}

impl<T, V, Extension> SupportsZeroLike<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn zero_like_operation() -> Self {
        ArrayOperation::ZeroLike
    }
}

impl<T, V, Extension> SupportsOneLike<T> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn one_like_operation() -> Self {
        ArrayOperation::OneLike
    }
}

impl<V: Value<ArrayType>, Extension> SupportsDot<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn dot_operation(dimensions: DotDimensionNumbers) -> Self {
        ArrayOperation::Dot { dimensions }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsTranspose<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn transpose_operation(permutation: Vec<usize>) -> Self {
        ArrayOperation::Transpose { permutation }
    }
}

impl<T, V, Extension> SupportsScale<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        ArrayOperation::Scale { factor }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsReshape<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn reshape_operation(output_shape: Shape) -> Self {
        ArrayOperation::Reshape { output_shape }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsReduce<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn reduce_operation(axes: Vec<usize>, kind: ReductionKind) -> Self {
        ArrayOperation::Reduce { axes, kind }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsCompare<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn compare_operation(direction: ComparisonDirection) -> Self {
        ArrayOperation::Compare { direction }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsNot<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn not_operation() -> Self {
        ArrayOperation::Not
    }
}

impl<V: Value<ArrayType>, Extension> SupportsAnd<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn and_operation() -> Self {
        ArrayOperation::And
    }
}

impl<V: Value<ArrayType>, Extension> SupportsOr<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn or_operation() -> Self {
        ArrayOperation::Or
    }
}

impl<V: Value<ArrayType>, Extension> SupportsXor<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn xor_operation() -> Self {
        ArrayOperation::Xor
    }
}

impl<V: Value<ArrayType>, Extension> SupportsCollective<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn collective_operation(axis_name: String, kind: CollectiveKind) -> Self {
        ArrayOperation::Collective { axis_name, kind }
    }
}

impl<T, V, Extension> MaybeDot for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    Extension: MaybeDot,
{
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        match self {
            Self::Dot { dimensions } => Some(dimensions),
            Self::Extension(extension) => extension.dot_dimensions(),
            _ => None,
        }
    }
}

impl MaybeDot for Infallible {
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        match *self {}
    }
}

impl<V: Value<ArrayType>, Extension> SupportsBroadcast<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn broadcast_operation(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        ArrayOperation::Broadcast { output_type, output_axes }
    }
}

impl<V: Value<ArrayType>, Extension> crate::operations::control_flow::SupportsSelect<ArrayType>
    for ArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn select_operation() -> Self {
        ArrayOperation::Select
    }
}

impl<T, V, C, Extension, F, O> SupportsAdd<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn add_operation() -> Self {
        LinearArrayOperation::Add
    }
}

impl<T, V, C, Extension, F, O> SupportsSub<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn sub_operation() -> Self {
        LinearArrayOperation::Sub
    }
}

impl<T, V, C, Extension, F, O> SupportsZero<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
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

impl<T, V, C, Extension, F, O> SupportsOne<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        LinearArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V, C, Extension, F, O> SupportsFill<T, f64> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn fill_operation(r#type: T, value: f64) -> Self {
        LinearArrayOperation::Fill(FillOperation::new(r#type, value))
    }

    #[inline]
    fn as_fill_operation(&self) -> Option<&FillOperation<T, f64>> {
        match self {
            Self::Fill(fill) => Some(fill),
            _ => None,
        }
    }
}

impl<T, V, C, Extension, F, O> SupportsConstant<T, V> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn constant_operation(value: V) -> Self {
        LinearArrayOperation::Constant(ConstantOperation::new(value))
    }

    #[inline]
    fn as_constant_operation(&self) -> Option<&ConstantOperation<T, V>> {
        match self {
            Self::Constant(constant) => Some(constant),
            _ => None,
        }
    }
}

impl<T, V, C, Extension, F, O> SupportsMul<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn mul_operation() -> Self {
        LinearArrayOperation::Mul
    }
}

impl<T, V, C, Extension, F, O> SupportsZeroLike<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn zero_like_operation() -> Self {
        LinearArrayOperation::ZeroLike
    }
}

impl<T, V, C, Extension, F, O> SupportsOneLike<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn one_like_operation() -> Self {
        LinearArrayOperation::OneLike
    }
}

impl<T, V, C, Extension, F, O> SupportsNeg<T> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn neg_operation() -> Self {
        LinearArrayOperation::Neg
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsTransferToMemory<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn transfer_to_memory_operation(destination: Memory) -> Self {
        LinearArrayOperation::TransferToMemory { destination }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsTranspose<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn transpose_operation(permutation: Vec<usize>) -> Self {
        LinearArrayOperation::Transpose { permutation }
    }
}

impl<T, V, C, Extension, F, O> SupportsScale<T, F> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
{
    #[inline]
    fn scale_operation(factor: F) -> Self {
        LinearArrayOperation::Scale { factor }
    }
}

impl<T, V, Extension> SupportsCustomJvp<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn custom_jvp_operation(operation: CustomJvpOperation<V, Self, T>) -> Self {
        Self::CustomJvp(Box::new(operation))
    }
}

impl<T, V, Extension> SupportsCustomVjp<T, V> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    #[inline]
    fn custom_vjp_operation(operation: CustomVjpOperation<V, Self, T>) -> Self {
        Self::CustomVjp(Box::new(operation))
    }
}

impl<T, V, C, Extension, F, O> SupportsCustomVjpCall<T, C, O, F> for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
    O: Operation<T>,
{
    #[inline]
    fn custom_vjp_call_operation(
        backward: crate::programs::Program<T, C, O, Vec<C>, Vec<C>>,
        tangent: Option<crate::programs::Program<T, C, O, Vec<C>, Vec<C>>>,
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

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    super::dot::SupportsLeftDot<ArrayType, F> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn left_dot_operation(factor: F, dimensions: DotDimensionNumbers) -> Self {
        LinearArrayOperation::LeftDot { factor, dimensions }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    super::dot::SupportsRightDot<ArrayType, F> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn right_dot_operation(factor: F, dimensions: DotDimensionNumbers) -> Self {
        LinearArrayOperation::RightDot { factor, dimensions }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsReshape<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn reshape_operation(output_shape: Shape) -> Self {
        LinearArrayOperation::Reshape { output_shape }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsBroadcast<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn broadcast_operation(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        LinearArrayOperation::Broadcast { output_type, output_axes }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsReduce<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn reduce_operation(axes: Vec<usize>, kind: ReductionKind) -> Self {
        LinearArrayOperation::Reduce { axes, kind }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsLinearSelect<ArrayType, F>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn linear_select_operation(condition: F) -> Self {
        LinearArrayOperation::Select { condition }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    From<ConditionOperation<V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>, ArrayType>>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn from(op: ConditionOperation<V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>, ArrayType>) -> Self {
        LinearArrayOperation::Condition(Box::new(op))
    }
}

impl<T, V, Extension> ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    Extension: Operation<T>,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::One(one) => one.name(),
            Self::Constant(constant) => constant.name(),
            Self::Fill(fill) => fill.name(),
            Self::ZeroLike => ZERO_LIKE_OPERATION_NAME,
            Self::OneLike => ONE_LIKE_OPERATION_NAME,
            Self::Add => ADD_OPERATION_NAME,
            Self::Sub => SUB_OPERATION_NAME,
            Self::Mul => MUL_OPERATION_NAME,
            Self::Div => DIV_OPERATION_NAME,
            Self::Neg => NEG_OPERATION_NAME,
            Self::Sin => SIN_OPERATION_NAME,
            Self::Cos => COS_OPERATION_NAME,
            Self::StopGradient => STOP_GRADIENT_OPERATION_NAME,
            Self::RematerializationName(_) => {
                crate::tracing_v2::rematerialization::REMATERIALIZATION_NAME_OPERATION_NAME
            }
            Self::TransferToMemory(_) => TRANSFER_TO_MEMORY_OPERATION_NAME,
            Self::Dot { .. } => "dot",
            Self::Transpose { .. } => "transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::Reshape { .. } => "reshape",
            Self::Broadcast { .. } => "broadcast",
            Self::Reduce { kind, .. } => match kind {
                ReductionKind::Sum => "reduce_sum",
                ReductionKind::Mean => "reduce_mean",
                ReductionKind::Max => "reduce_max",
                ReductionKind::Min => "reduce_min",
                ReductionKind::Any => "reduce_any",
                ReductionKind::All => "reduce_all",
            },
            Self::Compare { .. } => "compare",
            Self::Not => NOT_OPERATION_NAME,
            Self::And => AND_OPERATION_NAME,
            Self::Or => OR_OPERATION_NAME,
            Self::Xor => XOR_OPERATION_NAME,
            Self::Collective { kind, .. } => match kind {
                CollectiveKind::PSum => "psum",
                CollectiveKind::PMean => "pmean",
                CollectiveKind::PMax => "pmax",
            },
            Self::Select => "select",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::CustomJvp(_) => "custom_jvp",
            Self::CustomVjp(_) => "custom_vjp",
            Self::Extension(extension) => extension.name(),
        }
    }
}

impl<T, V, C, Extension, F, O> LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
    Extension: Operation<T>,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::One(one) => one.name(),
            Self::Constant(constant) => constant.name(),
            Self::Fill(fill) => fill.name(),
            Self::ZeroLike => ZERO_LIKE_OPERATION_NAME,
            Self::OneLike => ONE_LIKE_OPERATION_NAME,
            Self::Add => ADD_OPERATION_NAME,
            Self::Sub => SUB_OPERATION_NAME,
            Self::Mul => MUL_OPERATION_NAME,
            Self::TransferToMemory { .. } => TRANSFER_TO_MEMORY_OPERATION_NAME,
            Self::Neg => NEG_OPERATION_NAME,
            Self::Transpose { .. } => "transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::LeftDot { .. } => "left_dot",
            Self::RightDot { .. } => "right_dot",
            Self::Reshape { .. } => "reshape",
            Self::Broadcast { .. } => "broadcast",
            Self::Reduce { kind, .. } => match kind {
                ReductionKind::Sum => "reduce_sum",
                ReductionKind::Mean => "reduce_mean",
                ReductionKind::Max => "reduce_max",
                ReductionKind::Min => "reduce_min",
                ReductionKind::Any => "reduce_any",
                ReductionKind::All => "reduce_all",
            },
            Self::Select { .. } => SELECT_OPERATION_NAME,
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::CustomVjpCall(call) => {
                if call.transposed() {
                    "custom_vjp_backward"
                } else {
                    "custom_vjp_tangent"
                }
            }
            Self::Extension(extension) => extension.name(),
        }
    }
}

impl<T, V, Extension> Display for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    Extension: Operation<T>,
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T, V, C, Extension, F, O> Display for LinearArrayOperation<V, C, T, Extension, F, O>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
    C: Value<T>,
    F: Value<T>,
    Extension: Operation<T>,
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

fn unsupported_scalar_metadata_operation(operation_name: &'static str) -> TypeError {
    TypeError { message: format!("{operation_name} is not supported for scalar data type metadata") }
}

fn symbolic_zero_one_error<T: Type>(r#type: &T) -> TypeError {
    TypeError { message: format!("zero tangent space has no one value for {type}", type = r#type) }
}

fn symbolic_zero_constant_error<T: Type, F: Display>(r#type: &T, value: &F) -> TypeError {
    TypeError { message: format!("zero tangent space has no constant value {value} for {type}", type = r#type) }
}

fn infer_zero_only_tangent_output_types<T: Type, O: Operation<T>>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
) -> Result<Vec<T>, ProgramError> {
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    Ok(operation.infer_output_types(input_types.as_slice())?)
}

fn interpret_zero_only_tangent_operation<T: Type, O: Operation<T>>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
) -> Result<Vec<Tangent<T, Infallible>>, ProgramError> {
    Ok(infer_zero_only_tangent_output_types(operation, inputs)?.into_iter().map(Tangent::zero).collect())
}

fn reject_zero_only_tangent_one_operation<T: Type, O: Operation<T>>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
) -> Result<Vec<Tangent<T, Infallible>>, ProgramError> {
    let output_types = infer_zero_only_tangent_output_types(operation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
    Err(symbolic_zero_one_error(&output_types[0]).into())
}

fn reject_zero_only_tangent_constant_operation<T: Type, O: Operation<T>, F: Display>(
    operation: &O,
    inputs: &[Tangent<T, Infallible>],
    value: &F,
) -> Result<Vec<Tangent<T, Infallible>>, ProgramError> {
    let output_types = infer_zero_only_tangent_output_types(operation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
    Err(symbolic_zero_constant_error(&output_types[0], value).into())
}

fn infer_tangent_value_output_types<T: Type, V: Value<T>, O: Operation<T>>(
    operation: &O,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<T>, ProgramError> {
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    Ok(operation.infer_output_types(input_types.as_slice())?)
}

fn symbolic_zero_tangent_value_outputs<T: Type, V: Value<T>>(output_types: Vec<T>) -> Vec<Tangent<T, V>> {
    output_types.into_iter().map(Tangent::Zero).collect()
}

fn interpret_materialized_tangent_value_operation<T: Type, V: Value<T> + Zero<T>, O: InterpretableOperation<T, V>>(
    operation: &O,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError> {
    let materialized_inputs = inputs
        .iter()
        .map(|input| match input {
            Tangent::Zero(r#type) => V::zero(r#type),
            Tangent::Value(value) => Ok(value.clone()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(operation.interpret(materialized_inputs.as_slice())?.into_iter().map(Tangent::Value).collect())
}

fn tangent_value_type_matches<T: Parameter + PartialEq + Type, V: Value<T>>(value: &V, output_type: &T) -> bool {
    value.r#type().as_ref() == output_type
}

fn interpret_tangent_value_add<T: Parameter + PartialEq + Type, V: Value<T> + Add<Output = V> + Zero<T>>(
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    AddOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&AddOperation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
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

fn interpret_tangent_value_mul<T: Parameter + PartialEq + Type, V: Value<T> + Mul<Output = V> + Zero<T>>(
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    MulOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&MulOperation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
    // If either operand is symbolic zero, the product is zero (this is the linear-side rule that
    // multiplying by a zero constant yields zero).
    if inputs.iter().any(Tangent::is_zero) {
        return Ok(symbolic_zero_tangent_value_outputs(output_types));
    }
    interpret_materialized_tangent_value_operation(&MulOperation, inputs)
}

fn interpret_tangent_value_sub<
    T: Parameter + PartialEq + Type,
    V: Value<T> + Neg<Output = V> + Sub<Output = V> + Zero<T>,
>(
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    SubOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&SubOperation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
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

fn interpret_tangent_value_neg<T: Type, V: Value<T> + Neg<Output = V>>(
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    NegOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&NegOperation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
    match inputs {
        [Tangent::Zero(_)] => Ok(symbolic_zero_tangent_value_outputs(output_types)),
        [Tangent::Value(value)] => {
            Ok(NegOperation.interpret(std::slice::from_ref(value))?.into_iter().map(Tangent::Value).collect())
        }
        _ => unreachable!("neg output type inference validates the input count"),
    }
}

fn interpret_tangent_value_zero_like<T: Type, V: Value<T>, O: Operation<T>>(
    operation: &O,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError> {
    Ok(symbolic_zero_tangent_value_outputs(infer_tangent_value_output_types(operation, inputs)?))
}

fn interpret_tangent_value_constant<T, V>(
    operation: &ConstantOperation<T, Tangent<T, V>>,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    check_count!("input", inputs, 0, ProgramError);
    Ok(vec![operation.value().clone()])
}

fn interpret_tangent_value_one_like<T: Type, V: Value<T> + OneLike>(
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    OneLikeOperation: Operation<T>,
{
    let output_types = infer_tangent_value_output_types(&OneLikeOperation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
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
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    T: Type,
    V: Value<T> + Scale<Output = V>,
    O: Operation<T>,
    ScaleOperation<T, V>: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(operation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
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

/// Transposes a captured-condition select (the `Select` variant of [`LinearArrayOperation`]).
///
/// The forward linear map is `(t, f) ↦ select(condition, t, f)`. Its transpose routes the output cotangent into the
/// branch that the condition selected: the `on_true` cotangent is `select(condition, cotangent, 0)` and the
/// `on_false` cotangent is `select(condition, 0, cotangent)`. The zero operand is staged as a typed `Zero` operation
/// via [`stage_cotangent`](crate::tracing_v2::operations::control_flow::stage_cotangent), and `make_operation`
/// rebuilds the captured-condition select for staging into the transpose builder.
fn transpose_captured_condition_select<'transpose, T, V, O, MakeOperationFn>(
    make_operation: MakeOperationFn,
    context: &mut AbstractTracingContext<'transpose, T, V, O>,
    input_types: &[&T],
    output_cotangents: &[Cotangent<'transpose, T, V, O>],
) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError>
where
    T: Type,
    V: Value<T>,
    O: Operation<T> + SupportsZero<T>,
    MakeOperationFn: Fn() -> O,
{
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_cotangents, 1, ProgramError);
    match &output_cotangents[0] {
        Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
        Cotangent::Staged(cotangent) => {
            let zero =
                crate::tracing_v2::operations::control_flow::stage_cotangent(context, &Cotangent::Zero, input_types[0]);
            let on_true = context.stage_operation(make_operation(), &[cotangent.clone(), zero.clone()])?;
            check_count!("output", on_true, 1, ProgramError);
            let on_false = context.stage_operation(make_operation(), &[zero, cotangent.clone()])?;
            check_count!("output", on_false, 1, ProgramError);
            Ok(vec![
                Cotangent::Staged(on_true.into_iter().next().unwrap()),
                Cotangent::Staged(on_false.into_iter().next().unwrap()),
            ])
        }
    }
}

fn interpret_tangent_value_unary_value_or_zero<T, V, MetadataOperation, ConcreteOperation>(
    metadata_operation: &MetadataOperation,
    concrete_operation: &ConcreteOperation,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    T: Type,
    V: Value<T>,
    MetadataOperation: Operation<T>,
    ConcreteOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(metadata_operation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
    match inputs {
        [Tangent::Zero(_)] => Ok(symbolic_zero_tangent_value_outputs(output_types)),
        [Tangent::Value(input)] => {
            Ok(concrete_operation.interpret(std::slice::from_ref(input))?.into_iter().map(Tangent::Value).collect())
        }
        _ => unreachable!("unary output type inference validates the input count"),
    }
}

impl<V: Value<ArrayType>, Extension: Operation<ArrayType>> Operation<ArrayType>
    for ArrayOperation<V, ArrayType, Extension>
{
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::Constant(constant) => constant.infer_output_types(input_types),
            Self::Fill(fill) => fill.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Div => DivOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::StopGradient => StopGradientOperation.infer_output_types(input_types),
            Self::RematerializationName(operation) => operation.infer_output_types(input_types),
            Self::TransferToMemory(operation) => operation.infer_output_types(input_types),
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).infer_output_types(input_types),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).infer_output_types(input_types)
            }
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).infer_output_types(input_types)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).infer_output_types(input_types)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).infer_output_types(input_types),
            Self::Compare { direction } => CompareOperation::new(*direction).infer_output_types(input_types),
            Self::Not => NotOperation.infer_output_types(input_types),
            Self::And => AndOperation.infer_output_types(input_types),
            Self::Or => OrOperation.infer_output_types(input_types),
            Self::Xor => XorOperation.infer_output_types(input_types),
            Self::Collective { axis_name, kind } => {
                CollectiveOperation::new(axis_name.clone(), *kind).infer_output_types(input_types)
            }
            Self::Select => SelectOperation.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::CustomJvp(operation) => operation.infer_output_types(input_types),
            Self::CustomVjp(operation) => operation.infer_output_types(input_types),
            Self::Extension(extension) => extension.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).render(formatter, indentation),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).render(formatter, indentation)
            }
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).render(formatter, indentation)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).render(formatter, indentation)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).render(formatter, indentation),
            Self::Compare { direction } => {
                Operation::<ArrayType>::render(&CompareOperation::new(*direction), formatter, indentation)
            }
            Self::Collective { axis_name, kind } => {
                CollectiveOperation::new(axis_name.clone(), *kind).render(formatter, indentation)
            }
            Self::Scale { factor, .. } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Fill(fill) => fill.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => formatter.write_str(self.name()),
        }
    }
}

impl<V: Value<DataType>, Extension: Operation<DataType>> Operation<DataType>
    for ArrayOperation<V, DataType, Extension>
{
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::CustomJvp(_) | Self::CustomVjp(_) => {
                Err(unsupported_scalar_metadata_operation(self.operation_name()))
            }
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::Constant(constant) => constant.infer_output_types(input_types),
            Self::Fill(fill) => fill.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Div => DivOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::StopGradient => StopGradientOperation.infer_output_types(input_types),
            Self::RematerializationName(operation) => operation.infer_output_types(input_types),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Extension(extension) => extension.infer_output_types(input_types),
            Self::TransferToMemory(_)
            | Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Compare { .. }
            | Self::Not
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Collective { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor, .. } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Fill(fill) => fill.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => formatter.write_str(self.name()),
        }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension: Operation<ArrayType>, F: Value<ArrayType>, O>
    Operation<ArrayType> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
where
    O: Operation<ArrayType>,
{
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::Constant(constant) => constant.infer_output_types(input_types),
            Self::Fill(fill) => fill.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).infer_output_types(input_types)
            }
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::LeftDot { factor, dimensions } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone()).infer_output_types(input_types)
            }
            Self::RightDot { factor, dimensions } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone()).infer_output_types(input_types)
            }
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).infer_output_types(input_types)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).infer_output_types(input_types)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).infer_output_types(input_types),
            Self::Select { condition } => {
                check_count!("input", input_types, 2, TypeError);
                SelectOperation.infer_output_types(&[
                    condition.r#type().into_owned(),
                    input_types[0].clone(),
                    input_types[1].clone(),
                ])
            }
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::TransferToMemory { destination } => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone().with_memory(*destination)])
            }
            Self::CustomVjpCall(call) => call.infer_output_types(input_types),
            Self::Extension(extension) => extension.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).render(formatter, indentation)
            }
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).render(formatter, indentation)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).render(formatter, indentation)
            }
            Self::Scale { factor, .. } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Fill(fill) => fill.render(formatter, indentation),
            Self::LeftDot { factor, dimensions } | Self::RightDot { factor, dimensions } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("factor", factor)?;
                    operation.field("dimensions", dimensions)
                })
            }
            Self::Select { condition } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("condition", condition)),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => formatter.write_str(self.name()),
        }
    }
}

impl<V: Value<DataType>, C: Value<DataType>, Extension: Operation<DataType>, F: Value<DataType>, O> Operation<DataType>
    for LinearArrayOperation<V, C, DataType, Extension, F, O>
where
    O: Operation<DataType>,
{
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::CustomVjpCall(_) => Err(unsupported_scalar_metadata_operation(self.operation_name())),
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::Constant(constant) => constant.infer_output_types(input_types),
            Self::Fill(fill) => fill.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Extension(extension) => extension.infer_output_types(input_types),
            Self::TransferToMemory { .. }
            | Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Constant(constant) => constant.render(formatter, indentation),
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor, .. } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Fill(fill) => fill.render(formatter, indentation),
            Self::LeftDot { factor, dimensions } | Self::RightDot { factor, dimensions } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("factor", factor)?;
                    operation.field("dimensions", dimensions)
                })
            }
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => formatter.write_str(self.name()),
        }
    }
}

impl<V, C, Extension, F, O> FactorParameterizedOperation<ArrayType, F>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    F: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    type WithFactor<MappedFactor: Value<ArrayType>> = LinearArrayOperation<V, C, ArrayType, Extension, MappedFactor, O>;

    fn try_map_factors<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        match self {
            Self::CustomVjpCall(call) => {
                Ok(LinearArrayOperation::CustomVjpCall(Box::new(call.map_factors(map_factor)?)))
            }
            Self::Zero(zero) => Ok(LinearArrayOperation::Zero(zero.clone())),
            Self::One(one) => Ok(LinearArrayOperation::One(one.clone())),
            Self::Constant(constant) => Ok(LinearArrayOperation::Constant(constant.clone())),
            Self::Fill(fill) => Ok(LinearArrayOperation::Fill(fill.clone())),
            Self::ZeroLike => Ok(LinearArrayOperation::ZeroLike),
            Self::OneLike => Ok(LinearArrayOperation::OneLike),
            Self::Add => Ok(LinearArrayOperation::Add),
            Self::Sub => Ok(LinearArrayOperation::Sub),
            Self::Neg => Ok(LinearArrayOperation::Neg),
            Self::Mul => Ok(LinearArrayOperation::Mul),
            Self::TransferToMemory { destination } => {
                Ok(LinearArrayOperation::TransferToMemory { destination: *destination })
            }
            Self::Transpose { permutation } => Ok(LinearArrayOperation::Transpose { permutation: permutation.clone() }),
            Self::Scale { factor, .. } => Ok(LinearArrayOperation::Scale { factor: map_factor(factor)? }),
            Self::LeftDot { factor, dimensions } => {
                Ok(LinearArrayOperation::LeftDot { factor: map_factor(factor)?, dimensions: dimensions.clone() })
            }
            Self::RightDot { factor, dimensions } => {
                Ok(LinearArrayOperation::RightDot { factor: map_factor(factor)?, dimensions: dimensions.clone() })
            }
            Self::Reshape { output_shape } => Ok(LinearArrayOperation::Reshape { output_shape: output_shape.clone() }),
            Self::Broadcast { output_type, output_axes } => Ok(LinearArrayOperation::Broadcast {
                output_type: output_type.clone(),
                output_axes: output_axes.clone(),
            }),
            Self::Reduce { axes, kind } => Ok(LinearArrayOperation::Reduce { axes: axes.clone(), kind: *kind }),
            Self::Select { condition } => Ok(LinearArrayOperation::Select { condition: map_factor(condition)? }),
            Self::Condition(condition) => {
                let true_branch =
                    condition.true_branch().map_operations(|operation| operation.try_map_factors(map_factor))?;
                let false_branch =
                    condition.false_branch().map_operations(|operation| operation.try_map_factors(map_factor))?;
                let condition = match condition.predicate() {
                    ConditionPredicate::RuntimeInput(predicate_type) => {
                        ConditionOperation::new(predicate_type.clone(), true_branch, false_branch)?
                    }
                    ConditionPredicate::Captured(predicate) => {
                        ConditionOperation::with_captured_predicate(*predicate, true_branch, false_branch)?
                    }
                };
                Ok(LinearArrayOperation::Condition(Box::new(condition)))
            }
            Self::While(while_operation) => {
                let condition =
                    while_operation.condition().map_operations(|operation| operation.try_map_factors(map_factor))?;
                let body = while_operation.body().map_operations(|operation| operation.try_map_factors(map_factor))?;
                Ok(LinearArrayOperation::While(Box::new(WhileOperation::new(condition, body)?)))
            }
            Self::Extension(extension) => Ok(LinearArrayOperation::Extension(extension.clone())),
        }
    }
}

impl<V: Value<DataType>> InterpretableOperation<DataType, V> for ScalarOperation<V>
where
    V: BooleanLike
        + SupportsArithmeticOperations
        + SupportsTrigonometricOperations
        + SupportsConstantOperations<DataType>
        + Compare<Output = V>
        + Select<Condition = bool>,
    Vec<V>: Parameterized<
            V,
            Family: crate::parameters::ParameterizedFamily<V>,
            To<V> = Vec<V>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::Constant(constant) => constant.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Div => <DivOperation as InterpretableOperation<DataType, V>>::interpret(&DivOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Sin => <SinOperation as InterpretableOperation<DataType, V>>::interpret(&SinOperation, inputs),
            Self::Cos => <CosOperation as InterpretableOperation<DataType, V>>::interpret(&CosOperation, inputs),
            Self::Compare { direction } => <CompareOperation as InterpretableOperation<DataType, V>>::interpret(
                &CompareOperation::new(*direction),
                inputs,
            ),
            Self::Select => {
                <SelectOperation as InterpretableOperation<DataType, V>>::interpret(&SelectOperation, inputs)
            }
            Self::StopGradient => <StopGradientOperation as InterpretableOperation<DataType, V>>::interpret(
                &StopGradientOperation,
                inputs,
            ),
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as InterpretableOperation<DataType, V>>::interpret(operation, inputs)
            }
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::CustomJvp(operation) => operation.interpret(inputs),
            Self::CustomVjp(operation) => operation.interpret(inputs),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> InterpretableOperation<DataType, ZeroScalarTangent>
    for LinearScalarOperation<C, F>
{
    fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, ProgramError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Constant(constant) => reject_zero_only_tangent_constant_operation(self, inputs, constant.value()),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Value<DataType>> InterpretableOperation<DataType, Tangent<DataType, V>>
    for LinearScalarOperation<V, Tangent<DataType, V>>
where
    V: SupportsLinearArithmeticOperations + Zero<DataType> + One<DataType> + OneLike,
{
    fn interpret(&self, inputs: &[Tangent<DataType, V>]) -> Result<Vec<Tangent<DataType, V>>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback interpretation over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::Zero(zero) => Ok(vec![Tangent::Zero(*zero.r#type())]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(one.r#type())?)]),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Scale { factor, .. } => interpret_tangent_value_scale(self, factor, inputs),
        }
    }
}

impl<V: Value<DataType>, F> InterpretableOperation<DataType, V> for LinearScalarOperation<V, F>
where
    V: SupportsLinearArithmeticOperations + SupportsConstantOperations<DataType> + Scale<F, Output = V>,
    F: CustomVjpResidual<DataType, V>,
    ScalarOperation<V>: InterpretableOperation<DataType, V>,
    ScaleOperation<DataType, F>: InterpretableOperation<DataType, V>,
    ConstantOperation<DataType, F>: InterpretableOperation<DataType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret(inputs),
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::Constant(constant) => constant.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
        }
    }
}

impl<C, S> InterpretableOperation<DataType, Tracer<S>> for LinearScalarOperation<C, Tracer<S>>
where
    C: Value<DataType>,
    S: StagingContext<Type = DataType, Constant = C, Operation = ScalarOperation<C>>,
    Tracer<S>: Add<Output = Tracer<S>>
        + Sub<Output = Tracer<S>>
        + Neg<Output = Tracer<S>>
        + Mul<Output = Tracer<S>>
        + ZeroLike
        + OneLike,
    Vec<Tracer<S>>: Parameterized<Tracer<S>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret_over_tracers(inputs),
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    zero.r#type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    one.r#type()
                ),
            }
            .into()),
            Self::Constant(constant) => Err(TypeError {
                message: format!(
                    "linear constant operation over tracer values was not materialized before interpretation for {}",
                    constant.value().r#type()
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&SubOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor, .. } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
        }
    }
}

/// [`InterpretableOperation`] for [`ArrayOperation`] requires the full union of value capabilities exercised by the
/// closed default ordinary operation enum.
///
/// The value-side bound list is expressed via the orthogonal capability bundles defined in [`super::bounds`] (one
/// per operation category — arithmetic, trigonometric, constants, manipulation, comparison) plus the few singleton
/// traits ([`Fill<ArrayType, f64>`], [`DotOps`], [`Select`], [`BooleanLike`]) that the dispatcher requires
/// directly. Each impl site composes only the categories it actually exercises, so downstream consumers never
/// depend on a single monolithic value-bundle trait.
impl<V: Value<ArrayType>, Extension> InterpretableOperation<ArrayType, V> for ArrayOperation<V, ArrayType, Extension>
where
    V: Parameter
        + SupportsArithmeticOperations
        + SupportsTrigonometricOperations
        + SupportsConstantOperations<ArrayType>
        + Fill<ArrayType, f64>
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = V>
        + BooleanLike,
    Extension: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomJvp(operation) => operation.interpret(inputs),
            Self::CustomVjp(operation) => operation.interpret(inputs),
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::Constant(constant) => constant.interpret(inputs),
            Self::Fill(fill) => fill.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Sub => SubOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Div => DivOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::StopGradient => StopGradientOperation.interpret(inputs),
            Self::RematerializationName(operation) => operation.interpret(inputs),
            Self::TransferToMemory(operation) => operation.interpret(inputs),
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).interpret(inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).interpret(inputs),
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).interpret(inputs)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).interpret(inputs),
            Self::Compare { direction } => CompareOperation::new(*direction).interpret(inputs),
            Self::Not => NotOperation.interpret(inputs),
            Self::And => AndOperation.interpret(inputs),
            Self::Or => OrOperation.interpret(inputs),
            Self::Xor => XorOperation.interpret(inputs),
            Self::Collective { axis_name, kind } => {
                CollectiveOperation::new(axis_name.clone(), *kind).interpret(inputs)
            }
            Self::Select => SelectOperation.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<'domain, D, C, V, Extension> InterpretableOperation<ArrayType, Tracer<TracingContext<'domain, D, C>>>
    for ArrayOperation<V, ArrayType, Extension>
where
    D: Domain<Type = ArrayType, Value = V, Operation = ArrayOperation<V, ArrayType, Extension>>,
    V: Value<ArrayType>,
    Extension: Clone + InterpretableOperation<ArrayType, Tracer<TracingContext<'domain, D, C>>>,
{
    fn interpret(
        &self,
        inputs: &[Tracer<TracingContext<'domain, D, C>>],
    ) -> Result<Vec<Tracer<TracingContext<'domain, D, C>>>, ProgramError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "typed zero operation over tracer values was not materialized before interpretation for {}",
                    zero.r#type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "typed one operation over tracer values was not materialized before interpretation for {}",
                    one.r#type()
                ),
            }
            .into()),
            Self::Constant(constant) => Err(TypeError {
                message: format!(
                    "typed constant operation over tracer values was not materialized before interpretation for {}",
                    constant.value().r#type()
                ),
            }
            .into()),
            Self::Fill(fill) => Err(TypeError {
                message: format!(
                    "typed fill operation over tracer values was not materialized before interpretation for {}",
                    fill.r#type()
                ),
            }
            .into()),
            Self::Extension(extension) => extension.interpret(inputs),
            _ => {
                let exemplar = inputs.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
                exemplar.context().stage_operation(self.clone(), inputs)
            }
        }
    }
}

impl<V: Value<DataType>, Extension> InterpretableOperation<DataType, V> for ArrayOperation<V, DataType, Extension>
where
    V: Parameter
        + SupportsArithmeticOperations
        + SupportsTrigonometricOperations
        + SupportsConstantOperations<DataType>
        + Fill<DataType, f64>,
    Extension: InterpretableOperation<DataType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomJvp(_) | Self::CustomVjp(_) => {
                Err(unsupported_scalar_metadata_operation(self.operation_name()).into())
            }
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::Constant(constant) => constant.interpret(inputs),
            Self::Fill(fill) => fill.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Div => <DivOperation as InterpretableOperation<DataType, V>>::interpret(&DivOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Sin => <SinOperation as InterpretableOperation<DataType, V>>::interpret(&SinOperation, inputs),
            Self::Cos => <CosOperation as InterpretableOperation<DataType, V>>::interpret(&CosOperation, inputs),
            Self::StopGradient => <StopGradientOperation as InterpretableOperation<DataType, V>>::interpret(
                &StopGradientOperation,
                inputs,
            ),
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as InterpretableOperation<DataType, V>>::interpret(operation, inputs)
            }
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
            Self::TransferToMemory(_)
            | Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Compare { .. }
            | Self::Not
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Collective { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<C: Value<ArrayType>, Extension, O> InterpretableOperation<ArrayType, ZeroArrayTangent>
    for LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>
where
    Extension: InterpretableOperation<ArrayType, ZeroArrayTangent>,
    O: Operation<ArrayType>,
{
    fn interpret(&self, inputs: &[ZeroArrayTangent]) -> Result<Vec<ZeroArrayTangent>, ProgramError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::Fill(fill) if *fill.value() == 0.0 => interpret_zero_only_tangent_operation(self, inputs),
            Self::Fill(fill) => reject_zero_only_tangent_constant_operation(self, inputs, fill.value()),
            Self::Condition(condition) => {
                let output_types = infer_zero_only_tangent_output_types(self, inputs)?;
                let branch = match condition.predicate() {
                    ConditionPredicate::Captured(predicate) => {
                        if *predicate {
                            condition.true_branch()
                        } else {
                            condition.false_branch()
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
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::While(while_operation) => {
                let output_types = infer_zero_only_tangent_output_types(self, inputs)?;
                let condition_outputs = while_operation.condition().interpret(inputs.to_vec())?;
                check_count!("output", condition_outputs, 1, ProgramError);
                let outputs = while_operation.body().interpret(inputs.to_vec())?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::Extension(extension) => extension.interpret(inputs),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Value<ArrayType>, Extension, O> InterpretableOperation<ArrayType, Tangent<ArrayType, V>>
    for LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>
where
    V: Parameter
        + SupportsLinearArithmeticOperations
        + Zero<ArrayType>
        + One<ArrayType>
        + Fill<ArrayType, f64>
        + OneLike
        + SupportsLinearAlgebraOperations
        + SupportsManipulationOperations
        + Select<Condition = V>
        + BooleanLike,
    Extension: InterpretableOperation<ArrayType, Tangent<ArrayType, V>>,
    O: Operation<ArrayType>,
{
    fn interpret(&self, inputs: &[Tangent<ArrayType, V>]) -> Result<Vec<Tangent<ArrayType, V>>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback interpretation over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::TransferToMemory { destination } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![match &inputs[0] {
                    Tangent::Zero(r#type) => Tangent::Zero(r#type.clone().with_memory(*destination)),
                    Tangent::Value(value) => Tangent::Value(value.clone()),
                }])
            }
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type().clone())]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(one.r#type())?)]),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::Fill(fill) if *fill.value() == 0.0 => Ok(vec![Tangent::Zero(fill.r#type().clone())]),
            Self::Fill(fill) => Ok(vec![Tangent::Value(V::fill(fill.r#type(), *fill.value())?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Mul => interpret_tangent_value_mul(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Transpose { permutation } => {
                let op = TransposeOperation::new(permutation.clone());
                interpret_tangent_value_unary_value_or_zero(&op, &op, inputs)
            }
            Self::Scale { factor, .. } => interpret_tangent_value_scale(self, factor, inputs),
            Self::Broadcast { output_type, output_axes } => {
                let op = BroadcastOperation::new(output_type.clone(), output_axes.clone());
                interpret_tangent_value_unary_value_or_zero(&op, &op, inputs)
            }
            Self::LeftDot { factor, dimensions } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
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
                check_count!("output", output_types, 1, ProgramError);
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
            Self::Reshape { output_shape } => interpret_tangent_value_unary_value_or_zero(
                &ReshapeOperation::new(output_shape.clone()),
                &ReshapeOperation::new(output_shape.clone()),
                inputs,
            ),
            Self::Reduce { axes, kind } => {
                let op = ReduceOperation::new(axes.clone(), *kind);
                interpret_tangent_value_unary_value_or_zero(&op, &op, inputs)
            }
            Self::Select { condition } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                let Tangent::Value(condition) = condition else {
                    return Err(TypeError {
                        message: format!("captured select condition for {} must be a concrete value", output_types[0],),
                    }
                    .into());
                };
                Ok(vec![Tangent::select(condition.clone(), inputs[0].clone(), inputs[1].clone())?])
            }
            Self::Condition(condition) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let (predicate, operands) = match condition.predicate() {
                    ConditionPredicate::RuntimeInput(_) => {
                        let predicate = match &inputs[0] {
                            Tangent::Zero(_) => {
                                return Err(ControlFlowError::MissingTransformRule {
                                    transform: "runtime-predicate mixed symbolic-zero condition interpretation",
                                }
                                .into());
                            }
                            Tangent::Value(predicate) => predicate.boolean()?,
                        };
                        (predicate, &inputs[1..])
                    }
                    ConditionPredicate::Captured(predicate) => (*predicate, inputs),
                };
                let branch = if predicate { condition.true_branch() } else { condition.false_branch() };
                let outputs = branch.interpret(operands.to_vec())?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::While(while_operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let mut state = inputs.to_vec();
                loop {
                    let condition_outputs = while_operation.condition().interpret(state.clone())?;
                    check_count!("output", condition_outputs, 1, ProgramError);
                    let predicate = match &condition_outputs[0] {
                        Tangent::Zero(_) => {
                            return Err(ControlFlowError::MissingTransformRule {
                                transform: "mixed symbolic-zero while predicate interpretation",
                            }
                            .into());
                        }
                        Tangent::Value(predicate) => predicate.boolean()?,
                    };
                    if !predicate {
                        check_count!("output", state, output_types.len(), ProgramError);
                        return Ok(state);
                    }
                    state = while_operation.body().interpret(state)?;
                    check_count!("output", state, while_operation.state_types().len(), ProgramError);
                }
            }
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<V: Value<ArrayType>, Extension, F: Value<ArrayType>, O> InterpretableOperation<ArrayType, V>
    for LinearArrayOperation<V, V, ArrayType, Extension, F, O>
where
    V: Parameter
        + SupportsLinearArithmeticOperations
        + SupportsConstantOperations<ArrayType>
        + Fill<ArrayType, f64>
        + SupportsLinearAlgebraOperations
        + Scale<F, Output = V>
        + super::dot::LeftDot<F>
        + super::dot::RightDot<F>
        + SupportsManipulationOperations
        + Select<Condition = V>
        + BooleanLike,
    ScaleOperation<ArrayType, F>: InterpretableOperation<ArrayType, V>,
    super::dot::LeftDotOperation<F>: InterpretableOperation<ArrayType, V>,
    super::dot::RightDotOperation<F>: InterpretableOperation<ArrayType, V>,
    Extension: InterpretableOperation<ArrayType, V>,
    ArrayOperation<V, ArrayType, Extension>: InterpretableOperation<ArrayType, V>,
    F: CustomVjpResidual<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret(inputs),
            Self::TransferToMemory { .. } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone()])
            }
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::Constant(constant) => constant.interpret(inputs),
            Self::Fill(fill) => fill.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Sub => SubOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftDot { factor, dimensions } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::RightDot { factor, dimensions } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).interpret(inputs),
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).interpret(inputs)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).interpret(inputs),
            Self::Select { condition } => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![V::select(condition.residual_value()?, inputs[0].clone(), inputs[1].clone())?])
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<C: Value<DataType>, Extension, O> InterpretableOperation<DataType, ZeroScalarTangent>
    for LinearArrayOperation<ZeroScalarTangent, C, DataType, Extension, ZeroScalarTangent, O>
where
    Extension: InterpretableOperation<DataType, ZeroScalarTangent>,
    O: Operation<DataType>,
{
    fn interpret(&self, inputs: &[ZeroScalarTangent]) -> Result<Vec<ZeroScalarTangent>, ProgramError> {
        match self {
            Self::One(_) | Self::OneLike => reject_zero_only_tangent_one_operation(self, inputs),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::Fill(fill) if *fill.value() == 0.0 => interpret_zero_only_tangent_operation(self, inputs),
            Self::Fill(fill) => reject_zero_only_tangent_constant_operation(self, inputs, fill.value()),
            Self::Extension(extension) => extension.interpret(inputs),
            _ => interpret_zero_only_tangent_operation(self, inputs),
        }
    }
}

impl<V: Value<DataType>, Extension, O> InterpretableOperation<DataType, Tangent<DataType, V>>
    for LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>
where
    V: SupportsLinearArithmeticOperations + Zero<DataType> + One<DataType> + Fill<DataType, f64> + OneLike,
    Extension: InterpretableOperation<DataType, Tangent<DataType, V>>,
    O: Operation<DataType>,
{
    fn interpret(&self, inputs: &[Tangent<DataType, V>]) -> Result<Vec<Tangent<DataType, V>>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Zero(zero) => Ok(vec![Tangent::Zero(*zero.r#type())]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(one.r#type())?)]),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::Fill(fill) if *fill.value() == 0.0 => Ok(vec![Tangent::Zero(*fill.r#type())]),
            Self::Fill(fill) => Ok(vec![Tangent::Value(V::fill(fill.r#type(), *fill.value())?)]),
            Self::ZeroLike => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike => interpret_tangent_value_one_like(inputs),
            Self::Add => interpret_tangent_value_add(inputs),
            Self::Sub => interpret_tangent_value_sub(inputs),
            Self::Mul => interpret_tangent_value_mul(inputs),
            Self::Neg => interpret_tangent_value_neg(inputs),
            Self::Scale { factor, .. } => interpret_tangent_value_scale(self, factor, inputs),
            Self::TransferToMemory { .. }
            | Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<V: Value<DataType>, C: Value<DataType>, Extension, F: Value<DataType>, O> InterpretableOperation<DataType, V>
    for LinearArrayOperation<V, C, DataType, Extension, F, O>
where
    V: SupportsLinearArithmeticOperations
        + SupportsConstantOperations<DataType>
        + Fill<DataType, f64>
        + Scale<F, Output = V>,
    ScaleOperation<DataType, F>: InterpretableOperation<DataType, V>,
    Extension: InterpretableOperation<DataType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Operation<DataType>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::Constant(constant) => constant.interpret(inputs),
            Self::Fill(fill) => fill.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Sub => <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::TransferToMemory { .. }
            | Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<C, S, Extension, O> InterpretableOperation<ArrayType, Tracer<S>>
    for LinearArrayOperation<Tracer<S>, C, ArrayType, Extension, Tracer<S>, O>
where
    C: Value<ArrayType>,
    S: StagingContext<Type = ArrayType, Constant = C, Operation = O>,
    S::Operation: SupportsDot<ArrayType>,
    Extension: InterpretableOperation<ArrayType, Tracer<S>>,
    Tracer<S>: Add<Output = Tracer<S>>
        + Sub<Output = Tracer<S>>
        + Neg<Output = Tracer<S>>
        + Mul<Output = Tracer<S>>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + Broadcast<Output = Tracer<S>>
        + crate::tracing_v2::operations::reduce::Reduce
        + BooleanLike,
    Vec<Tracer<S>>:
        Parameterized<Tracer<S>, To<Tracer<S>> = Vec<Tracer<S>>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone + Operation<ArrayType> + SupportsTransferToMemory<ArrayType> + SupportsSelect<ArrayType>,
{
    fn interpret(&self, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret_over_tracers(inputs),
            Self::TransferToMemory { destination } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone().transfer_to_memory(*destination)])
            }
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    zero.r#type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    one.r#type()
                ),
            }
            .into()),
            Self::Constant(constant) => Err(TypeError {
                message: format!(
                    "linear constant operation over tracer values was not materialized before interpretation for {}",
                    constant.value().r#type()
                ),
            }
            .into()),
            Self::Fill(fill) => Err(TypeError {
                message: format!(
                    "linear fill operation over tracer values was not materialized before interpretation for {}",
                    fill.r#type()
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<ArrayType, Tracer<S>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<ArrayType, Tracer<S>>>::interpret(&SubOperation, inputs)
            }
            Self::Mul => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].clone() * inputs[1].clone()])
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<ArrayType, Tracer<S>>>::interpret(&NegOperation, inputs)
            }
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor, .. } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::LeftDot { factor, dimensions } => {
                use crate::tracing_v2::operations::dot::Dot;
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![factor.clone().dot(inputs[0].clone(), dimensions)])
            }
            Self::RightDot { factor, dimensions } => {
                use crate::tracing_v2::operations::dot::Dot;
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone().dot(factor.clone(), dimensions)])
            }
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).interpret(inputs),
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).interpret(inputs)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).interpret(inputs),
            Self::Select { condition } => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![Tracer::select(condition.clone(), inputs[0].clone(), inputs[1].clone())?])
            }
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<C, S, Extension, O> InterpretableOperation<DataType, Tracer<S>>
    for LinearArrayOperation<Tracer<S>, C, DataType, Extension, Tracer<S>, O>
where
    C: Value<DataType>,
    S: Context<Type = DataType>,
    Extension: InterpretableOperation<DataType, Tracer<S>>,
    Tracer<S>: Add<Output = Tracer<S>>
        + Sub<Output = Tracer<S>>
        + Neg<Output = Tracer<S>>
        + Mul<Output = Tracer<S>>
        + ZeroLike
        + OneLike,
    Vec<Tracer<S>>: Parameterized<Tracer<S>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Operation<DataType>,
{
    fn interpret(&self, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    zero.r#type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    one.r#type()
                ),
            }
            .into()),
            Self::Constant(constant) => Err(TypeError {
                message: format!(
                    "linear constant operation over tracer values was not materialized before interpretation for {}",
                    constant.value().r#type()
                ),
            }
            .into()),
            Self::Fill(fill) => Err(TypeError {
                message: format!(
                    "linear fill operation over tracer values was not materialized before interpretation for {}",
                    fill.r#type()
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&AddOperation, inputs)
            }
            Self::Sub => {
                <SubOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&SubOperation, inputs)
            }
            Self::Mul => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].clone() * inputs[1].clone()])
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor, .. } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::TransferToMemory { .. }
            | Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.interpret(inputs),
        }
    }
}

impl<V: Value<DataType>>
    TransposableOperation<DataType, Tangent<DataType, V>, LinearScalarOperation<V, Tangent<DataType, V>>>
    for LinearScalarOperation<V, Tangent<DataType, V>>
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearScalarOperation<V, Tangent<DataType, V>>,
        >,
        input_types: &[&DataType],
        output_cotangents: &[Cotangent<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearScalarOperation<V, Tangent<DataType, V>>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, DataType, Tangent<DataType, V>, LinearScalarOperation<V, Tangent<DataType, V>>>>,
        ProgramError,
    > {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback transposition over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Neg => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor, .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context
                            .stage_operation(Self::Scale { factor: factor.clone() }, std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
        }
    }
}

impl<C: Value<ArrayType>, Extension, O>
    TransposableOperation<
        ArrayType,
        ZeroArrayTangent,
        LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>,
    > for LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>
where
    Extension: TransposableOperation<
            ArrayType,
            ZeroArrayTangent,
            LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>,
        >,
    O: Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            ZeroArrayTangent,
            LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>,
        >,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            ZeroArrayTangent,
            LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                ArrayType,
                ZeroArrayTangent,
                LinearArrayOperation<ZeroArrayTangent, C, ArrayType, Extension, ZeroArrayTangent, O>,
            >,
        >,
        ProgramError,
    > {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback transposition over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add | Self::Sub => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Mul => {
                // For symbolic-zero tangents, multiplication propagates zero straight through to
                // both input cotangents.
                check_count!("output", output_cotangents, 2, ProgramError);
                Ok(vec![Cotangent::Zero, Cotangent::Zero])
            }
            Self::Neg | Self::Scale { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone()])
            }
            Self::Fill(fill) => fill.transpose(context, input_types, output_cotangents),
            Self::TransferToMemory { .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::TransferToMemory { destination: input_types[0].memory() },
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Transpose { permutation } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                let inverse = inverse_permutation(permutation);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().transpose(inverse))]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::LeftDot { .. } | Self::RightDot { .. } => {
                // Factor for ZeroArrayTangent is always symbolic zero, so dot(zero, t) is zero
                // and the cotangent for `t` is symbolic zero as well.
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![Cotangent::Zero])
            }
            Self::Select { .. } => {
                // Every value in the zero-only tangent space is zero, so the masked branch cotangents are
                // symbolic zeros as well.
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![Cotangent::Zero, Cotangent::Zero])
            }
            Self::Reshape { .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().reshape(input_types[0].shape().clone())?)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Broadcast { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ControlFlowError::MissingTransformRule {
                        transform: "broadcast transpose (would need reduce-sum)",
                    }
                    .into()),
                }
            }
            Self::Reduce { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ControlFlowError::MissingTransformRule {
                        transform: "reduce transpose (would need broadcast-back with stored input shape)",
                    }
                    .into()),
                }
            }
            Self::Condition(condition) => condition.transpose(context, input_types, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, input_types, output_cotangents),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<V: Value<ArrayType>, Extension, O>
    TransposableOperation<
        ArrayType,
        Tangent<ArrayType, V>,
        LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>,
    > for LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>
where
    V: crate::tracing_v2::operations::matrix::DotOps + Scale<f64, Output = V>,
    Extension: TransposableOperation<
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>,
        >,
    O: Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>,
        >,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                ArrayType,
                Tangent<ArrayType, V>,
                LinearArrayOperation<Tangent<ArrayType, V>, V, ArrayType, Extension, Tangent<ArrayType, V>, O>,
            >,
        >,
        ProgramError,
    > {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback transposition over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Mul => {
                // `LinearArrayOperation::Mul` is emitted only when one operand is the staged
                // output of a constant-producing op (e.g., [`Self::Fill`]). Transposing
                // it requires knowing which operand is the constant, which is not recoverable from
                // the op alone — defer to a higher-level pass that rewrites mul-by-constant into
                // [`Self::Scale`] before transposition.
                Err(ControlFlowError::MissingTransformRule {
                    transform: "linear `Mul` transpose (rewrite to `Scale` before transposition)",
                }
                .into())
            }
            Self::Neg => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::TransferToMemory { .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::TransferToMemory { destination: input_types[0].memory() },
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Transpose { permutation } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                let inverse = inverse_permutation(permutation);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().transpose(inverse))]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor, .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context
                            .stage_operation(Self::Scale { factor: factor.clone() }, std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Fill(fill) => fill.transpose(context, input_types, output_cotangents),
            Self::LeftDot { factor, dimensions } => {
                check_count!("output", output_cotangents, 1, ProgramError);
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
                check_count!("output", output_cotangents, 1, ProgramError);
                let Tangent::Value(_) = factor else {
                    return Ok(vec![Cotangent::Zero]);
                };
                let factor_rank = factor.r#type().as_ref().rank();
                let cotangent_rank = match &output_cotangents[0] {
                    Cotangent::Staged(value) => value.r#type().as_ref().rank(),
                    Cotangent::Zero => return Ok(vec![Cotangent::Zero]),
                };
                let t_rank = cotangent_rank + factor_rank
                    - 2 * dimensions.rhs_contracting_dimensions().len()
                    - dimensions.rhs_batching_dimensions().len();
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
            Self::Reshape { .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().reshape(input_types[0].shape().clone())?)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Broadcast { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ControlFlowError::MissingTransformRule {
                        transform: "broadcast transpose (would need reduce-sum)",
                    }
                    .into()),
                }
            }
            Self::Reduce { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ControlFlowError::MissingTransformRule {
                        transform: "reduce transpose (would need broadcast-back with stored input shape)",
                    }
                    .into()),
                }
            }
            Self::Select { condition } => transpose_captured_condition_select(
                || Self::Select { condition: condition.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Condition(condition) => condition.transpose(context, input_types, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, input_types, output_cotangents),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<V: Value<DataType>, Extension, O>
    TransposableOperation<
        DataType,
        Tangent<DataType, V>,
        LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>,
    > for LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>
where
    Extension: TransposableOperation<
            DataType,
            Tangent<DataType, V>,
            LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>,
        >,
    O: Operation<DataType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>,
        >,
        input_types: &[&DataType],
        output_cotangents: &[Cotangent<
            'transpose,
            DataType,
            Tangent<DataType, V>,
            LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                DataType,
                Tangent<DataType, V>,
                LinearArrayOperation<Tangent<DataType, V>, V, DataType, Extension, Tangent<DataType, V>, O>,
            >,
        >,
        ProgramError,
    > {
        match self {
            Self::CustomVjpCall(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Mul => Err(ControlFlowError::MissingTransformRule {
                transform: "linear `Mul` transpose (rewrite to `Scale` before transposition)",
            }
            .into()),
            Self::Neg => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor, .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context
                            .stage_operation(Self::Scale { factor: factor.clone() }, std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Fill(fill) => fill.transpose(context, input_types, output_cotangents),
            Self::TransferToMemory { .. }
            | Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<V: Value<DataType>, C: Value<DataType>, F: Value<DataType>>
    TransposableOperation<DataType, V, LinearScalarOperation<C, F>> for LinearScalarOperation<C, F>
where
    V: Add<Output = V> + Neg<Output = V> + ZeroLike + OneLike,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, DataType, V, LinearScalarOperation<C, F>>,
        input_types: &[&DataType],
        output_cotangents: &[Cotangent<'transpose, DataType, V, LinearScalarOperation<C, F>>],
    ) -> Result<Vec<Cotangent<'transpose, DataType, V, LinearScalarOperation<C, F>>>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.transpose(context, input_types, output_cotangents),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub => SubOperation.transpose(context, input_types, output_cotangents),
            Self::Neg => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale { factor, .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context
                            .stage_operation(Self::Scale { factor: factor.clone() }, std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
        }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    TransposableOperation<ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
where
    V: Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + BooleanLike,
    Extension: TransposableOperation<ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>>,
    ArrayOperation<V, ArrayType, Extension>: Clone + Operation<ArrayType>,
    ArrayOperation<C, ArrayType, Extension>: Clone + Operation<ArrayType>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone + Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            V,
            LinearArrayOperation<V, C, ArrayType, Extension, F, O>,
        >,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            V,
            LinearArrayOperation<V, C, ArrayType, Extension, F, O>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>>>,
        ProgramError,
    > {
        match self {
            Self::CustomVjpCall(call) => call.transpose(context, input_types, output_cotangents),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Fill(fill) => fill.transpose(context, input_types, output_cotangents),
            Self::Add => AddOperation.transpose(context, input_types, output_cotangents),
            Self::Sub => SubOperation.transpose(context, input_types, output_cotangents),
            Self::Mul => Err(ControlFlowError::MissingTransformRule {
                transform: "linear `Mul` transpose (rewrite to `Scale` before transposition)",
            }
            .into()),
            Self::Neg => NegOperation.transpose(context, input_types, output_cotangents),
            Self::TransferToMemory { .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::TransferToMemory { destination: input_types[0].memory() },
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Scale { factor, .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context
                            .stage_operation(Self::Scale { factor: factor.clone() }, std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::LeftDot { factor, dimensions } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                let factor_rank = factor.r#type().as_ref().rank();
                let adjoint_dims = super::dot::adjoint_dimensions_for_left_dot(dimensions, factor_rank);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::LeftDot { factor: factor.clone(), dimensions: adjoint_dims },
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::RightDot { factor, dimensions } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                let factor_rank = factor.r#type().as_ref().rank();
                let cotangent_rank = match &output_cotangents[0] {
                    Cotangent::Staged(value) => value.r#type().as_ref().rank(),
                    Cotangent::Zero => return Ok(vec![Cotangent::Zero]),
                };
                let t_rank = cotangent_rank
                    + 2 * dimensions.rhs_contracting_dimensions().len()
                    + dimensions.rhs_batching_dimensions().len()
                    - factor_rank;
                let adjoint_dims = super::dot::adjoint_dimensions_for_right_dot(dimensions, factor_rank, t_rank);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::RightDot { factor: factor.clone(), dimensions: adjoint_dims },
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Broadcast { output_type, output_axes } => BroadcastOperation::new(
                output_type.clone(),
                output_axes.clone(),
            )
            .transpose(context, input_types, output_cotangents),
            Self::Reduce { axes, kind } => {
                ReduceOperation::new(axes.clone(), *kind).transpose(context, input_types, output_cotangents)
            }
            Self::Select { condition } => transpose_captured_condition_select(
                || Self::Select { condition: condition.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Condition(condition) => condition.transpose(context, input_types, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, input_types, output_cotangents),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<V: Value<DataType>, C: Value<DataType>, Extension, F: Value<DataType>, O>
    TransposableOperation<DataType, V, LinearArrayOperation<V, C, DataType, Extension, F, O>>
    for LinearArrayOperation<V, C, DataType, Extension, F, O>
where
    V: Add<Output = V> + Neg<Output = V> + Mul<Output = V> + ZeroLike + OneLike,
    Extension: TransposableOperation<DataType, V, LinearArrayOperation<V, C, DataType, Extension, F, O>>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Operation<DataType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            DataType,
            V,
            LinearArrayOperation<V, C, DataType, Extension, F, O>,
        >,
        input_types: &[&DataType],
        output_cotangents: &[Cotangent<
            'transpose,
            DataType,
            V,
            LinearArrayOperation<V, C, DataType, Extension, F, O>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, DataType, V, LinearArrayOperation<V, C, DataType, Extension, F, O>>>,
        ProgramError,
    > {
        match self {
            Self::CustomVjpCall(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Zero(zero) => zero.transpose(context, input_types, output_cotangents),
            Self::One(one) => one.transpose(context, input_types, output_cotangents),
            Self::Constant(constant) => constant.transpose(context, input_types, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Fill(fill) => fill.transpose(context, input_types, output_cotangents),
            Self::Add => AddOperation.transpose(context, input_types, output_cotangents),
            Self::Sub => SubOperation.transpose(context, input_types, output_cotangents),
            Self::Mul => Err(ControlFlowError::MissingTransformRule {
                transform: "linear `Mul` transpose (rewrite to `Scale` before transposition)",
            }
            .into()),
            Self::Neg => NegOperation.transpose(context, input_types, output_cotangents),
            Self::Scale { factor, .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context
                            .stage_operation(Self::Scale { factor: factor.clone() }, std::slice::from_ref(cotangent))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::TransferToMemory { .. }
            | Self::Transpose { .. }
            | Self::LeftDot { .. }
            | Self::RightDot { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<F, D> DifferentiableOperation<D> for ScalarOperation<F>
where
    F: Value<DataType>,
    D: DifferentiationContext<Type = DataType, Constant = F>,
    D::Operation: SupportsZero<DataType> + SupportsOne<DataType>,
    D::Value: crate::tracing_v2::rematerialization::RematerializationName,
    D::Value: Add<Output = D::Value>
        + Sub<Output = D::Value>
        + Mul<Output = D::Value>
        + Div<Output = D::Value>
        + Neg<Output = D::Value>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + Compare<Output = D::Value>
        + Parameterized<D::Value>,
    <D::Value as Parameterized<D::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<D::Value>: Parameterized<D::Value, ParameterStructure: std::fmt::Debug + PartialEq>,
    ScaleOperation<DataType, F>: DifferentiableOperation<D>,
    LinearOperationOf<D>: SupportsLinearScalarOperation<DataType, ResidualFactor<DataType, D::Value>>
        + crate::tracing_v2::ResidualizedOperation<D>
        + SupportsCustomVjpCall<DataType, F, ScalarOperation<F>, ResidualFactor<DataType, D::Value>>,
    Vec<F>: Parameterized<
            F,
            Family: crate::parameters::ParameterizedFamily<D::Tangent>
                        + crate::parameters::ParameterizedFamily<D::Value>,
            To<D::Value> = Vec<D::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        match self {
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::Constant(constant) => constant.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::Compare { direction } => CompareOperation::new(*direction).jvp(context, inputs),
            Self::Select => {
                Err(TypeError { message: format!("{} does not support generic scalar jvp dispatch", self.name()) }
                    .into())
            }
            Self::StopGradient => StopGradientOperation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::CustomJvp(operation) => custom_jvp_rule(operation, context, inputs),
            Self::CustomVjp(operation) => custom_vjp_rule(operation, context, inputs),
        }
    }
}

impl<V: Value<ArrayType>, D, Extension> DifferentiableOperation<D> for ArrayOperation<V, ArrayType, Extension>
where
    D: DifferentiationContext<Type = ArrayType, Constant = V>,
    D::Operation: SupportsZero<ArrayType> + SupportsOne<ArrayType> + SupportsFill<ArrayType, f64>,
    D::Value: crate::tracing_v2::rematerialization::RematerializationName + TransferToMemory,
    D::Value: Add<Output = D::Value>
        + Sub<Output = D::Value>
        + Mul<Output = D::Value>
        + Div<Output = D::Value>
        + Neg<Output = D::Value>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + Compare<Output = D::Value>
        + BitAnd<Output = D::Value>
        + BitOr<Output = D::Value>
        + BitXor<Output = D::Value>
        + Not<Output = D::Value>
        + Select<Condition = D::Value>
        + BooleanLike
        + Parameterized<D::Value>,
    D::Tangent: Transpose + Broadcast<Output = D::Tangent> + super::reduce::Reduce,
    Extension: DifferentiableOperation<D>,
    ScaleOperation<ArrayType, V>: DifferentiableOperation<D>,
    <D::Value as Parameterized<D::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<
            V,
            Family: crate::parameters::ParameterizedFamily<D::Tangent>
                        + crate::parameters::ParameterizedFamily<D::Value>,
            To<V> = Vec<V>,
            To<D::Value> = Vec<D::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    Vec<D::Value>: Parameterized<
            D::Value,
            Family: crate::parameters::ParameterizedFamily<D::Tangent>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    LinearOperationOf<D>: SupportsLinearArrayOperation<ArrayType, ResidualFactor<ArrayType, D::Value>>
        + crate::tracing_v2::ResidualizedOperation<D>
        + SupportsCustomVjpCall<
            ArrayType,
            V,
            ArrayOperation<V, ArrayType, Extension>,
            ResidualFactor<ArrayType, D::Value>,
        > + SupportsTransferToMemory<ArrayType>
        + SupportsLinearSelect<ArrayType, ResidualFactor<ArrayType, D::Value>>,
    ArrayOperation<V, ArrayType, Extension>: Clone,
    D: Domain<Operation = ArrayOperation<V, ArrayType, Extension>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        match self {
            Self::CustomJvp(operation) => custom_jvp_rule(operation, context, inputs),
            Self::CustomVjp(operation) => custom_vjp_rule(operation, context, inputs),
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::Fill(fill) => fill.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::StopGradient => StopGradientOperation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::TransferToMemory(operation) => operation.jvp(context, inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Dot { dimensions } => DotOperation::new(dimensions.clone()).jvp(context, inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).jvp(context, inputs),
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).jvp(context, inputs),
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).jvp(context, inputs)
            }
            Self::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).jvp(context, inputs),
            Self::Compare { direction } => CompareOperation::new(*direction).jvp(context, inputs),
            Self::Not => NotOperation.jvp(context, inputs),
            Self::And => AndOperation.jvp(context, inputs),
            Self::Or => OrOperation.jvp(context, inputs),
            Self::Xor => XorOperation.jvp(context, inputs),
            Self::Select => SelectOperation.jvp(context, inputs),
            Self::Condition(condition) => condition.jvp(context, inputs),
            Self::While(while_operation) => while_operation.jvp(context, inputs),
            Self::Constant(_) | Self::Collective { .. } => {
                Err(TypeError { message: format!("{} does not support generic array jvp dispatch", self.name()) }
                    .into())
            }
            Self::Extension(extension) => extension.jvp(context, inputs),
        }
    }
}

impl<V: Value<DataType>, D, Extension> DifferentiableOperation<D> for ArrayOperation<V, DataType, Extension>
where
    D: DifferentiationContext<Type = DataType, Constant = V>,
    D::Operation: SupportsZero<DataType> + SupportsOne<DataType> + SupportsFill<DataType, f64>,
    D::Value: crate::tracing_v2::rematerialization::RematerializationName,
    D::Value: Add<Output = D::Value>
        + Sub<Output = D::Value>
        + Mul<Output = D::Value>
        + Div<Output = D::Value>
        + Neg<Output = D::Value>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + Parameterized<D::Value>,
    Extension: DifferentiableOperation<D>,
    ScaleOperation<DataType, V>: DifferentiableOperation<D>,
    <D::Value as Parameterized<D::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    Vec<D::Value>: Parameterized<D::Value, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearOperationOf<D>: SupportsLinearScalarOperation<DataType, ResidualFactor<DataType, D::Value>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        match self {
            Self::CustomJvp(_) | Self::CustomVjp(_) => {
                Err(unsupported_scalar_metadata_operation(self.operation_name()).into())
            }
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::Fill(fill) => fill.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::StopGradient => StopGradientOperation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Constant(_)
            | Self::TransferToMemory(_)
            | Self::Dot { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Reduce { .. }
            | Self::Compare { .. }
            | Self::Not
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Collective { .. }
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

/// Builds the common error for zero-input operation enum variants that must be handled by the staging path.
fn missing_zero_input_batch_rule(operation_enum: &str, kind: &str) -> ProgramError {
    BatchingError::UnsupportedOperation {
        message: format!(
            "{operation_enum}::{kind}: zero-input operations are lane-uniform by construction — stage them through the \
             active context, which handles the lane-uniform short-circuit, instead of invoking `batch` directly",
        ),
    }
    .into()
}

/// Dispatches non-control-flow [`ArrayOperation`] variants to their primitive batching rules.
///
/// Higher-order variants are intentionally returned as `None` so concrete impls can handle them with their specialized
/// recursive bounds instead of forcing the trait solver through one fully generic recursive operation impl.
fn batch_array_non_control_operation<F, V, E>(
    operation: &ArrayOperation<F, ArrayType, E>,
    inputs: &[ArrayBatch<V>],
) -> Result<Option<Vec<ArrayBatch<V>>>, ProgramError>
where
    F: Value<ArrayType>,
    V: Value<ArrayType>
        + SupportsArithmeticOperations<F>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = V>,
{
    let outputs = match operation {
        ArrayOperation::Add => AddOperation.batch(&(), inputs)?,
        ArrayOperation::Sub => SubOperation.batch(&(), inputs)?,
        ArrayOperation::Mul => MulOperation.batch(&(), inputs)?,
        ArrayOperation::Div => DivOperation.batch(&(), inputs)?,
        ArrayOperation::Neg => NegOperation.batch(&(), inputs)?,
        ArrayOperation::Sin => SinOperation.batch(&(), inputs)?,
        ArrayOperation::Cos => CosOperation.batch(&(), inputs)?,
        ArrayOperation::StopGradient => StopGradientOperation.batch(&(), inputs)?,
        ArrayOperation::RematerializationName(operation) => operation.batch(&(), inputs)?,
        ArrayOperation::Select => SelectOperation.batch(&(), inputs)?,
        ArrayOperation::ZeroLike => ZeroLikeOperation.batch(&(), inputs)?,
        ArrayOperation::OneLike => OneLikeOperation.batch(&(), inputs)?,
        ArrayOperation::Scale { factor } => ScaleOperation::new(factor.clone()).batch(&(), inputs)?,
        ArrayOperation::Dot { dimensions } => DotOperation::new(dimensions.clone()).batch(&(), inputs)?,
        ArrayOperation::Transpose { permutation } => TransposeOperation::new(permutation.clone()).batch(&(), inputs)?,
        ArrayOperation::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).batch(&(), inputs)?,
        ArrayOperation::Broadcast { output_type, output_axes } => {
            BroadcastOperation::new(output_type.clone(), output_axes.clone()).batch(&(), inputs)?
        }
        ArrayOperation::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).batch(&(), inputs)?,
        ArrayOperation::Compare { direction } => CompareOperation::new(*direction).batch(&(), inputs)?,
        ArrayOperation::Not => NotOperation.batch(&(), inputs)?,
        ArrayOperation::And => AndOperation.batch(&(), inputs)?,
        ArrayOperation::Or => OrOperation.batch(&(), inputs)?,
        ArrayOperation::Xor => XorOperation.batch(&(), inputs)?,
        ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Collective { .. }
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::CustomJvp(_)
        | ArrayOperation::CustomVjp(_)
        | ArrayOperation::Extension(_) => return Ok(None),
        ArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Zero")),
        ArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "One")),
        ArrayOperation::Constant(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Constant")),
        ArrayOperation::Fill(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Fill")),
    };
    Ok(Some(outputs))
}

/// Blanket value-level batching impl for the [`ArrayOperation`] sum type.
impl<V, E> BatchableOperation<V, ()> for ArrayOperation<V, ArrayType, E>
where
    V: Value<ArrayType>
        + SupportsArithmeticOperations
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + Fill<ArrayType, f64>
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = V>
        + BooleanLike,
    E: BatchableOperation<V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        if let Some(outputs) = batch_array_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            Self::Collective { axis_name, kind } => {
                CollectiveOperation::new(axis_name.clone(), *kind).batch(&(), inputs)
            }
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(&(), inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`ArrayOperation`] sum type.
///
/// The `Operation = Self` projection equality and the
/// [`SupportsProgramBatching`](crate::tracing_v2::batching::SupportsProgramBatching) / lane-alignment bounds exist
/// for the custom-derivative arms: their re-wrapping batch rules batch the captured programs and stage a new
/// custom-derivative call into the parent context, which is only expressible when the staged operation type is this
/// enum itself. Both extra bounds are leaf obligations (a structural type equality and a closed-enum capability
/// whose impl carries no batching-context obligations of its own), so instantiating this impl never recurses into
/// another batching-context obligation.
impl<C, V, E> BatchableOperation<Tracer<C>, BatchingContext<C>> for ArrayOperation<V, ArrayType, E>
where
    C: StagingContext<Type = ArrayType, Constant = V, Operation = ArrayOperation<V, ArrayType, E>>,
    V: Value<ArrayType> + BooleanLike,
    C::Operation: SupportsCollective<ArrayType> + SupportsFill<ArrayType, f64>,
    Tracer<C>: SupportsArithmeticOperations<V>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + Broadcast<Output = Tracer<C>>
        + Transpose,
    E: Clone + BatchableOperation<Tracer<C>, BatchingContext<C>>,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
    Self: crate::tracing_v2::batching::SupportsProgramBatching<V>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        if let Some(outputs) = batch_array_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so the operation is
            // staged unchanged on the physical batched value in the parent context and the lane axis is preserved.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let outputs = context
                    .parent_context()
                    .stage_operation(Self::TransferToMemory(*operation), &[inputs[0].value()])?;
                check_count!("output", outputs, 1, ProgramError);
                let tracer = outputs.into_iter().next().unwrap();
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The staged collective rule owns named-axis resolution: it consumes the lane axis when this
            // context's axis name matches and forwards the collective to the parent context otherwise.
            Self::Collective { axis_name, kind } => {
                CollectiveOperation::new(axis_name.clone(), *kind).batch(context, inputs)
            }
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Program-level batching for the [`ArrayOperation`] sum type, backing the re-wrapping `batch` rules of
/// [`CustomJvpOperation`] and [`CustomVjpOperation`]; see
/// [`SupportsProgramBatching`](crate::tracing_v2::batching::SupportsProgramBatching).
///
/// The where clauses here are deliberately the *leaf* closure of what `batch_flat_program::<V, Self>` needs — the
/// blanket traced batching impl's bounds instantiated at [`ProgramBatchingContext`] — rather than the
/// `Self: BatchableOperation<..>` bound itself. Spelling out the leaves keeps instantiating this impl free of
/// batching-context obligations, which is what lets the traced batching impl require
/// `Self: SupportsProgramBatching<..>` without sending the trait solver into an unbounded
/// batching-context recursion.
impl<V, E> crate::tracing_v2::batching::SupportsProgramBatching<V> for ArrayOperation<V, ArrayType, E>
where
    V: Value<ArrayType> + BooleanLike + 'static,
    E: Clone
        + Operation<ArrayType>
        + 'static
        + BatchableOperation<Tracer<ProgramBatchingContext<V, Self>>, BatchingContext<ProgramBatchingContext<V, Self>>>,
    Tracer<ProgramBatchingContext<V, Self>>: SupportsArithmeticOperations<V>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = Tracer<ProgramBatchingContext<V, Self>>>
        + BooleanLike
        + Broadcast<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Transpose,
    Vec<Tracer<ProgramBatchingContext<V, Self>>>: Parameterized<
            Tracer<ProgramBatchingContext<V, Self>>,
            To<Tracer<ProgramBatchingContext<V, Self>>> = Vec<Tracer<ProgramBatchingContext<V, Self>>>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn batch_flat_program(
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        axis_size: usize,
    ) -> Result<crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>, ProgramError> {
        crate::tracing_v2::batching::batch_flat_program(program, axis_size)
    }
}

/// Dispatches non-control-flow [`LinearArrayOperation`] variants to their primitive batching rules.
fn batch_linear_non_control_operation<F, C, V, E>(
    operation: &LinearArrayOperation<F, C, ArrayType, E>,
    inputs: &[ArrayBatch<V>],
) -> Result<Option<Vec<ArrayBatch<V>>>, ProgramError>
where
    F: Value<ArrayType>,
    C: Value<ArrayType>,
    V: Value<ArrayType>
        + SupportsLinearArithmeticOperations<F>
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations<F>
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>,
{
    let outputs = match operation {
        LinearArrayOperation::Add => AddOperation.batch(&(), inputs)?,
        LinearArrayOperation::Sub => SubOperation.batch(&(), inputs)?,
        LinearArrayOperation::Mul => MulOperation.batch(&(), inputs)?,
        LinearArrayOperation::Neg => NegOperation.batch(&(), inputs)?,
        LinearArrayOperation::ZeroLike => ZeroLikeOperation.batch(&(), inputs)?,
        LinearArrayOperation::OneLike => OneLikeOperation.batch(&(), inputs)?,
        LinearArrayOperation::Scale { factor } => ScaleOperation::new(factor.clone()).batch(&(), inputs)?,
        LinearArrayOperation::Transpose { permutation } => {
            TransposeOperation::new(permutation.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::LeftDot { factor, dimensions } => {
            LeftDotOperation::new(factor.clone(), dimensions.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::RightDot { factor, dimensions } => {
            RightDotOperation::new(factor.clone(), dimensions.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Reshape { output_shape } => {
            ReshapeOperation::new(output_shape.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Broadcast { output_type, output_axes } => {
            BroadcastOperation::new(output_type.clone(), output_axes.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Reduce { axes, kind } => ReduceOperation::new(axes.clone(), *kind).batch(&(), inputs)?,
        LinearArrayOperation::TransferToMemory { .. }
        | LinearArrayOperation::Select { .. }
        | LinearArrayOperation::Condition(_)
        | LinearArrayOperation::While(_)
        | LinearArrayOperation::CustomVjpCall(_)
        | LinearArrayOperation::Extension(_) => {
            return Ok(None);
        }
        LinearArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Zero")),
        LinearArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "One")),
        LinearArrayOperation::Constant(_) => {
            return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Constant"));
        }
        LinearArrayOperation::Fill(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Fill")),
    };
    Ok(Some(outputs))
}

/// Blanket value-level batching impl for the [`LinearArrayOperation`] sum type.
impl<V, E> BatchableOperation<V, ()> for LinearArrayOperation<V, V, ArrayType, E>
where
    ArrayOperation<V, ArrayType, E>: BatchableOperation<V>,
    V: Value<ArrayType>
        + SupportsLinearArithmeticOperations
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    E: BatchableOperation<V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        if let Some(outputs) = batch_linear_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory { .. } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            // The captured condition is lane-uniform: prepending it as an unbatched operand lets the elementwise
            // select batching rule broadcast it to the batched physical shape before selecting per lane.
            Self::Select { condition } => {
                check_count!("input", inputs, 2, ProgramError);
                SelectOperation
                    .batch(&(), &[ArrayBatch::unbatched(condition.clone()), inputs[0].clone(), inputs[1].clone()])
            }
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::CustomVjpCall(call) => call.batch(context, inputs),
            Self::Extension(extension) => extension.batch(&(), inputs),
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`LinearArrayOperation`] sum type.
impl<C, E> BatchableOperation<Tracer<C>, BatchingContext<C>>
    for LinearArrayOperation<C::Constant, C::Constant, ArrayType, E>
where
    ArrayOperation<C::Constant, ArrayType, E>: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType> + BooleanLike,
    Tracer<C>: SupportsLinearArithmeticOperations<C::Constant>
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations<C::Constant>
        + SupportsManipulationOperations
        + BitAnd<Output = Tracer<C>>
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + TransferToMemory,
    E: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        if let Some(outputs) = batch_linear_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so the transfer is
            // staged unchanged on the physical batched value (in its own parent context) and the lane axis is
            // preserved. The parent operation type is generic here, so the value-level capability stages it.
            Self::TransferToMemory { destination } => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().clone().transfer_to_memory(*destination);
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The captured condition is a lane-uniform parent-context constant: lift it into the parent trace and
            // let the elementwise select batching rule broadcast it to the batched physical shape.
            Self::Select { condition } => {
                check_count!("input", inputs, 2, ProgramError);
                let condition = context.parent_context().constant(condition.clone());
                SelectOperation.batch(&(), &[ArrayBatch::unbatched(condition), inputs[0].clone(), inputs[1].clone()])
            }
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::CustomVjpCall(call) => {
                if !call.transposed() {
                    return Err(crate::types::TypeError {
                        message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                            value_and_grad, or jacrev) instead"
                            .to_string(),
                    }
                    .into());
                }
                let mut values = call
                    .residuals()
                    .iter()
                    .map(|residual| ArrayBatch::unbatched(context.parent_context().constant(residual.clone())))
                    .collect::<Vec<_>>();
                values.extend(inputs.iter().cloned());
                context.interpret_program(call.backward(), values)
            }
            Self::Extension(extension) => extension.batch(context, inputs),
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

impl<V, E> BatchableOperation<Tangent<ArrayType, V>, ()> for LinearArrayOperation<V, V, ArrayType, E>
where
    ArrayOperation<V, ArrayType, E>: BatchableOperation<V>,
    V: Value<ArrayType>
        + SupportsLinearArithmeticOperations
        + SupportsConstantOperations<ArrayType>
        + SupportsLinearAlgebraOperations
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    E: BatchableOperation<V> + BatchableOperation<Tangent<ArrayType, V>, ()>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &(),
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
        match self {
            Self::Condition(condition) => {
                return <ConditionOperation<V, LinearArrayOperation<V, V, ArrayType, E>, ArrayType>
                    as BatchableOperation<Tangent<ArrayType, V>, ()>>::batch(
                    condition, context, inputs,
                );
            }
            Self::While(while_op) => {
                return <WhileOperation<V, LinearArrayOperation<V, V, ArrayType, E>, ArrayType> as BatchableOperation<
                    Tangent<ArrayType, V>,
                    (),
                >>::batch(while_op, context, inputs);
            }
            _ => {}
        }

        // First-order linear ops over tangent values: materialize `Tangent::Zero` to `V::zero`
        // once, dispatch to the V-level batching rule, and re-wrap as `Tangent::Value`. Symbolic
        // zero propagates through every Tangent V-trait impl (`Add`, `Sub`, `Neg`, `Scale`,
        // `LeftDot`, `RightDot`, `Reshape`, `Transpose`), so dispatching through `apply_with_axes`
        // on `lifted_op.interpret(tangent_values)` would also work — but the materialize-then-
        // dispatch path lets us reuse the V-level rule unchanged, which keeps the rule defined in
        // exactly one place.
        let always_materialize = matches!(self, LinearArrayOperation::ZeroLike | LinearArrayOperation::OneLike);
        if !always_materialize && inputs.iter().all(|input| input.value().is_zero()) {
            // Use the V-level rule purely for the lifted output types/axes; the value-level
            // interpret would have nothing to do for symbolic zeros.
            let materialized_zero_inputs = inputs
                .iter()
                .map(|input| -> Result<ArrayBatch<V>, ProgramError> {
                    ArrayBatch::new(input.r#type().into_owned(), V::zero(&input.r#type())?, input.batch_axis())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let v_outputs = <LinearArrayOperation<V, V, ArrayType, E> as BatchableOperation<V>>::batch(
                self,
                &(),
                materialized_zero_inputs.as_slice(),
            )?;
            return v_outputs
                .into_iter()
                .map(|v_batch| -> Result<ArrayBatch<Tangent<ArrayType, V>>, ProgramError> {
                    let output_type = v_batch.r#type().into_owned();
                    let output_axis = v_batch.batch_axis();
                    ArrayBatch::new(output_type.clone(), Tangent::zero(output_type), output_axis)
                })
                .collect();
        }

        let materialized = inputs
            .iter()
            .map(|input| -> Result<ArrayBatch<V>, ProgramError> {
                let materialized_value = match input.value() {
                    Tangent::Zero(zero_type) => V::zero(zero_type)?,
                    Tangent::Value(value) => value.clone(),
                };
                ArrayBatch::new(input.r#type().into_owned(), materialized_value, input.batch_axis())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let v_outputs = <LinearArrayOperation<V, V, ArrayType, E> as BatchableOperation<V>>::batch(
            self,
            &(),
            materialized.as_slice(),
        )?;
        v_outputs
            .into_iter()
            .map(|v_batch| -> Result<ArrayBatch<Tangent<ArrayType, V>>, ProgramError> {
                let output_type = v_batch.r#type().into_owned();
                let output_batch_axis = v_batch.batch_axis();
                let output_value = v_batch.into_value();
                ArrayBatch::new(output_type, Tangent::Value(output_value), output_batch_axis)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::InterpretableOperation as _;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tests::TestArray;
    use crate::types::Size;

    use super::*;

    type ZeroArrayOperation = LinearArrayOperation<ZeroArrayTangent, ZeroArrayTangent, ArrayType>;
    type ZeroArrayProgram =
        Program<ArrayType, ZeroArrayTangent, ZeroArrayOperation, Vec<ZeroArrayTangent>, Vec<ZeroArrayTangent>>;
    type MixedScalar = Tangent<DataType, f64>;
    type MixedScalarOperation = LinearScalarOperation<f64, MixedScalar>;
    type MixedArray = Tangent<ArrayType, TestArray>;
    type MixedArrayOperation = LinearArrayOperation<MixedArray, TestArray, ArrayType>;

    fn array_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
    }

    fn f64_array_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
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
        let no_inputs: &[ZeroScalarTangent] = &[];

        assert_eq!(add.interpret(&[tangent.clone(), tangent.clone()]), Ok(vec![tangent.clone()]));
        assert_eq!(neg.interpret(std::slice::from_ref(&tangent)), Ok(vec![tangent.clone()]));
        assert_eq!(zero.interpret(no_inputs), Ok(vec![tangent.clone()]));
        assert_eq!(one.interpret(no_inputs).unwrap_err().to_string(), "zero tangent space has no one value for f32");
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
            .add_instruction(ZeroArrayOperation::Reshape { output_shape: reshaped_type.shape().clone() }, vec![input])
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
        let input_zero = MixedArray::zero(input.array_type().clone());

        assert_eq!(
            MixedArrayOperation::Add.interpret(&[MixedArray::value(input.clone()), input_zero.clone()]),
            Ok(vec![MixedArray::value(input.clone())])
        );
        assert_eq!(MixedArrayOperation::Neg.interpret(std::slice::from_ref(&input_zero)), Ok(vec![input_zero.clone()]));

        let reshaped_type = f64_array_type(&[3, 2]);
        assert_eq!(
            (MixedArrayOperation::Reshape { output_shape: reshaped_type.shape().clone() })
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
            .interpret(&[MixedArray::value(input)]),
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

    #[test]
    fn test_batched_linear_operation_short_circuits_all_zero_inputs() {
        // Build an Add over two all-zero batched Tangent inputs and confirm the result is also
        // structurally zero — i.e., Tangent::Zero — without going through the underlying V::add.
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let zero_input =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type.clone()), Some(0))
                .unwrap();

        let op: LinearArrayOperation<TestArray, TestArray, ArrayType> = LinearArrayOperation::Add;
        let outputs = <LinearArrayOperation<TestArray, TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&op, &(), &[zero_input.clone(), zero_input])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero Add inputs");

        // Sanity-check that the same input type used through op.infer_output_types matches the
        // type reported on the symbolic-zero output.
        let expected_output_type = op.infer_output_types(&[batched_type.clone(), batched_type]).unwrap()[0].clone();
        assert_eq!(outputs[0].r#type().into_owned(), expected_output_type);
    }

    #[test]
    fn test_batched_linear_operation_short_circuit_uses_later_batched_input_axis() {
        let unbatched_type = ArrayType::scalar(DataType::F64);
        let unbatched_zero = ArrayBatch::unbatched(Tangent::<ArrayType, TestArray>::zero(unbatched_type));
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let batched_zero =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type.clone()), Some(0))
                .unwrap();

        let operation: LinearArrayOperation<TestArray, TestArray, ArrayType> = LinearArrayOperation::Add;
        let outputs = <LinearArrayOperation<TestArray, TestArray, ArrayType> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
        >>::batch(&operation, &(), &[unbatched_zero, batched_zero])
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().into_owned(), batched_type);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero Add inputs");
    }
}
