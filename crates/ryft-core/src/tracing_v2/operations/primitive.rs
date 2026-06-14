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

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::batching::BatchingError;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{Cotangent, SupportsTransposition, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::{check_count, check_types};
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
use crate::operations::control_flow::scan::{
    interpret_scan_lanes, read_scan_lane, scan_output_types, stacked_scan_type, validate_scan_unroll,
};
use crate::operations::control_flow::{
    CONDITION_OPERATION_NAME, ConditionOperation, SCAN_OPERATION_NAME, ScanOperation, WHILE_OPERATION_NAME,
    WhileOperation,
};
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
use crate::operations::manipulation::{CONCATENATE_OPERATION_NAME, ConcatenateOperation, SupportsConcatenate};
use crate::operations::manipulation::{
    DYNAMIC_SLICE_OPERATION_NAME, DYNAMIC_UPDATE_SLICE_OPERATION_NAME, DynamicSlice, DynamicSliceOperation,
    DynamicUpdateSlice, DynamicUpdateSliceOperation, PAD_OPERATION_NAME, PadOperation, SLICE_OPERATION_NAME, Slice,
    SliceOperation, SupportsDynamicSlice, SupportsDynamicUpdateSlice, SupportsPad, SupportsSlice, SupportsUpdateSlice,
    UPDATE_SLICE_OPERATION_NAME, UpdateSliceOperation,
};
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::operations::sharding::{
    ConstrainSharding, RESHARD_OPERATION_NAME, Reshard, ReshardOperation, SHARDING_CONSTRAINT_OPERATION_NAME,
    ShardingConstraintOperation, SupportsReshard, SupportsShardingConstraint,
};
use crate::operations::trigonometric::{
    COS_OPERATION_NAME, CosOperation, SIN_OPERATION_NAME, SinOperation, SupportsCos, SupportsSin,
};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::sharding::Sharding;
use crate::tracing::{AbstractTracingContext, Tracer, TracingContext};
use crate::tracing_v2::DifferentiableOperation;
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingContext, ProgramBatchableOperation, ProgramBatchingContext,
    ProgramBatchingOutputAxes,
};
use crate::tracing_v2::differentiation::{
    DifferentiationContext, FactorParameterizedOperation, JvpTracer, LinearOperationOf, LinearizationContextOf,
    NestedLinearization, ProgramLinearizableOperation, ResidualFactor, TangentContext,
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
use crate::types::{ArrayType, DataType, Memory, Shape, Size, Type, TypeError, Typed};

use super::bounds::{
    SupportsArithmeticOperations, SupportsComparisonOperations, SupportsConstantOperations,
    SupportsLinearAlgebraOperations, SupportsLinearArithmeticOperations, SupportsLinearArrayOperation,
    SupportsLinearScalarOperation, SupportsManipulationOperations, SupportsTrigonometricOperations,
};
use super::control_flow::{
    DefactorizedOperation, SupportsLinearCondition, SupportsLinearWhile, batch_condition_with_interpreter,
    batch_while_with_interpreter,
};
use super::dot::DotOps;
use super::scan::SupportsLinearScan;
use super::slicing::{SupportsLinearDynamicSlice, SupportsLinearDynamicUpdateSlice, static_update_sizes};
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

        /// Optional requested output sharding. Refer to the documentation of
        /// [`DotOperation::with_output_sharding`].
        output_sharding: Option<Sharding>,
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

    /// Tracked sharding transition over explicit/manual mesh axes. Lowers to the backend sharding-constraint
    /// operation. Refer to the documentation of [`ReshardOperation`].
    Reshard {
        /// Target sharding the input is resharded to.
        sharding: Sharding,
    },

    /// Untracked sharding-propagation hint over auto mesh axes. Lowers to the backend sharding-constraint
    /// operation. Refer to the documentation of [`ShardingConstraintOperation`].
    ShardingConstraint {
        /// Sharding hint for the backend's propagation over auto mesh axes.
        sharding: Sharding,
    },

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

    /// Statically indexed (possibly strided) slice. Lowers to StableHLO's `slice` op in the XLA
    /// backend.
    Slice {
        /// Inclusive start index for each input axis.
        start_indices: Vec<usize>,

        /// Exclusive limit index for each input axis.
        limit_indices: Vec<usize>,

        /// Stride for each input axis (every stride is at least `1`).
        strides: Vec<usize>,
    },

    /// Statically indexed contiguous update of the first operand with the second operand. Lowers
    /// to StableHLO's `dynamic_update_slice` op with constant start indices in the XLA backend.
    UpdateSlice {
        /// Inclusive start index for each input axis at which the update is written.
        start_indices: Vec<usize>,
    },

    /// Statically shaped slice at start indices computed at run time, with operands
    /// `[input, start_indices...]`. Lowers to StableHLO's `dynamic_slice` op in the XLA backend.
    DynamicSlice {
        /// Size of the extracted slice along each input axis.
        sizes: Vec<usize>,
    },

    /// Contiguous update at start indices computed at run time, with operands
    /// `[input, update, start_indices...]`. Lowers to StableHLO's `dynamic_update_slice` op in
    /// the XLA backend.
    DynamicUpdateSlice,

    /// Edge and interior padding of the first operand filled with the second (scalar) operand,
    /// with operands `[input, padding_value]`. Lowers to StableHLO's `pad` op in the XLA backend.
    Pad {
        /// Padding added before the first element of each input axis.
        edge_padding_low: Vec<usize>,

        /// Padding added after the last element of each input axis.
        edge_padding_high: Vec<usize>,

        /// Padding added between any two adjacent elements of each input axis.
        interior_padding: Vec<usize>,
    },

    /// Joins two or more operands end to end along `axis`. Lowers to StableHLO's `concatenate` op
    /// in the XLA backend.
    Concatenate {
        /// Axis along which the operands are joined.
        axis: usize,
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

        /// Optional requested output sharding. Refer to the documentation of
        /// [`ReduceOperation::with_output_sharding`].
        output_sharding: Option<Sharding>,
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

    /// Higher-order statically counted scan loop carrying its body program.
    Scan(Box<ScanOperation<V, ArrayOperation<V, T, Extension>, T>>),

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

        /// Optional requested output sharding. Refer to the documentation of
        /// [`DotOperation::with_output_sharding`].
        output_sharding: Option<Sharding>,
    },

    /// Captured-factor right dot: linear map `t ↦ dot(t, factor; dimensions)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::Dot`] when the RHS primal is held
    /// constant.
    RightDot {
        /// Captured constant factor (RHS of the underlying dot).
        factor: F,

        /// Dimension numbers of the underlying dot.
        dimensions: DotDimensionNumbers,

        /// Optional requested output sharding. Refer to the documentation of
        /// [`DotOperation::with_output_sharding`].
        output_sharding: Option<Sharding>,
    },

    /// Reshape from one shape to another.
    Reshape { output_shape: Shape },

    /// Tracked sharding transition over explicit/manual mesh axes. Lowers to the backend sharding-constraint
    /// operation. Refer to the documentation of [`ReshardOperation`].
    Reshard {
        /// Target sharding the input is resharded to.
        sharding: Sharding,
    },

    /// Untracked sharding-propagation hint over auto mesh axes. Lowers to the backend sharding-constraint
    /// operation. Refer to the documentation of [`ShardingConstraintOperation`].
    ShardingConstraint {
        /// Sharding hint for the backend's propagation over auto mesh axes.
        sharding: Sharding,
    },

    /// N-dimensional broadcast to a target shape; linear-side analogue of
    /// [`ArrayOperation::Broadcast`].
    Broadcast {
        /// Target output [`ArrayType`].
        output_type: T,

        /// For each input axis, the output axis it maps to.
        output_axes: Vec<usize>,
    },

    /// Statically indexed (possibly strided) slice; linear-side analogue of
    /// [`ArrayOperation::Slice`]. Its transpose scatters the cotangent back into the positions the
    /// forward slice read: a zero array overwritten via [`Self::UpdateSlice`] for unit strides, and
    /// a [`Self::Pad`] with a zero padding value and `interior_padding = stride - 1` for non-unit
    /// strides.
    Slice {
        /// Inclusive start index for each input axis.
        start_indices: Vec<usize>,

        /// Exclusive limit index for each input axis.
        limit_indices: Vec<usize>,

        /// Stride for each input axis (every stride is at least `1`).
        strides: Vec<usize>,
    },

    /// Statically indexed contiguous update; linear-side analogue of
    /// [`ArrayOperation::UpdateSlice`], jointly linear in its `(input, update)` operands. Its
    /// transpose splits the cotangent into the cotangent with the update window zeroed (for the
    /// input) and the [`Self::Slice`] of the cotangent at the update window (for the update).
    UpdateSlice {
        /// Inclusive start index for each input axis at which the update is written.
        start_indices: Vec<usize>,
    },

    /// Edge and interior padding; linear-side analogue of [`ArrayOperation::Pad`], jointly linear
    /// in its `(input, padding_value)` operands. Its transpose splits the cotangent into the
    /// strided [`Self::Slice`] of the cotangent at the pad geometry (for the input) and the sum of
    /// every padding position, computed as the full [`Self::Reduce`] sum minus the sliced region's
    /// sum (for the padding value).
    Pad {
        /// Padding added before the first element of each input axis.
        edge_padding_low: Vec<usize>,

        /// Padding added after the last element of each input axis.
        edge_padding_high: Vec<usize>,

        /// Padding added between any two adjacent elements of each input axis.
        interior_padding: Vec<usize>,
    },

    /// Joins two or more operands end to end along `axis`; linear-side analogue of
    /// [`ArrayOperation::Concatenate`], jointly linear in every operand (it carries no captured
    /// factors). Its transpose splits the output cotangent into per-operand pieces by slicing the
    /// cotangent at the cumulative operand offsets along `axis`.
    Concatenate {
        /// Axis along which the operands are joined.
        axis: usize,
    },

    /// Captured-index dynamic slice: linear map `t ↦ dynamic_slice(t, start_indices, sizes)`. Linear-side
    /// counterpart emitted by the JVP of [`ArrayOperation::DynamicSlice`], with the scalar integer start index
    /// primals captured as factors (integer indices have no tangent space, so the map is linear in the sliced
    /// operand). Its transpose scatters the cotangent into a zero array at the same captured indices via
    /// [`Self::DynamicUpdateSlice`].
    DynamicSlice {
        /// Captured scalar integer start index factors, one per input axis.
        start_indices: Vec<F>,

        /// Size of the extracted slice along each input axis.
        sizes: Vec<usize>,
    },

    /// Captured-index dynamic update-slice: linear map `(t, u) ↦ dynamic_update_slice(t, u, start_indices)`.
    /// Linear-side counterpart emitted by the JVP of [`ArrayOperation::DynamicUpdateSlice`], with the scalar
    /// integer start index primals captured as factors (integer indices have no tangent space, so the map is
    /// jointly linear in the `(input, update)` operands). Its transpose splits the cotangent into the cotangent
    /// with the update window zeroed (for the input) and the [`Self::DynamicSlice`] of the cotangent at the
    /// captured indices (for the update).
    DynamicUpdateSlice {
        /// Captured scalar integer start index factors, one per input axis.
        start_indices: Vec<F>,
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

        /// Optional requested output sharding. Refer to the documentation of
        /// [`ReduceOperation::with_output_sharding`].
        output_sharding: Option<Sharding>,
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

    /// Captured-factor residual injection: nullary map that materializes the captured factor as a program value.
    /// Emitted by the JVP of [`WhileOperation`] to feed the loop-entry primal state into the staged doubled-state
    /// linear loop, and by fused while bodies to materialize nested primal program constants. Like every captured
    /// factor, the payload is a residual of the primal computation rather than a linear operand, so this operation
    /// is not a linear map and rejects transposition (it is only reachable behind the while transpose error, which
    /// fires first).
    Residual {
        /// Captured factor materialized as this operation's single output.
        factor: F,
    },

    /// Recomputed primal operation embedded in a linear program. Fused while bodies use this variant to interleave
    /// primal state recomputation with tangent propagation (the loop-varying residuals a body pushforward needs are
    /// recomputed in-loop instead of being captured once, exactly like JAX's `while_loop` JVP). The wrapped
    /// operation is the *primal* operation type `O` already carried by this enum, so the recomputed computation can
    /// use the full primal surface (comparisons, divisions, nested control flow, and so on) without the linear enum
    /// mirroring each primal variant. Recomputed operations are not linear maps and reject transposition, which is
    /// only reachable behind the while transpose error.
    Recompute(O),

    /// Higher-order conditional restricted to linear branch programs, with the Boolean primal predicate captured as
    /// a factor (the predicate itself has no tangent space, so the map is linear in the branch operands). The
    /// operation inputs are exactly the branch operands, and the captured predicate selects which branch program
    /// runs. Because the predicate is a residual of the primal computation rather than a linear operand, it receives
    /// no cotangent and is carried verbatim through transposition.
    Condition {
        /// Captured Boolean predicate that selects the branch to run.
        predicate: F,

        /// Branch [`Program`] evaluated when the predicate is true.
        true_branch: Box<Program<T, V, LinearArrayOperation<V, C, T, Extension, F, O>, Vec<V>, Vec<V>>>,

        /// Branch [`Program`] evaluated when the predicate is false.
        false_branch: Box<Program<T, V, LinearArrayOperation<V, C, T, Extension, F, O>, Vec<V>, Vec<V>>>,
    },

    /// Operand-form counterpart of [`Condition`](Self::Condition) produced by
    /// [`SupportsLinearWhile::defactorize`] for fused while bodies: the Boolean predicate is operand `0` (recomputed
    /// in-loop, like every `Recompute`-wrapped primal) instead of a captured factor, and any loop-varying residuals
    /// the branch programs referenced are forwarded as additional trailing operands. The operands are therefore
    /// `[predicate, branch_operands..., forwarded_residuals...]`, and both branch programs consume
    /// `[branch_operands..., forwarded_residuals...]` with identical signatures (each branch receives the full
    /// forwarded union even when only one of them reads a given residual). The rewritten branches carry only closed
    /// [`ResidualFactor::Constant`] factors — their residual references were defactorized into operand form against
    /// the trailing inputs — but factor traversal stays total over them. Like recomputed primal operations, the
    /// operand-form condition is not a linear map in its predicate and forwarded-residual operands and rejects
    /// transposition, which is only reachable behind the while transpose error.
    OperandCondition {
        /// Branch [`Program`] evaluated when the predicate operand is true.
        true_branch: Box<Program<T, V, LinearArrayOperation<V, C, T, Extension, F, O>, Vec<V>, Vec<V>>>,

        /// Branch [`Program`] evaluated when the predicate operand is false.
        false_branch: Box<Program<T, V, LinearArrayOperation<V, C, T, Extension, F, O>, Vec<V>, Vec<V>>>,
    },

    /// Higher-order while loop restricted to linear condition and body programs, staged by the JVP rule of
    /// [`WhileOperation`](crate::operations::control_flow::WhileOperation) for *unbounded* loops.
    ///
    /// The wrapped [`WhileOperation`](crate::operations::control_flow::WhileOperation) is the fused doubled-state loop
    /// (recomputed primal interleaved with the defactorized body pushforward over `[primal_state..., tangent_state...]`)
    /// that drives interpretation, type inference, rendering, and `ryft-xla` lowering, so forward-mode execution and
    /// lowering see exactly the fused form. Transposition of an unbounded fused loop is rejected (the fused body
    /// recomputes primal state forward, which a while loop cannot run backwards); reverse mode through a while loop
    /// therefore requires an iteration bound, whose pushforward is a masked linear [`Scan`](Self::Scan) instead.
    While(Box<WhileOperation<V, LinearArrayOperation<V, C, T, Extension, F, O>, T>>),

    /// Higher-order statically counted linear scan staged by the JVP rule of
    /// [`ScanOperation`](crate::operations::control_flow::ScanOperation). The body is the scan body's residualized
    /// pushforward, mapping `[tangent_carry..., tangent_x_slice...]` to `[tangent_carry..., tangent_y_slice...]`,
    /// and `residual_stacks` are the stacked per-iteration residuals of the extended primal scan, captured as
    /// factors of the enclosing linearization.
    ///
    /// **Scan-local factor namespace.** The body's factor payloads are pinned to
    /// `ResidualFactor<T, V>` and form a namespace owned by this operation: reference index `i` resolves to slice
    /// `lane` of `residual_stacks[i]` while iteration `lane` runs, so the body stays fully linear in its tangent
    /// inputs with every captured primal entering through a per-lane residual slice. Enclosing factor passes
    /// ([`FactorParameterizedOperation::try_map_factors`] and everything built on it, such as residual compaction
    /// and instantiation) map **only** `residual_stacks` and never rewrite body-internal factors. This is also the
    /// transposability trick: transposing the body program and flipping `reverse` pairs cotangent lane `i` with
    /// residual lane `i` exactly when the forward scan consumed them, making linear-scan transposition total —
    /// where the linear [`While`](Self::While) must recompute its residuals forward and therefore rejects
    /// transposition.
    Scan {
        /// Residualized body pushforward with scan-local residual references.
        body: Box<Program<T, V, LinearArrayOperation<V, C, T, Extension, ResidualFactor<T, V>, O>, Vec<V>, Vec<V>>>,

        /// Stacked per-iteration residual factors indexed by the body's scan-local residual references; each
        /// stack's leading dimension is the scan length.
        residual_stacks: Vec<F>,

        /// Number of loop-carried tangent leaves at the front of the body's inputs and outputs.
        carry_count: usize,

        /// Static trip count of the scan.
        length: usize,

        /// Boolean indicating whether iterations visit the stacked slices in reverse order.
        reverse: bool,

        /// Lowering-only unroll factor inherited from the primal scan (see
        /// [`ScanOperation::with_unroll`](crate::operations::control_flow::ScanOperation::with_unroll)):
        /// interpretation and transform rules ignore it but preserve it.
        unroll: usize,
    },

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

impl<T: Type, F: Value<T>> FactorParameterizedOperation<T, F> for Infallible {
    type WithFactor<MappedFactor: Value<T>> = Infallible;

    fn try_map_factors<MappedFactor: Value<T>, MapFactorFn>(
        &self,
        _map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
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
    fn dot_operation(dimensions: DotDimensionNumbers, output_sharding: Option<Sharding>) -> Self {
        ArrayOperation::Dot { dimensions, output_sharding }
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
    fn reduce_operation(axes: Vec<usize>, kind: ReductionKind, output_sharding: Option<Sharding>) -> Self {
        ArrayOperation::Reduce { axes, kind, output_sharding }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsReshard for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn reshard_operation(sharding: Sharding) -> Self {
        ArrayOperation::Reshard { sharding }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsShardingConstraint for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn sharding_constraint_operation(sharding: Sharding) -> Self {
        ArrayOperation::ShardingConstraint { sharding }
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
            Self::Dot { dimensions, .. } => Some(dimensions),
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

impl<V: Value<ArrayType>, Extension> SupportsSlice<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn slice_operation(start_indices: Vec<usize>, limit_indices: Vec<usize>, strides: Vec<usize>) -> Self {
        ArrayOperation::Slice { start_indices, limit_indices, strides }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsUpdateSlice<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn update_slice_operation(start_indices: Vec<usize>) -> Self {
        ArrayOperation::UpdateSlice { start_indices }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsPad<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn pad_operation(
        edge_padding_low: Vec<usize>,
        edge_padding_high: Vec<usize>,
        interior_padding: Vec<usize>,
    ) -> Self {
        ArrayOperation::Pad { edge_padding_low, edge_padding_high, interior_padding }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsConcatenate<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn concatenate_operation(axis: usize) -> Self {
        ArrayOperation::Concatenate { axis }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsDynamicSlice<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn dynamic_slice_operation(sizes: Vec<usize>) -> Self {
        ArrayOperation::DynamicSlice { sizes }
    }
}

impl<V: Value<ArrayType>, Extension> SupportsDynamicUpdateSlice<ArrayType> for ArrayOperation<V, ArrayType, Extension> {
    #[inline]
    fn dynamic_update_slice_operation() -> Self {
        ArrayOperation::DynamicUpdateSlice
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
    fn left_dot_operation(factor: F, dimensions: DotDimensionNumbers, output_sharding: Option<Sharding>) -> Self {
        LinearArrayOperation::LeftDot { factor, dimensions, output_sharding }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    super::dot::SupportsRightDot<ArrayType, F> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn right_dot_operation(factor: F, dimensions: DotDimensionNumbers, output_sharding: Option<Sharding>) -> Self {
        LinearArrayOperation::RightDot { factor, dimensions, output_sharding }
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

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsReshard
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn reshard_operation(sharding: Sharding) -> Self {
        LinearArrayOperation::Reshard { sharding }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsShardingConstraint
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn sharding_constraint_operation(sharding: Sharding) -> Self {
        LinearArrayOperation::ShardingConstraint { sharding }
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
    fn reduce_operation(axes: Vec<usize>, kind: ReductionKind, output_sharding: Option<Sharding>) -> Self {
        LinearArrayOperation::Reduce { axes, kind, output_sharding }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsSlice<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn slice_operation(start_indices: Vec<usize>, limit_indices: Vec<usize>, strides: Vec<usize>) -> Self {
        LinearArrayOperation::Slice { start_indices, limit_indices, strides }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsUpdateSlice<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn update_slice_operation(start_indices: Vec<usize>) -> Self {
        LinearArrayOperation::UpdateSlice { start_indices }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsPad<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn pad_operation(
        edge_padding_low: Vec<usize>,
        edge_padding_high: Vec<usize>,
        interior_padding: Vec<usize>,
    ) -> Self {
        LinearArrayOperation::Pad { edge_padding_low, edge_padding_high, interior_padding }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> SupportsConcatenate<ArrayType>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn concatenate_operation(axis: usize) -> Self {
        LinearArrayOperation::Concatenate { axis }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    SupportsLinearDynamicSlice<ArrayType, F> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn linear_dynamic_slice_operation(start_indices: Vec<F>, sizes: Vec<usize>) -> Self {
        LinearArrayOperation::DynamicSlice { start_indices, sizes }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    SupportsLinearDynamicUpdateSlice<ArrayType, F> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn linear_dynamic_update_slice_operation(start_indices: Vec<F>) -> Self {
        LinearArrayOperation::DynamicUpdateSlice { start_indices }
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
    SupportsLinearCondition<ArrayType, V, F> for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
{
    #[inline]
    fn linear_condition_operation(
        predicate: F,
        true_branch: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        false_branch: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Self {
        LinearArrayOperation::Condition {
            predicate,
            true_branch: Box::new(true_branch),
            false_branch: Box::new(false_branch),
        }
    }
}

/// Disposition of one residual-reference index while defactorizing a nested linear program (see
/// [`defactorize_nested_linear_program`]).
#[derive(Copy, Clone)]
enum NestedResidualDisposition {
    /// The referenced residual enters the rewritten program as the trailing input at this position, and references
    /// to it are rewritten into operand form against that input.
    Operand(usize),

    /// The referenced residual stays a factor payload, re-indexed to this position.
    Factor(usize),
}

/// Rewrites a nested linear `program`'s residual references into operand form against new trailing inputs.
///
/// This is the whole-program counterpart of [`SupportsLinearWhile::defactorize`], used by the higher-order
/// defactorization arms: operand-form condition branches receive their forwarded while-body residuals as trailing
/// inputs, and operand-form scan bodies receive the lane slices of their moved residual stacks as trailing scanned
/// inputs. The returned program consumes `[original_inputs..., forwarded_inputs...]` with one trailing input per
/// entry of `forwarded_input_types`, and each instruction is rewritten according to `dispositions`, indexed by the
/// program's residual-reference namespace:
///
///   - Instructions whose references all map to [`NestedResidualDisposition::Factor`] keep their factor form with
///     the references re-indexed to the compacted factor positions.
///   - Instructions whose references all map to [`NestedResidualDisposition::Operand`] are rewritten into operand
///     form against the trailing input atoms through [`SupportsLinearWhile::defactorize`] (a nested residual
///     injection collapses to forwarding the trailing input).
///   - Instructions referencing both kinds are rejected, mirroring the mixed constant/reference index rejection of
///     the dynamic-slicing defactorization arms (defactorization stages exactly one instruction per source
///     instruction).
fn defactorize_nested_linear_program<V, C, Extension, R, O>(
    program: &Program<
        ArrayType,
        V,
        LinearArrayOperation<V, C, ArrayType, Extension, ResidualFactor<ArrayType, R>, O>,
        Vec<V>,
        Vec<V>,
    >,
    dispositions: &[Option<NestedResidualDisposition>],
    forwarded_input_types: &[ArrayType],
) -> Result<
    Program<
        ArrayType,
        V,
        LinearArrayOperation<V, C, ArrayType, Extension, ResidualFactor<ArrayType, R>, O>,
        Vec<V>,
        Vec<V>,
    >,
    ProgramError,
>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    R: Value<ArrayType>,
    O: Clone
        + Operation<ArrayType>
        + SupportsMul<ArrayType>
        + SupportsDot<ArrayType>
        + SupportsSelect<ArrayType>
        + SupportsDynamicSlice<ArrayType>
        + SupportsDynamicUpdateSlice<ArrayType>
        + SupportsConcatenate<ArrayType>,
{
    let mut builder = ProgramBuilder::<
        ArrayType,
        V,
        LinearArrayOperation<V, C, ArrayType, Extension, ResidualFactor<ArrayType, R>, O>,
    >::new();
    let mut atom_map: Vec<Option<AtomId>> = vec![None; program.atoms().len()];
    for (program_atom, input_type) in program.input_ids().iter().zip(program.input_types().into_iter()) {
        atom_map[program_atom.index()] = Some(builder.add_input(input_type));
    }
    let forwarded_atoms = forwarded_input_types
        .iter()
        .map(|forwarded_type| builder.add_input(forwarded_type.clone()))
        .collect::<Vec<_>>();
    for (atom_index, atom) in program.atoms().iter().enumerate() {
        if let Atom::Constant(constant) = atom {
            atom_map[atom_index] = Some(builder.add_constant(constant.clone()));
        }
    }
    let map_atom = |atom_map: &[Option<AtomId>], atom: AtomId| {
        atom_map.get(atom.index()).copied().flatten().ok_or(ProgramError::UnboundAtomId { id: atom })
    };
    let resolve_disposition = |index: usize| {
        dispositions.get(index).copied().flatten().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "nested linear program references residual {index} but only {} residuals were dispositioned",
                dispositions.len(),
            ))
        })
    };
    for instruction in program.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .map(|input| map_atom(atom_map.as_slice(), *input))
            .collect::<Result<Vec<_>, _>>()?;
        let mut references_operand_residual = false;
        let mut references_factor_residual = false;
        instruction.operation().try_map_factors_preserving_extensions(&mut |factor: &ResidualFactor<
            ArrayType,
            R,
        >| {
            if let ResidualFactor::Reference { index, .. } = factor {
                match resolve_disposition(*index)? {
                    NestedResidualDisposition::Operand(_) => references_operand_residual = true,
                    NestedResidualDisposition::Factor(_) => references_factor_residual = true,
                }
            }
            Ok(factor.clone())
        })?;
        if references_operand_residual && references_factor_residual {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "jvp of a while loop whose body pushforward stages {} over a mix of loop-varying and \
                     constant-stack residual references is not supported",
                    instruction.operation().name(),
                ),
            });
        }
        let remapped = instruction.operation().try_map_factors_preserving_extensions(&mut |factor| match factor {
            ResidualFactor::Reference { index, r#type } => {
                let position = match resolve_disposition(*index)? {
                    NestedResidualDisposition::Operand(position) => position,
                    NestedResidualDisposition::Factor(position) => position,
                };
                Ok(ResidualFactor::Reference { index: position, r#type: r#type.clone() })
            }
            ResidualFactor::Constant(value) => Ok(ResidualFactor::Constant(value.clone())),
        })?;
        if !references_operand_residual {
            let outputs = builder.add_instruction(remapped, inputs)?.to_vec();
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);
            for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
                atom_map[program_atom.index()] = Some(builder_atom);
            }
            continue;
        }
        match remapped.defactorize(forwarded_atoms.as_slice(), inputs)? {
            DefactorizedOperation::Operation { operation, inputs } => {
                let outputs = builder.add_instruction(operation, inputs)?.to_vec();
                check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
                    atom_map[program_atom.index()] = Some(builder_atom);
                }
            }
            DefactorizedOperation::Forward { atom } => {
                check_count!("output", instruction.outputs(), 1, ProgramError);
                atom_map[instruction.outputs()[0].index()] = Some(atom);
            }
        }
    }
    let outputs = program
        .output_ids()
        .iter()
        .map(|output| map_atom(atom_map.as_slice(), *output))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let input_count = program.input_ids().len() + forwarded_input_types.len();
    let output_count = outputs.len();
    builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])
}

impl<V, C, Extension, R, O> SupportsLinearWhile<ArrayType, V, ResidualFactor<ArrayType, R>, O>
    for LinearArrayOperation<V, C, ArrayType, Extension, ResidualFactor<ArrayType, R>, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    R: Value<ArrayType>,
    O: Clone
        + Operation<ArrayType>
        + SupportsMul<ArrayType>
        + SupportsDot<ArrayType>
        + SupportsSelect<ArrayType>
        + SupportsDynamicSlice<ArrayType>
        + SupportsDynamicUpdateSlice<ArrayType>
        + SupportsConcatenate<ArrayType>,
{
    #[inline]
    fn recompute_operation(operation: O) -> Self {
        LinearArrayOperation::Recompute(operation)
    }

    #[inline]
    fn residual_operation(factor: ResidualFactor<ArrayType, R>) -> Self {
        LinearArrayOperation::Residual { factor }
    }

    fn defactorize(
        &self,
        residual_atoms: &[AtomId],
        mut inputs: Vec<AtomId>,
    ) -> Result<DefactorizedOperation<Self>, ProgramError> {
        let resolve_residual_atom = |index: usize| {
            residual_atoms.get(index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "while body pushforward references residual {index} but only {} residuals were captured",
                    residual_atoms.len(),
                ))
            })
        };
        match self {
            // `Scale` by a loop-varying residual becomes a recomputed elementwise product against the recomputed
            // residual atom; `LeftDot` / `RightDot` become the recomputed operand-form dot with the residual spliced
            // in on the side the captured factor occupied. All three target `Recompute` so that every
            // recomputed-primal instruction in a fused while body carries the same provenance.
            Self::Scale { factor: ResidualFactor::Reference { index, .. } } => {
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::mul_operation()),
                    inputs,
                })
            }
            Self::LeftDot { factor: ResidualFactor::Reference { index, .. }, dimensions, output_sharding } => {
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::dot_operation(
                        dimensions.clone(),
                        output_sharding.clone(),
                    )),
                    inputs,
                })
            }
            Self::RightDot { factor: ResidualFactor::Reference { index, .. }, dimensions, output_sharding } => {
                inputs.push(resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::dot_operation(
                        dimensions.clone(),
                        output_sharding.clone(),
                    )),
                    inputs,
                })
            }
            // `DynamicSlice` / `DynamicUpdateSlice` over loop-varying residual start indices become the recomputed
            // operand-form primal operations with the residual atoms spliced in as index operands. Mixed
            // constant/reference index lists are rejected because defactorization stages exactly one instruction,
            // while constant indices would need their own materializing instructions.
            Self::DynamicSlice { start_indices, sizes }
                if start_indices.iter().any(|index| matches!(index, ResidualFactor::Reference { .. })) =>
            {
                for start_index in start_indices {
                    let ResidualFactor::Reference { index, .. } = start_index else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "jvp of a while loop whose body captures a mix of loop-varying and constant \
                                      dynamic_slice start indices is not supported"
                                .to_string(),
                        });
                    };
                    inputs.push(resolve_residual_atom(*index)?);
                }
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::dynamic_slice_operation(sizes.clone())),
                    inputs,
                })
            }
            Self::DynamicUpdateSlice { start_indices }
                if start_indices.iter().any(|index| matches!(index, ResidualFactor::Reference { .. })) =>
            {
                for start_index in start_indices {
                    let ResidualFactor::Reference { index, .. } = start_index else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "jvp of a while loop whose body captures a mix of loop-varying and constant \
                                      dynamic_update_slice start indices is not supported"
                                .to_string(),
                        });
                    };
                    inputs.push(resolve_residual_atom(*index)?);
                }
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::dynamic_update_slice_operation()),
                    inputs,
                })
            }
            // A nested loop's residual injection materializes a value the fused body already recomputes, so the
            // instruction collapses to forwarding the residual atom.
            Self::Residual { factor: ResidualFactor::Reference { index, .. } } => {
                Ok(DefactorizedOperation::Forward { atom: resolve_residual_atom(*index)? })
            }
            // `Select` over a loop-varying residual condition becomes the recomputed operand-form primal select
            // with the residual atom spliced in as the condition operand.
            Self::Select { condition: ResidualFactor::Reference { index, .. } } => {
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::select_operation()),
                    inputs,
                })
            }
            // A loop-varying condition predicate becomes operand 0 of an operand-form condition
            // (`OperandCondition`). The branch programs may carry their own references into the same while-body
            // residual table (the condition JVP rule remapped them onto the enclosing linearization environment), so
            // the union of the residual indices referenced by both branches is forwarded as additional trailing
            // operands — both branches receive the full union because their signatures must agree — and each branch
            // is recursively defactorized against the new trailing branch inputs.
            Self::Condition { predicate: ResidualFactor::Reference { index, .. }, true_branch, false_branch } => {
                let predicate_atom = resolve_residual_atom(*index)?;
                let mut forwarded_residuals = BTreeMap::new();
                for branch in [true_branch.as_ref(), false_branch.as_ref()] {
                    for instruction in branch.instructions() {
                        instruction.operation().try_map_factors_preserving_extensions(
                            &mut |factor: &ResidualFactor<ArrayType, R>| {
                                if let ResidualFactor::Reference { index, r#type } = factor {
                                    forwarded_residuals.entry(*index).or_insert_with(|| r#type.clone());
                                }
                                Ok(factor.clone())
                            },
                        )?;
                    }
                }
                let mut dispositions = vec![None; residual_atoms.len()];
                let mut forwarded_types = Vec::with_capacity(forwarded_residuals.len());
                let mut forwarded_atoms = Vec::with_capacity(forwarded_residuals.len());
                for (position, (residual_index, residual_type)) in forwarded_residuals.into_iter().enumerate() {
                    forwarded_atoms.push(resolve_residual_atom(residual_index)?);
                    dispositions[residual_index] = Some(NestedResidualDisposition::Operand(position));
                    forwarded_types.push(residual_type);
                }
                let true_branch = defactorize_nested_linear_program(
                    true_branch,
                    dispositions.as_slice(),
                    forwarded_types.as_slice(),
                )?;
                let false_branch = defactorize_nested_linear_program(
                    false_branch,
                    dispositions.as_slice(),
                    forwarded_types.as_slice(),
                )?;
                let mut condition_inputs = Vec::with_capacity(1 + inputs.len() + forwarded_atoms.len());
                condition_inputs.push(predicate_atom);
                condition_inputs.extend(inputs);
                condition_inputs.extend(forwarded_atoms);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::OperandCondition {
                        true_branch: Box::new(true_branch),
                        false_branch: Box::new(false_branch),
                    },
                    inputs: condition_inputs,
                })
            }
            // A linear scan whose residual stacks reference loop-varying residuals moves those stacks into operand
            // position: each referenced stack becomes one extra scanned input, the body gains one trailing lane
            // input per moved stack (the stack type minus its leading length axis), and the body's scan-local
            // references to moved stacks are rewritten into operand form against those inputs. Constant stacks stay
            // factor payloads, with the surviving body references re-indexed against the compacted constant-only
            // stack list.
            Self::Scan { body, residual_stacks, carry_count, length, reverse, unroll }
                if residual_stacks.iter().any(|stack| matches!(stack, ResidualFactor::Reference { .. })) =>
            {
                let mut dispositions = Vec::with_capacity(residual_stacks.len());
                let mut lane_types = Vec::new();
                let mut moved_stack_atoms = Vec::new();
                let mut surviving_stacks = Vec::new();
                for stack in residual_stacks {
                    match stack {
                        ResidualFactor::Reference { index, r#type } => {
                            dispositions.push(Some(NestedResidualDisposition::Operand(lane_types.len())));
                            lane_types.push(r#type.without_dimension(0)?.0);
                            moved_stack_atoms.push(resolve_residual_atom(*index)?);
                        }
                        constant_stack => {
                            dispositions.push(Some(NestedResidualDisposition::Factor(surviving_stacks.len())));
                            surviving_stacks.push(constant_stack.clone());
                        }
                    }
                }
                let body = defactorize_nested_linear_program(body, dispositions.as_slice(), lane_types.as_slice())?;
                inputs.extend(moved_stack_atoms);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Scan {
                        body: Box::new(body),
                        residual_stacks: surviving_stacks,
                        carry_count: *carry_count,
                        length: *length,
                        reverse: *reverse,
                        unroll: *unroll,
                    },
                    inputs,
                })
            }
            operation => {
                // Closed constant factors and factor-free operations pass through unchanged. Residual references
                // hidden in payloads this rule cannot splice operands into — custom VJP call residuals, factor-form
                // while payloads, and condition branches whose predicate factor is a closed constant (defactorization
                // stages exactly one instruction, so a constant predicate cannot be materialized as the operand the
                // rewritten branches would require) — are rejected with the offending operation's name.
                let mut references_residual = false;
                operation.try_map_factors_preserving_extensions(&mut |factor: &ResidualFactor<ArrayType, R>| {
                    if matches!(factor, ResidualFactor::Reference { .. }) {
                        references_residual = true;
                    }
                    Ok(factor.clone())
                })?;
                if references_residual {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "jvp of a while loop whose body pushforward stages {} over a loop-varying residual \
                             reference is not supported",
                            operation.name(),
                        ),
                    });
                }
                Ok(DefactorizedOperation::Operation { operation: operation.clone(), inputs })
            }
        }
    }

    fn linear_while_operation(
        condition: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        body: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        Ok(LinearArrayOperation::While(Box::new(WhileOperation::new(condition, body)?)))
    }
}

impl<V, C, Extension, R, O> SupportsLinearScan<ArrayType, V, ResidualFactor<ArrayType, R>>
    for LinearArrayOperation<V, C, ArrayType, Extension, ResidualFactor<ArrayType, R>, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    R: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    fn linear_scan_operation(
        body: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        residual_stacks: Vec<ResidualFactor<ArrayType, R>>,
        carry_count: usize,
        length: usize,
        reverse: bool,
        unroll: usize,
    ) -> Result<Self, ProgramError> {
        // Rebind the body's factor payloads into the scan-local residual-reference namespace pinned at
        // `ResidualFactor<ArrayType, V>`: references carry over index-for-index against `residual_stacks`, while
        // closed constants are rejected because their payloads live in the enclosing context's value family (the
        // scan JVP rule broadcasts every captured constant into a lane-uniform residual stack before staging, so
        // the rejection is unreachable from the rule).
        let body = body.map_operations(|operation| {
            operation.try_map_factors_preserving_extensions(&mut |factor| match factor {
                ResidualFactor::Reference { index, r#type } => {
                    Ok(ResidualFactor::Reference { index: *index, r#type: r#type.clone() })
                }
                ResidualFactor::Constant(_) => Err(ProgramError::UnsupportedOperation {
                    message: "scan body pushforwards must reference residual stacks instead of carrying closed \
                                  constant factors"
                        .to_string(),
                }),
            })
        })?;
        Ok(LinearArrayOperation::Scan { body: Box::new(body), residual_stacks, carry_count, length, reverse, unroll })
    }
}

impl<T, V, Extension> From<ConditionOperation<V, ArrayOperation<V, T, Extension>, T>>
    for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    fn from(operation: ConditionOperation<V, ArrayOperation<V, T, Extension>, T>) -> Self {
        Self::Condition(Box::new(operation))
    }
}

impl<T, V, Extension> From<WhileOperation<V, ArrayOperation<V, T, Extension>, T>> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    fn from(operation: WhileOperation<V, ArrayOperation<V, T, Extension>, T>) -> Self {
        Self::While(Box::new(operation))
    }
}

impl<T, V, Extension> From<ScanOperation<V, ArrayOperation<V, T, Extension>, T>> for ArrayOperation<V, T, Extension>
where
    T: Parameter + PartialEq + Type,
    V: Value<T>,
{
    fn from(operation: ScanOperation<V, ArrayOperation<V, T, Extension>, T>) -> Self {
        Self::Scan(Box::new(operation))
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
            Self::Reshard { .. } => RESHARD_OPERATION_NAME,
            Self::ShardingConstraint { .. } => SHARDING_CONSTRAINT_OPERATION_NAME,
            Self::Broadcast { .. } => "broadcast",
            Self::Slice { .. } => SLICE_OPERATION_NAME,
            Self::UpdateSlice { .. } => UPDATE_SLICE_OPERATION_NAME,
            Self::DynamicSlice { .. } => DYNAMIC_SLICE_OPERATION_NAME,
            Self::DynamicUpdateSlice => DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
            Self::Pad { .. } => PAD_OPERATION_NAME,
            Self::Concatenate { .. } => CONCATENATE_OPERATION_NAME,
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
            Self::Condition(_) => CONDITION_OPERATION_NAME,
            Self::While(_) => WHILE_OPERATION_NAME,
            Self::Scan(_) => SCAN_OPERATION_NAME,
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
    O: Operation<T>,
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
            Self::Reshard { .. } => RESHARD_OPERATION_NAME,
            Self::ShardingConstraint { .. } => SHARDING_CONSTRAINT_OPERATION_NAME,
            Self::Broadcast { .. } => "broadcast",
            Self::Slice { .. } => SLICE_OPERATION_NAME,
            Self::UpdateSlice { .. } => UPDATE_SLICE_OPERATION_NAME,
            Self::DynamicSlice { .. } => DYNAMIC_SLICE_OPERATION_NAME,
            Self::DynamicUpdateSlice { .. } => DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
            Self::Pad { .. } => PAD_OPERATION_NAME,
            Self::Concatenate { .. } => CONCATENATE_OPERATION_NAME,
            Self::Reduce { kind, .. } => match kind {
                ReductionKind::Sum => "reduce_sum",
                ReductionKind::Mean => "reduce_mean",
                ReductionKind::Max => "reduce_max",
                ReductionKind::Min => "reduce_min",
                ReductionKind::Any => "reduce_any",
                ReductionKind::All => "reduce_all",
            },
            Self::Select { .. } => SELECT_OPERATION_NAME,
            Self::Residual { .. } => "residual",
            Self::Recompute(operation) => operation.name(),
            Self::Condition { .. } | Self::OperandCondition { .. } => CONDITION_OPERATION_NAME,
            Self::While(_) => WHILE_OPERATION_NAME,
            Self::Scan { .. } => SCAN_OPERATION_NAME,
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

/// Renders a captured factor list as a bracketed, comma-separated sequence of `Display` renderings, for use in the
/// bracketed-attribute rendering of captured-index linear operations.
fn render_factor_list<F: Display>(factors: &[F]) -> String {
    format!("[{}]", factors.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
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

/// Extracts concrete values from captured tangent-wrapped start index factors, rejecting symbolic zeros: integer
/// start indices are residuals of the primal computation and must always be concrete at interpretation time. The
/// `operation_name` parameter selects the reported operation name because this helper serves both captured-index
/// dynamic slicing operations.
fn concrete_tangent_factor_indices<T: Type, V: Value<T>>(
    operation_name: &'static str,
    start_indices: &[Tangent<T, V>],
) -> Result<Vec<V>, ProgramError> {
    start_indices
        .iter()
        .map(|index| match index {
            Tangent::Value(value) => Ok(value.clone()),
            Tangent::Zero(_) => {
                Err(TypeError { message: format!("captured {operation_name} start indices must be concrete values") }
                    .into())
            }
        })
        .collect()
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

/// Transposes a linear condition (the `Condition` variant of [`LinearArrayOperation`]).
///
/// The forward linear map runs the linear branch program selected by the captured predicate factor over the branch
/// operands. The predicate is a residual of the primal computation rather than a linear operand, so it has no
/// cotangent and is carried verbatim into the transposed condition, which makes linear-condition transposition total
/// over all predicates: the transpose stages one condition over the transposed branch programs, selected by the same
/// predicate. Output cotangents are materialized via
/// [`stage_cotangent`](crate::tracing_v2::operations::control_flow::stage_cotangent) because the staged transposed
/// condition consumes all output cotangents jointly.
fn transpose_linear_condition<'transpose, V, C, Extension, F, O>(
    predicate: &F,
    true_branch: &Program<ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>, Vec<V>, Vec<V>>,
    false_branch: &Program<ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>, Vec<V>, Vec<V>>,
    context: &mut AbstractTracingContext<
        'transpose,
        ArrayType,
        V,
        LinearArrayOperation<V, C, ArrayType, Extension, F, O>,
    >,
    output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>>],
) -> Result<
    Vec<Cotangent<'transpose, ArrayType, V, LinearArrayOperation<V, C, ArrayType, Extension, F, O>>>,
    ProgramError,
>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    LinearArrayOperation<V, C, ArrayType, Extension, F, O>: SupportsTransposition<ArrayType, V>,
{
    // A condition with no outputs (or only zero output cotangents) is a zero linear map, so every input
    // cotangent is zero. Note that `all` is trivially true for an empty cotangent slice.
    if output_cotangents.iter().all(Cotangent::is_zero) {
        return Ok(vec![Cotangent::Zero; true_branch.input_types().len()]);
    }
    let transposed_condition = LinearArrayOperation::Condition {
        predicate: predicate.clone(),
        true_branch: Box::new(context.transpose_nested(true_branch)?),
        false_branch: Box::new(context.transpose_nested(false_branch)?),
    };
    let materialized = output_cotangents
        .iter()
        .zip(true_branch.output_types())
        .map(|(cotangent, output_type)| {
            crate::tracing_v2::operations::control_flow::stage_cotangent(context, cotangent, &output_type)
        })
        .collect::<Vec<_>>();
    let cotangents = context.stage_operation(transposed_condition, materialized.as_slice())?;
    check_count!("output", cotangents, true_branch.input_types().len(), ProgramError);
    Ok(cotangents.into_iter().map(Cotangent::Staged).collect())
}

/// Transposes a captured-index dynamic slice (the `DynamicSlice` variant of [`LinearArrayOperation`]).
///
/// The forward linear map is `t ↦ dynamic_slice(t, start_indices, sizes)`. Its transpose scatters the output
/// cotangent into a zero array of the input type at the same captured indices:
/// `cotangent ↦ dynamic_update_slice(zeros(input_type), cotangent, start_indices)`. The zero array is staged as a
/// typed `Zero` operation via [`stage_cotangent`](crate::tracing_v2::operations::control_flow::stage_cotangent),
/// and `make_dynamic_update_slice` rebuilds the captured-index dynamic update-slice for staging into the transpose
/// builder. Symbolic-zero cotangents propagate unchanged.
fn transpose_captured_index_dynamic_slice<'transpose, T, V, O, MakeOperationFn>(
    make_dynamic_update_slice: MakeOperationFn,
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
    check_count!("input", input_types, 1, ProgramError);
    check_count!("output", output_cotangents, 1, ProgramError);
    match &output_cotangents[0] {
        Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        Cotangent::Staged(cotangent) => {
            let zeros =
                crate::tracing_v2::operations::control_flow::stage_cotangent(context, &Cotangent::Zero, input_types[0]);
            let outputs = context.stage_operation(make_dynamic_update_slice(), &[zeros, cotangent.clone()])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
        }
    }
}

/// Transposes a captured-index dynamic update-slice (the `DynamicUpdateSlice` variant of [`LinearArrayOperation`]).
///
/// The forward linear map is `(t, u) ↦ dynamic_update_slice(t, u, start_indices)`. Its transpose splits the output
/// cotangent into two contributions at the same captured indices: the input cotangent is the cotangent with the
/// update window zeroed (`dynamic_update_slice(cotangent, zeros(update_type), start_indices)`) and the update
/// cotangent is the dynamic slice of the cotangent at the update window
/// (`dynamic_slice(cotangent, start_indices, update_shape)`). The zero update is staged as a typed `Zero` operation
/// via [`stage_cotangent`](crate::tracing_v2::operations::control_flow::stage_cotangent), and the two closures
/// rebuild the captured-index operations for staging into the transpose builder. Symbolic-zero cotangents propagate
/// unchanged.
fn transpose_captured_index_dynamic_update_slice<'transpose, V, O, MakeUpdateOperationFn, MakeSliceOperationFn>(
    make_dynamic_update_slice: MakeUpdateOperationFn,
    make_dynamic_slice: MakeSliceOperationFn,
    context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
    input_types: &[&ArrayType],
    output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + SupportsZero<ArrayType>,
    MakeUpdateOperationFn: Fn() -> O,
    MakeSliceOperationFn: Fn(Vec<usize>) -> O,
{
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_cotangents, 1, ProgramError);
    match &output_cotangents[0] {
        Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
        Cotangent::Staged(cotangent) => {
            let update_sizes = static_update_sizes("dynamic_update_slice transpose", input_types[1])?;
            let zeros =
                crate::tracing_v2::operations::control_flow::stage_cotangent(context, &Cotangent::Zero, input_types[1]);
            let input_cotangents = context.stage_operation(make_dynamic_update_slice(), &[cotangent.clone(), zeros])?;
            check_count!("output", input_cotangents, 1, ProgramError);
            let update_cotangents =
                context.stage_operation(make_dynamic_slice(update_sizes), std::slice::from_ref(cotangent))?;
            check_count!("output", update_cotangents, 1, ProgramError);
            Ok(vec![
                Cotangent::Staged(input_cotangents.into_iter().next().unwrap()),
                Cotangent::Staged(update_cotangents.into_iter().next().unwrap()),
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

/// Maps the error of a fallible staged-operation reconstruction (e.g., [`SliceOperation::with_strides`] or
/// [`PadOperation::new`] over enum-borne fields) into the [`TypeError`] domain that
/// [`Operation::infer_output_types`] reports, unwrapping the common `ProgramError::Type` payload.
fn reconstruction_type_error(error: ProgramError) -> TypeError {
    match error {
        ProgramError::Type(error) => error,
        error => TypeError { message: error.to_string() },
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
            Self::Dot { dimensions, output_sharding } => DotOperation::new(dimensions.clone())
                .with_output_sharding(output_sharding.clone())
                .infer_output_types(input_types),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).infer_output_types(input_types)
            }
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).infer_output_types(input_types)
            }
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).infer_output_types(input_types),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).infer_output_types(input_types)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).infer_output_types(input_types)
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())
                    .map_err(reconstruction_type_error)?
                    .infer_output_types(input_types)
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).infer_output_types(input_types)
            }
            Self::DynamicSlice { sizes } => DynamicSliceOperation::new(sizes.clone()).infer_output_types(input_types),
            Self::DynamicUpdateSlice => DynamicUpdateSliceOperation.infer_output_types(input_types),
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())
                    .map_err(reconstruction_type_error)?
                    .infer_output_types(input_types)
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).infer_output_types(input_types),
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .infer_output_types(input_types),
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
            Self::Scan(scan) => scan.infer_output_types(input_types),
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
            Self::Dot { dimensions, output_sharding } => DotOperation::new(dimensions.clone())
                .with_output_sharding(output_sharding.clone())
                .render(formatter, indentation),
            Self::Transpose { permutation } => {
                TransposeOperation::new(permutation.clone()).render(formatter, indentation)
            }
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).render(formatter, indentation)
            }
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).render(formatter, indentation),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).render(formatter, indentation)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).render(formatter, indentation)
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                match SliceOperation::new(start_indices.clone(), limit_indices.clone()).with_strides(strides.clone()) {
                    Ok(operation) => operation.render(formatter, indentation),
                    Err(_) => formatter.write_str(self.name()),
                }
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).render(formatter, indentation)
            }
            Self::DynamicSlice { sizes } => DynamicSliceOperation::new(sizes.clone()).render(formatter, indentation),
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                match PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone()) {
                    Ok(operation) => operation.render(formatter, indentation),
                    Err(_) => formatter.write_str(self.name()),
                }
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).render(formatter, indentation),
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .render(formatter, indentation),
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
            Self::Scan(scan) => scan.render(formatter, indentation),
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Compare { .. }
            | Self::Not
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Collective { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_)
            | Self::Scan(_) => Err(unsupported_scalar_metadata_operation(self.operation_name())),
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
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).render(formatter, indentation),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).render(formatter, indentation)
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
            Self::LeftDot { factor, dimensions, output_sharding } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone())
                    .with_output_sharding(output_sharding.clone())
                    .infer_output_types(input_types)
            }
            Self::RightDot { factor, dimensions, output_sharding } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone())
                    .with_output_sharding(output_sharding.clone())
                    .infer_output_types(input_types)
            }
            Self::Reshape { output_shape } => {
                ReshapeOperation::new(output_shape.clone()).infer_output_types(input_types)
            }
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).infer_output_types(input_types),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).infer_output_types(input_types)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).infer_output_types(input_types)
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())
                    .map_err(reconstruction_type_error)?
                    .infer_output_types(input_types)
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).infer_output_types(input_types)
            }
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())
                    .map_err(reconstruction_type_error)?
                    .infer_output_types(input_types)
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).infer_output_types(input_types),
            Self::DynamicSlice { start_indices, sizes } => {
                check_count!("input", input_types, 1, TypeError);
                let mut full_input_types = input_types.to_vec();
                full_input_types.extend(start_indices.iter().map(|index| index.r#type().into_owned()));
                DynamicSliceOperation::new(sizes.clone()).infer_output_types(full_input_types.as_slice())
            }
            Self::DynamicUpdateSlice { start_indices } => {
                check_count!("input", input_types, 2, TypeError);
                let mut full_input_types = input_types.to_vec();
                full_input_types.extend(start_indices.iter().map(|index| index.r#type().into_owned()));
                DynamicUpdateSliceOperation.infer_output_types(full_input_types.as_slice())
            }
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .infer_output_types(input_types),
            Self::Select { condition } => {
                check_count!("input", input_types, 2, TypeError);
                SelectOperation.infer_output_types(&[
                    condition.r#type().into_owned(),
                    input_types[0].clone(),
                    input_types[1].clone(),
                ])
            }
            Self::Residual { factor } => {
                check_count!("input", input_types, 0, TypeError);
                Ok(vec![factor.r#type().into_owned()])
            }
            Self::Recompute(operation) => operation.infer_output_types(input_types),
            Self::Condition { true_branch, false_branch, .. } => {
                let branch_input_types = true_branch.input_types();
                check_types!("condition branch input", &branch_input_types, &false_branch.input_types());
                let output_types = true_branch.output_types();
                check_types!("condition branch output", &output_types, &false_branch.output_types());
                check_count!("input", input_types, branch_input_types.len(), TypeError);
                check_types!("condition operand", &branch_input_types, input_types);
                Ok(output_types)
            }
            Self::OperandCondition { true_branch, false_branch } => {
                let branch_input_types = true_branch.input_types();
                check_types!("condition branch input", &branch_input_types, &false_branch.input_types());
                let output_types = true_branch.output_types();
                check_types!("condition branch output", &output_types, &false_branch.output_types());
                check_count!("input", input_types, 1 + branch_input_types.len(), TypeError);
                if !input_types[0].is_scalar() || input_types[0] != input_types[0].as_boolean() {
                    return Err(TypeError {
                        message: format!(
                            "condition predicate type must be a scalar boolean, but got {}",
                            input_types[0],
                        ),
                    });
                }
                check_types!("condition operand", &branch_input_types, &input_types[1..]);
                Ok(output_types)
            }
            Self::While(operation) => operation.infer_output_types(input_types),
            Self::Scan { body, residual_stacks, carry_count, length, unroll, .. } => {
                validate_scan_unroll(*unroll, *length)?;
                for (index, stack) in residual_stacks.iter().enumerate() {
                    let stack_type = stack.r#type();
                    if stack_type.rank() == 0 || stack_type.dimension(0) != Size::Static(*length) {
                        return Err(TypeError {
                            message: format!(
                                "scan residual stack {index} must have leading dimension {length} but has type \
                                 {stack_type}",
                                stack_type = stack_type.as_ref(),
                            ),
                        });
                    }
                }
                scan_output_types(
                    body.input_types().as_slice(),
                    body.output_types().as_slice(),
                    *carry_count,
                    *length,
                    input_types,
                )
            }
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
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).render(formatter, indentation),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).render(formatter, indentation)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).render(formatter, indentation)
            }
            Self::Scale { factor, .. } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Fill(fill) => fill.render(formatter, indentation),
            Self::LeftDot { factor, dimensions, output_sharding }
            | Self::RightDot { factor, dimensions, output_sharding } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("factor", factor)?;
                    operation.field("dimensions", dimensions)?;
                    if let Some(output_sharding) = output_sharding {
                        operation.field("output_sharding", output_sharding)?;
                    }
                    Ok(())
                })
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                match SliceOperation::new(start_indices.clone(), limit_indices.clone()).with_strides(strides.clone()) {
                    Ok(operation) => operation.render(formatter, indentation),
                    Err(_) => formatter.write_str(self.name()),
                }
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).render(formatter, indentation)
            }
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                match PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone()) {
                    Ok(operation) => operation.render(formatter, indentation),
                    Err(_) => formatter.write_str(self.name()),
                }
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).render(formatter, indentation),
            Self::DynamicSlice { start_indices, sizes } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("start_indices", format_args!("{}", render_factor_list(start_indices)))?;
                    operation.field("sizes", format_args!("{sizes:?}"))
                })
            }
            Self::DynamicUpdateSlice { start_indices } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("start_indices", format_args!("{}", render_factor_list(start_indices)))
                })
            }
            Self::Select { condition } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("condition", condition)),
            Self::Residual { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Recompute(operation) => operation.render(formatter, indentation),
            Self::Condition { predicate, true_branch, false_branch } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("predicate", predicate)?;
                    operation.program("true_branch", true_branch.as_ref())?;
                    operation.program("false_branch", false_branch.as_ref())
                })
            }
            Self::OperandCondition { true_branch, false_branch } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.program("true_branch", true_branch.as_ref())?;
                    operation.program("false_branch", false_branch.as_ref())
                })
            }
            Self::While(operation) => operation.render(formatter, indentation),
            Self::Scan { body, residual_stacks, carry_count, length, reverse, unroll } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("carry_count", carry_count)?;
                    operation.field("length", length)?;
                    operation.field("reverse", reverse)?;
                    if *unroll > 1 {
                        operation.field("unroll", unroll)?;
                    }
                    operation.field("residual_stacks", format_args!("{}", render_factor_list(residual_stacks)))?;
                    operation.program("body", body.as_ref())
                })
            }
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice { .. }
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Residual { .. }
            | Self::Recompute(_)
            | Self::Condition { .. }
            | Self::OperandCondition { .. }
            | Self::While(_)
            | Self::Scan { .. } => Err(unsupported_scalar_metadata_operation(self.operation_name())),
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
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).render(formatter, indentation),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).render(formatter, indentation)
            }
            Self::Scale { factor, .. } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Fill(fill) => fill.render(formatter, indentation),
            Self::LeftDot { factor, dimensions, output_sharding }
            | Self::RightDot { factor, dimensions, output_sharding } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("factor", factor)?;
                    operation.field("dimensions", dimensions)?;
                    if let Some(output_sharding) = output_sharding {
                        operation.field("output_sharding", output_sharding)?;
                    }
                    Ok(())
                })
            }
            Self::Condition { predicate, true_branch, false_branch } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.field("predicate", predicate)?;
                    operation.program("true_branch", true_branch.as_ref())?;
                    operation.program("false_branch", false_branch.as_ref())
                })
            }
            Self::OperandCondition { true_branch, false_branch } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?.bracketed(|operation| {
                    operation.program("true_branch", true_branch.as_ref())?;
                    operation.program("false_branch", false_branch.as_ref())
                })
            }
            Self::While(operation) => operation.render(formatter, indentation),
            Self::Extension(extension) => extension.render(formatter, indentation),
            _ => formatter.write_str(self.name()),
        }
    }
}

/// Rewriting strategy for backend extension payloads while mapping the factor payloads carried by one
/// [`LinearArrayOperation`] (see [`map_linear_array_operation_factors`]).
///
/// The two implementations cover the two factor spaces a linear program can be mapped in:
///
///   - [`RecurseExtensionFactors`] serves enclosing-space passes ([`FactorParameterizedOperation::try_map_factors`]
///     and everything built on it, such as residual compaction, rebasing, and instantiation): backend extensions
///     carry their captured primal payloads in the same enclosing factor space, so the strategy recurses into the
///     extension's own [`FactorParameterizedOperation::try_map_factors`].
///   - [`PreserveExtensionFactors`] serves *body-local* passes (scan-namespace rebinding and per-lane residual
///     instantiation), whose factor space is local to one control-flow body: extension captures never join such a
///     local namespace, so the strategy clones extensions unchanged.
trait ExtensionFactorMapping<Extension, F, MappedFactor>
where
    Extension: Clone + Operation<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
{
    /// Extension operation type produced by this strategy.
    type MappedExtension: Clone + Operation<ArrayType>;

    /// Maps one backend extension payload, with `map_factor` available for strategies that recurse into the
    /// extension's own factor payloads.
    fn map_extension(
        extension: &Extension,
        map_factor: &mut dyn FnMut(&F) -> Result<MappedFactor, ProgramError>,
    ) -> Result<Self::MappedExtension, ProgramError>;
}

/// [`ExtensionFactorMapping`] strategy that maps extension payloads through the extension's own
/// [`FactorParameterizedOperation::try_map_factors`]; see the trait documentation.
struct RecurseExtensionFactors;

impl<Extension, F, MappedFactor> ExtensionFactorMapping<Extension, F, MappedFactor> for RecurseExtensionFactors
where
    Extension: FactorParameterizedOperation<ArrayType, F>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
{
    type MappedExtension = Extension::WithFactor<MappedFactor>;

    fn map_extension(
        extension: &Extension,
        mut map_factor: &mut dyn FnMut(&F) -> Result<MappedFactor, ProgramError>,
    ) -> Result<Self::MappedExtension, ProgramError> {
        extension.try_map_factors(&mut map_factor)
    }
}

/// [`ExtensionFactorMapping`] strategy that clones extension payloads unchanged; see the trait documentation.
struct PreserveExtensionFactors;

/// [`ExtensionFactorMapping`] strategy used for the scan-body traversal of an enclosing factor pass: scan bodies
/// live in their own scan-local factor space that backend extensions never join, so the traversal only converts
/// the body's static extension type to the enclosing pass's `MappedExtension` and reports
/// [`ProgramError::UnsupportedOperation`] if an extension operation is actually present inside a scan body.
struct RejectExtensionFactors<MappedExtension>(PhantomData<MappedExtension>);

impl<Extension, F, MappedFactor, MappedExtension> ExtensionFactorMapping<Extension, F, MappedFactor>
    for RejectExtensionFactors<MappedExtension>
where
    Extension: Clone + Operation<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
    MappedExtension: Clone + Operation<ArrayType>,
{
    type MappedExtension = MappedExtension;

    fn map_extension(
        extension: &Extension,
        _map_factor: &mut dyn FnMut(&F) -> Result<MappedFactor, ProgramError>,
    ) -> Result<Self::MappedExtension, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "extension operation '{}' inside a linear scan body does not support factor mapping",
                extension.name(),
            ),
        })
    }
}

impl<Extension, F, MappedFactor> ExtensionFactorMapping<Extension, F, MappedFactor> for PreserveExtensionFactors
where
    Extension: Clone + Operation<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
{
    type MappedExtension = Extension;

    fn map_extension(
        extension: &Extension,
        _map_factor: &mut dyn FnMut(&F) -> Result<MappedFactor, ProgramError>,
    ) -> Result<Self::MappedExtension, ProgramError> {
        Ok(extension.clone())
    }
}

/// Clones one factor payload unchanged; used as a stable `fn`-pointer identity mapping by the scan-body
/// traversal of [`map_linear_array_operation_factors`].
fn clone_factor<F: Clone>(factor: &F) -> Result<F, ProgramError> {
    Ok(factor.clone())
}

/// Shared payload-mapping core behind [`FactorParameterizedOperation::try_map_factors`] for
/// [`LinearArrayOperation`], parameterized by an [`ExtensionFactorMapping`] strategy that decides how backend
/// extension payloads are rewritten (recursed into for enclosing-space passes, cloned for body-local passes).
fn map_linear_array_operation_factors<V, C, Extension, F, MappedFactor, O, MapFactorFn, Strategy>(
    operation: &LinearArrayOperation<V, C, ArrayType, Extension, F, O>,
    map_factor: &mut MapFactorFn,
) -> Result<LinearArrayOperation<V, C, ArrayType, Strategy::MappedExtension, MappedFactor, O>, ProgramError>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
    MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    Strategy: ExtensionFactorMapping<Extension, F, MappedFactor>,
{
    {
        match operation {
            LinearArrayOperation::CustomVjpCall(call) => {
                Ok(LinearArrayOperation::CustomVjpCall(Box::new(call.map_factors(map_factor)?)))
            }
            LinearArrayOperation::Zero(zero) => Ok(LinearArrayOperation::Zero(zero.clone())),
            LinearArrayOperation::One(one) => Ok(LinearArrayOperation::One(one.clone())),
            LinearArrayOperation::Constant(constant) => Ok(LinearArrayOperation::Constant(constant.clone())),
            LinearArrayOperation::Fill(fill) => Ok(LinearArrayOperation::Fill(fill.clone())),
            LinearArrayOperation::ZeroLike => Ok(LinearArrayOperation::ZeroLike),
            LinearArrayOperation::OneLike => Ok(LinearArrayOperation::OneLike),
            LinearArrayOperation::Add => Ok(LinearArrayOperation::Add),
            LinearArrayOperation::Sub => Ok(LinearArrayOperation::Sub),
            LinearArrayOperation::Neg => Ok(LinearArrayOperation::Neg),
            LinearArrayOperation::Mul => Ok(LinearArrayOperation::Mul),
            LinearArrayOperation::TransferToMemory { destination } => {
                Ok(LinearArrayOperation::TransferToMemory { destination: *destination })
            }
            LinearArrayOperation::Transpose { permutation } => {
                Ok(LinearArrayOperation::Transpose { permutation: permutation.clone() })
            }
            LinearArrayOperation::Scale { factor, .. } => {
                Ok(LinearArrayOperation::Scale { factor: map_factor(factor)? })
            }
            LinearArrayOperation::LeftDot { factor, dimensions, output_sharding } => {
                Ok(LinearArrayOperation::LeftDot {
                    factor: map_factor(factor)?,
                    dimensions: dimensions.clone(),
                    output_sharding: output_sharding.clone(),
                })
            }
            LinearArrayOperation::RightDot { factor, dimensions, output_sharding } => {
                Ok(LinearArrayOperation::RightDot {
                    factor: map_factor(factor)?,
                    dimensions: dimensions.clone(),
                    output_sharding: output_sharding.clone(),
                })
            }
            LinearArrayOperation::Reshape { output_shape } => {
                Ok(LinearArrayOperation::Reshape { output_shape: output_shape.clone() })
            }
            LinearArrayOperation::Reshard { sharding } => {
                Ok(LinearArrayOperation::Reshard { sharding: sharding.clone() })
            }
            LinearArrayOperation::ShardingConstraint { sharding } => {
                Ok(LinearArrayOperation::ShardingConstraint { sharding: sharding.clone() })
            }
            LinearArrayOperation::Broadcast { output_type, output_axes } => Ok(LinearArrayOperation::Broadcast {
                output_type: output_type.clone(),
                output_axes: output_axes.clone(),
            }),
            LinearArrayOperation::Slice { start_indices, limit_indices, strides } => Ok(LinearArrayOperation::Slice {
                start_indices: start_indices.clone(),
                limit_indices: limit_indices.clone(),
                strides: strides.clone(),
            }),
            LinearArrayOperation::UpdateSlice { start_indices } => {
                Ok(LinearArrayOperation::UpdateSlice { start_indices: start_indices.clone() })
            }
            LinearArrayOperation::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                Ok(LinearArrayOperation::Pad {
                    edge_padding_low: edge_padding_low.clone(),
                    edge_padding_high: edge_padding_high.clone(),
                    interior_padding: interior_padding.clone(),
                })
            }
            LinearArrayOperation::Concatenate { axis } => Ok(LinearArrayOperation::Concatenate { axis: *axis }),
            LinearArrayOperation::DynamicSlice { start_indices, sizes } => Ok(LinearArrayOperation::DynamicSlice {
                start_indices: start_indices.iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                sizes: sizes.clone(),
            }),
            LinearArrayOperation::DynamicUpdateSlice { start_indices } => {
                Ok(LinearArrayOperation::DynamicUpdateSlice {
                    start_indices: start_indices.iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                })
            }
            LinearArrayOperation::Reduce { axes, kind, output_sharding } => Ok(LinearArrayOperation::Reduce {
                axes: axes.clone(),
                kind: *kind,
                output_sharding: output_sharding.clone(),
            }),
            LinearArrayOperation::Select { condition } => {
                Ok(LinearArrayOperation::Select { condition: map_factor(condition)? })
            }
            LinearArrayOperation::Residual { factor } => {
                Ok(LinearArrayOperation::Residual { factor: map_factor(factor)? })
            }
            LinearArrayOperation::Recompute(operation) => Ok(LinearArrayOperation::Recompute(operation.clone())),
            LinearArrayOperation::Condition { predicate, true_branch, false_branch } => {
                Ok(LinearArrayOperation::Condition {
                    predicate: map_factor(predicate)?,
                    true_branch: Box::new(true_branch.map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                    false_branch: Box::new(false_branch.map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                })
            }
            // Operand-form condition branches carry only closed constant factors after defactorization, but the
            // traversal stays total over them like the factor-form variant's.
            LinearArrayOperation::OperandCondition { true_branch, false_branch } => {
                Ok(LinearArrayOperation::OperandCondition {
                    true_branch: Box::new(true_branch.map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                    false_branch: Box::new(false_branch.map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                })
            }
            LinearArrayOperation::While(while_operation) => {
                let condition = while_operation.condition().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                })?;
                let body = while_operation.body().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                })?;
                Ok(LinearArrayOperation::While(Box::new(
                    WhileOperation::new(condition, body)?.with_iteration_bound(while_operation.iteration_bound())?,
                )))
            }
            // The scan body's factor space is scan-local (references index `residual_stacks` per lane), so
            // enclosing factor passes map only the stack payloads and never rewrite body-internal factors; the
            // body traversal below merely converts the body's static extension type through
            // [`RejectExtensionFactors`], which fails only if an extension operation actually appears in the body.
            LinearArrayOperation::Scan { body, residual_stacks, carry_count, length, reverse, unroll } => {
                // The factor-cloning function is passed as a `fn` pointer (not a closure) so the recursive
                // monomorphization below reaches a fixed point: nested scans reuse the exact same
                // `(map_factor, Strategy)` instantiation instead of minting a fresh closure type per level.
                let mut clone_scan_local_factor = clone_factor::<ResidualFactor<ArrayType, V>>
                    as fn(&ResidualFactor<ArrayType, V>) -> Result<ResidualFactor<ArrayType, V>, ProgramError>;
                Ok(LinearArrayOperation::Scan {
                    body: Box::new(body.map_operations(|operation| {
                        map_linear_array_operation_factors::<
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            RejectExtensionFactors<Strategy::MappedExtension>,
                        >(operation, &mut clone_scan_local_factor)
                    })?),
                    residual_stacks: residual_stacks.iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                    carry_count: *carry_count,
                    length: *length,
                    reverse: *reverse,
                    unroll: *unroll,
                })
            }
            LinearArrayOperation::Extension(extension) => {
                Ok(LinearArrayOperation::Extension(Strategy::map_extension(extension, map_factor)?))
            }
        }
    }
}

impl<V, C, Extension, F, O> FactorParameterizedOperation<ArrayType, F>
    for LinearArrayOperation<V, C, ArrayType, Extension, F, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: FactorParameterizedOperation<ArrayType, F>,
    F: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    type WithFactor<MappedFactor: Value<ArrayType>> =
        LinearArrayOperation<V, C, ArrayType, Extension::WithFactor<MappedFactor>, MappedFactor, O>;

    fn try_map_factors<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        map_linear_array_operation_factors::<_, _, _, _, _, _, _, RecurseExtensionFactors>(self, map_factor)
    }
}

impl<V, C, Extension, F, O> LinearArrayOperation<V, C, ArrayType, Extension, F, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    F: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    /// Maps this operation's factor payloads through `map_factor` while cloning backend extension payloads
    /// unchanged.
    ///
    /// This is the *body-local* counterpart of [`FactorParameterizedOperation::try_map_factors`], used by passes
    /// whose factor space is local to one control-flow body (scan-namespace rebinding and per-lane residual
    /// instantiation): extension captures live in the enclosing residual environment rather than in any body-local
    /// namespace, so such passes must not rewrite them. The extension type is preserved exactly, which is also what
    /// keeps the rewritten operation embeddable in the same operation universe.
    pub fn try_map_factors_preserving_extensions<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<LinearArrayOperation<V, C, ArrayType, Extension, MappedFactor, O>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        map_linear_array_operation_factors::<_, _, _, _, _, _, _, PreserveExtensionFactors>(self, map_factor)
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
        + BooleanLike
        + TransferToMemory,
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
            Self::Dot { dimensions, .. } => DotOperation::new(dimensions.clone()).interpret(inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).interpret(inputs),
            Self::Scale { factor, .. } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).interpret(inputs),
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).interpret(inputs),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).interpret(inputs)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).interpret(inputs)
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .interpret(inputs)
            }
            Self::UpdateSlice { start_indices } => UpdateSliceOperation::new(start_indices.clone()).interpret(inputs),
            Self::DynamicSlice { sizes } => DynamicSliceOperation::new(sizes.clone()).interpret(inputs),
            Self::DynamicUpdateSlice => DynamicUpdateSliceOperation.interpret(inputs),
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .interpret(inputs)
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).interpret(inputs),
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .interpret(inputs),
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
            Self::Scan(scan) => scan.interpret(inputs),
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Compare { .. }
            | Self::Not
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Collective { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_)
            | Self::Scan(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
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
            Self::Condition { .. } | Self::OperandCondition { .. } => {
                // The captured predicate factor (or the predicate operand, in the operand form) lives in the
                // zero-only tangent space, so it is always a symbolic zero and can never select a branch.
                Err(ProgramError::UnsupportedOperation {
                    message: "symbolic-zero condition predicate interpretation is not supported".to_string(),
                })
            }
            Self::While(operation) => {
                let output_types = infer_zero_only_tangent_output_types(self, inputs)?;
                let condition_outputs = operation.condition().interpret(inputs.to_vec())?;
                check_count!("output", condition_outputs, 1, ProgramError);
                let outputs = operation.body().interpret(inputs.to_vec())?;
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
        + BooleanLike
        + TransferToMemory,
    Extension: Clone + InterpretableOperation<ArrayType, Tangent<ArrayType, V>>,
    O: Clone + Operation<ArrayType>,
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
                    Tangent::Value(value) => Tangent::Value(value.clone().transfer_to_memory(*destination)),
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
            Self::LeftDot { factor, dimensions, .. } => {
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
            Self::RightDot { factor, dimensions, .. } => {
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
            Self::Reshard { sharding } => interpret_tangent_value_unary_value_or_zero(
                &ReshardOperation::new(sharding.clone()),
                &ReshardOperation::new(sharding.clone()),
                inputs,
            ),
            Self::ShardingConstraint { sharding } => interpret_tangent_value_unary_value_or_zero(
                &ShardingConstraintOperation::new(sharding.clone()),
                &ShardingConstraintOperation::new(sharding.clone()),
                inputs,
            ),
            Self::Reduce { axes, kind, output_sharding } => {
                let op = ReduceOperation::new(axes.clone(), *kind).with_output_sharding(output_sharding.clone());
                interpret_tangent_value_unary_value_or_zero(&op, &op, inputs)
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                let op =
                    SliceOperation::new(start_indices.clone(), limit_indices.clone()).with_strides(strides.clone())?;
                interpret_tangent_value_unary_value_or_zero(&op, &op, inputs)
            }
            Self::UpdateSlice { start_indices } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                interpret_materialized_tangent_value_operation(
                    &UpdateSliceOperation::new(start_indices.clone()),
                    inputs,
                )
            }
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                interpret_materialized_tangent_value_operation(
                    &PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?,
                    inputs,
                )
            }
            Self::Concatenate { axis } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                interpret_materialized_tangent_value_operation(&ConcatenateOperation::new(*axis), inputs)
            }
            Self::DynamicSlice { start_indices, sizes } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                match inputs {
                    [input] if input.is_zero() => Ok(symbolic_zero_tangent_value_outputs(output_types)),
                    [Tangent::Value(input)] => {
                        let index_values =
                            concrete_tangent_factor_indices(DYNAMIC_SLICE_OPERATION_NAME, start_indices)?;
                        Ok(vec![Tangent::Value(input.clone().dynamic_slice(index_values, sizes.as_slice())?)])
                    }
                    _ => unreachable!("dynamic_slice output type inference validates the input count"),
                }
            }
            Self::DynamicUpdateSlice { start_indices } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                check_count!("input", inputs, 2, ProgramError);
                let index_values = concrete_tangent_factor_indices(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, start_indices)?;
                let materialize = |tangent: &Tangent<ArrayType, V>| match tangent {
                    Tangent::Zero(r#type) => V::zero(r#type),
                    Tangent::Value(value) => Ok(value.clone()),
                };
                Ok(vec![Tangent::Value(
                    materialize(&inputs[0])?.dynamic_update_slice(materialize(&inputs[1])?, index_values)?,
                )])
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
            Self::Residual { factor } => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![factor.clone()])
            }
            Self::Recompute(_) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "recomputed primal operation {} does not support tangent-wrapped interpretation",
                    self.operation_name(),
                ),
            }),
            Self::Condition { predicate, true_branch, false_branch } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let Tangent::Value(predicate) = predicate else {
                    return Err(TypeError {
                        message: "captured condition predicate must be a concrete value".to_string(),
                    }
                    .into());
                };
                let branch = if predicate.boolean()? { true_branch } else { false_branch };
                let outputs = branch.interpret(inputs.to_vec())?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::OperandCondition { true_branch, false_branch } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let Tangent::Value(predicate) = &inputs[0] else {
                    return Err(TypeError {
                        message: "operand-form condition predicate must be a concrete value".to_string(),
                    }
                    .into());
                };
                let branch = if predicate.boolean()? { true_branch } else { false_branch };
                let outputs = branch.interpret(inputs[1..].to_vec())?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::While(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let mut state = inputs.to_vec();
                let mut completed_iterations = 0;
                loop {
                    // A semantic iteration bound truncates the loop even while the condition still produces true,
                    // mirroring `WhileOperation`'s own interpretation.
                    if operation.iteration_bound().is_some_and(|bound| completed_iterations >= bound) {
                        check_count!("output", state, output_types.len(), ProgramError);
                        return Ok(state);
                    }
                    let condition_outputs = operation.condition().interpret(state.clone())?;
                    check_count!("output", condition_outputs, 1, ProgramError);
                    let predicate = match &condition_outputs[0] {
                        Tangent::Zero(_) => {
                            return Err(ProgramError::UnsupportedOperation {
                                message: "mixed symbolic-zero while predicate interpretation is not supported"
                                    .to_string(),
                            });
                        }
                        Tangent::Value(predicate) => predicate.boolean()?,
                    };
                    if !predicate {
                        check_count!("output", state, output_types.len(), ProgramError);
                        return Ok(state);
                    }
                    state = operation.body().interpret(state)?;
                    check_count!("output", state, operation.state_types().len(), ProgramError);
                    completed_iterations += 1;
                }
            }
            Self::Scan { body, residual_stacks, carry_count, length, reverse, .. } => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let y_slice_types = body.output_types().split_off(*carry_count);
                let outputs = interpret_scan_lanes(
                    *carry_count,
                    *length,
                    *reverse,
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| Ok(Tangent::Zero(stacked_type.clone())),
                    |lane, lane_inputs| {
                        // Bind the body's scan-local residual references against this lane's residual slices and
                        // interpret the resulting direct body.
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_factors_preserving_extensions(&mut |factor| {
                                factor.instantiate(lane_residuals.as_slice())
                            })
                        })?;
                        lane_body.interpret(lane_inputs)
                    },
                )?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
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
    Extension: Clone + InterpretableOperation<ArrayType, V>,
    ArrayOperation<V, ArrayType, Extension>: InterpretableOperation<ArrayType, V>,
    F: CustomVjpResidual<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone + InterpretableOperation<ArrayType, V>,
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
            Self::LeftDot { factor, dimensions, .. } => {
                super::dot::LeftDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::RightDot { factor, dimensions, .. } => {
                super::dot::RightDotOperation::new(factor.clone(), dimensions.clone()).interpret(inputs)
            }
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).interpret(inputs),
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).interpret(inputs),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).interpret(inputs)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).interpret(inputs)
            }
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .interpret(inputs),
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .interpret(inputs)
            }
            Self::UpdateSlice { start_indices } => UpdateSliceOperation::new(start_indices.clone()).interpret(inputs),
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .interpret(inputs)
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).interpret(inputs),
            Self::DynamicSlice { start_indices, sizes } => {
                check_count!("input", inputs, 1, ProgramError);
                let index_values =
                    start_indices.iter().map(|index| index.residual_value()).collect::<Result<Vec<_>, _>>()?;
                Ok(vec![inputs[0].clone().dynamic_slice(index_values, sizes.as_slice())?])
            }
            Self::DynamicUpdateSlice { start_indices } => {
                check_count!("input", inputs, 2, ProgramError);
                let index_values =
                    start_indices.iter().map(|index| index.residual_value()).collect::<Result<Vec<_>, _>>()?;
                Ok(vec![inputs[0].clone().dynamic_update_slice(inputs[1].clone(), index_values)?])
            }
            Self::Select { condition } => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![V::select(condition.residual_value()?, inputs[0].clone(), inputs[1].clone())?])
            }
            Self::Residual { factor } => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![factor.residual_value()?])
            }
            Self::Recompute(operation) => operation.interpret(inputs),
            Self::Condition { predicate, true_branch, false_branch } => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if predicate.residual_value()?.boolean()? { true_branch } else { false_branch };
                branch.interpret(inputs.to_vec())
            }
            Self::OperandCondition { true_branch, false_branch } => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if inputs[0].boolean()? { true_branch } else { false_branch };
                branch.interpret(inputs[1..].to_vec())
            }
            Self::While(operation) => operation.interpret(inputs),
            Self::Scan { body, residual_stacks, carry_count, length, reverse, .. } => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let stack_values =
                    residual_stacks.iter().map(|stack| stack.residual_value()).collect::<Result<Vec<_>, _>>()?;
                let y_slice_types = body.output_types().split_off(*carry_count);
                interpret_scan_lanes(
                    *carry_count,
                    *length,
                    *reverse,
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| V::zero(stacked_type),
                    |lane, lane_inputs| {
                        // Bind the body's scan-local residual references against this lane's residual slices and
                        // interpret the resulting direct body.
                        let lane_residuals = stack_values
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_factors_preserving_extensions(&mut |factor| {
                                factor.instantiate(lane_residuals.as_slice())
                            })
                        })?;
                        lane_body.interpret(lane_inputs)
                    },
                )
            }
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice { .. }
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Residual { .. }
            | Self::Recompute(_)
            | Self::Condition { .. }
            | Self::OperandCondition { .. }
            | Self::While(_)
            | Self::Scan { .. } => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice { .. }
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Residual { .. }
            | Self::Recompute(_)
            | Self::Condition { .. }
            | Self::OperandCondition { .. }
            | Self::While(_)
            | Self::Scan { .. } => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
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
    Extension: Clone + InterpretableOperation<ArrayType, Tracer<S>>,
    Tracer<S>: Add<Output = Tracer<S>>
        + Sub<Output = Tracer<S>>
        + Neg<Output = Tracer<S>>
        + Mul<Output = Tracer<S>>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::dot::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + Broadcast<Output = Tracer<S>>
        + crate::tracing_v2::operations::reduce::Reduce
        + BooleanLike,
    Vec<Tracer<S>>:
        Parameterized<Tracer<S>, To<Tracer<S>> = Vec<Tracer<S>>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone
        + Operation<ArrayType>
        + SupportsZero<ArrayType>
        + SupportsTransferToMemory<ArrayType>
        + SupportsSelect<ArrayType>
        + SupportsSlice<ArrayType>
        + SupportsUpdateSlice<ArrayType>
        + SupportsPad<ArrayType>
        + SupportsDynamicSlice<ArrayType>
        + SupportsDynamicUpdateSlice<ArrayType>
        + SupportsConcatenate<ArrayType>
        + SupportsReshard
        + SupportsShardingConstraint,
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
            Self::LeftDot { factor, dimensions, .. } => {
                use crate::tracing_v2::operations::dot::Dot;
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![factor.clone().dot(inputs[0].clone(), dimensions)])
            }
            Self::RightDot { factor, dimensions, .. } => {
                use crate::tracing_v2::operations::dot::Dot;
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone().dot(factor.clone(), dimensions)])
            }
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).interpret(inputs),
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).interpret(inputs),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).interpret(inputs)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).interpret(inputs)
            }
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .interpret(inputs),
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .interpret(inputs)
            }
            Self::UpdateSlice { start_indices } => UpdateSliceOperation::new(start_indices.clone()).interpret(inputs),
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .interpret(inputs)
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).interpret(inputs),
            Self::DynamicSlice { start_indices, sizes } => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone().dynamic_slice(start_indices.clone(), sizes.as_slice())?])
            }
            Self::DynamicUpdateSlice { start_indices } => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].clone().dynamic_update_slice(inputs[1].clone(), start_indices.clone())?])
            }
            Self::Select { condition } => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![Tracer::select(condition.clone(), inputs[0].clone(), inputs[1].clone())?])
            }
            Self::Residual { factor } => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![factor.clone()])
            }
            Self::Recompute(operation) => {
                // Recomputed primal operations replay over tracers by staging the wrapped primal operation into the
                // tracers' own staging context, which the operands provide; nullary recomputed operations carry no
                // operand and therefore no context to stage into.
                let Some(input) = inputs.first() else {
                    return Err(TypeError {
                        message: format!(
                            "nullary recomputed primal operation {} over tracer values was not materialized before \
                             interpretation",
                            operation.name(),
                        ),
                    }
                    .into());
                };
                input.context().stage_operation(operation.clone(), inputs)
            }
            Self::Condition { predicate, true_branch, false_branch } => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if predicate.boolean()? { true_branch } else { false_branch };
                branch.interpret(inputs.to_vec())
            }
            Self::OperandCondition { true_branch, false_branch } => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if inputs[0].boolean()? { true_branch } else { false_branch };
                branch.interpret(inputs[1..].to_vec())
            }
            Self::While(operation) => operation.interpret(inputs),
            Self::Scan { body, residual_stacks, carry_count, length, reverse, .. } => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                // Replaying a linear scan over tracers unrolls the statically counted loop: each lane's body
                // pushforward is bound against that lane's residual slices and inlined into the tracers' staging
                // context, mirroring how the linear condition inlines its captured branch. Stacked output
                // accumulators are staged as typed zero operations because tracer values cannot materialize
                // constants directly.
                let Some(exemplar) = inputs.first().or_else(|| residual_stacks.first()) else {
                    return Err(ProgramError::UnsupportedOperation {
                        message: "cannot replay a linear scan with no inputs and no residual stacks over tracer \
                                  values"
                            .to_string(),
                    });
                };
                let context = exemplar.context();
                let y_slice_types = body.output_types().split_off(*carry_count);
                interpret_scan_lanes(
                    *carry_count,
                    *length,
                    *reverse,
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| {
                        let mut outputs = context.stage_operation(
                            <O as SupportsZero<ArrayType>>::zero_operation(stacked_type.clone()),
                            &[] as &[Tracer<S>],
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    },
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_factors_preserving_extensions(&mut |factor| {
                                factor.instantiate(lane_residuals.as_slice())
                            })
                        })?;
                        lane_body.interpret(lane_inputs)
                    },
                )
            }
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice { .. }
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Residual { .. }
            | Self::Recompute(_)
            | Self::Condition { .. }
            | Self::OperandCondition { .. }
            | Self::While(_)
            | Self::Scan { .. } => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
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
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::Concatenate { axis } => {
                ConcatenateOperation::new(*axis).transpose(context, input_types, output_cotangents)
            }
            Self::DynamicSlice { start_indices, .. } => transpose_captured_index_dynamic_slice(
                || Self::DynamicUpdateSlice { start_indices: start_indices.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::DynamicUpdateSlice { start_indices } => transpose_captured_index_dynamic_update_slice(
                || Self::DynamicUpdateSlice { start_indices: start_indices.clone() },
                |sizes| Self::DynamicSlice { start_indices: start_indices.clone(), sizes },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Select { .. } => {
                // Every value in the zero-only tangent space is zero, so the masked branch cotangents are
                // symbolic zeros as well.
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![Cotangent::Zero, Cotangent::Zero])
            }
            Self::Residual { .. } => Err(ProgramError::UnsupportedOperation {
                message: "residual is not a linear map and does not support transposition".to_string(),
            }),
            Self::Recompute(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "recomputed primal operation {} is not a linear map and does not support transposition",
                    operation.name(),
                ),
            }),
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
            Self::Reshard { .. } => {
                // Reshard's transpose reshards the cotangent to the cotangent dual of the input's sharding; the
                // staged cotangent's operation type supports it, so the symbolic-zero tangent space inlines the same
                // rule as the general `ReshardOperation` transpose instead of delegating (which would require
                // `Infallible: Reshard`).
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let contribution = match input_types[0].sharding() {
                            Some(input_sharding) => cotangent.clone().reshard(&input_sharding.cotangent_dual()),
                            None => cotangent.clone(),
                        };
                        Ok(vec![Cotangent::Staged(contribution)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::ShardingConstraint { sharding } => {
                // The hint is self-adjoint: its transpose applies the same hint to the cotangent (see the general
                // `ShardingConstraintOperation` transpose). Inlined here for the same `Infallible: ConstrainSharding`
                // reason as `Reshard`.
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone().constrain_sharding(sharding))])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Broadcast { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ProgramError::UnsupportedOperation {
                        message: "broadcast transpose is not supported (would need reduce-sum)".to_string(),
                    }),
                }
            }
            Self::Reduce { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ProgramError::UnsupportedOperation {
                        message: "reduce transpose is not supported (would need broadcast-back with stored input \
                                  shape)"
                            .to_string(),
                    }),
                }
            }
            Self::Condition { predicate, true_branch, false_branch } => transpose_linear_condition(
                predicate,
                true_branch.as_ref(),
                false_branch.as_ref(),
                context,
                output_cotangents,
            ),
            Self::OperandCondition { .. } => Err(ProgramError::UnsupportedOperation {
                message: "operand-form condition inside a fused while body does not support transposition".to_string(),
            }),
            Self::While(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Scan { .. } => {
                // Every value in the zero-only tangent space is zero, so transposing the zero linear scan yields
                // symbolic zero cotangents for every input.
                Ok(vec![Cotangent::Zero; input_types.len()])
            }
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
    V: crate::tracing_v2::operations::dot::DotOps + Scale<f64, Output = V> + Reshard + ConstrainSharding,
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
                Err(ProgramError::UnsupportedOperation {
                    message: "linear `Mul` transpose is not supported (rewrite to `Scale` before transposition)"
                        .to_string(),
                })
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
            Self::LeftDot { factor, dimensions, .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                let Tangent::Value(_) = factor else {
                    return Ok(vec![Cotangent::Zero]);
                };
                let factor_rank = factor.r#type().as_ref().rank();
                let adjoint =
                    crate::tracing_v2::operations::dot::adjoint_dimensions_for_left_dot(dimensions, factor_rank);
                // The adjoint's output *is* the input's cotangent, so its sharding is pinned to the cotangent dual
                // of the input's sharding instead of being re-derived.
                let adjoint_output_sharding = input_types[0].sharding().map(Sharding::cotangent_dual);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let contribution = match &adjoint_output_sharding {
                            Some(output_sharding) => cotangent.clone().left_dot_with_output_sharding(
                                factor.clone(),
                                &adjoint,
                                output_sharding,
                            ),
                            None => cotangent.clone().left_dot(factor.clone(), &adjoint),
                        };
                        Ok(vec![Cotangent::Staged(contribution)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::RightDot { factor, dimensions, .. } => {
                check_count!("input", input_types, 1, ProgramError);
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
                // The adjoint's output *is* the input's cotangent, so its sharding is pinned to the cotangent dual
                // of the input's sharding instead of being re-derived.
                let adjoint_output_sharding = input_types[0].sharding().map(Sharding::cotangent_dual);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let contribution = match &adjoint_output_sharding {
                            Some(output_sharding) => cotangent.clone().right_dot_with_output_sharding(
                                factor.clone(),
                                &adjoint,
                                output_sharding,
                            ),
                            None => cotangent.clone().right_dot(factor.clone(), &adjoint),
                        };
                        Ok(vec![Cotangent::Staged(contribution)])
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
            Self::Reshard { sharding } => {
                ReshardOperation::new(sharding.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Broadcast { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ProgramError::UnsupportedOperation {
                        message: "broadcast transpose is not supported (would need reduce-sum)".to_string(),
                    }),
                }
            }
            Self::Reduce { .. } => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ProgramError::UnsupportedOperation {
                        message: "reduce transpose is not supported (would need broadcast-back with stored input \
                                  shape)"
                            .to_string(),
                    }),
                }
            }
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::Concatenate { axis } => {
                ConcatenateOperation::new(*axis).transpose(context, input_types, output_cotangents)
            }
            Self::DynamicSlice { start_indices, .. } => transpose_captured_index_dynamic_slice(
                || Self::DynamicUpdateSlice { start_indices: start_indices.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::DynamicUpdateSlice { start_indices } => transpose_captured_index_dynamic_update_slice(
                || Self::DynamicUpdateSlice { start_indices: start_indices.clone() },
                |sizes| Self::DynamicSlice { start_indices: start_indices.clone(), sizes },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Select { condition } => transpose_captured_condition_select(
                || Self::Select { condition: condition.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Residual { .. } => Err(ProgramError::UnsupportedOperation {
                message: "residual is not a linear map and does not support transposition".to_string(),
            }),
            Self::Recompute(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "recomputed primal operation {} is not a linear map and does not support transposition",
                    operation.name(),
                ),
            }),
            Self::Condition { predicate, true_branch, false_branch } => transpose_linear_condition(
                predicate,
                true_branch.as_ref(),
                false_branch.as_ref(),
                context,
                output_cotangents,
            ),
            Self::OperandCondition { .. } => Err(ProgramError::UnsupportedOperation {
                message: "operand-form condition inside a fused while body does not support transposition".to_string(),
            }),
            Self::While(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Scan { .. } => Err(ProgramError::UnsupportedOperation {
                message: "scan transposition over tangent-wrapped values is not supported".to_string(),
            }),
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
            Self::Mul => Err(ProgramError::UnsupportedOperation {
                message: "linear `Mul` transpose is not supported (rewrite to `Scale` before transposition)"
                    .to_string(),
            }),
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice { .. }
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Residual { .. }
            | Self::Recompute(_)
            | Self::Condition { .. }
            | Self::OperandCondition { .. }
            | Self::While(_)
            | Self::Scan { .. } => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
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
    // Scan bodies pin their factor payloads to the scan-local `ResidualFactor<ArrayType, V>` namespace, so
    // transposing a scan body re-instantiates this impl at that factor type; spelling the extension obligation at
    // that fixed point (instead of requiring the body operation's `SupportsTransposition` directly) keeps the trait
    // solver's recursion finite.
    Extension: TransposableOperation<
            ArrayType,
            V,
            LinearArrayOperation<V, C, ArrayType, Extension, ResidualFactor<ArrayType, V>, O>,
        >,
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
            Self::Mul => Err(ProgramError::UnsupportedOperation {
                message: "linear `Mul` transpose is not supported (rewrite to `Scale` before transposition)"
                    .to_string(),
            }),
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
            Self::LeftDot { factor, dimensions, .. } => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                let factor_rank = factor.r#type().as_ref().rank();
                let adjoint_dims = super::dot::adjoint_dimensions_for_left_dot(dimensions, factor_rank);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            // The adjoint's output *is* the input's cotangent, so its sharding is pinned to the
                            // cotangent dual of the input's sharding instead of being re-derived.
                            Self::LeftDot {
                                factor: factor.clone(),
                                dimensions: adjoint_dims,
                                output_sharding: input_types[0].sharding().map(Sharding::cotangent_dual),
                            },
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::RightDot { factor, dimensions, .. } => {
                check_count!("input", input_types, 1, ProgramError);
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
                            // The adjoint's output *is* the input's cotangent, so its sharding is pinned to the
                            // cotangent dual of the input's sharding instead of being re-derived.
                            Self::RightDot {
                                factor: factor.clone(),
                                dimensions: adjoint_dims,
                                output_sharding: input_types[0].sharding().map(Sharding::cotangent_dual),
                            },
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
            Self::Reshard { sharding } => {
                ReshardOperation::new(sharding.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Broadcast { output_type, output_axes } => BroadcastOperation::new(
                output_type.clone(),
                output_axes.clone(),
            )
            .transpose(context, input_types, output_cotangents),
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .transpose(context, input_types, output_cotangents),
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).transpose(context, input_types, output_cotangents)
            }
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::Concatenate { axis } => {
                ConcatenateOperation::new(*axis).transpose(context, input_types, output_cotangents)
            }
            Self::DynamicSlice { start_indices, .. } => transpose_captured_index_dynamic_slice(
                || Self::DynamicUpdateSlice { start_indices: start_indices.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::DynamicUpdateSlice { start_indices } => transpose_captured_index_dynamic_update_slice(
                || Self::DynamicUpdateSlice { start_indices: start_indices.clone() },
                |sizes| Self::DynamicSlice { start_indices: start_indices.clone(), sizes },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Select { condition } => transpose_captured_condition_select(
                || Self::Select { condition: condition.clone() },
                context,
                input_types,
                output_cotangents,
            ),
            Self::Residual { .. } => Err(ProgramError::UnsupportedOperation {
                message: "residual is not a linear map and does not support transposition".to_string(),
            }),
            Self::Recompute(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "recomputed primal operation {} is not a linear map and does not support transposition",
                    operation.name(),
                ),
            }),
            Self::Condition { predicate, true_branch, false_branch } => transpose_linear_condition(
                predicate,
                true_branch.as_ref(),
                false_branch.as_ref(),
                context,
                output_cotangents,
            ),
            Self::OperandCondition { .. } => Err(ProgramError::UnsupportedOperation {
                message: "operand-form condition inside a fused while body does not support transposition".to_string(),
            }),
            Self::While(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Scan { body, residual_stacks, carry_count, length, reverse, unroll } => {
                // A scan with only zero output cotangents is a zero linear map, so every input cotangent is zero.
                if output_cotangents.iter().all(Cotangent::is_zero) {
                    return Ok(vec![Cotangent::Zero; input_types.len()]);
                }
                // Linear-scan transposition is total: the body pushforward maps `[carry..., x_slice...]` to
                // `[carry..., y_slice...]`, so its program transpose maps `[carry_cotangent..., y_slice_cotangent...]`
                // to `[carry_cotangent..., x_slice_cotangent...]` — the same scan-body signature with the same carry
                // count. Flipping `reverse` pairs cotangent lane `i` with residual stack lane `i` exactly when the
                // forward scan consumed them, so the same residual stacks (and the lowering-only unroll factor)
                // carry over verbatim.
                let transposed = LinearArrayOperation::Scan {
                    body: Box::new(body.transpose()?),
                    residual_stacks: residual_stacks.clone(),
                    carry_count: *carry_count,
                    length: *length,
                    reverse: !*reverse,
                    unroll: *unroll,
                };
                let mut output_types = body.output_types();
                let y_slice_types = output_types.split_off(*carry_count);
                output_types.extend(y_slice_types.iter().map(|slice_type| stacked_scan_type(slice_type, *length)));
                check_count!("output", output_cotangents, output_types.len(), ProgramError);
                let materialized = output_cotangents
                    .iter()
                    .zip(output_types.iter())
                    .map(|(cotangent, output_type)| {
                        crate::tracing_v2::operations::control_flow::stage_cotangent(context, cotangent, output_type)
                    })
                    .collect::<Vec<_>>();
                let cotangents = context.stage_operation(transposed, materialized.as_slice())?;
                check_count!("output", cotangents, input_types.len(), ProgramError);
                Ok(cotangents.into_iter().map(Cotangent::Staged).collect())
            }
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
            Self::Mul => Err(ProgramError::UnsupportedOperation {
                message: "linear `Mul` transpose is not supported (rewrite to `Scale` before transposition)"
                    .to_string(),
            }),
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice { .. }
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Select { .. }
            | Self::Residual { .. }
            | Self::Recompute(_)
            | Self::Condition { .. }
            | Self::OperandCondition { .. }
            | Self::While(_)
            | Self::Scan { .. } => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<F, D> DifferentiableOperation<D> for ScalarOperation<F>
where
    F: Value<DataType>,
    D: DifferentiationContext<Type = DataType, Constant = F> + Domain<Operation = ScalarOperation<F>>,
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
    ScalarOperation<F>: Clone + ProgramLinearizableOperation<D>,
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
    D::Tangent: Transpose
        + Broadcast<Output = D::Tangent>
        + super::reduce::Reduce
        + Slice<Output = D::Tangent>
        + Reshard
        + ConstrainSharding,
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
        + SupportsConcatenate<ArrayType>
        + SupportsLinearSelect<ArrayType, ResidualFactor<ArrayType, D::Value>>
        + SupportsLinearDynamicSlice<ArrayType, ResidualFactor<ArrayType, D::Value>>
        + SupportsLinearDynamicUpdateSlice<ArrayType, ResidualFactor<ArrayType, D::Value>>
        + SupportsLinearCondition<ArrayType, D::Tangent, ResidualFactor<ArrayType, D::Value>>
        + SupportsLinearWhile<
            ArrayType,
            D::Tangent,
            ResidualFactor<ArrayType, D::Value>,
            ArrayOperation<V, ArrayType, Extension>,
        > + SupportsLinearScan<ArrayType, D::Tangent, ResidualFactor<ArrayType, D::Value>>,
    ArrayOperation<V, ArrayType, Extension>: Clone + ProgramLinearizableOperation<D>,
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
            Self::Dot { dimensions, output_sharding } => DotOperation::new(dimensions.clone())
                .with_output_sharding(output_sharding.clone())
                .jvp(context, inputs),
            Self::Transpose { permutation } => TransposeOperation::new(permutation.clone()).jvp(context, inputs),
            Self::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).jvp(context, inputs),
            Self::Reshard { sharding } => ReshardOperation::new(sharding.clone()).jvp(context, inputs),
            Self::ShardingConstraint { sharding } => {
                ShardingConstraintOperation::new(sharding.clone()).jvp(context, inputs)
            }
            Self::Broadcast { output_type, output_axes } => {
                BroadcastOperation::new(output_type.clone(), output_axes.clone()).jvp(context, inputs)
            }
            Self::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
                .with_output_sharding(output_sharding.clone())
                .jvp(context, inputs),
            Self::Slice { start_indices, limit_indices, strides } => {
                SliceOperation::new(start_indices.clone(), limit_indices.clone())
                    .with_strides(strides.clone())?
                    .jvp(context, inputs)
            }
            Self::UpdateSlice { start_indices } => {
                UpdateSliceOperation::new(start_indices.clone()).jvp(context, inputs)
            }
            Self::DynamicSlice { sizes } => DynamicSliceOperation::new(sizes.clone()).jvp(context, inputs),
            Self::DynamicUpdateSlice => DynamicUpdateSliceOperation.jvp(context, inputs),
            Self::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
                PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                    .jvp(context, inputs)
            }
            Self::Concatenate { axis } => ConcatenateOperation::new(*axis).jvp(context, inputs),
            Self::Compare { direction } => CompareOperation::new(*direction).jvp(context, inputs),
            Self::Not => NotOperation.jvp(context, inputs),
            Self::And => AndOperation.jvp(context, inputs),
            Self::Or => OrOperation.jvp(context, inputs),
            Self::Xor => XorOperation.jvp(context, inputs),
            Self::Select => SelectOperation.jvp(context, inputs),
            // These two dispatches use fully-qualified syntax because method probing on the boxed receiver would
            // have to evaluate the rules' nested-linearization bounds against an unconstrained context type; pinning
            // the differentiation context to `D` resolves them against this impl's where clauses instead.
            Self::Condition(condition) => {
                <ConditionOperation<V, Self, ArrayType> as DifferentiableOperation<D>>::jvp(condition, context, inputs)
            }
            Self::While(while_operation) => <WhileOperation<V, Self, ArrayType> as DifferentiableOperation<D>>::jvp(
                while_operation,
                context,
                inputs,
            ),
            Self::Scan(scan) => {
                <ScanOperation<V, Self, ArrayType> as DifferentiableOperation<D>>::jvp(scan, context, inputs)
            }
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
        LinearOperationOf<D>: SupportsZero<DataType>,
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
            | Self::Reshard { .. }
            | Self::ShardingConstraint { .. }
            | Self::Broadcast { .. }
            | Self::Slice { .. }
            | Self::UpdateSlice { .. }
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice
            | Self::Pad { .. }
            | Self::Concatenate { .. }
            | Self::Reduce { .. }
            | Self::Compare { .. }
            | Self::Not
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Collective { .. }
            | Self::Select
            | Self::Condition(_)
            | Self::While(_)
            | Self::Scan(_) => {
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
        ArrayOperation::Dot { dimensions, output_sharding } => DotOperation::new(dimensions.clone())
            .with_output_sharding(output_sharding.clone())
            .batch(&(), inputs)?,
        ArrayOperation::Transpose { permutation } => TransposeOperation::new(permutation.clone()).batch(&(), inputs)?,
        ArrayOperation::Reshape { output_shape } => ReshapeOperation::new(output_shape.clone()).batch(&(), inputs)?,
        ArrayOperation::Reshard { sharding } => ReshardOperation::new(sharding.clone()).batch(&(), inputs)?,
        ArrayOperation::ShardingConstraint { sharding } => {
            ShardingConstraintOperation::new(sharding.clone()).batch(&(), inputs)?
        }
        ArrayOperation::Broadcast { output_type, output_axes } => {
            BroadcastOperation::new(output_type.clone(), output_axes.clone()).batch(&(), inputs)?
        }
        ArrayOperation::Slice { start_indices, limit_indices, strides } => {
            SliceOperation::new(start_indices.clone(), limit_indices.clone())
                .with_strides(strides.clone())?
                .batch(&(), inputs)?
        }
        ArrayOperation::UpdateSlice { start_indices } => {
            UpdateSliceOperation::new(start_indices.clone()).batch(&(), inputs)?
        }
        ArrayOperation::DynamicSlice { sizes } => DynamicSliceOperation::new(sizes.clone()).batch(&(), inputs)?,
        ArrayOperation::DynamicUpdateSlice => DynamicUpdateSliceOperation.batch(&(), inputs)?,
        ArrayOperation::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
            PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                .batch(&(), inputs)?
        }
        ArrayOperation::Concatenate { axis } => ConcatenateOperation::new(*axis).batch(&(), inputs)?,
        ArrayOperation::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
            .with_output_sharding(output_sharding.clone())
            .batch(&(), inputs)?,
        ArrayOperation::Compare { direction } => CompareOperation::new(*direction).batch(&(), inputs)?,
        ArrayOperation::Not => NotOperation.batch(&(), inputs)?,
        ArrayOperation::And => AndOperation.batch(&(), inputs)?,
        ArrayOperation::Or => OrOperation.batch(&(), inputs)?,
        ArrayOperation::Xor => XorOperation.batch(&(), inputs)?,
        ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Collective { .. }
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::Scan(_)
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
        + Zero<ArrayType>
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
            Self::Scan(scan) => scan.batch(context, inputs),
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
/// [`ProgramBatchableOperation`](crate::tracing_v2::batching::ProgramBatchableOperation) / lane-alignment bounds exist
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
    Self: ProgramBatchableOperation<V>,
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
            Self::Scan(scan) => scan.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Program-level batching for the [`ArrayOperation`] sum type, backing the re-wrapping `batch` rules of
/// [`CustomJvpOperation`] and [`CustomVjpOperation`]; see
/// [`ProgramBatchableOperation`](crate::tracing_v2::batching::ProgramBatchableOperation).
///
/// The where clauses here are deliberately the *leaf* closure of what `batch_program::<V, Self>` needs — the
/// blanket traced batching impl's bounds instantiated at [`ProgramBatchingContext`] — rather than the
/// `Self: BatchableOperation<..>` bound itself. Spelling out the leaves keeps instantiating this impl free of
/// batching-context obligations, which is what lets the traced batching impl require
/// `Self: ProgramBatchableOperation<..>` without sending the trait solver into an unbounded
/// batching-context recursion.
impl<V, E> ProgramBatchableOperation<V> for ArrayOperation<V, ArrayType, E>
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
    fn batch_program(
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        axis_size: usize,
        input_batch_axes: &[Option<usize>],
        output_batch_axes: ProgramBatchingOutputAxes,
    ) -> Result<(crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>, Vec<Option<usize>>), ProgramError> {
        crate::tracing_v2::batching::batch_program::<V, Self>(program, axis_size, input_batch_axes, output_batch_axes)
    }
}

/// Nested symbolic linearization for the [`ArrayOperation`] sum type, backing the staged-condition JVP rule of
/// [`ConditionOperation`]; see [`ProgramLinearizableOperation`](crate::tracing_v2::ProgramLinearizableOperation).
///
/// The where clauses here are deliberately the *leaf* closure of what
/// [`linearize_program`](crate::tracing_v2::linearize_program)`::<E, Self>` needs — the generic JVP
/// dispatch impl's bounds instantiated at [`LinearizationContextOf`] — rather than the
/// `Self: DifferentiableOperation<LinearizationContextOf<E, Self>>` bound itself. Spelling out the leaves keeps
/// instantiating this impl free of derived-context differentiation obligations (the recursive obligation is
/// discharged once, as a definition-time body check), which is what lets the JVP dispatch impl require
/// `Self: ProgramLinearizableOperation<E>` without sending the trait solver into an unbounded nested-context
/// recursion. The `WithFactor<V> = ..` equality pins the canonical linear operation as a fixed point of factor
/// reparameterization, which is what collapses `LinearizationContextOf<LinearizationContextOf<E, ..>, ..>`
/// to `LinearizationContextOf<E, ..>` and keeps the obligations finite for nested conditions.
impl<V, E, Extension> ProgramLinearizableOperation<E> for ArrayOperation<V, ArrayType, Extension>
where
    V: Value<ArrayType>,
    E: DifferentiationContext<Type = ArrayType, Constant = V>,
    E::Tangent: Transpose
        + Broadcast<Output = E::Tangent>
        + super::reduce::Reduce
        + Slice<Output = E::Tangent>
        + Reshard
        + ConstrainSharding,
    E::LinearOperation<E::Tangent, V>:
        FactorParameterizedOperation<ArrayType, V, WithFactor<V> = E::LinearOperation<E::Tangent, V>>,
    Extension: Clone + Operation<ArrayType> + DifferentiableOperation<LinearizationContextOf<E, Self>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: SupportsLinearArrayOperation<ArrayType, ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + crate::tracing_v2::ResidualizedOperation<LinearizationContextOf<E, Self>>
        + SupportsZero<ArrayType>
        + SupportsCustomVjpCall<ArrayType, V, Self, ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + SupportsTransferToMemory<ArrayType>
        + SupportsConcatenate<ArrayType>
        + SupportsLinearSelect<ArrayType, ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + SupportsLinearDynamicSlice<ArrayType, ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + SupportsLinearDynamicUpdateSlice<ArrayType, ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + SupportsLinearCondition<
            ArrayType,
            E::Tangent,
            ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
        > + SupportsLinearWhile<
            ArrayType,
            E::Tangent,
            ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            Self,
        > + SupportsLinearScan<ArrayType, E::Tangent, ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: FactorParameterizedOperation<
            ArrayType,
            ResidualFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            WithFactor<ResidualFactor<ArrayType, E::Value>> = LinearOperationOf<E>,
        >,
{
    fn linearize_program(
        differentiable: &E,
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<NestedLinearization<E, Self>, ProgramError> {
        crate::tracing_v2::differentiation::linearize_program(differentiable, program)
    }
}

/// Nested symbolic linearization for the [`ScalarOperation`] sum type, mirroring the [`ArrayOperation`] impl above
/// (refer to its documentation for why the where clauses spell the *leaf* closure of what
/// [`linearize_program`](crate::tracing_v2::linearize_program)`::<E, Self>` needs instead of the recursive
/// `Self: DifferentiableOperation<LinearizationContextOf<E, Self>>` bound).
impl<F, E> ProgramLinearizableOperation<E> for ScalarOperation<F>
where
    F: Value<DataType>,
    E: DifferentiationContext<Type = DataType, Constant = F>,
    E::LinearOperation<E::Tangent, F>:
        FactorParameterizedOperation<DataType, F, WithFactor<F> = E::LinearOperation<E::Tangent, F>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: SupportsLinearScalarOperation<DataType, ResidualFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>>
        + crate::tracing_v2::ResidualizedOperation<LinearizationContextOf<E, Self>>
        + SupportsZero<DataType>
        + SupportsCustomVjpCall<DataType, F, Self, ResidualFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: FactorParameterizedOperation<
            DataType,
            ResidualFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>,
            WithFactor<ResidualFactor<DataType, E::Value>> = LinearOperationOf<E>,
        >,
{
    fn linearize_program(
        differentiable: &E,
        program: &crate::programs::Program<DataType, F, Self, Vec<F>, Vec<F>>,
    ) -> Result<NestedLinearization<E, Self>, ProgramError> {
        crate::tracing_v2::differentiation::linearize_program(differentiable, program)
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
        LinearArrayOperation::LeftDot { factor, dimensions, output_sharding } => {
            LeftDotOperation::new(factor.clone(), dimensions.clone())
                .with_output_sharding(output_sharding.clone())
                .batch(&(), inputs)?
        }
        LinearArrayOperation::RightDot { factor, dimensions, output_sharding } => {
            RightDotOperation::new(factor.clone(), dimensions.clone())
                .with_output_sharding(output_sharding.clone())
                .batch(&(), inputs)?
        }
        LinearArrayOperation::Reshape { output_shape } => {
            ReshapeOperation::new(output_shape.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Reshard { sharding } => ReshardOperation::new(sharding.clone()).batch(&(), inputs)?,
        LinearArrayOperation::ShardingConstraint { sharding } => {
            ShardingConstraintOperation::new(sharding.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Broadcast { output_type, output_axes } => {
            BroadcastOperation::new(output_type.clone(), output_axes.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Reduce { axes, kind, output_sharding } => ReduceOperation::new(axes.clone(), *kind)
            .with_output_sharding(output_sharding.clone())
            .batch(&(), inputs)?,
        LinearArrayOperation::Slice { start_indices, limit_indices, strides } => {
            SliceOperation::new(start_indices.clone(), limit_indices.clone())
                .with_strides(strides.clone())?
                .batch(&(), inputs)?
        }
        LinearArrayOperation::UpdateSlice { start_indices } => {
            UpdateSliceOperation::new(start_indices.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Pad { edge_padding_low, edge_padding_high, interior_padding } => {
            PadOperation::new(edge_padding_low.clone(), edge_padding_high.clone(), interior_padding.clone())?
                .batch(&(), inputs)?
        }
        LinearArrayOperation::Concatenate { axis } => ConcatenateOperation::new(*axis).batch(&(), inputs)?,
        LinearArrayOperation::TransferToMemory { .. }
        | LinearArrayOperation::DynamicSlice { .. }
        | LinearArrayOperation::DynamicUpdateSlice { .. }
        | LinearArrayOperation::Select { .. }
        | LinearArrayOperation::Residual { .. }
        | LinearArrayOperation::Recompute(_)
        | LinearArrayOperation::Condition { .. }
        | LinearArrayOperation::OperandCondition { .. }
        | LinearArrayOperation::While(_)
        | LinearArrayOperation::Scan { .. }
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
        + Zero<ArrayType>
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    E: Clone + BatchableOperation<V>,
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
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice { start_indices, sizes } => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(start_indices.iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicSliceOperation::new(sizes.clone()).batch(&(), lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice { start_indices } => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(start_indices.iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicUpdateSliceOperation.batch(&(), lifted_inputs.as_slice())
            }
            // The captured factor is lane-uniform by construction: the same residual value applies to every lane.
            Self::Residual { factor } => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(factor.clone())])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => operation.batch(&(), inputs),
            // The captured predicate is lane-uniform: prepending it as an unbatched input lets the condition
            // batching helper read the branch choice from input 0, exactly like an ordinary runtime predicate.
            Self::Condition { predicate, true_branch, false_branch } => {
                let mut condition_inputs = Vec::with_capacity(inputs.len() + 1);
                condition_inputs.push(ArrayBatch::unbatched(predicate.clone()));
                condition_inputs.extend(inputs.iter().cloned());
                batch_condition_with_interpreter(
                    true_branch.as_ref(),
                    false_branch.as_ref(),
                    condition_inputs.as_slice(),
                    |program, program_inputs| {
                        program.interpret_with(
                            program_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
                        )
                    },
                )
            }
            // The operand-form condition already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate.
            Self::OperandCondition { true_branch, false_branch } => batch_condition_with_interpreter(
                true_branch.as_ref(),
                false_branch.as_ref(),
                inputs,
                |program, program_inputs| {
                    program.interpret_with(
                        program_inputs,
                        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                        |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
                    )
                },
            ),
            Self::While(operation) => operation.batch(context, inputs),
            // Each lane's body pushforward is bound against that lane's residual slices and batched through the
            // shared scan loop; the residual stacks are concrete values in the direct linear form.
            Self::Scan { body, residual_stacks, carry_count, length, reverse, .. } => {
                let y_slice_types = body.output_types().split_off(*carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    *carry_count,
                    *length,
                    *reverse,
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| V::zero(stacked_type),
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_factors_preserving_extensions(&mut |factor| {
                                factor.instantiate(lane_residuals.as_slice())
                            })
                        })?;
                        lane_body.interpret_with(
                            lane_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
                        )
                    },
                )
            }
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
    C::Constant: Value<ArrayType> + BooleanLike + Slice<Output = C::Constant> + Reshape<Output = C::Constant>,
    C::Operation: SupportsZero<ArrayType>,
    Tracer<C>: SupportsLinearArithmeticOperations<C::Constant>
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations<C::Constant>
        + SupportsManipulationOperations
        + BitAnd<Output = Tracer<C>>
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + TransferToMemory,
    E: Clone + BatchableOperation<Tracer<C>, BatchingContext<C>>,
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
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice { start_indices, sizes } => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    start_indices
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicSliceOperation::new(sizes.clone()).batch(&(), lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice { start_indices } => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    start_indices
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicUpdateSliceOperation.batch(&(), lifted_inputs.as_slice())
            }
            // The captured factor is a lane-uniform parent-context constant: lift it into the parent trace.
            Self::Residual { factor } => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(context.parent_context().constant(factor.clone()))])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => operation.batch(context, inputs),
            // The captured predicate is a lane-uniform parent-context constant, so the branch choice is concrete:
            // extract it from the factor and batch only the selected branch. Prepending a lifted predicate tracer
            // would defeat the lane-uniform extraction because tracers cannot be concretized.
            Self::Condition { predicate, true_branch, false_branch } => {
                let branch = if predicate.boolean()? { true_branch } else { false_branch };
                context.interpret_program(branch.as_ref(), inputs.to_vec())
            }
            // The operand-form condition already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate (lane-uniform predicates extract
            // concretely, lane-varying ones run both branches and select per lane).
            Self::OperandCondition { true_branch, false_branch } => {
                batch_condition_with_interpreter::<C::Constant, Tracer<C>, _, _>(
                    true_branch.as_ref(),
                    false_branch.as_ref(),
                    inputs,
                    |program, program_inputs| context.interpret_program(program, program_inputs),
                )
            }
            // The fused doubled-state linear while keeps the operational masked-unrolling rule even under tracing:
            // its condition recomputes the loop predicate from captured residual injections (parent-context
            // constants), so the per-iteration predicate extraction stays concrete and the loop unrolls through the
            // batched tracers. The staged batching rule on the primal `WhileOperation` does not apply here because
            // the loop's nested operation type is this linear enum, not the staged program's operation type.
            Self::While(operation) => {
                batch_while_with_interpreter(operation.as_ref(), inputs, |program, program_inputs| {
                    context.interpret_program(program, program_inputs)
                })
            }
            // Each lane's body pushforward is bound against that lane's residual slices at the constant level
            // (the stacks are lane-uniform parent-context constants) and batched over the traced lanes through
            // the shared scan loop; stacked output accumulators are staged as typed zeros in the parent trace.
            Self::Scan { body, residual_stacks, carry_count, length, reverse, .. } => {
                let y_slice_types = body.output_types().split_off(*carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    *carry_count,
                    *length,
                    *reverse,
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| {
                        let mut outputs = context.parent_context().stage_operation(
                            <C::Operation as SupportsZero<ArrayType>>::zero_operation(stacked_type.clone()),
                            &[] as &[Tracer<C>],
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    },
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_factors_preserving_extensions(&mut |factor| {
                                factor.instantiate(lane_residuals.as_slice())
                            })
                        })?;
                        context.interpret_program(&lane_body, lane_inputs)
                    },
                )
            }
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
    E: Clone + BatchableOperation<V> + BatchableOperation<Tangent<ArrayType, V>, ()>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        _context: &(),
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
        // First-order linear ops over tangent values: materialize `Tangent::Zero` to `V::zero`
        // once, dispatch to the V-level batching rule, and re-wrap as `Tangent::Value`. Symbolic
        // zero propagates through every Tangent V-trait impl (`Add`, `Sub`, `Neg`, `Scale`,
        // `LeftDot`, `RightDot`, `Reshape`, `Transpose`), so dispatching through `apply_with_axes`
        // on `lifted_op.interpret(tangent_values)` would also work — but the materialize-then-
        // dispatch path lets us reuse the V-level rule unchanged, which keeps the rule defined in
        // exactly one place.
        //
        // `Residual` is nullary, so the all-zero shortcut would fire vacuously and zero out the materialized
        // factor; `While` runs its loop during the V-level dispatch, so lifting output types from a zero-state run
        // would execute the loop at a primal point it was never staged for. Both always take the materialize path.
        let always_materialize = matches!(
            self,
            LinearArrayOperation::ZeroLike
                | LinearArrayOperation::OneLike
                | LinearArrayOperation::Residual { .. }
                | LinearArrayOperation::While(_),
        );
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
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::domains::AbstractDomain;
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
            output_sharding: None,
        };

        assert_eq!(
            right_dot.interpret(&[Tangent::zero(input_type.clone())]),
            Ok(vec![Tangent::zero(array_type(&[2, 4]))])
        );

        let left_factor_type = array_type(&[4, 2]);
        let left_dot = ZeroArrayOperation::LeftDot {
            factor: Tangent::zero(left_factor_type),
            dimensions: DotDimensionNumbers::matmul(),
            output_sharding: None,
        };

        assert_eq!(left_dot.interpret(&[Tangent::zero(input_type)]), Ok(vec![Tangent::zero(array_type(&[4, 3]))]));
    }

    #[test]
    fn test_linear_array_zero_only_tangent_control_flow_interprets_nested_programs() {
        let state_type = array_type(&[2, 3]);
        // The captured condition predicate factor lives in the zero-only tangent space, so it is always a symbolic
        // zero and can never select a branch.
        let condition = ZeroArrayOperation::Condition {
            predicate: Tangent::zero(ArrayType::scalar(DataType::Boolean)),
            true_branch: Box::new(identity_zero_array_program(state_type.clone())),
            false_branch: Box::new(one_zero_array_program(state_type.clone(), state_type.clone())),
        };
        assert_eq!(
            condition.interpret(&[Tangent::zero(state_type.clone())]).unwrap_err().to_string(),
            "symbolic-zero condition predicate interpretation is not supported"
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
    fn test_linear_condition_transpose_supports_runtime_predicates() {
        // Linear-condition transposition is total: the captured predicate factor is a residual of the primal
        // computation rather than a linear operand, so it is carried verbatim into one staged transposed condition
        // over the transposed branch programs. Runtime (factor) predicates used to be rejected with an
        // `UnsupportedOperation` error.
        type TestLinearOperation = LinearArrayOperation<TestArray, TestArray, ArrayType>;
        let scale_branch = |factor: f64| {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let output = builder
                .add_instruction(TestLinearOperation::Scale { factor: TestArray::scalar(factor) }, vec![input])
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = TestLinearOperation::Condition {
            predicate: TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
            true_branch: Box::new(scale_branch(2.0)),
            false_branch: Box::new(scale_branch(3.0)),
        };

        let domain = AbstractDomain::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new()));
        let cotangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = AbstractTracingContext::new(&domain, builder.clone());
        let cotangent = context.tracer(cotangent_input, None);
        let cotangents = operation
            .transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Staged(cotangent)])
            .unwrap();
        assert_eq!(cotangents.len(), 1);
        assert!(!cotangents[0].is_zero());
        let pullback_output = cotangents[0].as_staged().unwrap().atom_id().unwrap();
        assert!(matches!(builder.borrow().instructions()[0].operation(), TestLinearOperation::Condition { .. }));

        // Interpreting the pullback applies the transposed branch selected by the carried predicate (scale by 2).
        drop(cotangents);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let pullback = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![pullback_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outputs = pullback.interpret(vec![TestArray::scalar(5.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![10.0]);
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
        let input_zero = MixedArray::zero(input.r#type().into_owned());

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
                output_sharding: None,
            })
            .interpret(&[MixedArray::value(input)]),
            Ok(vec![MixedArray::zero(f64_array_type(&[4, 3]))])
        );

        let right_factor = TestArray::matrix(3, 4, vec![0.0; 12]);
        assert_eq!(
            (MixedArrayOperation::RightDot {
                factor: MixedArray::value(right_factor),
                dimensions: DotDimensionNumbers::matmul(),
                output_sharding: None,
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
