//! Closed operation families and array-owned operation implementations.
//!
//! [`ArrayOperation`], [`DimensionOperation`], and [`ArrayIrOperation`] are the staged operation families for the
//! array universe. Most child modules group private mixed array-IR machinery by the semantic families used under
//! [`crate::operations`].

// TODO(eaplatanios): Review this module.

use std::ops::{Add as StandardAdd, Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg, Sub as StandardSub};

use ryft_macros::Operation;

use crate::arrays::arrays::Array;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::reference_views::ArrayReferenceViewTransform;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, DimensionType};
use crate::arrays::types::ir::ArrayIrType;
use crate::axes::AxisIndexOperation;
use crate::contexts::{Context, ProjectedContext};
use crate::differentiation::{
    CustomJvpOperation, CustomVjpOperation, DifferentiableOperation, DifferentiableType, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, LinearCallOperation, MemberDifferentiableOperation,
    ResidualZeroProvider, StopGradient, StopGradientOperation, jvp_projected_operation,
};
use crate::operations::attention::{
    DotProductAttention, DotProductAttentionBackwardOperation, DotProductAttentionOperation,
};
use crate::operations::collectives::{AllGatherOperation, AllToAllOperation, PSumScatterOperation, PpermuteOperation};
use crate::operations::complex::{
    Complex, ComplexOperation, Conjugate, ConjugateOperation, Imaginary, ImaginaryOperation, Real, RealOperation,
};
use crate::operations::custom_call::CustomCallOperation;
use crate::operations::random::RngBitGeneratorOperation;
use crate::operations::sort::{Sort, SortOperation};
use crate::operations::{
    Abs, AbsOperation, Add, AddOperation, And, AndOperation, Atan2, Atan2Operation, Broadcast, BroadcastOperation,
    Ceil, CeilOperation, CollectiveOperation, Compare, CompareOperation, Concatenate, ConcatenateOperation,
    ConditionOperation, ConstantOperation, ConvertElementType, ConvertElementTypeOperation, Cos, CosOperation,
    DimensionAddOperation, DimensionArithmetic, DimensionDivFloorOperation, DimensionFromScalar,
    DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin, DimensionMinOperation,
    DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation, DimensionRequirement,
    DimensionRequirementOperation, DimensionSaturatingSub, DimensionSaturatingSubOperation, DimensionSize,
    DimensionSizeOperation, DimensionSubOperation, DimensionToScalar, DimensionToScalarOperation, Div, DivOperation,
    Dot, DotOperation, DynamicBroadcast, DynamicBroadcastOperation, DynamicReshape, DynamicReshapeOperation,
    DynamicShapeSliceOperation, DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation,
    Erf, ErfOperation, Exp, ExpOperation, Floor, FloorOperation, Gather, GatherOperation, IotaOperation, Log,
    LogOperation, Logistic, LogisticOperation, Max, MaxOperation, Min, MinOperation, Mul, MulOperation, Neg,
    NegOperation, Not, NotOperation, OneLike, OneLikeOperation, OneOperation, Or, OrOperation, Pad, PadOperation, Pow,
    PowOperation, PrintOperation, Reduce, ReduceOperation, Rem, RemOperation, Reshape, ReshapeOperation,
    ReshardOperation, Round, RoundOperation, Rsqrt, RsqrtOperation, ScaledDot, ScaledDotOperation, ScanOperation,
    Scatter, ScatterOperation, Select, SelectOperation, ShardingConstraintOperation, Sign, SignOperation, Sin,
    SinOperation, Slice, SliceOperation, Sqrt, SqrtOperation, Sub, SubOperation, TagOperation, Tanh, TanhOperation,
    TransferToMemoryOperation, Transpose, TransposeOperation, UpdateSlice, UpdateSliceOperation, WhileOperation, Xor,
    XorOperation, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::programs::{
    FreezeReference, FreezeReferenceOperation, MaybeZero, NewReference, NewReferenceOperation, Operation,
    OperationProjection, ProgramError, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceRead,
    ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation, Type, TypeError, TypeIdentityPosition, Typed, Value,
    ValueProjection,
};
use crate::tracing::TracingContext;
use crate::tracing_v2::RematerializeOperation;

mod attention;
mod collectives;
mod compare;
mod complex;
mod constants;
mod control_flow;
mod custom_call;
mod dimensions;
mod logical;
mod manipulation;
mod math;
mod memory;
mod quantization;
mod random;
mod references;
mod sharding;
mod sort;
mod tag;

// TODO(eaplatanios): This seems a bit weirdly placed.
pub use references::{
    REFERENCE_INDEX_OPERATION_NAME, REFERENCE_SLICE_OPERATION_NAME, ReferenceIndex, ReferenceIndexOperation,
    ReferenceSlice, ReferenceSliceOperation,
};

/// Reusable [`Operation`] enum for ordinary staged programs over arrays.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates, pairing with [`Array`]. Most
/// variants are thin tags around one semantic primitive defined in [`crate::operations`] or
/// [`crate::differentiation::operations`].
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`] and
/// [`Dot`](Self::Dot) a [`DotOperation`].
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(batching, differentiation, transposition))]
pub enum ArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation<ArrayType>),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation<ArrayType>),
    Constant(ConstantOperation<Array>),
    Iota(IotaOperation<ArrayType>),
    Abs(AbsOperation<ArrayType>),
    Neg(NegOperation<ArrayType>),
    Add(AddOperation<ArrayType>),
    Sub(SubOperation<ArrayType>),
    Mul(MulOperation<ArrayType>),
    Div(DivOperation<ArrayType>),
    Sin(SinOperation<ArrayType>),
    Cos(CosOperation<ArrayType>),
    Atan2(Atan2Operation<ArrayType>),
    Exp(ExpOperation<ArrayType>),
    Log(LogOperation<ArrayType>),
    Sqrt(SqrtOperation<ArrayType>),
    Rsqrt(RsqrtOperation<ArrayType>),
    Tanh(TanhOperation<ArrayType>),
    Logistic(LogisticOperation<ArrayType>),
    Erf(ErfOperation<ArrayType>),
    Pow(PowOperation<ArrayType>),
    Sign(SignOperation<ArrayType>),
    Floor(FloorOperation<ArrayType>),
    Ceil(CeilOperation<ArrayType>),
    Round(RoundOperation<ArrayType>),
    Max(MaxOperation<ArrayType>),
    Min(MinOperation<ArrayType>),
    Rem(RemOperation<ArrayType>),
    Not(NotOperation<ArrayType>),
    And(AndOperation<ArrayType>),
    Or(OrOperation<ArrayType>),
    Xor(XorOperation<ArrayType>),
    Complex(ComplexOperation<ArrayType>),
    Conjugate(ConjugateOperation<ArrayType>),
    Real(RealOperation<ArrayType>),
    Imaginary(ImaginaryOperation<ArrayType>),
    Dot(DotOperation),
    ScaledDot(ScaledDotOperation),
    DotProductAttention(DotProductAttentionOperation),
    DotProductAttentionBackward(DotProductAttentionBackwardOperation),
    Reduce(ReduceOperation),
    Sort(SortOperation),
    RngBitGenerator(RngBitGeneratorOperation<ArrayType>),
    Collective(CollectiveOperation),
    AllGather(AllGatherOperation),
    PSumScatter(PSumScatterOperation),
    Ppermute(PpermuteOperation),
    AllToAll(AllToAllOperation),
    AxisIndex(AxisIndexOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    Broadcast(BroadcastOperation),
    Pad(PadOperation<ArrayType>),
    Concatenate(ConcatenateOperation<ArrayType>),
    Gather(GatherOperation),
    Scatter(ScatterOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(DynamicSliceOperation),
    DynamicUpdateSlice(DynamicUpdateSliceOperation),
    Compare(CompareOperation<ArrayType>),
    Select(SelectOperation<ArrayType>),
    Condition(ConditionOperation<V>),
    While(WhileOperation<ArrayType>),
    Scan(ScanOperation<V>),
    ConvertElementType(ConvertElementTypeOperation<ArrayType>),
    TransferToMemory(TransferToMemoryOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    StopGradient(StopGradientOperation<ArrayType>),
    Tag(TagOperation<ArrayType>),
    Rematerialize(RematerializeOperation<ArrayType>),
    Print(PrintOperation<ArrayType>),
    CustomCall(CustomCallOperation<ArrayType>),
    CustomJvp(CustomJvpOperation<ArrayType>),
    CustomVjp(CustomVjpOperation<ArrayType>),
    LinearCall(LinearCallOperation<ArrayType>),
}

/// Value-level capability bundle paired with the [`ArrayOperation`] family.
///
/// [`ArrayOperations`] collects, as supertraits, the value-level capabilities through which a value materializes the
/// [`ArrayOperation`] variants, so that generic array code states one bound instead of re-listing every capability it
/// happens to use. It is a pure bundle: the blanket implementation below covers every value that satisfies the same
/// supertrait list, so this trait must never be implemented manually.
///
/// # Membership
///
/// Membership is limited to *value-level capabilities of the family*: traits whose methods take and return values of
/// the implementing type and stage or execute one [`ArrayOperation`] variant. Everything that a variant needs in
/// order to exist, but that a value does not itself perform, stays out:
///
///   - type-semantics plumbing such as [`WhileTypeSemantics`](crate::operations::control_flow::WhileTypeSemantics),
///     [`ScanTypeSemantics`](crate::operations::control_flow::scan::ScanTypeSemantics),
///     [`ConditionTypeSemantics`](crate::operations::control_flow::condition::ConditionTypeSemantics), and
///     [`WhilePredicate`](crate::operations::control_flow::WhilePredicate);
///   - staging machinery such as [`Constant`](crate::operations::constants::Constant) and
///     [`Tag`](crate::operations::tag::Tag), and the context-side constructors [`Zero`],
///     [`One`](crate::operations::constants::One), [`Fill`](crate::operations::constants::Fill), and
///     [`Iota`](crate::operations::constants::Iota), whose value-driven counterparts [`ZeroLike`] and [`OneLike`] are
///     members instead;
///   - effects and debugging such as [`Print`](crate::operations::debugging::Print), and the foreign-kernel escape
///     hatch [`CustomCall`](crate::operations::custom_call::CustomCall);
///   - first-class dimension plumbing, which belongs to [`ArrayIrOperations`] rather than to the homogeneous array
///     family;
///   - random bit generation, whose [`RngBitGenerator`](crate::operations::random::RngBitGenerator) contract threads
///     explicit algorithm state rather than shaping a value-to-value capability;
///   - the collectives ([`AllGather`](crate::operations::collectives::AllGather),
///     [`AllToAll`](crate::operations::collectives::AllToAll),
///     [`PSumScatter`](crate::operations::collectives::PSumScatter),
///     [`PSwapAxes`](crate::operations::collectives::PSwapAxes),
///     [`Pshuffle`](crate::operations::collectives::Pshuffle), [`Reshard`](crate::operations::sharding::Reshard),
///     [`ConstrainSharding`](crate::operations::sharding::ConstrainSharding),
///     [`Collective`](crate::operations::collectives::Collective), and
///     [`TransferToMemory`](crate::operations::memory::TransferToMemory)), so that single-device generic code never
///     carries sharded-programming obligations; and
///   - differentiation plumbing such as [`ReverseModeDifferentiate`](crate::differentiation::ReverseModeDifferentiate)
///     and the operation-family `From` bounds that transforms require of a domain.
///
/// Derived conveniences that a member already implies are also left out, because bounding them would only duplicate
/// solver work: [`TopK`](crate::operations::sort::TopK), [`ArgMax`](crate::operations::sort::ArgMax), and
/// [`ArgMin`](crate::operations::sort::ArgMin) all follow from [`Sort`], [`Slice`], and [`Reshape`], and
/// [`DotOps`](crate::operations::dot::DotOps) follows from [`Dot`] and [`Transpose`].
///
/// # Tracers
///
/// This bundle deliberately does *not* imply anything about the tracers derived from an implementing value. Generic
/// code that traces (for example, code that differentiates) states the tracer requirement as its own separate bound,
/// such as `LinearizationTracer<A::ExecutionDomain>: ArrayOperations`. Making the bundle recursively imply its own
/// tracer bounds would make the trait solver chase an unbounded tower of nested tracer types.
pub trait ArrayOperations:
    Value<Type = ArrayType>
    // Arithmetic, in both the panicking operator sugar and the fallible capability forms.
    + StandardNeg<Output = Self> + StandardAdd<Output = Self> + StandardSub<Output = Self>
    + StandardMul<Output = Self> + StandardDiv<Output = Self>
    + Neg + Add + Sub + Mul + Div + Rem + Pow + Max + Min + Abs + Sign
    // Elementwise math and logic.
    + Sin + Cos + Atan2 + Exp + Log + Sqrt + Rsqrt + Tanh + Logistic + Erf + Floor + Ceil + Round
    + Not + And + Or + Xor
    // Complex numbers.
    + Complex + Conjugate + Real + Imaginary
    // Comparison and selection.
    + Compare + Select
    // Shape and layout manipulation.
    + Transpose + Reshape + Broadcast + Pad + Concatenate + Gather + Scatter + Slice + UpdateSlice
    + DynamicSlice + DynamicUpdateSlice + ConvertElementType + Sort
    // Linear algebra and reduction.
    + Dot + ScaledDot + DotProductAttention + Reduce
    // Constants and differentiation barriers.
    + ZeroLike + OneLike + StopGradient
{
}

// The predicates below restate the supertrait list of `ArrayOperations`, one predicate per category, so that the
// bundle is satisfied exactly when every one of its member capabilities is.
impl<V> ArrayOperations for V
where
    V: Value<Type = ArrayType>,
    V: StandardNeg<Output = V> + StandardAdd<Output = V> + StandardSub<Output = V> + StandardMul<Output = V>,
    V: StandardDiv<Output = V> + Neg + Add + Sub + Mul + Div + Rem + Pow + Max + Min + Abs + Sign,
    V: Sin + Cos + Atan2 + Exp + Log + Sqrt + Rsqrt + Tanh + Logistic + Erf + Floor + Ceil + Round,
    V: Not + And + Or + Xor + Complex + Conjugate + Real + Imaginary + Compare + Select,
    V: Transpose + Reshape + Broadcast + Pad + Concatenate + Gather + Scatter + Slice + UpdateSlice,
    V: DynamicSlice + DynamicUpdateSlice + ConvertElementType + Sort,
    V: Dot + ScaledDot + DotProductAttention + Reduce + ZeroLike + OneLike + StopGradient,
{
}

/// [`Operation`] family used for staged [`DimensionValue`] [`Program`](crate::Program)s.
#[derive(Clone, Debug, Operation)]
pub enum DimensionOperation<V: Value<Type = DimensionType>> {
    Constant(ConstantOperation<V>),
    Add(DimensionAddOperation),
    Sub(DimensionSubOperation),
    SaturatingSub(DimensionSaturatingSubOperation),
    Mul(DimensionMulOperation),
    Pow(DimensionPowOperation),
    DivFloor(DimensionDivFloorOperation),
    Rem(DimensionRemOperation),
    Min(DimensionMinOperation),
    Max(DimensionMaxOperation),
    Requirement(DimensionRequirementOperation),
}

/// Value-level capability bundle paired with the [`DimensionOperation`] family.
///
/// [`DimensionOperations`] is to [`DimensionOperation`] what [`ArrayOperations`] is to [`ArrayOperation`]: a pure
/// bundle of the value-level capabilities through which a [`DimensionType`]-typed value materializes the family's
/// variants, blanket-implemented for every value that satisfies the same supertrait list and therefore never
/// implemented manually. Generic first-class-dimension code states this one bound instead of re-listing checked
/// dimension arithmetic capability by capability, and it is also the dimension member profile that
/// [`ArrayIrOperations`] pins on its [`ValueProjection<DimensionType>`](ValueProjection) supertrait.
///
/// # Membership
///
/// Membership follows the rule documented on [`ArrayOperations`]: a member is a value-level capability whose methods
/// take [`DimensionType`]-typed values and stage or execute one [`DimensionOperation`] variant. Checked arithmetic
/// reaches the shared capabilities [`Add`], [`Sub`], [`Mul`], [`Div`] (flooring division), and [`Rem`] pinned to
/// dimension-typed values, alongside the dedicated [`DimensionSaturatingSub`], [`DimensionPow`], [`DimensionMin`],
/// and [`DimensionMax`]. [`DimensionRequirement`] is the family's assertion surface and is a member even though its
/// methods return no value, because it is still performed by a dimension-typed value.
///
/// What a variant needs in order to exist, but that a dimension value does not itself perform, stays out:
///
///   - staging machinery, namely [`DimensionOperation::Constant`], exactly as
///     [`Constant`](crate::operations::constants::Constant) stays out of [`ArrayOperations`]; and
///   - the mixed conversions [`DimensionSize`], [`DimensionFromScalar`], and [`DimensionToScalar`], whose signatures
///     cross the array and first-class-dimension member kinds. They live in the composite family and belong to
///     [`ArrayIrOperations`]. [`DimensionToScalar`] does have a dimension-typed receiver, but its output is an array
///     representation the member universe cannot name, so it is reached through the composite value instead.
///
/// The composite counterpart of this bundle's arithmetic is [`DimensionArithmetic`], which spells the same operations
/// directly on [`ArrayIrType`]-typed values so that composite shape arithmetic needs no projection vocabulary.
///
/// # Tracers
///
/// As with [`ArrayOperations`], this bundle deliberately implies nothing about the tracers derived from an
/// implementing value; a tracer requirement stays a separate explicit bound.
pub trait DimensionOperations:
    Value<Type = DimensionType>
    // Checked first-class-dimension arithmetic.
    + Add + Sub + Mul + Div + Rem + DimensionSaturatingSub + DimensionPow + DimensionMin + DimensionMax
    // Runtime assertions over first-class dimensions.
    + DimensionRequirement
{
}

// The predicates below restate the supertrait list of `DimensionOperations`, so that the bundle is satisfied exactly
// when every one of its member capabilities is.
impl<V> DimensionOperations for V
where
    V: Value<Type = DimensionType> + Add + Sub + Mul + Div + Rem + DimensionSaturatingSub,
    V: DimensionPow + DimensionMin + DimensionMax + DimensionRequirement,
{
}

/// Closed [`Operation`] family for Ryft's array IR, whose values include ordinary arrays,
/// first-class runtime dimensions, and references to arrays. This dispatcher preserves the homogeneous contracts of
/// [`ArrayOperation`] and [`DimensionOperation`]: it selects the member family, projects the composite type boundary
/// once, delegates to that family, and lifts the inferred result types back into [`ArrayIrType`]. Reference operations
/// remain composite-native because their signatures cross the array/reference boundary and their ordered state
/// semantics must remain visible to generic program passes.
///
/// Operations whose signatures mix arrays and dimensions are represented as explicit variants because no homogeneous
/// member family can express such a signature. For example, [`DimensionSizeOperation`] consumes an array and produces
/// a first-class dimension without changing either homogeneous family.
#[derive(Clone, Debug, Operation)]
#[ryft(
    crate = "crate",
    type = ArrayIrType,
    constant = ArrayIrValue<A>,
    members(ArrayType, structural(DimensionType)),
    dispatch(discharge, batching, differentiation, transposition),
)]
pub enum ArrayIrOperation<A: Value<Type = ArrayType>> {
    /// Mixed zero constructor whose stored [`ArrayType`] defines the array result and whose dynamic dimensions are
    /// consumed as explicit first-class dimension operands, one per dynamic axis in axis order. A static stored type
    /// therefore consumes no operands and remains valid, although canonical lifts prefer the homogeneous
    /// [`ArrayOperation`] encoding. This constructor lives at the composite-family level because its dynamic signature
    /// crosses member kinds: a homogeneous [`ArrayOperation`] cannot consume dimension operands, while the stored
    /// structural type carries identities and bounds but not the concrete runtime extents required to materialize the
    /// result.
    #[ryft(mixed(structural), skip_from)]
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor whose stored [`ArrayType`] fully defines the output type and whose dynamic dimensions
    /// are consumed as explicit first-class dimension operands, one per dynamic axis in axis order. A static stored
    /// type consumes no operands and remains valid, although canonical lifts prefer the homogeneous [`ArrayOperation`]
    /// encoding.
    #[ryft(mixed(structural), skip_from)]
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor whose stored [`ArrayType`] and iota axis define the complete output, and whose dynamic
    /// dimensions are consumed as explicit first-class dimension operands in axis order. A static stored type consumes
    /// no operands and remains valid, although canonical lifts prefer the homogeneous [`ArrayOperation`] encoding.
    #[ryft(mixed(structural), skip_from)]
    DynamicIota(IotaOperation<ArrayType>),

    /// Homogeneous array operation whose complete boundary is projected into the array member family. Every transform
    /// reaches the member rule through that projection, which carries no region access: an operation's attached
    /// regions are programs in the *composite* universe, and no projected view can present them in the member
    /// universe. This variant therefore holds only region-free array operations: the array-operation lift promotes
    /// every region-carrying member payload to its composite carrier — [`Condition`](Self::Condition),
    /// [`While`](Self::While), [`Scan`](Self::Scan), [`CustomJvp`](Self::CustomJvp), [`CustomVjp`](Self::CustomVjp),
    /// [`LinearCall`](Self::LinearCall) (both interface forms), and [`Rematerialize`](Self::Rematerialize).
    ///
    /// No region-carrying array payload therefore reaches this variant. Should one ever be constructed directly, it is
    /// not silently mis-transformed: each transform rejects a region-carrying projected payload with an exact
    /// diagnostic naming the operation.
    #[ryft(projected(ArrayType), skip_from)]
    Array(ArrayOperation<A>),

    /// Homogeneous first-class-dimension operation.
    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two first-class dimensions that produces ordinary rank-zero Boolean array data.
    ///
    /// This variant has the precise composite member signature
    /// `(Dimension, Dimension) -> Array(Boolean scalar)`. It lives directly in [`ArrayIrOperation`] because
    /// [`DimensionOperation`] is intentionally homogeneous: its inputs and outputs are all first-class dimensions.
    /// Storing comparison there would break that invariant because a predicate is ordinary data rather than a
    /// first-class dimension.
    ///
    /// Homogeneous array comparison remains [`ArrayIrOperation::Array`] wrapping [`ArrayOperation::Compare`]. This
    /// variant does not permit array-dimension or dimension-array comparisons; it reuses [`CompareOperation`] for the
    /// dimension-dimension signature whose result crosses from the dimension member kind to the array member kind.
    Compare(CompareOperation<ArrayIrType>),

    /// Mixed operation that reads an array axis as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Creates a new whole-array reference root.
    NewReference(NewReferenceOperation<ArrayType, ArrayIrType>),

    /// Reads the array value selected by a root reference or derived view.
    ReferenceRead(ReferenceReadOperation<ArrayType, ArrayIrType>),

    /// Derives an axis-removing indexed view of a reference.
    ReferenceIndex(ReferenceIndexOperation),

    /// Derives a rank-preserving static slice view of a reference.
    ReferenceSlice(ReferenceSliceOperation),

    /// Replaces the array value selected by a root reference or derived view and returns its previous value.
    ReferenceSwap(ReferenceSwapOperation<ArrayType, ArrayIrType>),

    /// Adds an array update into the value selected by a root reference or derived view in program order.
    ReferenceAddUpdate(ReferenceAddUpdateOperation<ArrayType, ArrayIrType>),

    /// Consumes a whole-array reference and returns its final value.
    FreezeReference(FreezeReferenceOperation<ArrayType, ArrayIrType>),

    /// Mixed operation that converts ordinary scalar-array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Mixed operation that converts a first-class dimension into ordinary scalar-array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Mixed operation that reshapes an array using one first-class dimension operand per output axis.
    Reshape(DynamicReshapeOperation),

    /// Mixed operation that broadcasts an array using one first-class dimension operand per output axis.
    Broadcast(DynamicBroadcastOperation),

    /// Mixed operation that concatenates array operands using one trailing result-extent operand.
    Concatenate(ConcatenateOperation<ArrayIrType>),

    /// Mixed foreign-kernel call whose trailing dimension operands define its dynamic output axes.
    CustomCall(CustomCallOperation<ArrayIrType>),

    /// Mixed padding operation with one explicit result-extent operand per output axis.
    Pad(PadOperation<ArrayIrType>),

    /// Mixed slice whose starts and output sizes are first-class dimension operands.
    DynamicShapeSlice(DynamicShapeSliceOperation),

    /// Mixed bit generator whose trailing dimension operands define its dynamic bits-output axes.
    RngBitGenerator(RngBitGeneratorOperation<ArrayIrType>),

    /// Mixed all-gather whose trailing dimension operands define every result axis in axis order.
    #[ryft(mixed)]
    AllGather(AllGatherOperation),

    /// Mixed sum-scatter whose trailing dimension operands define every result axis in axis order.
    #[ryft(mixed)]
    PSumScatter(PSumScatterOperation),

    /// Mixed all-to-all whose trailing dimension operands define every result axis in axis order.
    #[ryft(mixed)]
    AllToAll(AllToAllOperation),

    /// Composite condition whose attached branches use the complete array IR storage universe. Validated local,
    /// nonescaping reference state can execute eagerly; reference-valued boundaries and generic transforms/backends
    /// remain unsupported until discharge.
    Condition(ConditionOperation<ArrayIrValue<A>>),

    /// Composite while loop whose condition and body use the complete array IR storage universe. Validated local,
    /// nonescaping reference state can execute eagerly; reference-valued carries/results and generic
    /// transforms/backends remain unsupported until discharge.
    While(WhileOperation<ArrayIrType>),

    /// Composite scan whose body uses the complete array IR storage universe. Validated local, nonescaping reference
    /// state can execute eagerly; reference-valued sequences/carries/results and generic transforms/backends remain
    /// unsupported until discharge.
    Scan(ScanOperation<ArrayIrValue<A>>),

    /// Composite custom-JVP call whose primal and JVP regions use the complete array IR storage universe. Generic
    /// differentiation rejects reference members because they have no tangent representation.
    CustomJvp(CustomJvpOperation<ArrayIrType>),

    /// Composite custom-VJP call whose primal, forward, and backward regions use the complete array IR storage
    /// universe. Generic differentiation rejects reference members because they have no cotangent representation.
    CustomVjp(CustomVjpOperation<ArrayIrType>),

    /// Differentiation-owned linear call with ordinary trailing residual operands, in either its executable
    /// forward-and-transpose form or its reverse-only transpose-only form.
    LinearCall(LinearCallOperation<ArrayIrType>),

    /// Composite rematerialized call whose primal, forward, backward, and tangent regions use the complete array IR
    /// storage universe. Unresolved state remains a non-recomputable boundary and is rejected before ordinary replay.
    Rematerialize(RematerializeOperation<ArrayIrType>),
}

/// Operation-family contract for array-reference analysis.
///
/// Generic [`Operation::reference_semantics`] identifies roots and aliases. This array-owned extension supplies the
/// exact coordinate mapping for aliases classified as reference views, keeping array indexing metadata out of the
/// generic program layer while giving every public analysis artifact a fully validated view mapping.
pub trait ArrayReferenceOperation: Operation<Type = ArrayIrType> {
    /// Returns the coordinate transform carried by a reference-view operation.
    #[inline]
    fn reference_view_transform(&self) -> Option<ArrayReferenceViewTransform> {
        None
    }
}

/// Operation-family constructors for the canonical array operations that one array-reference view traversal stages.
///
/// Mapping between a reference root and one derived handle's coordinates is a sequence of slices, reshapes, and
/// update-slices, and both consumers of that mapping — the eager handles and the
/// [`ArrayReferenceDischarge`](crate::ArrayReferenceDischarge) policy — walk the same
/// [`ArrayReferenceView`](crate::ArrayReferenceView). This contract is what lets the staging consumer put those
/// operations into a closed operation family it does not otherwise know the shape of, so core array IR and
/// backend-owned supersets share one traversal without matching operation names.
///
/// It is orthogonal to [`ArrayReferenceOperation`], which reports the view transform a member derives rather than
/// constructing one: a consumer that only stages view accesses states this contract alone.
pub trait ArrayReferenceViewOperation: Operation<Type = ArrayIrType> + Sized {
    /// Wraps a canonical homogeneous array reshape for reference-view staging.
    fn from_reference_reshape(operation: ReshapeOperation) -> Self;

    /// Wraps a canonical homogeneous array slice for reference-view staging.
    fn from_reference_slice(operation: SliceOperation) -> Self;

    /// Wraps a canonical homogeneous array update-slice for reference-view staging.
    fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self;
}

impl<A: Value<Type = ArrayType>> ArrayReferenceOperation for ArrayIrOperation<A> {
    fn reference_view_transform(&self) -> Option<ArrayReferenceViewTransform> {
        match self {
            Self::ReferenceIndex(operation) => Some(operation.transform()),
            Self::ReferenceSlice(operation) => Some(operation.transform()),
            _ => None,
        }
    }
}

impl<A: Value<Type = ArrayType>> ArrayReferenceViewOperation for ArrayIrOperation<A> {
    fn from_reference_reshape(operation: ReshapeOperation) -> Self {
        Self::Array(ArrayOperation::Reshape(operation))
    }

    fn from_reference_slice(operation: SliceOperation) -> Self {
        Self::Array(ArrayOperation::Slice(operation))
    }

    fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
        Self::Array(ArrayOperation::UpdateSlice(operation))
    }
}

/// Value-level capability bundle paired with the [`ArrayIrOperation`] family.
///
/// [`ArrayIrOperations`] is to [`ArrayIrOperation`] what [`ArrayOperations`] is to [`ArrayOperation`]: a pure bundle
/// of the value-level capabilities that the family's variants expose, blanket-implemented for every value that
/// satisfies the same supertrait list and therefore never implemented manually. It obeys the same membership rule,
/// documented on [`ArrayOperations`], but its inventory is deliberately *not* the array inventory pinned to
/// [`ArrayIrType`], because the composite universe reaches the two surfaces differently:
///
///   - Mixed capabilities, whose signatures cross the array and first-class-dimension member kinds, exist only at
///     the composite level and are therefore the bundle's members: [`Compare`] of two first-class dimensions,
///     [`DimensionSize`], [`DimensionFromScalar`], [`DimensionToScalar`], [`DynamicBroadcast`], and
///     [`DynamicReshape`], the whole-value reference capabilities [`NewReference`], [`ReferenceRead`],
///     [`ReferenceSwap`], [`ReferenceAddUpdate`], and [`FreezeReference`], and the reference view derivations
///     [`ReferenceIndex`] and [`ReferenceSlice`].
///   - Homogeneous array capabilities such as [`Add`], [`Dot`], and [`Reshape`] are *not* members. The composite
///     family carries the array member payloads through [`ArrayIrOperation::Array`], so a composite value performs
///     them through its [`ValueProjection`] view onto [`ArrayType`]. Bounding them here would demand
///     `From<AddOperation<ArrayIrType>>`-style conversions of every array operation, which the composite family
///     intentionally does not provide.
///
/// # Member Profiles
///
/// The two computational member projections are themselves supertraits, each pinned to the sibling bundle of the
/// member family it projects onto, so one bound carries the complete surface: the array member satisfies
/// [`ArrayOperations`] and the first-class-dimension member satisfies [`DimensionOperations`]. References deliberately
/// have no homogeneous operation family or projection bundle; their cross-kind capabilities are direct supertraits
/// because the corresponding operations remain composite-native.
/// Generic composite code therefore states `V: ArrayIrOperations` alone instead of restating
/// `ValueProjection<ArrayType, Projected: ArrayOperations>`-style bounds at every call site.
///
/// Checked arithmetic over two first-class dimensions is a bundle member in its own right, through
/// [`DimensionArithmetic`], so composite shape arithmetic needs no projection vocabulary at all:
///
/// ```rust
/// use ryft_core::arrays::DimensionValue;
/// use ryft_core::{Array, ArrayIrValue, DimensionArithmetic, ProgramError};
///
/// # fn main() -> Result<(), ProgramError> {
/// let rows: ArrayIrValue<Array> = ArrayIrValue::Dimension(DimensionValue::constant(2)?);
/// let columns: ArrayIrValue<Array> = ArrayIrValue::Dimension(DimensionValue::constant(3)?);
/// let ArrayIrValue::Dimension(elements) = rows.dimension_mul(&columns)? else {
///     unreachable!("dimension arithmetic returns a first-class dimension");
/// };
/// assert_eq!(elements.extent(), 6);
/// # Ok(())
/// # }
/// ```
///
/// Every other member surface is reached by the general mechanism instead: project the composite value, use the
/// member capability, and inject the member result back. For the array member that reads
/// `value.into_projected()?.mul(&other.into_projected()?)`, followed by `from_projected`:
///
/// ```rust
/// use ryft_core::arrays::ArrayType;
/// use ryft_core::{Array, ArrayIrValue, Mul, ProgramError, ValueProjection};
///
/// # fn main() -> Result<(), ProgramError> {
/// let left: ArrayIrValue<Array> = ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0]));
/// let right: ArrayIrValue<Array> = ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0]));
/// let product = ValueProjection::<ArrayType>::into_projected(left)?
///     .mul(&ValueProjection::<ArrayType>::into_projected(right)?)?;
/// let product = <ArrayIrValue<Array> as ValueProjection<ArrayType>>::from_projected(product);
/// assert_eq!(product, ArrayIrValue::Array(Array::vector(vec![8.0_f64, 15.0])));
/// # Ok(())
/// # }
/// ```
///
/// The [`ParameterProjection`](crate::ParameterProjection) extension trait applies the same projection to a whole
/// [`Parameterized`](crate::parameters::Parameterized) tree at a boundary (i.e.,
/// `model.project_parameters::<ArrayType>()?` and `projected.lift_parameters::<ArrayIrValue<A>>()?`), which is how
/// hand-written composite code computes with ordinary array capabilities without projecting leaf by leaf. Dense
/// derivative terminals ([`Jacobian`](crate::Jacobian) and [`Hessian`](crate::Hessian)) use that same route, because
/// their coordinate machinery is defined on [`ArrayType`].
///
/// As with [`ArrayOperations`], this bundle never implies anything about the tracers derived from an implementing
/// value; a tracer requirement stays a separate explicit bound.
pub trait ArrayIrOperations:
    Value<Type = ArrayIrType>
    // Array member profile.
    + ValueProjection<ArrayType, Projected: ArrayOperations>
    // First-class-dimension member profile.
    + ValueProjection<DimensionType, Projected: DimensionOperations>
    // Comparison of first-class dimensions, producing ordinary Boolean array data.
    + Compare
    // First-class dimensions.
    + DimensionArithmetic + DimensionSize + DimensionFromScalar + DimensionToScalar
    + DynamicBroadcast + DynamicReshape
    // Whole-value references.
    + NewReference + ReferenceIndex + ReferenceSlice + ReferenceRead + ReferenceSwap + ReferenceAddUpdate
    + FreezeReference
{
}

// The predicates below restate the supertrait list of `ArrayIrOperations`, so that the bundle is satisfied exactly
// when every one of its member capabilities is.
impl<V> ArrayIrOperations for V
where
    V: Value<Type = ArrayIrType> + Compare,
    V: DimensionArithmetic + DimensionSize + DimensionFromScalar + DimensionToScalar,
    V: DynamicBroadcast + DynamicReshape,
    V: NewReference + ReferenceIndex + ReferenceSlice + ReferenceRead + ReferenceSwap + ReferenceAddUpdate,
    V: FreezeReference,
    V: ValueProjection<ArrayType, Projected: ArrayOperations>,
    V: ValueProjection<DimensionType, Projected: DimensionOperations>,
{
}

/// [`TracingContext`] over the array universe, pairing [`ArrayType`] types and [`Array`] staged constants with the
/// [`ArrayOperation`] family.
pub type ArrayTracingContext = TracingContext<Array, ArrayOperation<Array>>;

/// [`TracingContext`] over [`DimensionValue`]s and [`DimensionOperation`]s.
pub type DimensionTracingContext = TracingContext<DimensionValue, DimensionOperation<DimensionValue>>;

impl<A: Value<Type = ArrayType>> From<ArrayOperation<A>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ArrayOperation<A>) -> Self {
        match operation {
            ArrayOperation::Zero(operation) => Self::from(operation),
            ArrayOperation::Condition(_) => Self::Condition(ConditionOperation::new()),
            ArrayOperation::While(operation) => {
                Self::While(WhileOperation::new().with_iteration_bound(operation.iteration_bound()).unwrap())
            }
            ArrayOperation::Scan(operation) => {
                let captures = operation.captures().iter().cloned().map(ArrayIrValue::Array).collect();
                Self::Scan(operation.with_captures(captures))
            }
            ArrayOperation::CustomJvp(operation) => Self::CustomJvp(
                CustomJvpOperation::new().with_non_differentiated_count(operation.non_differentiated_count()),
            ),
            ArrayOperation::CustomVjp(operation) => Self::CustomVjp(
                CustomVjpOperation::new().with_non_differentiated_count(operation.non_differentiated_count()),
            ),
            // The executable form stores no types. The transpose-only form maps its unavailable forward interface into
            // the composite universe, so both reach the carrier that owns the extent-threaded region rule.
            ArrayOperation::LinearCall(operation) => Self::LinearCall(operation.map_types(ArrayIrType::Array)),
            ArrayOperation::Rematerialize(operation) => Self::Rematerialize(
                RematerializeOperation::new()
                    .with_prevent_cse(operation.prevent_cse())
                    .with_non_differentiated_count(operation.non_differentiated_count()),
            ),
            operation => Self::Array(operation),
        }
    }
}

impl<A: Value<Type = ArrayType>> From<ConcatenateOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ConcatenateOperation<ArrayType>) -> Self {
        Self::Concatenate(operation.into())
    }
}

impl<A: Value<Type = ArrayType>> From<CustomCallOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: CustomCallOperation<ArrayType>) -> Self {
        Self::CustomCall(operation.into())
    }
}

impl<A: Value<Type = ArrayType>> From<PadOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: PadOperation<ArrayType>) -> Self {
        Self::Pad(operation.into())
    }
}

// Cotangent accumulation adds two composite cotangents by binding an `AddOperation<ArrayIrType>` (refer to
// `Linearization::pullback` and the reverse-mode `From<AddOperation<C::Type>>` bounds), so the composite family lifts
// the type-generic add into the homogeneous array member that owns elementwise addition. The source payload is
// stateless, so no operand type survives the conversion, and member type inference rejects a dimension operand.
impl<A: Value<Type = ArrayType>> From<AddOperation<ArrayIrType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(_operation: AddOperation<ArrayIrType>) -> Self {
        Self::Array(ArrayOperation::Add(AddOperation::new()))
    }
}

// Dimension constants additionally lift directly, so that generic staging code (e.g., `ExactShape::dimensions`)
// can bound only `From<ConstantOperation<DimensionValue>>` without naming this family's dimension member.
impl<A: Value<Type = ArrayType>> From<ConstantOperation<DimensionValue>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ConstantOperation<DimensionValue>) -> Self {
        Self::Dimension(DimensionOperation::Constant(operation))
    }
}

/// Replicates the operands of an implicitly broadcasting elementwise [`ArrayOperation`] into its result geometry, when
/// that geometry carries a runtime extent, so that the projected member rule differentiates operands that already have
/// the result shape.
///
/// The member-family alignment ([`ElementwiseDerivativeAlignment`](crate::ElementwiseDerivativeAlignment)) cannot
/// replicate into a runtime extent itself, because its [`BroadcastOperation`] carries the complete output geometry as
/// payload metadata and program replay never refines a payload type. Such an extent is therefore only reachable
/// through an operand edge, which this composite arm supplies: it reads each runtime extent off the operand that owns
/// that axis as a first-class dimension value and replicates with [`DynamicBroadcastOperation`]. Linearization then
/// keeps one scalar dimension value per runtime axis alive as a residual, instead of one result-shaped array per
/// aligned operand.
///
/// Returns `None` when the replication does not apply, in which case the caller differentiates the operands as they
/// are: the operation is not one of the implicitly broadcasting elementwise variants, its result has no tangent space
/// (so no operand is ever aligned), its result geometry is fully static, or every operand already has the result
/// shape.
///
/// # Parameters
///
///   - `context`: [`Context`] that owns the primal trace, in which the extent reads and the replications are bound.
///   - `operation`: Elementwise [`ArrayOperation`] whose operands are being replicated.
///   - `inputs`: Input [`DifferentiationDual`]s that the rule received.
fn replicated_elementwise_duals<A, C>(
    context: &C,
    operation: &ArrayOperation<A>,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Option<Vec<DifferentiationDual<C::Value>>>, DifferentiationError>
where
    A: Value<Type = ArrayType>,
    C: Context<Type = ArrayIrType>,
    C::Operation:
        From<DynamicBroadcastOperation> + From<DimensionSizeOperation> + From<ConstantOperation<DimensionValue>>,
    ArrayOperation<A>: Operation<Type = ArrayType>,
{
    // The variants whose type inference broadcasts several operands into one result, and whose differentiation rules
    // therefore align narrower operands (both live tangents and primal coefficients) with that result type.
    if !matches!(
        operation,
        ArrayOperation::Add(_)
            | ArrayOperation::Sub(_)
            | ArrayOperation::Mul(_)
            | ArrayOperation::Div(_)
            | ArrayOperation::Rem(_)
            | ArrayOperation::Pow(_)
            | ArrayOperation::Max(_)
            | ArrayOperation::Min(_)
            | ArrayOperation::Atan2(_)
            | ArrayOperation::And(_)
            | ArrayOperation::Or(_)
            | ArrayOperation::Xor(_)
            | ArrayOperation::Complex(_)
            | ArrayOperation::Compare(_)
            | ArrayOperation::Select(_),
    ) {
        return Ok(None);
    }

    let input_types = inputs
        .iter()
        .map(|input| Ok(<&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone()))
        .collect::<Result<Vec<_>, TypeError>>()?;
    let output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
    let [output_type] = output_types.as_slice() else {
        return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
    };
    let output_shape = output_type.shape();
    if output_type.tangent()?.is_zero_space()
        || output_shape.dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_)))
        || input_types.iter().all(|input_type| input_type.shape() == output_shape)
    {
        return Ok(None);
    }

    // One first-class dimension operand per result axis, as the mixed broadcast requires. A repeated dimension denotes
    // one runtime extent, so it is read once and shared by every axis that carries it.
    let mut extents = Vec::<C::Value>::with_capacity(output_shape.rank());
    for (axis, dimension) in output_shape.dimensions().iter().enumerate() {
        if let Some(previous) = output_shape.dimensions()[..axis].iter().position(|earlier| earlier == dimension) {
            extents.push(extents[previous].clone());
            continue;
        }
        let extent = match dimension {
            Dimension::Static(extent) => {
                let extent = DimensionValue::constant(*extent).map_err(ProgramError::from)?;
                context.bind(ConstantOperation::new(extent), Vec::new(), &[])?.remove(0)
            }
            Dimension::Dynamic(_) => {
                let source = input_types.iter().enumerate().find_map(|(index, input_type)| {
                    let input_axis = axis.checked_sub(output_shape.rank() - input_type.rank())?;
                    (&input_type.dimension(input_axis) == dimension).then_some((index, input_axis))
                });
                let Some((index, input_axis)) = source else {
                    return Err(TypeError::invalid(format!(
                        "cannot replicate `{}` operands into result shape {output_shape} because no operand carries \
                         its runtime axis {axis}",
                        operation.name(),
                    ))
                    .into());
                };
                context
                    .bind(
                        DimensionSizeOperation::new(&input_types[index], input_axis)?,
                        Vec::new(),
                        std::slice::from_ref(inputs[index].primal()),
                    )?
                    .remove(0)
            }
        };
        extents.push(extent);
    }

    // Replication is structurally linear, so the primal and the tangent of a narrower operand ride the same mixed
    // broadcast, and a structural-zero tangent stays structural at the replicated tangent type.
    inputs
        .iter()
        .zip(input_types.iter())
        .map(|(input, input_type)| {
            if input_type.shape() == output_shape {
                return Ok(input.clone());
            }
            let offset = output_shape.rank() - input_type.rank();
            let replication =
                DynamicBroadcastOperation::new((0..input_type.rank()).map(|axis| axis + offset).collect());
            let mut replication_inputs = Vec::with_capacity(1 + extents.len());
            replication_inputs.push(input.primal().clone());
            replication_inputs.extend(extents.iter().cloned());
            let primal = context.bind(replication.clone(), Vec::new(), replication_inputs.as_slice())?.remove(0);
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => {
                    replication_inputs[0] = tangent.clone();
                    MaybeZero::Value(context.bind(replication, Vec::new(), replication_inputs.as_slice())?.remove(0))
                }
            };
            Ok(DifferentiationDual::new(primal, tangent)?)
        })
        .collect::<Result<Vec<_>, DifferentiationError>>()
        .map(Some)
}

impl<A, C> MemberDifferentiableOperation<C> for ArrayOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayIrType,
            Constant: ValueProjection<ArrayType, Projected = A>,
            Operation: ResidualZeroProvider<ArrayIrType>
                           + From<ArrayIrOperation<A>>
                           + From<DynamicBroadcastOperation>
                           + From<DimensionSizeOperation>
                           + From<DimensionToScalarOperation>
                           + From<LinearCallOperation<ArrayIrType>>
                           + From<ZeroOperation<ArrayType>>
                           + From<ConstantOperation<DimensionValue>>
                           + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
                           + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
        > + Zero<C::Value>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    ArrayOperation<A>: Operation<Type = ArrayType> + DifferentiableOperation<ProjectedContext<C, ArrayType>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_duals = match self {
            Self::Slice(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::DynamicSlice(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::DynamicUpdateSlice(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::Gather(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::Scatter(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            Self::Reduce(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            operation => match replicated_elementwise_duals(context, operation, inputs)? {
                Some(duals) => jvp_projected_operation(context, operation, duals.as_slice())?,
                None => jvp_projected_operation(context, operation, inputs)?,
            },
        };
        output_duals
            .into_iter()
            .map(|output| {
                let tangent_type = output.tangent().r#type().into_owned();
                if !output.tangent().is_zero()
                    || tangent_type.identities().all(|(position, _)| position != TypeIdentityPosition::Reference)
                {
                    return Ok(output);
                }

                // A projected array rule can return a structural zero even when its result has runtime extents. Use
                // the primal result as its geometry exemplar before lifting the dual into the composite family.
                let (primal, _) = output.into_parts();
                let tangent_array_type = <&ArrayType>::try_from(&tangent_type)?;
                let primal_type = primal.r#type();
                let primal_data_type = <&ArrayType>::try_from(primal_type.as_ref())?.data_type();
                let exemplar = if tangent_array_type.data_type() == primal_data_type {
                    primal.clone()
                } else {
                    context
                        .bind(
                            ArrayIrOperation::<A>::Array(ArrayOperation::ConvertElementType(
                                ConvertElementTypeOperation::new(tangent_array_type.data_type()),
                            )),
                            Vec::new(),
                            std::slice::from_ref(&primal),
                        )?
                        .remove(0)
                };
                let tangent = context
                    .bind(
                        ArrayIrOperation::<A>::Array(ArrayOperation::ZeroLike(ZeroLikeOperation::new())),
                        Vec::new(),
                        &[exemplar],
                    )?
                    .remove(0);
                DifferentiationDual::new(primal, MaybeZero::Value(tangent)).map_err(Into::into)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::addressing::ArraySliceAxis;
    use crate::arrays::arrays::Array;
    use crate::arrays::batching::{ArrayBatching, ArrayIrBatch, ArrayIrBatching};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{
        Dimension, DimensionBounds, DimensionError, DimensionType, DimensionVariable, Shape,
    };
    use crate::arrays::types::ir::ArrayIrType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::arrays::types::memories::Memory;
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer, batch};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableType, ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::{AllGatherOutputVariance, CollectiveMode, CollectiveOptions};
    use crate::operations::random::RandomAlgorithm;
    use crate::operations::{
        AddOperation, ComparisonDirection, ConcatenateOperation, ConditionOperation, DimensionAddOperation,
        DimensionMulOperation, DimensionRequirementOperation, DimensionSizeOperation, DynamicBroadcastOperation,
        DynamicReshapeOperation, MulOperation, ReduceOperation, ReductionKind, ScanOperation, WhileOperation,
        ZeroOperation, ZeroOperationProvider,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{
        AtomId, Effect, Effects, EmptyRegionDriver, OperationProjection, Program, ProgramBuilder, ProgramError,
        ReferenceType, RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    #[test]
    fn test_array_operations_holds_for_every_canonical_array_value() {
        // The bundle is satisfied exactly when every member capability is, so instantiating this function is a
        // compile-time assertion that the listed value families implement the complete `ArrayOperation` surface.
        fn requires_array_operations<V: ArrayOperations>() {}

        requires_array_operations::<Array>();
        requires_array_operations::<Tracer<ArrayTracingContext>>();
        requires_array_operations::<LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>>();
        requires_array_operations::<BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>>();
    }

    #[test]
    fn test_dimension_operations_holds_for_every_canonical_dimension_value() {
        // The bundle is satisfied exactly when every member capability is, so instantiating this function is a
        // compile-time assertion that the listed value families implement the complete `DimensionOperation` surface.
        fn requires_dimension_operations<V: DimensionOperations>() {}

        requires_dimension_operations::<DimensionValue>();
        requires_dimension_operations::<Tracer<DimensionTracingContext>>();

        // `ArrayIrOperations` pins this bundle on its first-class-dimension member profile, so the dimension member
        // projected out of every canonical composite value must satisfy it as well.
        requires_dimension_operations::<<ArrayIrValue<Array> as ValueProjection<DimensionType>>::Projected>();
        requires_dimension_operations::<
            <Tracer<TracingContext<TestValue, TestOperation>> as ValueProjection<DimensionType>>::Projected,
        >();
    }

    #[test]
    fn test_array_ir_operations_holds_for_every_canonical_array_ir_value() {
        fn requires_array_ir_operations<V: ArrayIrOperations>() {}

        requires_array_ir_operations::<ArrayIrValue<Array>>();
        requires_array_ir_operations::<Tracer<TracingContext<TestValue, TestOperation>>>();
        requires_array_ir_operations::<LinearizationTracer<EagerContext<TestValue, TestOperation>>>();

        // The bundle's member profiles make one bound enough to reach ordinary array math through projection, checked
        // dimension arithmetic, and composite-native reference capabilities, with no projection bounds restated here.
        fn square_and_element_count<V: ArrayIrOperations>(value: &V) -> Result<(V, V), ProgramError> {
            // The array member carries both the fallible capability and its panicking operator sugar, so the
            // capability is named explicitly here.
            let array = <V as ValueProjection<ArrayType>>::into_projected(value.clone())?;
            let square = <V as ValueProjection<ArrayType>>::from_projected(Mul::mul(&array, &array)?);
            let elements = value.dimension_size(0)?.dimension_mul(&value.dimension_size(1)?)?;
            Ok((square, elements))
        }

        fn reference_round_trip<V: ArrayIrOperations>(value: &V) -> Result<V, ProgramError> {
            value.new_reference()?.freeze()
        }

        let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let (square, elements) = square_and_element_count(&input).unwrap();
        assert_eq!(square, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 4.0, 9.0, 16.0, 25.0, 36.0])));
        let ArrayIrValue::Dimension(elements) = elements else {
            panic!("expected a first-class dimension member");
        };
        assert_eq!(elements.extent(), 6);
        assert_eq!(reference_round_trip(&input), Ok(input));
    }

    #[test]
    fn test_composite_pullback_materializes_a_dynamic_zero_space_input_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let context = TracingContext::<TestValue, TestOperation>::new();
        let key = context.input(key_type.clone().into());
        let accumulator = context.input(ArrayType::scalar(DataType::F64).into());
        let (_, pullback) =
            context.vjp(|inputs: Vec<_>, ()| Ok(inputs[1].clone()), vec![key, accumulator], ()).unwrap();
        let cotangent = context.input(ArrayType::scalar(DataType::F64).into());

        // The compact pullback has no result slot for the key's zero differential space. Rebuilding the public result
        // must use the key extent captured at linearization time rather than attempt a nullary dynamic zero.
        let cotangents = pullback.apply(cotangent).unwrap();
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayIrType::Array(key_type.tangent().unwrap()));
        assert_eq!(cotangents[1].r#type().as_ref(), &ArrayType::scalar(DataType::F64).into());
    }

    #[test]
    fn test_composite_pushforward_materializes_a_dynamic_zero_space_output_tangent() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let key_type =
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let context = TracingContext::<TestValue, TestOperation>::new();
        let extent = context.input(extent_type.into());
        let key = context.input(key_type.clone().into());
        let (_, pushforward) =
            context.linearize(|inputs: Vec<_>, ()| Ok(inputs[1].clone()), vec![extent, key], ()).unwrap();
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let key_tangent = context.input(key_type.tangent().unwrap().into());

        // The compact pushforward has no output slot for the key's zero differential space. Rebuilding its public
        // result must consume the key extent captured at linearization time.
        let tangent = pushforward.apply(vec![extent_tangent, key_tangent]).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayIrType::Array(key_type.tangent().unwrap()));
    }

    #[test]
    fn test_array_ir_operation() {
        fn assert_projection<T: Type, O: Operation<Type = T>, C: OperationProjection<T, Projected = O>>() {}

        assert_projection::<ArrayType, ArrayOperation<Array>, ArrayIrOperation<Array>>();
        assert_projection::<DimensionType, DimensionOperation<DimensionValue>, ArrayIrOperation<Array>>();

        let array_type = ArrayType::scalar(DataType::F32);
        let array_operation = ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new()));
        assert!(matches!(array_operation, ArrayIrOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(array_operation.name(), "add");
        assert_eq!(array_operation.to_string(), "add");
        assert_eq!(
            array_operation.infer_output_types(&[array_type.clone().into(), array_type.clone().into()], &[],),
            Ok(vec![array_type.clone().into()]),
        );
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        assert_eq!(
            array_operation.infer_output_types(&[reference_type.clone(), reference_type], &[]),
            Err(TypeError::invalid("expected array type but got reference type")),
        );

        // Member control-flow operations promote to their direct composite carriers. Scan promotion also lifts its
        // capture values while preserving every semantic and lowering attribute.
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new())),
            ArrayIrOperation::Condition(_),
        ));
        let while_operation = WhileOperation::new().with_iteration_bound(7).unwrap();
        let promoted_while = ArrayIrOperation::<Array>::from(ArrayOperation::While(while_operation.clone()));
        assert!(matches!(
            promoted_while,
            ArrayIrOperation::While(operation)
                if operation.iteration_bound() == while_operation.iteration_bound()
        ));
        let capture = Array::vector(vec![3.0_f32, 4.0, 5.0, 6.0]);
        let scan_operation = ScanOperation::<Array>::new(1, 4)
            .with_reverse(true)
            .with_unroll(2)
            .unwrap()
            .with_captures(vec![capture.clone()]);
        let promoted_scan = ArrayIrOperation::<Array>::from(ArrayOperation::Scan(scan_operation));
        let ArrayIrOperation::Scan(promoted_scan) = promoted_scan else {
            panic!("expected a direct composite scan operation");
        };
        assert_eq!(promoted_scan.carry_count(), 1);
        assert_eq!(promoted_scan.length(), &Dimension::Static(4));
        assert!(promoted_scan.reverse());
        assert_eq!(promoted_scan.unroll(), 2);
        assert_eq!(promoted_scan.captures(), &[ArrayIrValue::Array(capture)]);

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let dimension_operation = ArrayIrOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&left_type, &right_type).unwrap(),
        ));
        assert!(matches!(dimension_operation, ArrayIrOperation::Dimension(DimensionOperation::Add(_)),));
        assert_eq!(dimension_operation.name(), "dimension_add");
        let result_types = dimension_operation
            .infer_output_types(&[left_type.clone().into(), right_type.clone().into()], &[])
            .unwrap();
        let [ArrayIrType::Dimension(result_type)] = result_types.as_slice() else {
            panic!("expected one dimension result type");
        };
        assert_eq!(result_type.bounds(), DimensionBounds::new(2, Some(17)).unwrap());
        let requirement = ArrayIrOperation::<Array>::from(DimensionOperation::Requirement(
            DimensionRequirementOperation::equal(&left_type, &right_type),
        ));
        assert_eq!(requirement.effects(), Effects::single(Effect::OrderedAssertion));

        // Each dimension operation also lifts directly, so generic composite code never has to name the member family.
        let product = ArrayIrOperation::<Array>::from(DimensionMulOperation::new(&left_type, &right_type).unwrap());
        assert!(matches!(product, ArrayIrOperation::Dimension(DimensionOperation::Mul(_))));
        assert_eq!(product.name(), "dimension_mul");

        // Every wrong-kind path uses the same checked type projection and therefore reports the canonical diagnostic.
        assert_eq!(
            array_operation.infer_output_types(&[left_type.clone().into(), right_type.clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            dimension_operation.infer_output_types(&[array_type.clone().into(), array_type.clone().into()], &[]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(
            ArrayIrOperation::<Array>::zero_operation(left_type.clone().into()).unwrap_err(),
            ProgramError::Type(TypeError::invalid("cannot materialize a zero for a first-class dimension type")),
        );
        let reference_type = ReferenceType::new(array_type.clone());
        assert_eq!(
            ArrayIrOperation::<Array>::zero_operation(reference_type.clone().into()).unwrap_err(),
            ProgramError::Type(TypeError::invalid(format!(
                "cannot materialize a zero for reference type `{reference_type}`; references must be discharged first",
            ))),
        );

        // The direct composite condition preserves the complete higher-order interface, including effects.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let interface = RegionInterface::new(
            vec![array_type.clone().into()],
            vec![array_type.clone().into()],
            Effects::single(Effect::OrderedIo),
        );
        let condition = ArrayIrOperation::<Array>::Condition(ConditionOperation::new());
        assert!(matches!(condition, ArrayIrOperation::Condition(_)));
        assert_eq!(
            condition.infer_output_types(
                &[predicate_type.into(), array_type.clone().into()],
                &[interface.clone(), interface],
            ),
            Ok(vec![array_type.clone().into()]),
        );
        assert_eq!(
            condition.infer_region_input_types(
                &[ArrayType::scalar(DataType::Boolean).into(), array_type.clone().into()],
                &[
                    RegionInterface::new(vec![array_type.clone().into()], vec![], Effects::PURE),
                    RegionInterface::new(vec![array_type.clone().into()], vec![], Effects::PURE),
                ],
            ),
            Ok(vec![None, None]),
        );
        assert_eq!(condition.region_slots(), ConditionOperation::<ArrayIrValue<Array>>::new().region_slots());
        assert_eq!(
            condition.output_region_provenance(0),
            ConditionOperation::<ArrayIrValue<Array>>::new().output_region_provenance(0),
        );

        // Canonical lifts keep identity-free zeros homogeneous, while explicit static mixed constructors are also
        // valid and identity-bearing zeros require the mixed encoding with ordinary identity renaming.
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let zero = ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(dynamic_type)));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();
        let ArrayIrOperation::Zero(zero) = zero.rename_type_identities(&renaming).unwrap() else {
            panic!("expected a renamed mixed zero operation");
        };
        assert_eq!(zero.r#type().shape().dimensions(), &[Dimension::Dynamic(target)]);

        let static_zero_type = ArrayType::scalar(DataType::F32);
        let static_zero = ArrayIrOperation::<Array>::from(ZeroOperation::new(static_zero_type.clone()));
        assert!(matches!(static_zero, ArrayIrOperation::Array(ArrayOperation::Zero(_))));
        let mixed_static_zero = ArrayIrOperation::<Array>::Zero(ZeroOperation::new(static_zero_type.clone()));
        assert_eq!(mixed_static_zero.infer_output_types(&[], &[]), Ok(vec![static_zero_type.clone().into()]));

        // Canonical lifts keep identity-free ones homogeneous, while explicit static mixed constructors are also
        // valid and identity-bearing ones require the mixed encoding.
        let static_one = ArrayIrOperation::<Array>::from(OneOperation::new(static_zero_type.clone()));
        assert!(matches!(static_one, ArrayIrOperation::Array(ArrayOperation::One(_))));
        let mixed_static_one = ArrayIrOperation::<Array>::DynamicOne(OneOperation::new(static_zero_type.clone()));
        assert_eq!(mixed_static_one.infer_output_types(&[], &[]), Ok(vec![static_zero_type.into()]));
        let dynamic_one_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let dynamic_one = ArrayIrOperation::<Array>::from(OneOperation::new(dynamic_one_type.clone()));
        assert!(matches!(dynamic_one, ArrayIrOperation::DynamicOne(_)));
        assert_eq!(
            dynamic_one.infer_output_types(&[DimensionType::new(source.clone()).into()], &[]),
            Ok(vec![dynamic_one_type.into()]),
        );
        assert_eq!(
            dynamic_one.infer_output_types(&[], &[]),
            Err(TypeError::invalid(
                "`one` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
        let other = DimensionVariable::new("other", bounds);
        assert_eq!(
            dynamic_one.infer_output_types(&[DimensionType::new(other).into()], &[]),
            Err(TypeError::invalid(
                "`one` operand 0 has type dimension<other ∈ [1, 9)> but the output shape requires \
                 dimension<source ∈ [1, 9)>",
            )),
        );
        assert_eq!(
            dynamic_one.infer_output_types(
                &[DimensionType::new(source.clone()).into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("`one` expects no regions but got 1")),
        );

        // Iota follows the same static-versus-dynamic routing while retaining and validating its varying axis.
        let static_iota_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)]));
        let static_iota = ArrayIrOperation::<Array>::from(IotaOperation::new(static_iota_type.clone(), 0).unwrap());
        assert!(matches!(static_iota, ArrayIrOperation::Array(ArrayOperation::Iota(_))));
        let mixed_static_iota =
            ArrayIrOperation::<Array>::DynamicIota(IotaOperation::new(static_iota_type.clone(), 0).unwrap());
        assert_eq!(mixed_static_iota.infer_output_types(&[], &[]), Ok(vec![static_iota_type.into()]));
        let dynamic_iota_type =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(2)]));
        let dynamic_iota = ArrayIrOperation::<Array>::from(IotaOperation::new(dynamic_iota_type.clone(), 0).unwrap());
        assert!(matches!(dynamic_iota, ArrayIrOperation::DynamicIota(_)));
        assert_eq!(
            dynamic_iota.infer_output_types(&[DimensionType::new(source.clone()).into()], &[]),
            Ok(vec![dynamic_iota_type.clone().into()]),
        );
        assert_eq!(
            IotaOperation::new(dynamic_iota_type, 2).unwrap_err(),
            TypeError::invalid("`iota` dimension 2 is out of bounds for rank 2"),
        );

        let renamed_left = DimensionVariable::new("renamed_left", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left_type.variable().clone(), renamed_left.clone()).unwrap();
        let ArrayIrOperation::Dimension(DimensionOperation::Add(add)) =
            dimension_operation.rename_type_identities(&renaming).unwrap()
        else {
            panic!("expected a renamed dimension addition operation");
        };
        assert_eq!(add.left_type().variable(), &renamed_left);
        assert_eq!(add.right_type(), &right_type);

        // A genuinely mixed operation is represented directly by the outer family rather than either homogeneous
        // member projection.
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let dimension_size = ArrayIrOperation::<Array>::from(DimensionSizeOperation::new(&dynamic_type, 0).unwrap());
        assert!(matches!(dimension_size, ArrayIrOperation::DimensionSize(_)));
        assert_eq!(dimension_size.name(), "dimension_size");
        assert_eq!(
            dimension_size.infer_output_types(&[dynamic_type.into()], &[]),
            Ok(vec![DimensionType::new(source).into()]),
        );

        // Canonical reshape derives its entire result shape from its ordered first-class dimension operand types.
        let reshape = ArrayIrOperation::<Array>::from(DynamicReshapeOperation::new());
        assert!(matches!(reshape, ArrayIrOperation::Reshape(_)));
        let two = DimensionValue::constant(2).unwrap();
        let three = DimensionValue::constant(3).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)]));
        assert_eq!(
            reshape.infer_output_types(
                &[input_type.clone().into(), two.r#type().into_owned().into(), three.r#type().into_owned().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),).into()
            ]),
        );
        let output_extent =
            DimensionType::new(DimensionVariable::new("output", DimensionBounds::new(1, Some(7)).unwrap()));
        assert_eq!(
            reshape.infer_output_types(
                &[input_type.into(), output_extent.clone().into(), three.r#type().into_owned().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(output_extent.variable().clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            reshape.infer_output_types(&[two.r#type().into_owned().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            reshape.infer_output_types(
                &[ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::I64).into()],
                &[]
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let placed_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
                .with_memory(Memory::Host { pinned: true });
        let permuted = ArrayIrOperation::<Array>::from(DynamicReshapeOperation::new().with_dimensions([1, 0]));
        assert_eq!(permuted.to_string(), "reshape [dimensions=[1, 0]]");
        assert_eq!(
            permuted.infer_output_types(
                &[placed_input_type.into(), DimensionValue::constant(6).unwrap().r#type().into_owned().into(),],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)]))
                    .with_memory(Memory::Host { pinned: true })
                    .into()
            ]),
        );
    }

    #[test]
    fn test_array_ir_operation_forwards_payload_effects() {
        // A statically proven mixed concatenate is pure. The derived dispatcher must read that payload classification
        // rather than declaring the composite family effectful.
        let concatenate = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into(),
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ],
        )
        .unwrap();
        let operation = ArrayIrOperation::<Array>::Concatenate(concatenate.clone());
        assert_eq!(operation.effects(), concatenate.effects());
        assert_eq!(operation.effects(), Effects::PURE);

        // A dynamic axis sum remains an ordered assertion and reaches the outer family unchanged.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(9)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::positive(Some(12)).unwrap());
        let concatenate = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])).into(),
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                DimensionType::new(result).into(),
            ],
        )
        .unwrap();
        let operation = ArrayIrOperation::<Array>::Concatenate(concatenate.clone());
        assert_eq!(operation.effects(), concatenate.effects());
        assert_eq!(operation.effects(), Effects::single(Effect::OrderedAssertion));

        // A dimension requirement is likewise pure when provable and otherwise needs an ordered runtime assertion.
        // Both states must reach the composite family unchanged.
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));

        // Provable: the same dimension variable is trivially equal to itself.
        let proven = DimensionRequirementOperation::equal(&left_type, &left_type);
        let operation = ArrayIrOperation::<Array>::from(DimensionOperation::Requirement(proven.clone()));
        assert_eq!(operation.effects(), proven.effects());
        assert_eq!(operation.effects(), Effects::PURE);

        // Unprovable: two distinct variables whose `[1, 9)` bounds admit both equal and unequal extents.
        let inconclusive = DimensionRequirementOperation::equal(&left_type, &right_type);
        let operation = ArrayIrOperation::<Array>::from(DimensionOperation::Requirement(inconclusive.clone()));
        assert_eq!(operation.effects(), inconclusive.effects());
        assert_eq!(operation.effects(), Effects::single(Effect::OrderedAssertion));
    }

    #[test]
    fn test_array_ir_operation_interpretation() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(
                ArrayOperation::Add(AddOperation::new()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![1.0, 2.0])),
                    ArrayIrValue::Array(Array::vector(vec![3.0, 4.0])),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0, 6.0]))]),
        );

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let operation = DimensionOperation::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap());
        let result = context
            .bind(
                operation,
                Vec::new(),
                &[
                    ArrayIrValue::Dimension(DimensionValue::new(left_type, 3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::new(right_type, 4).unwrap()),
                ],
            )
            .unwrap();
        let [ArrayIrValue::Dimension(result)] = result.as_slice() else {
            panic!("expected one dimension result");
        };
        assert_eq!(result.extent(), 7);

        let reshape_input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let reshape = context
            .bind(
                DynamicReshapeOperation::new(),
                Vec::new(),
                &[
                    reshape_input,
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(reshape, vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],))],);

        let rows = DimensionValue::constant(2).unwrap();
        let columns = DimensionValue::constant(3).unwrap();
        let zero = context
            .bind(
                ZeroOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![
                        Dimension::Dynamic(rows.r#type().variable().clone()),
                        Dimension::Dynamic(columns.r#type().variable().clone()),
                    ]),
                )),
                Vec::new(),
                &[ArrayIrValue::Dimension(rows), ArrayIrValue::Dimension(columns)],
            )
            .unwrap();
        assert_eq!(zero, vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32; 6]))]);

        let extent = DimensionValue::constant(3).unwrap();
        let one = context
            .bind(
                OneOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(extent.r#type().variable().clone())]),
                )),
                Vec::new(),
                &[ArrayIrValue::Dimension(extent)],
            )
            .unwrap();
        assert_eq!(one, vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]))]);
        // Explicit static mixed constructors consume no dimension operands and interpret identically to their
        // preferred homogeneous encodings.
        let static_float_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            context.bind(ArrayIrOperation::Zero(ZeroOperation::new(static_float_type.clone())), Vec::new(), &[],),
            Ok(vec![ArrayIrValue::Array(Array::scalar(0.0f32))]),
        );
        assert_eq!(
            context.bind(ArrayIrOperation::DynamicOne(OneOperation::new(static_float_type)), Vec::new(), &[],),
            Ok(vec![ArrayIrValue::Array(Array::scalar(1.0f32))]),
        );
        let static_iota_type = ArrayType::new_static(DataType::I32, [3]);
        assert_eq!(
            context.bind(
                ArrayIrOperation::DynamicIota(IotaOperation::new(static_iota_type.clone(), 0).unwrap()),
                Vec::new(),
                &[],
            ),
            Ok(vec![ArrayIrValue::Array(Array::from_elements(static_iota_type.clone(), &[0i32, 1, 2]).unwrap(),)]),
        );
        assert_eq!(
            context.bind(
                ArrayIrOperation::DynamicIota(IotaOperation::new(static_iota_type, 0).unwrap()),
                Vec::new(),
                &[ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())],
            ),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );

        let rows = DimensionValue::constant(2).unwrap();
        let dynamic_iota = context
            .bind(
                IotaOperation::new(
                    ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Dynamic(rows.r#type().variable().clone()), Dimension::Static(3)]),
                    ),
                    0,
                )
                .unwrap(),
                Vec::new(),
                &[ArrayIrValue::Dimension(rows)],
            )
            .unwrap();
        assert_eq!(
            dynamic_iota,
            vec![ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),),
                    &[0i32, 0, 0, 1, 1, 1],
                )
                .unwrap(),
            )],
        );
        let extent_type =
            DimensionType::new(DimensionVariable::new("iota_extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let extent_program_type = extent.r#type().into_owned();
        let output = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])), &[0i32, 1, 2])
                .unwrap(),
        );
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ArrayIrOperation::from(IotaOperation::new(
                ArrayType::new(
                    DataType::I32,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                ),
                0,
            )
            .unwrap()),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_program_type, replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        // A runtime extent outside the stored output axis's authoritative bounds is rejected before allocation,
        // even though eager binds skip inference: the operand's own variable admits the extent, so only the stored
        // axis's bounds can catch it. The input may use an equivalent dimension identity supplied by the calling
        // program, so its identity does not need to exactly match the one stored on the operation.
        let bounded = DimensionVariable::new("bounded", DimensionBounds::new(1, Some(4)).unwrap());
        let error = context
            .bind(
                ZeroOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(bounded.clone())]),
                )),
                Vec::new(),
                &[ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap())],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: DimensionBounds::new(1, Some(4)).unwrap(),
            }),
        );
        let error = context
            .bind(
                OneOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(bounded.clone())]))),
                Vec::new(),
                &[ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap())],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: DimensionBounds::new(1, Some(4)).unwrap(),
            }),
        );
        let error = context
            .bind(
                IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(bounded)])), 0)
                    .unwrap(),
                Vec::new(),
                &[ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap())],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: DimensionBounds::new(1, Some(4)).unwrap(),
            }),
        );

        let condition = ArrayIrOperation::<Array>::Condition(ConditionOperation::new());
        assert_eq!(
            condition.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::MalformedProgram("condition interpretation requires a predicate input".to_string(),)),
        );
    }

    #[test]
    fn test_array_ir_operation_tracing_has_only_explicit_dependencies() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = TestContext::new();
        let array = context.input(ArrayType::scalar(DataType::F32).into());
        let array_atom = array.atom_id().unwrap();
        let array = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(array).unwrap();
        array.dispatch_domain().bind(AddOperation::new(), Vec::new(), &[array.clone(), array]).unwrap();

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let left = context.input(left_type.clone().into());
        let right = context.input(right_type.clone().into());
        let left_atom = left.atom_id().unwrap();
        let right_atom = right.atom_id().unwrap();
        let left = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(left).unwrap();
        let right = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(right).unwrap();
        left.dispatch_domain()
            .bind(DimensionAddOperation::new(&left_type, &right_type).unwrap(), Vec::new(), &[left, right])
            .unwrap();

        let builder = context.builder().borrow();
        let [array_instruction, dimension_instruction] = builder.instructions() else {
            panic!("expected one array instruction and one dimension instruction");
        };
        assert_eq!(array_instruction.inputs(), &[array_atom, array_atom]);
        assert!(array_instruction.regions().is_empty());
        assert!(matches!(array_instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(dimension_instruction.inputs(), &[left_atom, right_atom]);
        assert!(dimension_instruction.regions().is_empty());
        assert!(matches!(dimension_instruction.operation(), ArrayIrOperation::Dimension(DimensionOperation::Add(_)),));

        let reshape_context = TestContext::new();
        let reshape_input =
            reshape_context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent = reshape_context.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let second_extent = reshape_context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let reshape_input_atom = reshape_input.atom_id().unwrap();
        let first_extent_atom = first_extent.atom_id().unwrap();
        let second_extent_atom = second_extent.atom_id().unwrap();
        let reshape_input = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(reshape_input)
            .unwrap()
            .into_value();
        let reshape_output = reshape_context
            .bind(DynamicReshapeOperation::new(), Vec::new(), &[reshape_input, first_extent, second_extent])
            .unwrap()
            .remove(0);
        let reshape_builder = reshape_context.builder().borrow();
        let [reshape_instruction] = reshape_builder.instructions() else {
            panic!("expected one reshape instruction");
        };
        assert_eq!(reshape_instruction.inputs(), &[reshape_input_atom, first_extent_atom, second_extent_atom],);
        assert!(matches!(reshape_instruction.operation(), ArrayIrOperation::Reshape(_)));
        drop(reshape_builder);
        let reshape_program = reshape_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reshape_output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            reshape_program.to_string(),
            indoc! {"
                lambda %0:f32[6] .
                let %1:dimension<2> = const 2
                    %2:dimension<3> = const 3
                    %3:f32[2, 3] = reshape %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        let zero_context = TestContext::new();
        let rows_value = DimensionValue::constant(2).unwrap();
        let columns_value = DimensionValue::constant(3).unwrap();
        let zero_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows_value.r#type().variable().clone()),
                Dimension::Dynamic(columns_value.r#type().variable().clone()),
            ]),
        );
        let rows = zero_context.constant(ArrayIrValue::Dimension(rows_value));
        let columns = zero_context.constant(ArrayIrValue::Dimension(columns_value));
        let rows_atom = rows.atom_id().unwrap();
        let columns_atom = columns.atom_id().unwrap();
        let zero_output =
            zero_context.bind(ZeroOperation::new(zero_type), Vec::new(), &[rows, columns]).unwrap().remove(0);
        let zero_builder = zero_context.builder().borrow();
        let [zero_instruction] = zero_builder.instructions() else {
            panic!("expected one shaped-zero instruction");
        };
        assert_eq!(zero_instruction.inputs(), &[rows_atom, columns_atom]);
        assert!(matches!(zero_instruction.operation(), ArrayIrOperation::Zero(_)));
        drop(zero_builder);
        let zero_program = zero_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![zero_output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            zero_program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<2> = const 2
                    %1:dimension<3> = const 3
                    %2:f32[2, 3] = zero [type=f32[2, 3]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        let one_context = TestContext::new();
        let extent_value = DimensionValue::constant(3).unwrap();
        let extent = one_context.constant(ArrayIrValue::Dimension(extent_value.clone()));
        let extent_atom = extent.atom_id().unwrap();
        let one_output = one_context
            .bind(
                OneOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
                )),
                Vec::new(),
                &[extent],
            )
            .unwrap()
            .remove(0);
        let one_builder = one_context.builder().borrow();
        let [one_instruction] = one_builder.instructions() else {
            panic!("expected one dynamic-one instruction");
        };
        assert_eq!(one_instruction.inputs(), &[extent_atom]);
        assert!(matches!(one_instruction.operation(), ArrayIrOperation::DynamicOne(_)));
        drop(one_builder);
        let one_program = one_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![one_output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            one_program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<3> = const 3
                    %1:f32[3] = one [type=f32[3]] %0
                in (%1)
            "}
            .trim_end(),
        );

        let iota_context = TestContext::new();
        let extent_value = DimensionValue::constant(3).unwrap();
        let extent = iota_context.constant(ArrayIrValue::Dimension(extent_value.clone()));
        let extent_atom = extent.atom_id().unwrap();
        let output = iota_context
            .bind(
                IotaOperation::new(
                    ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
                    ),
                    0,
                )
                .unwrap(),
                Vec::new(),
                &[extent],
            )
            .unwrap()
            .remove(0);
        let iota_builder = iota_context.builder().borrow();
        let [instruction] = iota_builder.instructions() else {
            panic!("expected one dynamic-iota instruction");
        };
        assert_eq!(instruction.inputs(), &[extent_atom]);
        assert!(matches!(instruction.operation(), ArrayIrOperation::DynamicIota(_)));
        drop(iota_builder);
        let iota_program = iota_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            iota_program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<3> = const 3
                    %1:i32[3] = iota [type=i32[3], dimension=0] %0
                in (%1)
            "}
            .trim_end(),
        );

        let concatenate_context = TestContext::new();
        let left =
            concatenate_context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let right =
            concatenate_context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into());
        let extent = concatenate_context.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let left_atom = left.atom_id().unwrap();
        let right_atom = right.atom_id().unwrap();
        let extent_atom = extent.atom_id().unwrap();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[left.r#type().into_owned(), right.r#type().into_owned(), extent.r#type().into_owned()],
        )
        .unwrap();
        let output = concatenate_context.bind(operation, Vec::new(), &[left, right, extent]).unwrap().remove(0);
        let concatenate_builder = concatenate_context.builder().borrow();
        let [instruction] = concatenate_builder.instructions() else {
            panic!("expected one concatenate instruction");
        };
        assert_eq!(instruction.inputs(), &[left_atom, right_atom, extent_atom]);
        assert!(matches!(instruction.operation(), ArrayIrOperation::Concatenate(_)));
        drop(concatenate_builder);
        let concatenate_program = concatenate_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            concatenate_program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[1] .
                let %2:dimension<3> = const 3
                    %3:f32[3] = concatenate [axis=0] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    /// Builds a composite scalar `f64` [`Program`] from `build`, which receives the region builder together with its
    /// `input_count` scalar inputs and returns the region's output atoms.
    fn composite_scalar_program<BuildFn>(input_count: usize, build: BuildFn) -> TestProgram
    where
        BuildFn: FnOnce(&mut ProgramBuilder<TestValue, TestOperation>, &[AtomId]) -> Vec<AtomId>,
    {
        let array_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inputs = (0..input_count).map(|_| builder.add_input(array_type.clone().into())).collect::<Vec<_>>();
        let outputs = build(&mut builder, inputs.as_slice());
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count]).unwrap()
    }

    /// Stages `input * scale` in `builder` and returns its result atom.
    fn composite_scaled(builder: &mut ProgramBuilder<TestValue, TestOperation>, input: AtomId, scale: f64) -> AtomId {
        let scale = builder.add_constant(ArrayIrValue::Array(Array::scalar(scale)));
        builder
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, scale])
            .unwrap()[0]
    }

    /// Builds the `["primal", "jvp"]` regions of a composite `custom_jvp` implementing `primal(x) = 2 * x` with the
    /// deliberately wrong rule `jvp(x, dx) = (2 * x, 3 * dx)`, so a surviving custom-derivative boundary stays
    /// detectable in a transformed program's numbers.
    fn composite_custom_jvp_regions() -> Vec<TestProgram> {
        vec![
            composite_scalar_program(1, |builder, inputs| vec![composite_scaled(builder, inputs[0], 2.0)]),
            composite_scalar_program(2, |builder, inputs| {
                vec![composite_scaled(builder, inputs[0], 2.0), composite_scaled(builder, inputs[1], 3.0)]
            }),
        ]
    }

    /// Builds the `["primal", "forward", "backward"]` regions of a composite `custom_vjp` implementing
    /// `primal(x) = 2 * x`, `forward(x) = (2 * x, x)`, and the deliberately wrong rule `backward(r, ȳ) = 3 * ȳ`.
    fn composite_custom_vjp_regions() -> Vec<TestProgram> {
        vec![
            composite_scalar_program(1, |builder, inputs| vec![composite_scaled(builder, inputs[0], 2.0)]),
            composite_scalar_program(1, |builder, inputs| vec![composite_scaled(builder, inputs[0], 2.0), inputs[0]]),
            composite_scalar_program(2, |builder, inputs| vec![composite_scaled(builder, inputs[1], 3.0)]),
        ]
    }

    /// Builds the `["primal", "forward", "backward", "tangent"]` regions of a composite `rematerialize` implementing
    /// `primal(x) = 2 * x`, `forward(x) = (2 * x, x)`, `backward(r, ȳ) = 3 * ȳ`, and `tangent(r, dx) = 3 * dx`.
    fn composite_rematerialize_regions() -> Vec<TestProgram> {
        let mut regions = composite_custom_vjp_regions();
        regions.push(composite_scalar_program(2, |builder, inputs| vec![composite_scaled(builder, inputs[1], 3.0)]));
        regions
    }

    /// Batches one composite region-carrying payload over a single operand mapped along axis zero and returns the
    /// rendered staged program together with the result's [`BatchAxis`].
    fn batched_composite_payload(operation: ArrayOperation<Array>, regions: Vec<TestProgram>) -> (String, BatchAxis) {
        let extent = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TracingContext::<TestValue, TestOperation>::new();
        let extent_input = trace.input(DimensionType::new(extent.clone()).into());
        let mapped = trace.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), extent_input);
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0)).unwrap());
        let output = context.bind(operation, regions, std::slice::from_ref(&input)).unwrap().remove(0);
        let batch_axis = output.batch().batch_axis();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![output.batch().value().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        (program.to_string(), batch_axis)
    }

    #[test]
    fn test_composite_lift_promotes_every_region_carrying_array_payload() {
        // Every region-carrying array payload has a composite carrier, so none of them reaches the region-free
        // projected `Array` variant. The custom-derivative wrappers and rematerialization carry their
        // non-differentiated operand split (and rematerialization its lowering hint) across the lift.
        assert!(matches!(
            TestOperation::from(ArrayOperation::CustomJvp(CustomJvpOperation::new().with_non_differentiated_count(1))),
            ArrayIrOperation::CustomJvp(operation) if operation.non_differentiated_count() == 1,
        ));
        assert!(matches!(
            TestOperation::from(ArrayOperation::CustomVjp(CustomVjpOperation::new().with_non_differentiated_count(2))),
            ArrayIrOperation::CustomVjp(operation) if operation.non_differentiated_count() == 2,
        ));
        assert!(matches!(
            TestOperation::from(ArrayOperation::Rematerialize(
                RematerializeOperation::new().with_prevent_cse(true).with_non_differentiated_count(1),
            )),
            ArrayIrOperation::Rematerialize(operation)
                if operation.prevent_cse() && operation.non_differentiated_count() == 1,
        ));
        assert!(matches!(
            TestOperation::from(ArrayOperation::LinearCall(LinearCallOperation::new(1))),
            ArrayIrOperation::LinearCall(operation) if operation.residual_count() == 1,
        ));

        // The transpose-only form stores the member types of the forward map it does not attach, so its lift maps
        // each of them into the composite type universe instead of dropping the interface.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let promoted = TestOperation::from(ArrayOperation::LinearCall(LinearCallOperation::transpose_only(
            1,
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
        )));
        let ArrayIrOperation::LinearCall(promoted) = promoted else {
            panic!("expected a promoted composite linear call");
        };
        assert_eq!(
            promoted,
            LinearCallOperation::transpose_only(
                1,
                vec![ArrayIrType::Array(scalar_type.clone())],
                vec![ArrayIrType::Array(scalar_type)],
            ),
        );
    }

    #[test]
    fn test_composite_batching_of_a_custom_jvp_payload() {
        // The composite carrier batches both regions structurally and threads the first-class mapped extent into
        // them as one additional leading non-differentiated operand of the batched call.
        let (program, batch_axis) = batched_composite_payload(
            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
            composite_custom_jvp_regions(),
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = custom_jvp [non_differentiated_count=1] %0 %1 [
                    primal={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[] = const 2.0
                            %3:f64[batch] = broadcast [output_axes=[]] %2 %0
                            %4:f64[batch] = mul %1 %3
                        in (%4)
                    },
                    jvp={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch], %2:f64[batch] .
                        let %3:f64[] = const 2.0
                            %4:f64[] = const 3.0
                            %5:f64[batch] = broadcast [output_axes=[]] %3 %0
                            %6:f64[batch] = mul %1 %5
                            %7:f64[batch] = broadcast [output_axes=[]] %4 %0
                            %8:f64[batch] = mul %2 %7
                        in (%6, %8)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_batching_of_a_custom_vjp_payload() {
        // The backward region receives the threaded extent ahead of its residuals, and its result cotangents align
        // with the differentiated operands only.
        let (program, batch_axis) = batched_composite_payload(
            ArrayOperation::CustomVjp(CustomVjpOperation::new()),
            composite_custom_vjp_regions(),
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = custom_vjp [non_differentiated_count=1] %0 %1 [
                    primal={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[] = const 2.0
                            %3:f64[batch] = broadcast [output_axes=[]] %2 %0
                            %4:f64[batch] = mul %1 %3
                        in (%4)
                    },
                    forward={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[] = const 2.0
                            %3:f64[batch] = broadcast [output_axes=[]] %2 %0
                            %4:f64[batch] = mul %1 %3
                        in (%4, %1)
                    },
                    backward={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch], %2:f64[batch] .
                        let %3:f64[] = const 3.0
                            %4:f64[batch] = broadcast [output_axes=[]] %3 %0
                            %5:f64[batch] = mul %2 %4
                        in (%5)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_batching_of_a_rematerialize_payload() {
        // All four regions are rebuilt around the threaded extent, keeping the rematerialization boundary intact.
        let (program, batch_axis) = batched_composite_payload(
            ArrayOperation::Rematerialize(RematerializeOperation::new()),
            composite_rematerialize_regions(),
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = rematerialize [non_differentiated_count=1] %0 %1 [
                    primal={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[] = const 2.0
                            %3:f64[batch] = broadcast [output_axes=[]] %2 %0
                            %4:f64[batch] = mul %1 %3
                        in (%4)
                    },
                    forward={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[] = const 2.0
                            %3:f64[batch] = broadcast [output_axes=[]] %2 %0
                            %4:f64[batch] = mul %1 %3
                        in (%4, %1)
                    },
                    backward={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch], %2:f64[batch] .
                        let %3:f64[] = const 3.0
                            %4:f64[batch] = broadcast [output_axes=[]] %3 %0
                            %5:f64[batch] = mul %2 %4
                        in (%5)
                    },
                    tangent={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch], %2:f64[batch] .
                        let %3:f64[] = const 3.0
                            %4:f64[batch] = broadcast [output_axes=[]] %3 %0
                            %5:f64[batch] = mul %2 %4
                        in (%5)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_batching_of_a_linear_call_payload() {
        // The executable linear call threads the mapped extent as one more leading residual, which is the precedent
        // the custom-derivative carriers follow with their non-differentiated operand split.
        let array_type = ArrayType::scalar(DataType::F64);
        let identity_program = || -> TestProgram {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            builder.add_input(array_type.clone().into());
            let linear = builder.add_input(array_type.clone().into());
            builder.build(vec![linear], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let linear = ArrayIrValue::Array(Array::vector(vec![2.0_f64, 5.0]));
        let output: TestValue = batch(
            |(residual, linear): (BatchingTracer<_, ArrayIrBatching>, BatchingTracer<_, ArrayIrBatching>)| {
                let context = residual.context().clone();
                Ok(context
                    .bind(
                        ArrayOperation::LinearCall(LinearCallOperation::new(1)),
                        vec![identity_program(), identity_program()],
                        &[residual, linear],
                    )?
                    .remove(0))
            },
            (ArrayIrValue::Array(Array::scalar(3.0_f64)), linear.clone()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, linear);
    }

    /// Builds a composite program holding one `custom_jvp` call over a scalar input.
    fn composite_custom_jvp_program() -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let regions = composite_custom_jvp_regions()
            .into_iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output = builder
            .add_instruction(ArrayOperation::CustomJvp(CustomJvpOperation::new()), regions, vec![input])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_composite_differentiation_of_a_region_carrying_array_payload() {
        // Forward mode reaches the composite carrier with its regions attached, so it replays the user JVP rule
        // instead of failing without region access. The rule's deliberately wrong tangent scale of three is what
        // proves the custom derivative governed the staged result.
        assert_eq!(
            composite_custom_jvp_program().jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const 2.0
                    %3:f64[] = const 3.0
                    %4:f64[] = mul %0 %2
                    %5:f64[] = mul %1 %3
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_transposition_of_a_region_carrying_array_payload() {
        // Reverse mode linearizes first, and the composite carrier's JVP rule replaces the call with plain staged
        // operations, so transposition reaches a tangent program the payload has already left. The gradient is the
        // user rule's deliberately wrong scale of three rather than the primal's true derivative of two.
        let context = EagerContext::<TestValue, TestOperation>::new();
        let (value, pullback) = context
            .vjp(
                |input: LinearizationTracer<_>, ()| {
                    let context = input.context().clone();
                    Ok(context
                        .bind(
                            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
                            composite_custom_jvp_regions(),
                            &[input],
                        )?
                        .remove(0))
                },
                ArrayIrValue::Array(Array::scalar(5.0_f64)),
                (),
            )
            .unwrap();
        let gradient = pullback.apply(ArrayIrValue::Array(Array::scalar(1.0_f64))).unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(10.0_f64)));
        assert_eq!(gradient, ArrayIrValue::Array(Array::scalar(3.0_f64)));

        // Transposing the raw, un-linearized call directly is still rejected, but now by the composite payload's own
        // non-transposable rule instead of by a projected adapter that never received its regions.
        assert_eq!(
            composite_custom_jvp_program().transpose_with_respect_to(&[0]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "operation `custom_jvp` is not transposable".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_ir_homogeneous_differentiation_dispatch() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = TestContext::new();
        let (primal, tangent) = context
            .jvp(
                |input, ()| {
                    let context = input.context().clone();
                    let factor = context.lift(ArrayIrValue::Array(Array::scalar(3.0_f64)))?;
                    Ok(context
                        .bind(
                            ArrayIrOperation::<Array>::Array(ArrayOperation::Mul(MulOperation::new())),
                            Vec::new(),
                            &[input, factor],
                        )?
                        .remove(0))
                },
                ArrayIrValue::Array(Array::scalar(2.0_f64)),
                ArrayIrValue::Array(Array::scalar(4.0_f64)),
                (),
            )
            .unwrap();
        assert_eq!(primal, ArrayIrValue::Array(Array::scalar(6.0_f64)));
        assert_eq!(tangent, ArrayIrValue::Array(Array::scalar(12.0_f64)));

        // Reverse mode composes the same projected JVP with projected transposition. The constant factor is a known
        // replay input to the homogeneous multiply transpose rule.
        let (primal, pullback) = context
            .vjp(
                |input, ()| {
                    let context = input.context().clone();
                    let factor = context.lift(ArrayIrValue::Array(Array::scalar(3.0_f64)))?;
                    Ok(context
                        .bind(
                            ArrayIrOperation::<Array>::Array(ArrayOperation::Mul(MulOperation::new())),
                            Vec::new(),
                            &[input, factor],
                        )?
                        .remove(0))
                },
                ArrayIrValue::Array(Array::scalar(2.0_f64)),
                (),
            )
            .unwrap();
        assert_eq!(primal, ArrayIrValue::Array(Array::scalar(6.0_f64)));
        assert_eq!(
            pullback.apply(ArrayIrValue::Array(Array::scalar(5.0_f64))),
            Ok(ArrayIrValue::Array(Array::scalar(15.0_f64))),
        );

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let factor = builder.add_constant(ArrayIrValue::Array(Array::scalar(3.0_f64)));
        let output = builder
            .add_instruction(
                ArrayIrOperation::<Array>::Array(ArrayOperation::Mul(MulOperation::new())),
                Vec::new(),
                vec![input, factor],
            )
            .unwrap()[0];
        let program = builder
            .build::<ArrayIrValue<Array>, ArrayIrValue<Array>>(vec![output], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            program
                .transpose_with_respect_to(&[0])
                .unwrap()
                .interpret(vec![ArrayIrValue::Array(Array::scalar(5.0_f64))]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(15.0_f64))]),
        );
    }

    #[test]
    fn test_array_ir_reduce_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        for (kind, expected_primal, expected_tangent, expected_cotangent) in
            [(ReductionKind::Sum, 6.0, 15.0, 6.0), (ReductionKind::Mean, 2.0, 5.0, 2.0)]
        {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(input_type.clone().into());
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![0], kind))),
                    Vec::new(),
                    vec![input],
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize().unwrap();

            assert_eq!(linearization.residual_count(), 1);
            assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]))])
                .unwrap();
            assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::scalar(expected_primal)));
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]))];
            tangent_inputs.extend(residuals.clone());
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![ArrayIrValue::Array(Array::scalar(expected_tangent))]),
            );
            let mut pullback_inputs = vec![ArrayIrValue::Array(Array::scalar(6.0_f64))];
            pullback_inputs.extend(residuals);
            assert_eq!(
                linearization.pullback().unwrap().interpret(pullback_inputs),
                Ok(vec![ArrayIrValue::Array(Array::vector(vec![
                    expected_cotangent,
                    expected_cotangent,
                    expected_cotangent,
                ]))]),
            );
        }

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(Vec::<f64>::new()))])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::scalar(0.0_f64)));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(Vec::<f64>::new()))];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::scalar(0.0_f64))]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::scalar(3.0_f64))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(Vec::<f64>::new()))]),
        );

        for (kind, values, expected_primal, expected_tangent, expected_cotangent) in [
            (ReductionKind::Max, vec![1.0, 5.0, 5.0, 2.0], 5.0, 25.0, vec![0.0, 4.0, 4.0, 0.0]),
            (ReductionKind::Min, vec![1.0, 1.0, 5.0, 2.0], 1.0, 15.0, vec![4.0, 4.0, 0.0, 0.0]),
        ] {
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(input_type.into());
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![0], kind))),
                    Vec::new(),
                    vec![input],
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize().unwrap();

            assert_eq!(linearization.residual_count(), 2);
            let mut primal_outputs =
                linearization.primal().interpret(vec![ArrayIrValue::Array(Array::vector(values))]).unwrap();
            assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::scalar(expected_primal)));
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]))];
            tangent_inputs.extend(residuals.clone());
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![ArrayIrValue::Array(Array::scalar(expected_tangent))]),
            );
            let mut pullback_inputs = vec![ArrayIrValue::Array(Array::scalar(8.0_f64))];
            pullback_inputs.extend(residuals);
            assert_eq!(
                linearization.pullback().unwrap().interpret(pullback_inputs),
                Ok(vec![ArrayIrValue::Array(Array::vector(expected_cotangent))]),
            );
        }
    }

    #[test]
    fn test_array_ir_explicit_shape_vertical_slice() {
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let extent_variable = DimensionVariable::new("extent", bounds);
        let extent_type = DimensionType::new(extent_variable.clone());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_variable.clone())]));

        // Build one stored program in which ordinary dimension arithmetic supplies explicit reshape and broadcast
        // operands. The repeated extent edge deliberately feeds both shape operations.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder.add_input(extent_type.clone().into());
        let one_value = DimensionValue::constant(1).unwrap();
        let one_type = one_value.r#type().into_owned();
        let one = builder.add_constant(ArrayIrValue::Dimension(one_value));
        let repeated_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&extent_type, &one_type).unwrap()),
                Vec::new(),
                vec![extent, one],
            )
            .unwrap()[0];
        let two = builder
            .add_instruction(
                DimensionOperation::Add(DimensionAddOperation::new(&one_type, &one_type).unwrap()),
                Vec::new(),
                vec![one, one],
            )
            .unwrap()[0];
        let reshaped = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, one, repeated_extent])
            .unwrap()[0];
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![0, 1]),
                Vec::new(),
                vec![reshaped, two, repeated_extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let [multiply_instruction, add_instruction, reshape_instruction, broadcast_instruction] =
            program.instructions()
        else {
            panic!("expected dimension arithmetic followed by reshape and broadcast");
        };
        assert_eq!(reshape_instruction.inputs(), &[input, one, multiply_instruction.outputs()[0]]);
        assert_eq!(
            broadcast_instruction.inputs(),
            &[reshape_instruction.outputs()[0], add_instruction.outputs()[0], multiply_instruction.outputs()[0]],
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 5)> .
                let %2:dimension<1> = const 1
                    %3:dimension<extent * 1 ∈ [1, 5)> = dimension_mul %1 %2
                    %4:dimension<2> = dimension_add %2 %2
                    %5:f64[1, extent * 1] = reshape %0 %2 %3
                    %6:f64[2, extent * 1] = broadcast [output_axes=[0, 1]] %5 %4 %3
                in (%6)
            "}
            .trim_end(),
        );

        let extent_value = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let input_value = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]));
        let expected = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0]));
        assert_eq!(program.interpret(vec![input_value.clone(), extent_value.clone()]), Ok(vec![expected.clone()]));

        // Known dimension arithmetic folds during partial evaluation while the two shape operations retain their
        // explicit extent inputs in the residual program.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(input_type.clone().into()),
                PartialValue::Known(extent_value.clone()),
            ])
            .unwrap();
        assert_eq!(evaluation.program().instructions().len(), 2);
        assert!(matches!(evaluation.program().instructions()[0].operation(), ArrayIrOperation::Reshape(_),));
        assert!(matches!(evaluation.program().instructions()[1].operation(), ArrayIrOperation::Broadcast(_),));
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                std::slice::from_ref(&input_value),
            ),
            Ok(vec![expected.clone()]),
        );

        // Forward differentiation replays both shape operations over the live array tangent while every dimension
        // value remains structural.
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f64, 5.0, 6.0]));
        let expected_tangent = ArrayIrValue::Array(Array::matrix(2, 3, vec![4.0_f64, 5.0, 6.0, 4.0, 5.0, 6.0]));
        assert_eq!(
            program.jvp().unwrap().interpret(vec![input_value.clone(), extent_value.clone(), tangent,]),
            Ok(vec![expected.clone(), expected_tangent]),
        );

        // Batching inserts one physical leading axis while the extent remains a replicated shape value.
        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let batched_input = BatchingTracer::new(
            batching_context.clone(),
            ArrayIrBatch::new(
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let batched_extent =
            BatchingTracer::new(batching_context.clone(), ArrayIrBatch::replicated(extent_value.clone()));
        let batched_output = program
            .interpret_in_context(&batching_context, vec![batched_input, batched_extent])
            .unwrap()
            .remove(0);
        assert_eq!(batched_output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            batched_output.batch().value(),
            &ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(3)]),
                ),
                vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
            )),
        );

        // Instantiation and import rename the boundary identity while preserving the internal arithmetic result and
        // both consumers of its SSA value.
        let target_variable = DimensionVariable::new("target", bounds);
        let target_type = DimensionType::new(target_variable.clone());
        let target_array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(target_variable.clone())]));
        let instantiated = program
            .with_instantiated_type_identities(&[target_array_type.clone().into(), target_type.clone().into()])
            .unwrap()
            .into_owned();
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_input = destination.add_input(target_array_type.into());
        let imported_extent = destination.add_input(target_type.into());
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input, imported_extent]).unwrap();
        assert_eq!(destination.instructions().len(), 4);
        let [_, _, imported_reshape, imported_broadcast] = destination.instructions() else {
            panic!("expected the complete imported vertical slice");
        };
        assert_eq!(imported_reshape.inputs()[0], imported_input);
        assert_eq!(imported_broadcast.inputs()[0], imported_reshape.outputs()[0]);
        assert_eq!(imported_broadcast.outputs(), imported_outputs.as_slice());
    }

    #[test]
    fn test_array_ir_batching_stages_one_composite_graph() {
        // Batching a body that mixes ordinary array arithmetic with first-class dimension arithmetic stages exactly
        // one composite graph. The array instructions gain a packed batch axis, the dimension instructions stay
        // replicated shape values, and both kinds live in the same program with a single flat region: nothing is
        // split into a second universe and no shape is recovered from ambient metadata.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let per_item_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let extent_operation = DimensionSizeOperation::new(&per_item_type, 0).unwrap();
        let extent_type = extent_operation.result_type().clone();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(per_item_type.clone().into());
        let doubled = builder
            .add_instruction(ArrayOperation::Add(AddOperation::new()), Vec::new(), vec![input, input])
            .unwrap()[0];
        let extent = builder.add_instruction(extent_operation, Vec::new(), vec![doubled]).unwrap()[0];
        let total = builder
            .add_instruction(
                DimensionOperation::Add(DimensionAddOperation::new(&extent_type, &extent_type).unwrap()),
                Vec::new(),
                vec![extent, extent],
            )
            .unwrap()[0];
        let widened = builder
            .add_instruction(DynamicBroadcastOperation::new(vec![1]), Vec::new(), vec![doubled, total, extent])
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                Vec::new(),
                vec![widened],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[rows] .
                let %1:f32[rows] = add %0 %0
                    %2:dimension<rows ∈ [1, 5)> = dimension_size [axis=0] %1
                    %3:dimension<rows + rows ∈ [2, 9)> = dimension_add %2 %2
                    %4:f32[rows + rows, rows] = broadcast [output_axes=[1]] %1 %3 %2
                    %5:f32[rows] = reduce_sum [axes=[0]] %4
                in (%5)
            "}
            .trim_end(),
        );

        // Batch the body under a staging parent so that the composite graph batching produces is observable as a
        // program rather than as concrete per-item values. The per-item extent stays symbolic, so the batched graph
        // remains shape polymorphic in `rows` even though the mapped axis itself has a known size.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let packed = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(rows.clone())]))
                .into(),
        );
        let axis_extent_id = axis_extent.atom_id().unwrap();
        let packed_id = packed.atom_id().unwrap();
        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), axis_extent);
        let batched_output = program
            .interpret_in_context(
                &batching_context,
                vec![BatchingTracer::new(
                    batching_context.clone(),
                    ArrayIrBatch::new(packed, BatchAxis::new(0)).unwrap(),
                )],
            )
            .unwrap()
            .remove(0);
        assert_eq!(batched_output.batch().batch_axis(), BatchAxis::new(0));
        let batched_output_id = batched_output.into_batch().into_value().atom_id().unwrap();

        // One composite graph: the batched body is a flat instruction sequence over the single composite operation
        // family, in source order, with array and dimension instructions interleaved and no nested region.
        let builder = trace.builder().borrow();
        assert_eq!(
            builder.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["add", "dimension_size", "dimension_add", "broadcast", "reduce_sum"],
        );
        assert!(builder.instructions().iter().all(|instruction| instruction.regions().is_empty()));
        let [add, dimension_size, dimension_add, broadcast, reduce] = builder.instructions() else {
            panic!("expected one batched composite graph with five instructions");
        };
        assert_eq!(add.inputs(), &[packed_id, packed_id]);
        assert_eq!(dimension_size.inputs(), &[add.outputs()[0]]);
        assert_eq!(dimension_add.inputs(), &[dimension_size.outputs()[0], dimension_size.outputs()[0]]);
        assert_eq!(
            broadcast.inputs(),
            &[add.outputs()[0], axis_extent_id, dimension_add.outputs()[0], dimension_size.outputs()[0]],
        );
        assert_eq!(reduce.inputs(), &[broadcast.outputs()[0]]);
        drop(builder);

        let batched_program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![batched_output_id],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            batched_program.to_string(),
            indoc! {"
                lambda %1:f32[2, rows] .
                let %0:dimension<2> = const 2
                    %2:f32[2, rows] = add %1 %1
                    %3:dimension<rows ∈ [1, 5)> = dimension_size [axis=1] %2
                    %4:dimension<rows + rows ∈ [2, 9)> = dimension_add %3 %3
                    %5:f32[2, rows + rows, rows] = broadcast [output_axes=[0, 2]] %2 %0 %4 %3
                    %6:f32[2, rows] = reduce_sum [axes=[1]] %5
                in (%6)
            "}
            .trim_end(),
        );
        assert_eq!(
            batched_program.interpret(vec![ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ))]),
            Ok(vec![ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                vec![12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
            ))]),
        );
    }

    /// Member-kind signature of one [`ArrayIrOperation`] variant, in terms of which member kinds it consumes and
    /// produces as *values*.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum MemberKindSignature {
        /// Homogeneous array boundary, reached through the [`ArrayType`] operation projection.
        ArrayToArray,

        /// Homogeneous first-class-dimension boundary, reached through the [`DimensionType`] projection.
        DimensionToDimension,

        /// Converts a first-class dimension value into ordinary array data.
        DimensionToArrayGateway,

        /// Converts ordinary array data into a first-class dimension value.
        ArrayToDimensionGateway,

        /// Creates a new reference root initialized from ordinary array data.
        ArrayToReference,

        /// Derives a root-preserving reference view from another reference handle.
        ReferenceToReference,

        /// Reads or consumes ordinary array data from a reference.
        ReferenceToArray,

        /// Replaces a reference from an ordinary array and returns the previous ordinary array value.
        ReferenceAndArrayToArray,

        /// Updates a reference from an ordinary array without producing a value.
        ReferenceAndArrayToUnit,

        /// Consumes first-class dimensions as geometry (or reads an array's geometry as a dimension) without ever
        /// converting a value from one member kind into the other.
        GeometryMixed,

        /// Forwards composite regions whose bodies may carry every admitted member kind.
        RegionForwarding,
    }

    /// Classifies one composite operation by its member-kind signature.
    ///
    /// The match is deliberately exhaustive with no wildcard arm, so adding an [`ArrayIrOperation`] variant fails to
    /// compile until its member-kind signature is declared here and recorded in the expected table of
    /// `test_array_ir_operation_member_kinds_are_a_closed_family`. That is the compile-forced drift gate for explicit
    /// member-kind crossings: dimension-to-data and data-to-dimension conversion use their checked gateways, while
    /// array-to-reference and reference-to-array crossings use their explicitly classified reference operations.
    fn member_kind_signature(operation: &ArrayIrOperation<Array>) -> MemberKindSignature {
        match operation {
            ArrayIrOperation::Zero(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::DynamicOne(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::DynamicIota(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::Array(_) => MemberKindSignature::ArrayToArray,
            ArrayIrOperation::Dimension(_) => MemberKindSignature::DimensionToDimension,
            ArrayIrOperation::Compare(_) => MemberKindSignature::DimensionToArrayGateway,
            ArrayIrOperation::DimensionSize(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::NewReference(_) => MemberKindSignature::ArrayToReference,
            ArrayIrOperation::ReferenceIndex(_) | ArrayIrOperation::ReferenceSlice(_) => {
                MemberKindSignature::ReferenceToReference
            }
            ArrayIrOperation::ReferenceRead(_) => MemberKindSignature::ReferenceToArray,
            ArrayIrOperation::ReferenceSwap(_) => MemberKindSignature::ReferenceAndArrayToArray,
            ArrayIrOperation::ReferenceAddUpdate(_) => MemberKindSignature::ReferenceAndArrayToUnit,
            ArrayIrOperation::FreezeReference(_) => MemberKindSignature::ReferenceToArray,
            ArrayIrOperation::DimensionFromScalar(_) => MemberKindSignature::ArrayToDimensionGateway,
            ArrayIrOperation::DimensionToScalar(_) => MemberKindSignature::DimensionToArrayGateway,
            ArrayIrOperation::Reshape(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::Broadcast(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::Concatenate(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::CustomCall(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::Pad(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::DynamicShapeSlice(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::RngBitGenerator(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::AllGather(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::PSumScatter(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::AllToAll(_) => MemberKindSignature::GeometryMixed,
            ArrayIrOperation::Condition(_) => MemberKindSignature::RegionForwarding,
            ArrayIrOperation::While(_) => MemberKindSignature::RegionForwarding,
            ArrayIrOperation::Scan(_) => MemberKindSignature::RegionForwarding,
            ArrayIrOperation::CustomJvp(_) => MemberKindSignature::RegionForwarding,
            ArrayIrOperation::CustomVjp(_) => MemberKindSignature::RegionForwarding,
            ArrayIrOperation::LinearCall(_) => MemberKindSignature::RegionForwarding,
            ArrayIrOperation::Rematerialize(_) => MemberKindSignature::RegionForwarding,
        }
    }

    #[test]
    fn test_array_ir_operation_member_kinds_are_a_closed_family() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let first_type = DimensionType::new(first.clone());
        let second_type = DimensionType::new(second.clone());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(first.clone())]));
        let dynamic_integer_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(first.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        // One instance of every variant, in declaration order, paired with its hand-maintained expected member-kind
        // signature. Exactly two variants may turn a dimension value into array data (the explicit
        // `dimension_to_scalar` gateway and the deliberately composite-level dimension comparison) and exactly one may
        // turn array data into a dimension value (the checked `dimension_from_scalar` gateway). Everything else either
        // stays inside one homogeneous member family, treats dimensions as geometry, or forwards regions.
        let expected: Vec<(ArrayIrOperation<Array>, MemberKindSignature)> = vec![
            (ArrayIrOperation::Zero(ZeroOperation::new(dynamic_type.clone())), MemberKindSignature::GeometryMixed),
            (ArrayIrOperation::DynamicOne(OneOperation::new(dynamic_type.clone())), MemberKindSignature::GeometryMixed),
            (
                ArrayIrOperation::DynamicIota(IotaOperation::new(dynamic_integer_type, 0).unwrap()),
                MemberKindSignature::GeometryMixed,
            ),
            (ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())), MemberKindSignature::ArrayToArray),
            (
                ArrayIrOperation::Dimension(DimensionOperation::Add(
                    DimensionAddOperation::new(&first_type, &second_type).unwrap(),
                )),
                MemberKindSignature::DimensionToDimension,
            ),
            (
                ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                MemberKindSignature::DimensionToArrayGateway,
            ),
            (
                ArrayIrOperation::DimensionSize(DimensionSizeOperation::new(&dynamic_type, 0).unwrap()),
                MemberKindSignature::GeometryMixed,
            ),
            (ArrayIrOperation::NewReference(NewReferenceOperation::new()), MemberKindSignature::ArrayToReference),
            (
                ArrayIrOperation::ReferenceIndex(ReferenceIndexOperation::new(0, 0)),
                MemberKindSignature::ReferenceToReference,
            ),
            (
                ArrayIrOperation::ReferenceSlice(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 1, 1)])),
                MemberKindSignature::ReferenceToReference,
            ),
            (ArrayIrOperation::ReferenceRead(ReferenceReadOperation::new()), MemberKindSignature::ReferenceToArray),
            (
                ArrayIrOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                MemberKindSignature::ReferenceAndArrayToArray,
            ),
            (
                ArrayIrOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                MemberKindSignature::ReferenceAndArrayToUnit,
            ),
            (ArrayIrOperation::FreezeReference(FreezeReferenceOperation::new()), MemberKindSignature::ReferenceToArray),
            (
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(second.clone())),
                MemberKindSignature::ArrayToDimensionGateway,
            ),
            (
                ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation),
                MemberKindSignature::DimensionToArrayGateway,
            ),
            (ArrayIrOperation::Reshape(DynamicReshapeOperation::new()), MemberKindSignature::GeometryMixed),
            (ArrayIrOperation::Broadcast(DynamicBroadcastOperation::new(vec![0])), MemberKindSignature::GeometryMixed),
            (
                ArrayIrOperation::Concatenate(
                    ConcatenateOperation::<ArrayIrType>::from_input_types(
                        0,
                        &[
                            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into(),
                            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                            DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                        ],
                    )
                    .unwrap(),
                ),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::CustomCall(
                    CustomCallOperation::new("ryft.test.identity", vec![scalar_type.clone()]).into(),
                ),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::Pad(PadOperation::new(vec![0], vec![0], vec![0]).unwrap().into()),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::DynamicShapeSlice(DynamicShapeSliceOperation::new(1)),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::RngBitGenerator(RngBitGeneratorOperation::new(
                    RandomAlgorithm::ThreeFry,
                    dynamic_type.clone(),
                )),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::AllGather(AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled),
                    AllGatherOutputVariance::Varying,
                )),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::PSumScatter(PSumScatterOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled),
                )),
                MemberKindSignature::GeometryMixed,
            ),
            (
                ArrayIrOperation::AllToAll(AllToAllOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled),
                )),
                MemberKindSignature::GeometryMixed,
            ),
            (ArrayIrOperation::Condition(ConditionOperation::new()), MemberKindSignature::RegionForwarding),
            (ArrayIrOperation::While(WhileOperation::new()), MemberKindSignature::RegionForwarding),
            (ArrayIrOperation::Scan(ScanOperation::new(1, 4)), MemberKindSignature::RegionForwarding),
            (ArrayIrOperation::CustomJvp(CustomJvpOperation::new()), MemberKindSignature::RegionForwarding),
            (ArrayIrOperation::CustomVjp(CustomVjpOperation::new()), MemberKindSignature::RegionForwarding),
            (ArrayIrOperation::LinearCall(LinearCallOperation::new(0)), MemberKindSignature::RegionForwarding),
            (ArrayIrOperation::Rematerialize(RematerializeOperation::new()), MemberKindSignature::RegionForwarding),
        ];

        assert_eq!(
            expected.iter().map(|(operation, _)| member_kind_signature(operation)).collect::<Vec<_>>(),
            expected.iter().map(|(_, signature)| *signature).collect::<Vec<_>>(),
        );

        // The table must stay complete: every variant that `member_kind_signature` can classify appears above exactly
        // once, so the two enumeration claims above are enumerated rather than sampled.
        assert_eq!(expected.len(), 33);
        assert_eq!(
            expected
                .iter()
                .filter(|(_, signature)| *signature == MemberKindSignature::DimensionToArrayGateway)
                .count(),
            2,
        );
        assert_eq!(
            expected
                .iter()
                .filter(|(_, signature)| *signature == MemberKindSignature::ArrayToDimensionGateway)
                .count(),
            1,
        );
        assert_eq!(
            expected.iter().filter(|(_, signature)| *signature == MemberKindSignature::ArrayToReference).count(),
            1,
        );
        assert_eq!(
            expected.iter().filter(|(_, signature)| *signature == MemberKindSignature::ReferenceToArray).count(),
            2,
        );
        assert_eq!(
            expected
                .iter()
                .filter(|(_, signature)| *signature == MemberKindSignature::ReferenceToReference)
                .count(),
            2,
        );
        assert_eq!(
            expected
                .iter()
                .filter(|(_, signature)| *signature == MemberKindSignature::ReferenceAndArrayToArray)
                .count(),
            1,
        );
        assert_eq!(
            expected
                .iter()
                .filter(|(_, signature)| *signature == MemberKindSignature::ReferenceAndArrayToUnit)
                .count(),
            1,
        );
    }
}
