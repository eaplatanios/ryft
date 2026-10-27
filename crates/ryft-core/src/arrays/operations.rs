use ryft_macros::Operation;

use crate::arrays::arrays::Array;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::references::ArrayReferenceTransform;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::DimensionType;
use crate::arrays::types::ir::ArrayIrType;
use crate::operations::{
    AbsOperation, Add, AddOperation, AndOperation, ArgMaxOperation, ArgMinOperation, ArithmeticOperations, Assert,
    AssertOperation, Atan2Operation, AxisIndexOperation, BroadcastOperation, Capability, CastOperation, CeilOperation,
    ClampOperation, CollectiveOperations, Compare, CompareOperation, ComplexOperation, ComplexOperations,
    CompositeOperation, CompositeTransposeOperation, ConcatenateOperation, ConditionOperation, ConjugateOperation,
    ConstantOperation, ConstantOperations, ConstrainShardingOperation, ConvertElementTypeOperation, CosOperation,
    Cumulative, CumulativeOperation, CustomCall, CustomCallOperation, DimensionAddOperation, DimensionDivOperation,
    DimensionFromScalar, DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin,
    DimensionMinOperation, DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation,
    DimensionSize, DimensionSizeOperation, DimensionSubOperation, DimensionToScalar, DimensionToScalarOperation, Div,
    DivOperation, DotOperation, DotOperations, DotProductAttention, DotProductAttentionBackwardOperation,
    DotProductAttentionOperation, DynamicBroadcastOperation, DynamicManipulationOperations, DynamicReshapeOperation,
    DynamicRngBitGenerator, DynamicSliceOperation, DynamicUpdateSliceOperation, Erf, ErfOperation, ExpM1Operation,
    ExpOperation, ExponentialOperations, ExtremaOperations, FloorOperation, GatherOperation, GuardOperation,
    ImaginaryOperation, IotaOperation, LiftedCustomRules, LinearCallOperation, Ln1pOperation, LogAddExpOperation,
    LogOperation, LogicalOperations, LogisticOperation, ManipulationOperations, MaxOperation, MinOperation, Mul,
    MulOperation, NegOperation, NotOperation, OneLikeOperation, OneOperation, OrOperation, PadOperation,
    ParallelAllGatherOperation, ParallelAllToAllOperation, ParallelPermuteOperation, ParallelRaggedAllToAllOperation,
    ParallelReduceOperation, ParallelSumScatterOperation, ParallelVaryOperation, PowOperation, Print, PrintOperation,
    RaggedDotOperation, RealOperation, ReduceOperation, ReducePrecisionOperation, ReductionOperations,
    ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation,
    ReferenceOperations, ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation, Rem, RemOperation,
    RematerializeOperation, ReshapeOperation, ReshardOperation, ReverseOperation, RngBitGenerator,
    RngBitGeneratorOperation, RoundOperation, RoundingOperations, RsqrtOperation, SaturatingSub,
    SaturatingSubOperation, ScaledDot, ScaledDotOperation, ScanOperation, ScatterOperation, Select, SelectOperation,
    ShardMapOperation, ShardingOperations, SignOperation, SinOperation, SliceOperation, Softmax, Sort, SortOperation,
    SqrtOperation, StopGradient, StopGradientOperation, Sub, SubOperation, Tag, TagOperation, TanOperation,
    TanhOperation, TransferToMemoryOperation, TransposeOperation, TrigonometricOperations, UpdateSliceOperation,
    WhileOperation, XorOperation, ZeroLikeOperation, ZeroOperation,
};
use crate::programs::{Value, ValueProjection};

#[cfg(doc)]
use crate::programs::{Operation, TypeError, Typed};

/// [`Operation`] family for ordinary staged [`Program`](crate::Program)s over [`Array`]s.
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(identity, batching, differentiation, transposition))]
pub enum ArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation<ArrayType>),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation<ArrayType>),
    Constant(ConstantOperation<Array>),
    Iota(IotaOperation<ArrayType>),
    Min(MinOperation<ArrayType>),
    Max(MaxOperation<ArrayType>),
    Clamp(ClampOperation<ArrayType>),
    Sign(SignOperation<ArrayType>),
    Neg(NegOperation<ArrayType>),
    Abs(AbsOperation<ArrayType>),
    Add(AddOperation<ArrayType>),
    Sub(SubOperation<ArrayType>),
    SaturatingSub(SaturatingSubOperation<ArrayType>),
    Mul(MulOperation<ArrayType>),
    Div(DivOperation<ArrayType>),
    Rem(RemOperation<ArrayType>),
    Pow(PowOperation<ArrayType>),
    Sqrt(SqrtOperation<ArrayType>),
    Rsqrt(RsqrtOperation<ArrayType>),
    Sin(SinOperation<ArrayType>),
    Cos(CosOperation<ArrayType>),
    Tan(TanOperation<ArrayType>),
    Tanh(TanhOperation<ArrayType>),
    Atan2(Atan2Operation<ArrayType>),
    Exp(ExpOperation<ArrayType>),
    ExpM1(ExpM1Operation<ArrayType>),
    Log(LogOperation<ArrayType>),
    Ln1p(Ln1pOperation<ArrayType>),
    LogAddExp(LogAddExpOperation<ArrayType>),
    Logistic(LogisticOperation<ArrayType>),
    Erf(ErfOperation<ArrayType>),
    Floor(FloorOperation<ArrayType>),
    Ceil(CeilOperation<ArrayType>),
    Round(RoundOperation<ArrayType>),
    Not(NotOperation<ArrayType>),
    And(AndOperation<ArrayType>),
    Or(OrOperation<ArrayType>),
    Xor(XorOperation<ArrayType>),
    Complex(ComplexOperation<ArrayType>),
    Real(RealOperation<ArrayType>),
    Imaginary(ImaginaryOperation<ArrayType>),
    Conjugate(ConjugateOperation<ArrayType>),
    Reduce(ReduceOperation),
    ArgMax(ArgMaxOperation),
    ArgMin(ArgMinOperation),
    Cumulative(CumulativeOperation),
    Sort(SortOperation),
    Dot(DotOperation),
    RaggedDot(RaggedDotOperation),
    ScaledDot(ScaledDotOperation),
    DotProductAttention(DotProductAttentionOperation),
    DotProductAttentionBackward(DotProductAttentionBackwardOperation),
    AxisIndex(AxisIndexOperation),
    ParallelReduce(ParallelReduceOperation),
    ParallelVary(ParallelVaryOperation),
    ParallelAllGather(ParallelAllGatherOperation),
    ParallelSumScatter(ParallelSumScatterOperation),
    ParallelPermute(ParallelPermuteOperation),
    ParallelAllToAll(ParallelAllToAllOperation),
    ParallelRaggedAllToAll(ParallelRaggedAllToAllOperation),
    Compare(CompareOperation<ArrayType>),
    Select(SelectOperation<ArrayType>),
    Transpose(TransposeOperation),
    Reverse(ReverseOperation),
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
    ConvertElementType(ConvertElementTypeOperation<ArrayType>),
    ReducePrecision(ReducePrecisionOperation<ArrayType>),
    TransferToMemory(TransferToMemoryOperation),
    Reshard(ReshardOperation),
    ConstrainSharding(ConstrainShardingOperation),
    Condition(ConditionOperation<ArrayType>),
    While(WhileOperation<ArrayType>),
    Scan(ScanOperation<ArrayType>),
    StopGradient(StopGradientOperation<ArrayType>),
    Tag(TagOperation<ArrayType>),
    Rematerialize(RematerializeOperation<ArrayType>),
    LinearCall(LinearCallOperation<ArrayType>),
    RngBitGenerator(RngBitGeneratorOperation<ArrayType>),
    Print(PrintOperation<ArrayType>),
    Assert(AssertOperation<ArrayType>),
    Guard(GuardOperation<ArrayType>),
    Cast(CastOperation<ArrayType>),
    CustomCall(CustomCallOperation),
    Composite(CompositeOperation<V, ArrayOperation<V>>),
    CompositeTranspose(CompositeTransposeOperation<V, ArrayOperation<V>>),
}

/// Value-level capability bundle that groups every operation Ryft supports on arrays, so that generic array code can
/// state one bound instead of listing each capability it uses. Its members are the capability groups and individual
/// capabilities through which values stage or execute [`ArrayOperation`] variants, listed in variant order. It is
/// implemented automatically for every value that implements all of its members, and so it must never be implemented
/// manually.
///
/// The bundle is parameterized by the [`Capability`] universe `T` of its values, which defaults to that of the
/// implementing value. Homogeneous array values (e.g., [`Array`] and the tracers over [`ArrayType`]) implement
/// `ArrayOperations<ArrayType>`, and array IR values (e.g., [`ArrayIrValue`] and the tracers over [`ArrayIrType`])
/// implement `ArrayOperations<ArrayIrType>` by applying the array capabilities to their array members (a first-class
/// dimension or reference member fails with the projection's [`TypeError`]). Generic code bounded by
/// `V: ArrayOperations` therefore works for values of either universe without projecting them.
///
/// The bundle deliberately does not pin `Value<Type = T>`. A value's capability universe is its own [`Typed::Type`] by
/// convention (refer to [`Capability`]), and an equality between the two in the bound would make the trait solver
/// normalize `V::Type` to the universe projection, so that bounds on `V::Type` in generic code (e.g.,
/// `V::Type: DifferentiableType`) would no longer apply.
pub trait ArrayOperations<T = <Self as Capability>::Universe>:
    Value
    + ConstantOperations<T>
    + ExtremaOperations<T>
    + ArithmeticOperations<T>
    + SaturatingSub<T>
    + TrigonometricOperations<T>
    + ExponentialOperations<T>
    + Softmax<T>
    + Erf<T>
    + RoundingOperations<T>
    + LogicalOperations<T>
    + ComplexOperations<T>
    + ReductionOperations<T>
    + Cumulative<T>
    + Sort<T>
    + DotOperations<T>
    + ScaledDot<T>
    + DotProductAttention<T>
    + CollectiveOperations<T>
    + Compare<Self, T>
    + Select<T>
    + ManipulationOperations<T>
    + ShardingOperations<Self, T>
    + StopGradient<T>
    + Tag<T>
    + RngBitGenerator<T>
    + Print<T>
    + Assert<T>
    + CustomCall<T>
{
}

impl<
    T,
    V: Value
        + ConstantOperations<T>
        + ExtremaOperations<T>
        + ArithmeticOperations<T>
        + SaturatingSub<T>
        + TrigonometricOperations<T>
        + ExponentialOperations<T>
        + Softmax<T>
        + Erf<T>
        + RoundingOperations<T>
        + LogicalOperations<T>
        + ComplexOperations<T>
        + ReductionOperations<T>
        + Cumulative<T>
        + Sort<T>
        + DotOperations<T>
        + ScaledDot<T>
        + DotProductAttention<T>
        + CollectiveOperations<T>
        + Compare<V, T>
        + Select<T>
        + ManipulationOperations<T>
        + ShardingOperations<V, T>
        + StopGradient<T>
        + Tag<T>
        + RngBitGenerator<T>
        + Print<T>
        + Assert<T>
        + CustomCall<T>,
> ArrayOperations<T> for V
{
}

/// [`Operation`] family used for staged [`Program`](crate::Program)s over [`DimensionValue`]s.
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(identity))]
pub enum DimensionOperation<V: Value<Type = DimensionType>> {
    Constant(ConstantOperation<V>),
    Max(DimensionMaxOperation),
    Min(DimensionMinOperation),
    Add(DimensionAddOperation),
    Sub(DimensionSubOperation),
    // Saturating subtraction is total over valid dimension extents, so it uses the shared stateless operation.
    // Checked dimension operations such as `DimensionAddOperation` instead store input-dependent assertion metadata:
    // `Operation::effects(&self)` has no input types from which to decide whether an ordered runtime check is needed.
    SaturatingSub(SaturatingSubOperation<DimensionType>),
    Mul(DimensionMulOperation),
    Div(DimensionDivOperation),
    Rem(DimensionRemOperation),
    Pow(DimensionPowOperation),
}

/// Value-level capability bundle that groups every operation Ryft supports on first-class dimension values, so that
/// generic first-class-dimension code can state one bound instead of listing each capability it uses. Its members are
/// the capabilities through which values stage or execute [`DimensionOperation`] variants, listed in variant order.
/// Checked arithmetic reuses the shared [`Add`], [`Sub`], [`SaturatingSub`], [`Mul`], [`Div`] (flooring division), and
/// [`Rem`] capabilities, alongside the dedicated [`DimensionMin`], [`DimensionMax`], and [`DimensionPow`] capabilities.
/// It is implemented automatically for every value that implements all of its members, and so it must never be
/// implemented manually.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor and is passed to every member,
/// so that first-class dimension values implement this bundle for [`DimensionType`].
pub trait DimensionOperations<T = <Self as Capability>::Universe>:
    Value
    + DimensionMax<T>
    + DimensionMin<T>
    + Add<T>
    + Sub<T>
    + SaturatingSub<T>
    + Mul<T>
    + Div<T>
    + Rem<T>
    + DimensionPow<T>
{
}

impl<
    T,
    V: Value
        + DimensionMin<T>
        + DimensionMax<T>
        + Add<T>
        + Sub<T>
        + SaturatingSub<T>
        + Mul<T>
        + Div<T>
        + Rem<T>
        + DimensionPow<T>,
> DimensionOperations<T> for V
{
}

/// [`Operation`] family for Ryft's array Intermediate Representation (IR), whose values include arrays, first-class
/// dimensions, and array references. Homogeneous operations project their complete boundary into [`ArrayOperation`]
/// or [`DimensionOperation`], delegate to that family's rule, and lift the result types back into [`ArrayIrType`].
/// Mixed signatures need explicit variants. For example, [`DimensionSizeOperation`] consumes an array and produces
/// a dimension. Reference operations also stay family-native so that their array/reference boundaries,
/// [`ArrayReferenceTransform`] access paths, and ordered effects remain visible to program passes.
///
/// Mixed array constructors and shape-changing operations receive runtime extents as explicit first-class dimension
/// inputs. Stored types carry their identities and bounds, but not the values of dynamic extents. [`Zero`](Self::Zero),
/// [`One`](Self::One), and [`Iota`](Self::Iota) store the complete result [`ArrayType`] and take one extent per dynamic
/// result axis, in axis order. Static result types need no inputs and remain valid, although canonical lifts prefer
/// the homogeneous array encoding. Collective operations like [`ParallelAllGather`](Self::ParallelAllGather),
/// [`ParallelSumScatter`](Self::ParallelSumScatter), and [`ParallelAllToAll`](Self::ParallelAllToAll) take trailing
/// extent inputs for every result axis, in axis order. Other mixed variants describe their extent-input conventions
/// in their documentation.
///
/// Attached regions use the full [`ArrayIrType`] universe. Refer to the corresponding payload operation documentation
/// for reference-state validation and transform restrictions.
#[derive(Clone, Debug, Operation)]
#[ryft(
    type(ArrayIrType),
    constant(ArrayIrValue<A>),
    members(ArrayType, structural(DimensionType)),
    dispatch(identity, discharge, batching, differentiation, transposition),
)]
pub enum ArrayIrOperation<A: Value<Type = ArrayType>> {
    /// Constructs a zero-filled array of the stored type using explicit runtime extents.
    #[ryft(mixed(structural), skip_from)]
    Zero(ZeroOperation<ArrayType>),

    /// Constructs a one-filled array of the stored type using explicit runtime extents.
    #[ryft(mixed(structural), skip_from)]
    One(OneOperation<ArrayType>),

    /// Constructs an index array using the stored type, iota axis, and explicit runtime extents.
    #[ryft(mixed(structural), skip_from)]
    Iota(IotaOperation<ArrayType>),

    /// Region-free [`ArrayOperation`] member, reached by projecting the complete instruction boundary to [`ArrayType`].
    /// Attached regions belong to the array IR universe and cannot pass through this projection. Region-carrying
    /// operations must instead use dedicated array IR family variants that retain access to their regions. Transforms
    /// reject a region-carrying payload constructed here directly, with a diagnostic naming its operation.
    #[ryft(projected(ArrayType), skip_from)]
    Array(ArrayOperation<A>),

    /// Homogeneous [`DimensionOperation`] member.
    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),

    /// Reads an array axis extent as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Checks and converts scalar-array data into a first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Converts a first-class dimension into scalar-array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Creates a whole-array reference root.
    ReferenceNew(ReferenceNewOperation<ArrayType, ArrayIrType>),

    /// Reads the array addressed by a reference path.
    ReferenceRead(ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Writes through a reference path without reading its previous value.
    ReferenceWrite(ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Writes through a reference path and returns the previous value.
    ReferenceSwap(ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Adds an update through a reference path in program order.
    ReferenceAddUpdate(ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Atomically adds an update through a reference path.
    ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Consumes a whole-array reference and returns its final value.
    ReferenceFreeze(ReferenceFreezeOperation<ArrayType, ArrayIrType>),

    /// All-gather with explicit result extents.
    #[ryft(mixed)]
    ParallelAllGather(ParallelAllGatherOperation),

    /// Sum-scatter with explicit result extents.
    #[ryft(mixed)]
    ParallelSumScatter(ParallelSumScatterOperation),

    /// All-to-all with explicit result extents.
    #[ryft(mixed)]
    ParallelAllToAll(ParallelAllToAllOperation),

    /// Array-only ragged all-to-all. It needs an explicit carrier because [`Array`](Self::Array) is already the single
    /// canonical projected carrier for [`ArrayType`].
    #[ryft(mixed)]
    ParallelRaggedAllToAll(ParallelRaggedAllToAllOperation),

    /// Compares two dimensions, producing a rank-zero Boolean array: `(Dimension, Dimension) -> Array(Boolean scalar)`.
    /// This crossing cannot belong to the homogeneous [`DimensionOperation`] family. Array comparisons use
    /// [`ArrayOperation::Compare`] through [`Array`](Self::Array) while array-dimension and dimension-array
    /// pairs are rejected.
    Compare(CompareOperation<ArrayIrType>),

    /// Reshapes using one first-class dimension input per output axis.
    Reshape(DynamicReshapeOperation),

    /// Broadcasts using one first-class dimension input per output axis.
    Broadcast(DynamicBroadcastOperation),

    /// Pads using one explicit result-extent input per output axis.
    Pad(PadOperation<ArrayIrType>),

    /// Concatenates arrays using one trailing result-extent input.
    Concatenate(ConcatenateOperation<ArrayIrType>),

    /// Slices using first-class dimension inputs for starts and output sizes.
    DynamicSlice(DynamicSliceOperation<ArrayIrType>),

    /// Manual Single Program Multiple Data (SPMD) computation whose array IR body runs once per device coordinate along
    /// its active manual axes. Refer to [`ShardMapOperation`] for array/reference boundaries and first-class dimensions
    /// inside the body.
    ShardMap(ShardMapOperation),

    /// Branches between array IR regions.
    Condition(ConditionOperation<ArrayIrType>),

    /// Loops over array IR condition and body regions.
    While(WhileOperation<ArrayIrType>),

    /// Scans an array IR body, including first-class dimension carries, dynamic trip counts, reference carries,
    /// and reference stacks.
    Scan(ScanOperation<ArrayIrType>),

    /// Rematerialized call with an array IR body.
    Rematerialize(RematerializeOperation<ArrayIrType>),

    /// Differentiation-owned linear call with ordinary trailing residual inputs.
    LinearCall(LinearCallOperation<ArrayIrType>),

    /// Bit generator with trailing dimension inputs for its dynamic bits-output axes.
    RngBitGenerator(RngBitGeneratorOperation<ArrayIrType>),

    /// Asserts a Boolean array predicate with optional array or dimension diagnostics.
    Assert(AssertOperation<ArrayIrType>),

    /// Checks unproven entry identity relationships at runtime and returns inputs narrowed by those facts.
    Guard(GuardOperation<ArrayIrType>),

    /// Returns its first input at a compatible array or dimension type, using further shape and proof witnesses.
    Cast(CastOperation<ArrayIrType>),

    /// Foreign-kernel call with trailing dimension inputs for its dynamic output axes.
    #[ryft(mixed)]
    CustomCall(CustomCallOperation),

    /// Calls an array IR primal region with custom transform rules, supplied as attached program regions or retained
    /// callbacks registered for this array IR family. Converting an [`ArrayOperation::Composite`] with attached rules
    /// uses this variant because those regions convert with the primal. Calls with retained array-only callbacks
    /// instead use [`LiftedComposite`](Self::LiftedComposite). Reference inputs are allowed only in the leading
    /// non-differentiated segment and reference outputs are rejected.
    Composite(CompositeOperation<ArrayIrValue<A>, ArrayIrOperation<A>>),

    /// Temporary reverse-mode carrier of [`Composite`](Self::Composite) calls with custom backward rules. It represents
    /// the tangent map whose transpose is the backward rule, and has no executable forward program. Transposition
    /// replaces it with the backward computation. The corresponding carrier for retained array-only rules is
    /// [`LiftedCompositeTranspose`](Self::LiftedCompositeTranspose).
    CompositeTranspose(CompositeTransposeOperation<ArrayIrValue<A>, ArrayIrOperation<A>>),

    /// Calls an array IR primal region while retaining custom transform callbacks registered for the original
    /// [`ArrayOperation`] family. The primal region has already been converted into array IR, but the callbacks still
    /// accept array-only tracers. When a rule is needed, [`LiftedCustomRules`] requests its specialization from the
    /// original definition and converts the resulting program into array IR, caching the conversion for reuse.
    ///
    /// This uses the same [`CompositeOperation`] as [`Composite`](Self::Composite), with a different rule-source type
    /// that adapts the retained array-only rules. Calls with attached rule programs need no such adapter: their rules
    /// convert with the primal, so they use [`Composite`](Self::Composite) even when promoted from [`ArrayOperation`].
    LiftedComposite(CompositeOperation<ArrayIrValue<A>, ArrayIrOperation<A>, LiftedCustomRules<A, ArrayOperation<A>>>),

    /// Temporary reverse-mode carrier of [`LiftedComposite`](Self::LiftedComposite), with the same role as
    /// [`CompositeTranspose`](Self::CompositeTranspose). It retains the original array-only rule definition through
    /// [`LiftedCustomRules`], which specializes and converts the backward program into array IR during transposition.
    /// Like the native carrier, it has no executable forward program and is replaced by the backward computation.
    LiftedCompositeTranspose(
        CompositeTransposeOperation<ArrayIrValue<A>, ArrayIrOperation<A>, LiftedCustomRules<A, ArrayOperation<A>>>,
    ),
}

/// Value-level capability bundle that groups every operation Ryft supports on array IR values, so that generic array
/// IR code can state one bound instead of listing each capability it uses. It is implemented automatically for every
/// value that implements all of its members, and so it must never be implemented manually.
///
/// The bundle extends [`ArrayOperations<ArrayIrType>`](ArrayOperations), through which array IR values apply every
/// array capability to their array members, with the mixed capabilities whose signatures cross the array and
/// first-class-dimension member kinds (e.g., [`DimensionSize`] and [`DynamicManipulationOperations`]), the
/// [`ReferenceOperations`] over [`ArrayReferenceTransform`] paths, and the dynamic random bit generator. It also
/// requires that the [`ValueProjection<ArrayType>`](ValueProjection) of a value satisfies [`ArrayOperations`] and that
/// its [`ValueProjection<DimensionType>`](ValueProjection) satisfies [`DimensionOperations`], so that generic array IR
/// code that projects explicitly (e.g., to compute in the array universe over several operations) does not have to
/// restate `ValueProjection<ArrayType, Projected: ArrayOperations>`-style bounds. That projected guarantee is not
/// implied by the member bound: a blanket implementation that derives `V: X` from `V::Projected: X` cannot be inverted.
///
/// Variants without a value-level capability are necessarily absent. These are the context-side constructors
/// [`DynamicZero`](crate::DynamicZero), [`DynamicOne`](crate::DynamicOne), and [`DynamicIota`](crate::DynamicIota),
/// and the variants that function-level APIs and transforms stage on behalf of values, namely control flow,
/// rematerialization, linear calls, composites, and shard maps.
///
/// Array capabilities apply directly to array IR values, and checked arithmetic over first-class dimensions uses
/// their member projection:
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DimensionType, DimensionValue, Mul, ProgramError, ValueProjection};
/// # fn main() -> Result<(), ProgramError> {
/// // Array capabilities apply directly to array IR values containing arrays.
/// let left = ArrayIrValue::Array(Array::vector(vec![2.0f64, 3.0]).unwrap());
/// let right = ArrayIrValue::Array(Array::vector(vec![4.0f64, 5.0]).unwrap());
/// assert_eq!(left.mul(&right)?, ArrayIrValue::Array(Array::vector(vec![8.0f64, 15.0]).unwrap()));
///
/// // The same array IR universe also holds first-class dimension values representing extents.
/// let rows = ArrayIrValue::Dimension(DimensionValue::constant(2)?);
/// let columns = ArrayIrValue::Dimension(DimensionValue::constant(3)?);
///
/// // Project those members so that `mul` uses checked dimension arithmetic.
/// let rows = ValueProjection::<DimensionType>::into_projected(rows)?;
/// let columns = ValueProjection::<DimensionType>::into_projected(columns)?;
/// let elements = rows.mul(&columns)?;
/// assert_eq!(elements.extent(), 6);
/// # Ok(())
/// # }
/// ```
///
/// The [`ParameterProjection`](crate::ParameterProjection) extension trait applies the member projection to a whole
/// [`Parameterized`](crate::Parameterized) tree at a boundary (i.e., `model.project_parameters::<ArrayType>()?` and
/// `projected.lift_parameters::<ArrayIrValue<A>>()?`), which code whose machinery is defined only on [`ArrayType`]
/// uses instead of projecting leaf by leaf. Dense derivative terminals (i.e., [`Jacobian`](crate::Jacobian) and
/// [`Hessian`](crate::Hessian)) use that route for their coordinate machinery.
///
/// As with [`ArrayOperations`], this bundle implies nothing about the tracers derived from an implementing value,
/// and a tracer requirement stays a separate bound.
pub trait ArrayIrOperations:
    Value<Type = ArrayIrType>
    + ArrayOperations<ArrayIrType>
    + ValueProjection<ArrayType, Projected: ArrayOperations<ArrayType>>
    + ValueProjection<DimensionType, Projected: DimensionOperations<DimensionType>>
    + DimensionSize
    + DimensionFromScalar
    + DimensionToScalar
    + ReferenceOperations<ArrayReferenceTransform, ArrayIrType>
    + DynamicManipulationOperations<ArrayIrType>
    + DynamicRngBitGenerator<ArrayIrType>
{
}

impl<
    V: Value<Type = ArrayIrType>
        + ArrayOperations<ArrayIrType>
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>
        + DimensionSize
        + DimensionFromScalar
        + DimensionToScalar
        + ReferenceOperations<ArrayReferenceTransform, ArrayIrType>
        + DynamicManipulationOperations<ArrayIrType>
        + DynamicRngBitGenerator<ArrayIrType>,
> ArrayIrOperations for V
where
    <V as ValueProjection<ArrayType>>::Projected: ArrayOperations<ArrayType>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionOperations<DimensionType>,
{
}

impl<A: Value<Type = ArrayType>> From<ArrayOperation<A>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ArrayOperation<A>) -> Self {
        match operation {
            ArrayOperation::Zero(operation) => Self::from(operation),
            ArrayOperation::ParallelRaggedAllToAll(operation) => Self::from(operation),
            ArrayOperation::Condition(operation) => Self::Condition(operation.into()),
            ArrayOperation::While(operation) => Self::While(operation.into()),
            ArrayOperation::Scan(operation) => Self::Scan(operation.into()),
            ArrayOperation::LinearCall(operation) => Self::LinearCall(operation.into()),
            ArrayOperation::Rematerialize(operation) => Self::Rematerialize(operation.into()),
            ArrayOperation::Composite(operation) => {
                // The family chooses the native or lifted carrier; payload retyping preserves its rule metadata.
                match operation.into_attached_family() {
                    Ok(operation) => Self::Composite(operation),
                    Err(operation) => Self::LiftedComposite(operation.into_family(LiftedCustomRules::Member)),
                }
            }
            ArrayOperation::CompositeTranspose(operation) => {
                // The family chooses the native or lifted carrier; payload retyping preserves its rule metadata.
                match operation.into_attached_family() {
                    Ok(operation) => Self::CompositeTranspose(operation),
                    Err(operation) => Self::LiftedCompositeTranspose(operation.into_family(LiftedCustomRules::Member)),
                }
            }
            operation => {
                // Preserve existing instruction inputs. Mixed payload conversions such as `PadOperation` require
                // extra extent inputs, which lifting an existing homogeneous instruction cannot introduce.
                Self::Array(operation)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hash::Hash;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::batching::{ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::references::ArrayReferenceTransform;
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{
        Dimension, DimensionBounds, DimensionError, DimensionMemberType, DimensionType, DimensionVariable, Shape,
    };
    use crate::arrays::types::ir::ArrayIrType;
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{
        DOTS_SAVABLE_POLICY_NAME, DifferentiationContext, DotsSavable, LinearizationTracer, NothingSavable,
        PartitionedDifferentiationPolicy,
    };
    use crate::macros::check_operation_type_inference;
    use crate::operations::{
        AddOperation, AssertOperation, CollectiveMode, CollectiveOptions, ComparisonDirection, CompositeJvpRule,
        CompositeOperation, CompositeTransposeOperation, ConcatenateOperation, ConditionOperation,
        CustomRuleDefinition, CustomRuleRegistration, DimensionAddOperation, DimensionFromScalarOperation,
        DimensionMulOperation, DimensionSizeOperation, DynamicBroadcastOperation, DynamicReshapeOperation,
        MulOperation, ParallelAllGatherOutputVariance, RandomAlgorithm, ReduceOperation, ReductionKind,
        RematerializationOptimizationBarrier, ScanOperation, ShardMap, WhileOperation, ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialValue, ResidualPolicyReference};
    use crate::programs::{
        AtomId, EffectClass, EffectClasses, Operation, OperationProjection, Program, ProgramBuilder, ProgramError,
        ReferenceType, RegionInterface, Type, TypeIdentityPosition, TypeIdentityRenaming, Typed, Value,
        ValueProjection,
    };
    use crate::tests::hash_of;
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    /// Asserts at compile time that an operation family implements equality and hashing.
    fn assert_identity<T: Eq + Hash>() {}

    /// Builds an array IR scalar `f64` [`Program`] from `build`, which receives the region builder together with its
    /// `input_count` scalar inputs and returns the region's output atoms.
    fn array_ir_scalar_program<BuildFn>(input_count: usize, build: BuildFn) -> TestProgram
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
    fn array_ir_scaled(builder: &mut ProgramBuilder<TestValue, TestOperation>, input: AtomId, scale: f64) -> AtomId {
        let scale = builder.add_constant(ArrayIrValue::Array(Array::scalar(scale).unwrap()));
        builder
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, scale], None)
            .unwrap()[0]
    }

    /// Builds the `["primal", "jvp"]` regions of an array IR `composite` call with a JVP rule, implementing
    /// `primal(x) = 2 * x` with the deliberately distinct rule `jvp(x, dx) = (2 * x, 3 * dx)`, so a surviving composite
    /// boundary stays detectable in a transformed program's numbers.
    fn array_ir_custom_jvp_regions() -> Vec<TestProgram> {
        vec![
            array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 2.0)]),
            array_ir_scalar_program(2, |builder, inputs| {
                vec![array_ir_scaled(builder, inputs[0], 2.0), array_ir_scaled(builder, inputs[1], 3.0)]
            }),
        ]
    }

    /// Builds the `["primal", "forward", "backward"]` regions of an array IR `composite` call with reverse-mode rules,
    /// implementing `primal(x) = 2 * x`, `forward(x) = (2 * x, x)`, and the deliberately distinct rule
    /// `backward(r, ȳ) = 3 * ȳ`.
    fn array_ir_custom_vjp_regions() -> Vec<TestProgram> {
        vec![
            array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 2.0)]),
            array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 2.0), inputs[0]]),
            array_ir_scalar_program(2, |builder, inputs| vec![array_ir_scaled(builder, inputs[1], 3.0)]),
        ]
    }

    /// Batches one array IR region-carrying payload over a single input mapped along axis zero and returns the
    /// rendered staged program together with the result's [`BatchAxis`].
    fn batched_array_ir_payload(operation: ArrayOperation<Array>, regions: Vec<TestProgram>) -> (String, BatchAxis) {
        let extent = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TracingContext::<TestValue, TestOperation>::new();
        let extent_input = trace.input(DimensionType::from(extent.clone()).into());
        let mapped = trace.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), extent_input);
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

    /// Builds an array IR program holding one `composite` call with a JVP rule over a scalar input.
    fn array_ir_custom_jvp_program() -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let regions = array_ir_custom_jvp_regions()
            .into_iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output = builder
            .add_instruction(
                ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
                regions,
                vec![input],
                None,
            )
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a shape-polymorphic program whose dimension product supplies the same explicit extent to a reshape
    /// and a broadcast. The stored multiplication remains visible until staging or partial evaluation folds it.
    fn array_ir_shared_extent_program() -> TestProgram {
        let extent_type = DimensionType::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));

        // Build one stored program in which ordinary dimension arithmetic supplies explicit reshape and broadcast
        // inputs. The repeated extent edge deliberately feeds both shape operations.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(input_type.into());
        let extent = builder.add_input(extent_type.clone().into());
        let one_value = DimensionValue::constant(1).unwrap();
        let one_type = one_value.r#type().into_owned();
        let one = builder.add_constant(ArrayIrValue::Dimension(one_value));
        let repeated_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&extent_type, &one_type).unwrap()),
                Vec::new(),
                vec![extent, one],
                None,
            )
            .unwrap()[0];
        let two = builder
            .add_instruction(
                DimensionOperation::Add(DimensionAddOperation::new(&one_type, &one_type).unwrap()),
                Vec::new(),
                vec![one, one],
                None,
            )
            .unwrap()[0];
        let reshaped = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, one, repeated_extent], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                DynamicBroadcastOperation::new(vec![0, 1]),
                Vec::new(),
                vec![reshaped, two, repeated_extent],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    #[test]
    fn test_array_operation_identity() {
        assert_identity::<ArrayOperation<Array>>();

        // Operations of the same variant with equal payloads are equal and hash identically, so they work as map keys.
        let add = ArrayOperation::<Array>::Add(AddOperation::new());
        assert_eq!(add, ArrayOperation::Add(AddOperation::new()));
        assert_eq!(hash_of(&add), hash_of(&ArrayOperation::<Array>::Add(AddOperation::new())));
        assert_ne!(add, ArrayOperation::Sub(SubOperation::new()));
        let sum = ArrayOperation::<Array>::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum));
        let same_sum = ArrayOperation::<Array>::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum));
        assert_eq!(sum, same_sum);
        assert_eq!(hash_of(&sum), hash_of(&same_sum));
        let operations = HashMap::from([(sum.clone(), "sum")]);
        assert_eq!(operations.get(&same_sum), Some(&"sum"));
        assert_eq!(operations.get(&add), None);

        // Payload attributes participate in the identity.
        assert_ne!(sum, ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)));
        assert_ne!(sum, ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Product)));

        // Constant payloads follow literal identity rather than IEEE value equality.
        let constant =
            |value: f64| ArrayOperation::<Array>::Constant(ConstantOperation::new(Array::scalar(value).unwrap()));
        assert_eq!(constant(f64::NAN), constant(f64::NAN));
        assert_eq!(hash_of(&constant(f64::NAN)), hash_of(&constant(f64::NAN)));
        assert_ne!(constant(-0.0), constant(0.0));
    }

    #[test]
    fn test_array_operations() {
        // The bundle is satisfied exactly when every member capability is, so instantiating this function is a
        // compile-time assertion that the listed value families implement the complete `ArrayOperation` surface in
        // their own universe, including nested tracers over both homogeneous and array IR contexts.
        fn requires_array_operations<V: ArrayOperations>() {}

        requires_array_operations::<Array>();
        requires_array_operations::<ArrayIrValue<Array>>();
        requires_array_operations::<Tracer<TracingContext<Array, ArrayOperation<Array>>>>();
        requires_array_operations::<Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>();
        requires_array_operations::<LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>>();
        requires_array_operations::<LinearizationTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>();
        requires_array_operations::<BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>>();
        requires_array_operations::<
            BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatchingPolicy>,
        >();
        requires_array_operations::<
            BatchingTracer<
                DifferentiationContext<
                    PartialEvaluationContext<TracingContext<Array, ArrayOperation<Array>>>,
                    PartitionedDifferentiationPolicy,
                >,
                ArrayBatchingPolicy,
            >,
        >();
        requires_array_operations::<
            BatchingTracer<
                DifferentiationContext<
                    PartialEvaluationContext<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>,
                    PartitionedDifferentiationPolicy,
                >,
                ArrayIrBatchingPolicy,
            >,
        >();
    }

    #[test]
    fn test_dimension_operation_identity() {
        assert_identity::<DimensionOperation<DimensionValue>>();

        // Operations of the same variant with equal payloads are equal and hash identically.
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        let add =
            DimensionOperation::<DimensionValue>::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap());
        let same_add =
            DimensionOperation::<DimensionValue>::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap());
        assert_eq!(add, same_add);
        assert_eq!(hash_of(&add), hash_of(&same_add));
        assert_ne!(add, DimensionOperation::Mul(DimensionMulOperation::new(&left_type, &right_type).unwrap()));
        assert_ne!(add, DimensionOperation::Add(DimensionAddOperation::new(&right_type, &left_type).unwrap()));

        // Constant payloads follow the dimension literal identity, which is the literal's extent.
        let constant = DimensionOperation::<DimensionValue>::Constant(ConstantOperation::new(
            DimensionValue::constant(3).unwrap(),
        ));
        let same_constant = DimensionOperation::<DimensionValue>::Constant(ConstantOperation::new(
            DimensionValue::constant(3).unwrap(),
        ));
        assert_eq!(constant, same_constant);
        assert_eq!(hash_of(&constant), hash_of(&same_constant));
        assert_ne!(
            constant,
            DimensionOperation::Constant(ConstantOperation::new(DimensionValue::constant(4).unwrap())),
        );
        let operations = HashMap::from([(add, "addition"), (constant, "constant")]);
        assert_eq!(operations.get(&same_add), Some(&"addition"));
        assert_eq!(operations.get(&same_constant), Some(&"constant"));
        assert_eq!(
            operations.get(&DimensionOperation::Mul(DimensionMulOperation::new(&left_type, &right_type).unwrap())),
            None,
        );
    }

    #[test]
    fn test_dimension_operation_staging_operators() {
        let left_type = DimensionType::new("left", DimensionBounds::new(3, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(4)).unwrap());
        let (output_type, program) = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| {
                let sum = left.clone() + right.clone();
                let product = sum * right.clone();
                let difference = product - left.clone();
                let quotient = difference / right.clone();
                Ok(quotient % left)
            },
            (left_type.clone(), right_type.clone()),
        )
        .unwrap();

        assert_eq!(output_type.bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(
            program
                .interpret((DimensionValue::new(left_type, 7).unwrap(), DimensionValue::new(right_type, 3).unwrap(),))
                .unwrap()
                .extent(),
            0,
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<left ∈ [3, 9)>, %1:dimension<right ∈ [1, 4)> .
                let %2:dimension<left + right ∈ [4, 12)> = dimension_add %0 %1
                    %3:dimension<left + right * right ∈ [4, 34)> = dimension_mul %2 %1
                    %4:dimension<left + right * right - left ∈ [0, 31)> = \
                        dimension_sub [requires_runtime_assertion=true] %3 %0
                    %5:dimension<left + right * right - left / right ∈ [0, 31)> = dimension_div %4 %1
                    %6:dimension<left + right * right - left / right % left ∈ [0, 8)> = dimension_rem %5 %0
                in (%6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dimension_operation_staging_operator_error() {
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(2)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(3, Some(5)).unwrap());
        let result = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| Ok(left - right),
            (left_type, right_type),
        );

        let Err(error) = result else {
            panic!("expected impossible subtraction bounds to fail tracing");
        };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left >= right is impossible from declared bounds".to_string(),
            }),
        );
    }

    #[test]
    fn test_dimension_operations() {
        // The bundle is satisfied exactly when every member capability is, so instantiating this function is a
        // compile-time assertion that the listed value families implement the complete `DimensionOperation` surface.
        fn requires_dimension_operations<V: DimensionOperations>() {}

        requires_dimension_operations::<DimensionValue>();
        requires_dimension_operations::<Tracer<TracingContext<DimensionValue, DimensionOperation<DimensionValue>>>>();

        // `ArrayIrOperations` pins this bundle on its first-class-dimension member profile, so the dimension member
        // projected out of every canonical array IR value must satisfy it as well.
        requires_dimension_operations::<<ArrayIrValue<Array> as ValueProjection<DimensionType>>::Projected>();
        requires_dimension_operations::<
            <Tracer<TracingContext<TestValue, TestOperation>> as ValueProjection<DimensionType>>::Projected,
        >();
    }

    #[test]
    fn test_array_ir_operation_identity() {
        assert_identity::<ArrayIrOperation<Array>>();

        // Operations of the same variant with equal payloads are equal and hash identically.
        let add = ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new()));
        let same_add = ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new()));
        assert_eq!(add, same_add);
        assert_eq!(hash_of(&add), hash_of(&same_add));
        assert_ne!(add, ArrayIrOperation::from(ArrayOperation::Mul(MulOperation::new())));
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        let dimension_add = ArrayIrOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&left_type, &right_type).unwrap(),
        ));
        let same_dimension_add = ArrayIrOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&left_type, &right_type).unwrap(),
        ));
        assert_eq!(dimension_add, same_dimension_add);
        assert_eq!(hash_of(&dimension_add), hash_of(&same_dimension_add));
        assert_ne!(dimension_add, add);

        // Region-carrying marker payloads are identical within their family; their regions live on the instruction.
        let condition = ArrayIrOperation::<Array>::Condition(ConditionOperation::new());
        let same_condition = ArrayIrOperation::<Array>::Condition(ConditionOperation::new());
        assert_eq!(condition, same_condition);
        assert_eq!(hash_of(&condition), hash_of(&same_condition));
        assert_ne!(condition, add);

        // The identity distinguishes encodings, so a mixed zero constructor differs from the homogeneous one.
        let zero_type = ArrayType::scalar(DataType::F32);
        assert_ne!(
            ArrayIrOperation::<Array>::Zero(ZeroOperation::new(zero_type.clone())),
            ArrayIrOperation::Array(ArrayOperation::Zero(ZeroOperation::new(zero_type))),
        );

        // Constant payloads nested in the array member follow literal identity.
        let constant = |value: f64| {
            ArrayIrOperation::<Array>::Array(ArrayOperation::Constant(ConstantOperation::new(
                Array::scalar(value).unwrap(),
            )))
        };
        assert_eq!(constant(f64::NAN), constant(f64::NAN));
        assert_eq!(hash_of(&constant(f64::NAN)), hash_of(&constant(f64::NAN)));
        assert_ne!(constant(-0.0), constant(0.0));
        let operations = HashMap::from([(add, "array"), (dimension_add, "dimension"), (condition, "condition")]);
        assert_eq!(operations.get(&same_add), Some(&"array"));
        assert_eq!(operations.get(&same_dimension_add), Some(&"dimension"));
        assert_eq!(operations.get(&same_condition), Some(&"condition"));
        assert_eq!(operations.get(&ArrayIrOperation::<Array>::from(ArrayOperation::Mul(MulOperation::new()))), None);
    }

    #[test]
    fn test_array_ir_operation_member_kinds() {
        /// Member-kind signature of one [`ArrayIrOperation`] variant, in terms of which member kinds it consumes and
        /// produces as *values*.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        enum MemberKindSignature {
            /// Array-only boundary through the [`ArrayType`] operation projection or an explicit direct carrier.
            ArrayToArray,

            /// Homogeneous first-class-dimension boundary, reached through the [`DimensionType`] projection.
            DimensionToDimension,

            /// Converts a first-class dimension value into ordinary array data.
            DimensionToArrayGateway,

            /// Converts ordinary array data into a first-class dimension value.
            ArrayToDimensionGateway,

            /// Creates a new reference root initialized from ordinary array data.
            ArrayToReference,

            /// Reads or consumes ordinary array data from a reference.
            ReferenceToArray,

            /// Replaces a reference from an ordinary array and returns the previous ordinary array value.
            ReferenceAndArrayToArray,

            /// Updates a reference from an ordinary array without producing a value.
            ReferenceAndArrayToUnit,

            /// Consumes first-class dimensions as extents (or reads an array's extent as a dimension) without ever
            /// converting a value from one member kind into the other.
            DimensionMixed,

            /// Checks a Boolean array condition with optional scalar array or dimension observations and no outputs.
            AssertionToUnit,

            /// Returns each input at another type of the same member kind, never converting between member kinds.
            KindPreserving,

            /// Forwards array IR regions whose bodies may carry every admitted member kind.
            RegionForwarding,
        }

        macro_rules! check_member_kinds {
            // One case declares a variant's fixture and category, generating both exhaustive dispatch and coverage.
            ($($variant:ident => ($fixture:expr, $signature:ident $(,)?)),+ $(,)?) => {{
                /// Classifies every admitted array IR operation variant from the fixture list.
                #[deny(unreachable_patterns)]
                fn member_kind_signature(operation: &ArrayIrOperation<Array>) -> MemberKindSignature {
                    match operation {
                        $(ArrayIrOperation::$variant(_) => MemberKindSignature::$signature,)+
                    }
                }

                let operations: Vec<ArrayIrOperation<Array>> = vec![
                    $({
                        let operation: ArrayIrOperation<Array> = $fixture;
                        assert!(
                            matches!(&operation, ArrayIrOperation::$variant(_)),
                            "fixture for `{}` has the wrong variant",
                            stringify!($variant),
                        );
                        operation
                    },)+
                ];
                let signatures = operations.iter().map(member_kind_signature).collect::<Vec<_>>();

                // Exhaustive dispatch forbids omissions, and the denied unreachable-pattern lint forbids duplicates.
                assert_eq!(operations.len(), 40);
                assert_eq!(
                    signatures
                        .iter()
                        .filter(|signature| **signature == MemberKindSignature::DimensionToArrayGateway)
                        .count(),
                    2,
                );
                assert_eq!(
                    signatures
                        .iter()
                        .filter(|signature| **signature == MemberKindSignature::ArrayToDimensionGateway)
                        .count(),
                    1,
                );
                assert_eq!(
                    signatures.iter().filter(|signature| **signature == MemberKindSignature::AssertionToUnit).count(),
                    1,
                );
                assert_eq!(
                    signatures.iter().filter(|signature| **signature == MemberKindSignature::KindPreserving).count(),
                    2,
                );
                assert_eq!(
                    signatures
                        .iter()
                        .filter(|signature| **signature == MemberKindSignature::ArrayToReference)
                        .count(),
                    1,
                );
                assert_eq!(
                    signatures
                        .iter()
                        .filter(|signature| **signature == MemberKindSignature::ReferenceToArray)
                        .count(),
                    2,
                );
                assert_eq!(
                    signatures
                        .iter()
                        .filter(|signature| **signature == MemberKindSignature::ReferenceAndArrayToArray)
                        .count(),
                    1,
                );
                assert_eq!(
                    signatures
                        .iter()
                        .filter(|signature| **signature == MemberKindSignature::ReferenceAndArrayToUnit)
                        .count(),
                    3,
                );
            }};
        }

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let first_type = DimensionType::from(first.clone());
        let second_type = DimensionType::from(second.clone());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(first.clone())]));
        let array_ir_rules = CustomRuleRegistration::new(CustomRuleDefinition::<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
        >::new("ir"));
        let member_rules =
            CustomRuleRegistration::new(CustomRuleDefinition::<Array, ArrayOperation<Array>>::new("array"));
        let dynamic_integer_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(first.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        // This single declaration-order case list drives the exhaustive classifier and matching fixtures together.
        // Dimension comparison and conversion are the two dimension-to-data gateways; scalar conversion is the sole
        // data-to-dimension gateway. Other signatures preserve member kinds, read extents, or forward regions.
        check_member_kinds!(
            Zero => (
                ArrayIrOperation::Zero(ZeroOperation::new(dynamic_type.clone())),
                DimensionMixed,
            ),
            One => (
                ArrayIrOperation::One(OneOperation::new(dynamic_type.clone())),
                DimensionMixed,
            ),
            Iota => (
                ArrayIrOperation::Iota(IotaOperation::new(dynamic_integer_type, 0).unwrap()),
                DimensionMixed,
            ),
            Array => (
                ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                ArrayToArray,
            ),
            Dimension => (
                ArrayIrOperation::Dimension(DimensionOperation::Add(
                    DimensionAddOperation::new(&first_type, &second_type).unwrap(),
                )),
                DimensionToDimension,
            ),
            DimensionSize => (
                ArrayIrOperation::DimensionSize(DimensionSizeOperation::new(&dynamic_type, 0).unwrap()),
                DimensionMixed,
            ),
            DimensionFromScalar => (
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(second.clone().into())),
                ArrayToDimensionGateway,
            ),
            DimensionToScalar => (
                ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation),
                DimensionToArrayGateway,
            ),
            ReferenceNew => (
                ArrayIrOperation::ReferenceNew(ReferenceNewOperation::new()),
                ArrayToReference,
            ),
            ReferenceRead => (
                ArrayIrOperation::ReferenceRead(ReferenceReadOperation::new()),
                ReferenceToArray,
            ),
            ReferenceWrite => (
                ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                ReferenceAndArrayToUnit,
            ),
            ReferenceSwap => (
                ArrayIrOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                ReferenceAndArrayToArray,
            ),
            ReferenceAddUpdate => (
                ArrayIrOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                ReferenceAndArrayToUnit,
            ),
            ReferenceAtomicAddUpdate => (
                ArrayIrOperation::ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation::new()),
                ReferenceAndArrayToUnit,
            ),
            ReferenceFreeze => (
                ArrayIrOperation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                ReferenceToArray,
            ),
            ParallelAllGather => (
                ArrayIrOperation::ParallelAllGather(ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled),
                    ParallelAllGatherOutputVariance::Varying,
                )),
                DimensionMixed,
            ),
            ParallelSumScatter => (
                ArrayIrOperation::ParallelSumScatter(ParallelSumScatterOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled),
                )),
                DimensionMixed,
            ),
            ParallelAllToAll => (
                ArrayIrOperation::ParallelAllToAll(ParallelAllToAllOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    0,
                    CollectiveOptions::new(CollectiveMode::Untiled),
                )),
                DimensionMixed,
            ),
            ParallelRaggedAllToAll => (
                ArrayIrOperation::ParallelRaggedAllToAll(ParallelRaggedAllToAllOperation::new("x".to_string(), 2)),
                ArrayToArray,
            ),
            Compare => (
                ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                DimensionToArrayGateway,
            ),
            Reshape => (
                ArrayIrOperation::Reshape(DynamicReshapeOperation::new()),
                DimensionMixed,
            ),
            Broadcast => (
                ArrayIrOperation::Broadcast(DynamicBroadcastOperation::new(vec![0])),
                DimensionMixed,
            ),
            Pad => (
                ArrayIrOperation::Pad(PadOperation::new(vec![0], vec![0], vec![0]).unwrap()),
                DimensionMixed,
            ),
            Concatenate => (
                ArrayIrOperation::Concatenate(
                    ConcatenateOperation::<ArrayIrType>::new(
                        0,
                        &[
                            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into(),
                            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                            DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                        ],
                    )
                    .unwrap(),
                ),
                DimensionMixed,
            ),
            DynamicSlice => (
                ArrayIrOperation::DynamicSlice(DynamicSliceOperation::<ArrayIrType>::from_rank(1)),
                DimensionMixed,
            ),
            ShardMap => (
                {
                    let body = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new()
                        .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(Vec::new(), Vec::new(), Vec::new())
                        .unwrap();
                    let shard_map = ShardMap::new(
                        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
                        Vec::new(),
                        Vec::new(),
                        vec!["x".to_string()],
                    )
                    .unwrap();
                    ArrayIrOperation::ShardMap(ShardMapOperation::from_program(&body, Vec::new(), shard_map).unwrap())
                },
                RegionForwarding,
            ),
            Condition => (
                ArrayIrOperation::Condition(ConditionOperation::new()),
                RegionForwarding,
            ),
            While => (
                ArrayIrOperation::While(WhileOperation::new()),
                RegionForwarding,
            ),
            Scan => (
                ArrayIrOperation::Scan(ScanOperation::new(1, 4)),
                RegionForwarding,
            ),
            Rematerialize => (
                ArrayIrOperation::Rematerialize(RematerializeOperation::new(ResidualPolicyReference::new(
                    NothingSavable,
                ))),
                RegionForwarding,
            ),
            LinearCall => (
                ArrayIrOperation::LinearCall(LinearCallOperation::new(0)),
                RegionForwarding,
            ),
            RngBitGenerator => (
                ArrayIrOperation::RngBitGenerator(RngBitGeneratorOperation::new(
                    RandomAlgorithm::ThreeFry,
                    dynamic_type.clone(),
                )),
                DimensionMixed,
            ),
            Assert => (
                ArrayIrOperation::Assert(AssertOperation::new("dimensions must match")),
                AssertionToUnit,
            ),
            Guard => (
                ArrayIrOperation::Guard(GuardOperation::new("region", vec![dynamic_type.clone().into()])),
                KindPreserving,
            ),
            Cast => (
                ArrayIrOperation::Cast(
                    CastOperation::unchecked(dynamic_type.clone().into(), vec![dynamic_type.clone().into()]).unwrap(),
                ),
                KindPreserving,
            ),
            CustomCall => (
                ArrayIrOperation::CustomCall(
                    CustomCallOperation::new("ryft.test.identity", vec![scalar_type.clone()]).into(),
                ),
                DimensionMixed,
            ),
            Composite => (
                ArrayIrOperation::Composite(CompositeOperation::new(array_ir_rules.reference())),
                RegionForwarding,
            ),
            CompositeTranspose => (
                ArrayIrOperation::CompositeTranspose(CompositeTransposeOperation::new(
                    array_ir_rules.reference(),
                    0,
                    Vec::new(),
                    Vec::new(),
                )),
                RegionForwarding,
            ),
            LiftedComposite => (
                ArrayIrOperation::from(ArrayOperation::Composite(CompositeOperation::new(member_rules.reference()))),
                RegionForwarding,
            ),
            LiftedCompositeTranspose => (
                ArrayIrOperation::from(ArrayOperation::CompositeTranspose(CompositeTransposeOperation::new(
                    member_rules.reference(),
                    0,
                    Vec::new(),
                    Vec::new(),
                ))),
                RegionForwarding,
            ),
        );
    }

    #[test]
    fn test_array_ir_operation_projection() {
        fn assert_projection<T: Type, O: Operation<Type = T>, Family: OperationProjection<T, Projected = O>>() {}

        assert_projection::<ArrayType, ArrayOperation<Array>, ArrayIrOperation<Array>>();
        assert_projection::<DimensionType, DimensionOperation<DimensionValue>, ArrayIrOperation<Array>>();

        let array = ArrayOperation::<Array>::Add(AddOperation::new());
        let lifted = ArrayIrOperation::from(array.clone());
        assert!(matches!(lifted, ArrayIrOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(lifted.name(), array.name());
        assert_eq!(lifted.to_string(), array.to_string());

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
        let dimension = DimensionOperation::<DimensionValue>::Add(DimensionAddOperation::new(&left, &right).unwrap());
        let lifted = ArrayIrOperation::<Array>::from(dimension.clone());
        assert!(matches!(lifted, ArrayIrOperation::Dimension(DimensionOperation::Add(_))));
        assert_eq!(lifted.name(), dimension.name());
        assert_eq!(lifted.to_string(), dimension.to_string());

        // Direct dimension payload conversion reaches the same canonical member family.
        let product = ArrayIrOperation::<Array>::from(DimensionMulOperation::new(&left, &right).unwrap());
        assert!(matches!(product, ArrayIrOperation::Dimension(DimensionOperation::Mul(_))));
    }

    #[test]
    fn test_array_ir_operation_infer_output_types() {
        let array = ArrayType::scalar(DataType::F32);
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
        let reference = ArrayIrType::Reference(ReferenceType::new(array.clone()));
        check_operation_type_inference!(
            operation = ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new())),
            cases = [
                {
                    input_types = [array.clone().into(), array.clone().into()],
                    output_types = [array.clone().into()],
                },
                {
                    input_types = [left.clone().into(), right.clone().into()],
                    error = "expected array type but got dimension type",
                },
                {
                    input_types = [array.clone().into(), right.clone().into()],
                    error = "expected array type but got dimension type",
                },
                {
                    input_types = [reference.clone(), array.clone().into()],
                    error = "expected array type but got reference type",
                },
                {
                    input_types = [array.clone().into(), reference.clone()],
                    error = "expected array type but got reference type",
                },
            ],
        );

        // The wrapper delegates inference after projecting every input, and lifts the result back into array IR.
        let addition = ArrayIrOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&left, &right).unwrap(),
        ));
        let output_types = addition.infer_output_types(&[left.clone().into(), right.clone().into()], &[]).unwrap();
        let [ArrayIrType::Dimension(result)] = output_types.as_slice() else {
            panic!("expected one first-class dimension output");
        };
        // Each inference defines a fresh nominal result, so compare its authoritative bounds rather than another ID.
        assert_eq!(result.bounds(), DimensionBounds::new(2, Some(17)).unwrap());
        check_operation_type_inference!(
            operation = addition,
            cases = [
                {
                    input_types = [array.clone().into(), array.clone().into()],
                    error = "expected dimension type but got array type",
                },
                {
                    input_types = [left.into(), reference],
                    error = "expected dimension type but got reference type",
                },
            ],
        );

        // A mixed signature remains in the enclosing universe instead of projecting its array-to-dimension boundary.
        let extent = DimensionVariable::new("extent", bounds);
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        check_operation_type_inference!(
            operation = ArrayIrOperation::<Array>::from(DimensionSizeOperation::new(&dynamic, 0).unwrap()),
            cases = [{
                input_types = [dynamic.into()],
                output_types = [DimensionType::from(extent).into()],
            }],
        );
    }

    #[test]
    fn test_array_ir_operation_region_interfaces() {
        let array = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input_types = [ArrayType::scalar(DataType::Boolean).into(), array.clone()];
        let interface = RegionInterface::new(
            vec![array.clone()],
            vec![array.clone()],
            EffectClasses::single(EffectClass::OrderedIo),
        );
        let interfaces = [interface.clone(), interface];
        let payload = ConditionOperation::<ArrayIrType>::new();
        let operation = ArrayIrOperation::<Array>::Condition(payload.clone());

        assert_eq!(operation.region_slots(), payload.region_slots());
        assert_eq!(operation.infer_output_types(&input_types, &interfaces), Ok(vec![array.clone()]));
        assert_eq!(
            operation.infer_region_input_types(&input_types, &interfaces),
            payload.infer_region_input_types(&input_types, &interfaces),
        );
        assert_eq!(operation.input_region_provenance(0, 0), payload.input_region_provenance(0, 0));
        assert_eq!(operation.output_region_provenance(0), payload.output_region_provenance(0));
    }

    #[test]
    fn test_array_ir_operation_effects() {
        // A statically proven mixed concatenate is pure. The derived dispatcher must read that payload classification
        // rather than declaring the array IR family effectful.
        let concatenate = ConcatenateOperation::<ArrayIrType>::new(
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
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);

        // A dynamic axis sum remains an ordered assertion and reaches the outer family unchanged.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(9)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::positive(Some(12)).unwrap());
        let concatenate = ConcatenateOperation::<ArrayIrType>::new(
            0,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])).into(),
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                DimensionType::from(result).into(),
            ],
        )
        .unwrap();
        let operation = ArrayIrOperation::<Array>::Concatenate(concatenate.clone());
        assert_eq!(operation.effects(), concatenate.effects());
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));

        // Assertion effects reach the array IR family unchanged. Predicate proofs belong to comparison and
        // assertion staging rather than the operation's unconditional effect declaration.
        let assertion = AssertOperation::<ArrayIrType>::new("dimensions must match");
        let operation = ArrayIrOperation::<Array>::from(assertion.clone());
        assert_eq!(operation.effects(), assertion.effects());
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_array_ir_operation_rename_type_identities() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();

        // Identity-bearing payloads of a mixed constructor are renamed in the enclosing family.
        let zero = ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(source.clone())]),
        ))));
        let ArrayIrOperation::Zero(zero) = zero.rename_type_identities(&renaming).unwrap() else {
            panic!("expected a renamed mixed zero operation");
        };
        assert_eq!(zero.r#type().shape().dimensions(), &[Dimension::Dynamic(target.clone())]);

        // Projected dimension payloads use the same renaming while preserving unrelated identities.
        let right = DimensionType::new("right", bounds);
        let addition = ArrayIrOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&DimensionType::from(source), &right).unwrap(),
        ));
        let ArrayIrOperation::Dimension(DimensionOperation::Add(addition)) =
            addition.rename_type_identities(&renaming).unwrap()
        else {
            panic!("expected a renamed dimension addition operation");
        };
        assert_eq!(addition.left_type().variable(), Some(&target));
        assert_eq!(addition.right_type(), &right);
    }

    #[test]
    fn test_array_ir_operation_type_identity_import_shared_extents() {
        let program = array_ir_shared_extent_program();
        let target_type = DimensionType::new("target", DimensionBounds::new(1, Some(5)).unwrap());
        let target_array_type = ArrayType::new(DataType::F64, Shape::new(vec![target_type.dimension().clone()]));
        let instantiated = program
            .with_instantiated_type_identities(&[target_array_type.clone().into(), target_type.clone().into()])
            .unwrap()
            .into_owned();
        assert_eq!(instantiated.input_types(), vec![target_array_type.clone().into(), target_type.clone().into()]);
        let expected_rendering = indoc! {r#"
            lambda %0:f64[target], %1:dimension<target ∈ [1, 5)> .
            let %2:dimension<1> = const 1
                %3:dimension<extent * 1 ∈ [1, 5)> = dimension_mul %1 %2
                %4:dimension<2> = dimension_add %2 %2
                %5:f64[1, extent * 1] = reshape %0 %2 %3
                %6:f64[2, extent * 1] = broadcast [output_axes=[0, 1]] %5 %4 %3
            in (%6)
        "#}
        .trim_end();
        assert_eq!(instantiated.to_string(), expected_rendering);

        // Import preserves the renamed boundary and the one arithmetic result shared by both shape consumers.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(target_array_type.into());
        let extent = builder.add_input(target_type.clone().into());
        let outputs = builder.splice_program(&instantiated, &[input, extent]).unwrap();
        let imported = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(imported.to_string(), expected_rendering);
        assert_eq!(imported.input_types(), instantiated.input_types());

        // Import freshens the internal identity. The final array must refer to that new definition, whose printed
        // name alone cannot distinguish it from the source program's dimension.
        let shared_extent_id = imported.instructions()[0].outputs()[0];
        let shared_extent_type = imported.atoms()[shared_extent_id.index()].r#type();
        let expected_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), shared_extent_type.as_dimension_type().unwrap().dimension().clone()]),
        );
        assert_eq!(imported.output_types(), vec![expected_type.into()]);
        assert_ne!(imported.output_types(), instantiated.output_types());
        assert_eq!(
            imported.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(target_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap(),)]),
        );
    }

    #[test]
    fn test_array_ir_operation_interpretation() {
        let context = EagerContext::<TestValue, TestOperation>::new();

        // Representative projected array, projected dimension, and direct mixed boundaries all bind in one context.
        assert_eq!(
            context.bind(
                ArrayOperation::Add(AddOperation::new()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![1f32, 2.0]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![3f32, 4.0]).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4f32, 6.0]).unwrap())]),
        );
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
        let output = context
            .bind(
                DimensionOperation::Add(DimensionAddOperation::new(&left, &right).unwrap()),
                Vec::new(),
                &[
                    ArrayIrValue::Dimension(DimensionValue::new(left, 3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::new(right, 4).unwrap()),
                ],
            )
            .unwrap();
        let [ArrayIrValue::Dimension(output)] = output.as_slice() else {
            panic!("expected one dimension result");
        };
        assert_eq!(output.extent(), 7);
        assert_eq!(
            context.bind(
                DynamicReshapeOperation::new(),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_operation_interpretation_shared_extents() {
        let program = array_ir_shared_extent_program();
        let extent_type = program.input_types()[1].as_dimension_type().unwrap().clone();
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap(),)]),
        );
    }

    #[test]
    fn test_array_ir_operation_staging_member_projections() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        let addition = DimensionAddOperation::new(&left_type, &right_type).unwrap();
        let (_, program) = TracingContext::<TestValue, TestOperation>::trace(
            |(array, left, right): (
                Tracer<TracingContext<TestValue, TestOperation>>,
                Tracer<TracingContext<TestValue, TestOperation>>,
                Tracer<TracingContext<TestValue, TestOperation>>,
            )| {
                let array = ValueProjection::<ArrayType>::into_projected(array)?;
                let array_sum =
                    array.domain().bind(AddOperation::new(), Vec::new(), &[array.clone(), array])?.remove(0);
                let left = ValueProjection::<DimensionType>::into_projected(left)?;
                let right = ValueProjection::<DimensionType>::into_projected(right)?;
                let dimension_sum = left.domain().bind(addition, Vec::new(), &[left.clone(), right])?.remove(0);
                Ok((array_sum.into_value(), dimension_sum.into_value()))
            },
            (
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::Dimension(left_type),
                ArrayIrType::Dimension(right_type),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:dimension<left ∈ [1, 9)>, %2:dimension<right ∈ [1, 9)> .
                let %3:f32[] = add %0 %0
                    %4:dimension<left + right ∈ [2, 17)> = dimension_add %1 %2
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_operation_staging_dimension_projection() {
        let rows = DimensionType::new("rows", DimensionBounds::new(5, Some(9)).unwrap());
        let columns = DimensionType::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(rows, columns)| {
                let rows = ValueProjection::<DimensionType>::into_projected(rows)?;
                let columns = ValueProjection::<DimensionType>::into_projected(columns)?;
                let padded = Add::add(&Mul::mul(&rows, &columns)?, &columns)?;
                let trimmed = padded.saturating_sub(&rows)?;
                Ok((
                    Div::div(&trimmed, &columns)?.into_value(),
                    Rem::rem(&trimmed, &columns)?.into_value(),
                    rows.dimension_pow(&columns)?.into_value(),
                    Sub::sub(&rows.dimension_max(&columns)?, &rows.dimension_min(&columns)?)?.into_value(),
                ))
            },
            (ArrayIrType::Dimension(rows), ArrayIrType::Dimension(columns)),
        )
        .unwrap();
        let expected = indoc! {"
            lambda %0:dimension<rows ∈ [5, 9)>, %1:dimension<columns ∈ [1, 5)> .
            let %2:dimension<rows * columns ∈ [5, 33)> = dimension_mul %0 %1
                %3:dimension<rows * columns + columns ∈ [6, 37)> = dimension_add %2 %1
                %4:dimension<max(0, rows * columns + columns - rows) ∈ [0, 32)> = saturating_sub %3 %0
                %5:dimension<max(0, rows * columns + columns - rows) / columns ∈ [0, 32)> = dimension_div %4 %1
                %6:dimension<max(0, rows * columns + columns - rows) % columns ∈ [0, 4)> = dimension_rem %4 %1
                %7:dimension<rows ^ columns ∈ [5, 4097)> = dimension_pow %0 %1
                %8:dimension<rows - columns ∈ [1, 8)> = dimension_sub %0 %1
            in (%5, %6, %7, %8)
        "};
        assert_eq!(program.to_string(), expected.trim_end());
    }

    #[test]
    fn test_array_ir_operation_staging_extent_dependencies() {
        // One representative mixed program: a dynamically shaped input is read with `dimension_size`, the extent is
        // combined with ordinary dimension arithmetic, and the derived extent is then consumed by two shape-carrying
        // operations as an explicit input.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let size_operation = DimensionSizeOperation::new(&input_type, 0).unwrap();
        let size_type = size_operation.output_type().clone();
        let sum_operation = DimensionAddOperation::new(&size_type, &size_type).unwrap();
        let sum_type = sum_operation.infer_output_types(&[size_type.clone(), size_type], &[]).unwrap().remove(0);
        let one_value = DimensionValue::constant(1).unwrap();
        let concatenate_operation = ConcatenateOperation::<ArrayIrType>::new(
            0,
            &[input_type.clone().into(), input_type.clone().into(), sum_type.into()],
        )
        .unwrap();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let one = builder.add_constant(ArrayIrValue::Dimension(one_value));
        let size = builder.add_instruction(size_operation, Vec::new(), vec![input], None).unwrap()[0];
        let sum = builder
            .add_instruction(DimensionOperation::Add(sum_operation), Vec::new(), vec![size, size], None)
            .unwrap()[0];
        let concatenated =
            builder.add_instruction(concatenate_operation, Vec::new(), vec![input, input, sum], None).unwrap()[0];
        let reshaped = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![concatenated, sum, one], None)
            .unwrap()[0];
        let program = builder
            .build::<ArrayIrValue<Array>, ArrayIrValue<Array>>(vec![reshaped], Placeholder, Placeholder)
            .unwrap();

        // Every derived extent is visible in the rendered program: one `dimension_size` read plus explicit input
        // edges into the shape-carrying operations. No shape is recovered from ambient or type-level metadata.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[rows] .
                let %1:dimension<1> = const 1
                    %2:dimension<rows ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<rows + rows ∈ [2, 9)> = dimension_add %2 %2
                    %4:f32[rows + rows] = concatenate [axis=0, requires_runtime_assertion=true] %0 %0 %3
                    %5:f32[rows + rows, 1] = reshape %4 %3 %1
                in (%5)"},
        );

        // Drift gate: an instruction output may only *reference* a dimension identity that one of its own inputs
        // carries. A rule that recovered extents from stored metadata instead of an input edge would produce an output
        // type naming an identity that reaches the instruction through no rendered edge.
        let atoms = program.atoms();
        for instruction in program.instructions() {
            let mut input_identities = Vec::new();
            for input in instruction.inputs() {
                let r#type = atoms[input.index()].r#type();
                input_identities.extend(r#type.identities().map(|(_, identity)| identity.clone()));
            }
            for output in instruction.outputs() {
                let r#type = atoms[output.index()].r#type();
                for (position, identity) in r#type.identities() {
                    assert!(
                        position != TypeIdentityPosition::Reference || input_identities.contains(identity),
                        "instruction `{}` output type {} references identity {identity} that no input carries",
                        instruction.operation().name(),
                        r#type.as_ref(),
                    );
                }
            }
        }
    }

    #[test]
    fn test_array_ir_operation_staging_shared_extents() {
        let program = array_ir_shared_extent_program();
        // Both shape operations consume the same arithmetic output, rather than independently derived extents.
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 5)> .
                let %2:dimension<1> = const 1
                    %3:dimension<extent * 1 ∈ [1, 5)> = dimension_mul %1 %2
                    %4:dimension<2> = dimension_add %2 %2
                    %5:f64[1, extent * 1] = reshape %0 %2 %3
                    %6:f64[2, extent * 1] = broadcast [output_axes=[0, 1]] %5 %4 %3
                in (%6)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_operation_staging_dimension_member_folding() {
        let value_type = DimensionType::new("value", DimensionBounds::new(2, Some(9)).unwrap());
        let one_type = DimensionValue::constant(1).unwrap().r#type().into_owned();
        let operation = DimensionMulOperation::new(&value_type, &one_type).unwrap();
        let (_, program) = TracingContext::<TestValue, TestOperation>::trace(
            |(value, one)| Ok(value.context().bind(operation, Vec::new(), &[value.clone(), one])?.remove(0)),
            (ArrayIrType::Dimension(value_type), ArrayIrType::Dimension(one_type)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [2, 9)>, %1:dimension<1> .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_operation_partial_evaluation_shared_extents() {
        let program = array_ir_shared_extent_program();
        let input_types = program.input_types();
        let extent_type = input_types[1].as_dimension_type().unwrap().clone();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        let expected = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap());

        // Unknown extents retain their boundary identity, and folding substitutes it into both shape consumers.
        let unknown = program
            .partially_evaluate(&[
                PartialValue::Unknown(input_types[0].clone()),
                PartialValue::Unknown(input_types[1].clone()),
            ])
            .unwrap();
        assert_eq!(
            unknown.program().to_string(),
            indoc! {r#"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 5)> .
                let %2:dimension<1> = const 1
                    %3:f64[1, extent] = reshape %0 %2 %1
                    %4:dimension<2> = const 2
                    %5:f64[2, extent] = broadcast [output_axes=[0, 1]] %3 %4 %1
                in (%5)
            "#}
            .trim_end(),
        );
        assert_eq!(
            unknown.interpret(&EagerContext::<TestValue, TestOperation>::new(), &[input.clone(), extent.clone()],),
            Ok(vec![expected.clone()]),
        );

        // A known extent narrows the array through an entry guard before either shape operation reads its data.
        let known = program
            .partially_evaluate(&[PartialValue::Unknown(input_types[0].clone()), PartialValue::Known(extent)])
            .unwrap();
        assert_eq!(
            known.program().to_string(),
            indoc! {r#"
                lambda %0:f64[extent], %1:dimension<3> .
                let %2:f64[3], %3:dimension<3> = guard [
                    boundary="partial evaluation",
                    declared_types=[f64[extent], dimension<extent ∈ [1, 5)>],
                ] %0 %1
                    %4:dimension<1> = const 1
                    %5:f64[1, 3] = reshape %2 %4 %1
                    %6:dimension<2> = const 2
                    %7:f64[2, 3] = broadcast [output_axes=[0, 1]] %5 %6 %1
                in (%7)
            "#}
            .trim_end(),
        );
        assert_eq!(
            known.interpret(&EagerContext::<TestValue, TestOperation>::new(), std::slice::from_ref(&input)),
            Ok(vec![expected]),
        );
    }

    #[test]
    fn test_array_ir_operation_batching_shared_extents() {
        let program = array_ir_shared_extent_program();
        let extent_type = program.input_types()[1].as_dimension_type().unwrap().clone();
        let trace = TracingContext::<TestValue, TestOperation>::new();
        let packed = trace.input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), extent_type.dimension().clone()]))
                .into(),
        );
        let extent = trace.input(extent_type.clone().into());
        let batch_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);

        // Only array data gain a leading physical batch axis; the shared shape value remains replicated.
        let output = program
            .interpret_in_context(
                &context,
                vec![
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(packed, BatchAxis::new(0)).unwrap()),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let batched = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![output.into_batch().into_value().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            batched.to_string(),
            indoc! {r#"
                lambda %0:f64[2, extent], %1:dimension<extent ∈ [1, 5)> .
                let %2:dimension<2> = const 2
                    %3:dimension<1> = const 1
                    %4:dimension<2> = dimension_add %3 %3
                    %5:f64[2, 1, extent] = reshape [requires_runtime_assertion=false] %0 %2 %3 %1
                    %6:f64[2, 2, extent] = broadcast [output_axes=[0, 1, 2]] %5 %2 %4 %1
                in (%6)
            "#}
            .trim_end(),
        );
        assert_eq!(
            batched.interpret(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::F64, [2, 2, 3]),
                    &[1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn test_array_ir_operation_batching_composite_jvp() {
        // The array IR carrier batches both regions structurally and threads the first-class mapped extent into
        // them as one additional leading non-differentiated input of the batched call.
        let (program, batch_axis) = batched_array_ir_payload(
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
            array_ir_custom_jvp_regions(),
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = composite [non_differentiated_count=1] %0 %1 [
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
    fn test_array_ir_operation_batching_composite_vjp() {
        // The backward region receives the threaded extent ahead of its residuals, and its result cotangents align
        // with the differentiated inputs only.
        let (program, batch_axis) = batched_array_ir_payload(
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true)),
            array_ir_custom_vjp_regions(),
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = composite [non_differentiated_count=1] %0 %1 [
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
    fn test_array_ir_operation_batching_composite() {
        // A call with both a JVP rule and reverse-mode rules rebuilds all four regions around the threaded extent,
        // keeping every rule attached in region order.
        let mut regions = array_ir_custom_jvp_regions();
        regions.extend(array_ir_custom_vjp_regions().into_iter().skip(1));
        let (program, batch_axis) = batched_array_ir_payload(
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, true)),
            regions,
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = composite [non_differentiated_count=1] %0 %1 [
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
    fn test_array_ir_operation_batching_rematerialize() {
        // The body is rebuilt around the threaded extent, which becomes a leading body input, keeping the
        // rematerialization boundary intact.
        let (program, batch_axis) = batched_array_ir_payload(
            ArrayOperation::Rematerialize(RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable))),
            vec![array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 2.0)])],
        );
        assert_eq!(batch_axis, BatchAxis::new(0));
        assert_eq!(
            program,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[batch] = rematerialize %0 %1 [
                    body={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[] = const 2.0
                            %3:f64[batch] = broadcast [output_axes=[]] %2 %0
                            %4:f64[batch] = mul %1 %3
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_operation_batching_linear_call() {
        // Both regions use the known scalar residual; batching threads its symbolic extent as another residual.
        let multiply = array_ir_scalar_program(2, |builder, inputs| {
            vec![
                builder
                    .add_instruction(
                        ArrayOperation::Mul(MulOperation::new()),
                        Vec::new(),
                        vec![inputs[0], inputs[1]],
                        None,
                    )
                    .unwrap()[0],
            ]
        });
        let extent_type = DimensionType::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TracingContext::<TestValue, TestOperation>::new();
        let extent = trace.input(extent_type.clone().into());
        let mapped =
            trace.input(ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()])).into());
        let residual = trace.constant(ArrayIrValue::Array(Array::scalar(3.0).unwrap()));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), extent);
        let residual =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(residual, BatchAxis::replicated()).unwrap());
        let mapped = BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0)).unwrap());
        let operation = LinearCallOperation::new(1);
        let output = context
            .bind(ArrayOperation::LinearCall(operation.clone()), vec![multiply.clone(), multiply], &[residual, mapped])
            .unwrap()
            .remove(0);
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:f64[] = const 3.0
                    %3:f64[batch] = linear_call [residual_count=2] %0 %2 %1 [
                        forward={
                            lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[], %2:f64[batch] .
                            let %3:f64[batch] = broadcast [output_axes=[]] %1 %0
                                %4:f64[batch] = mul %3 %2
                            in (%4)
                        },
                        transpose={
                            lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[], %2:f64[batch] .
                            let %3:f64[batch] = broadcast [output_axes=[]] %1 %0
                                %4:f64[batch] = mul %3 %2
                            in (%4)
                        },
                    ]
                in (%3)
            "}
            .trim_end(),
        );
        let [instruction] = program.instructions() else {
            panic!("expected one batched linear call");
        };
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 2).unwrap());
        let linear = ArrayIrValue::Array(Array::vector(vec![2.0f64, 5.0]).unwrap());
        assert_eq!(
            program.interpret(vec![extent.clone(), linear]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![6.0f64, 15.0]).unwrap())]),
        );
        let transpose = program.region_ref(instruction.regions()[1]).unwrap().to_program();
        assert_eq!(
            transpose.interpret(vec![
                extent,
                ArrayIrValue::Array(Array::scalar(3.0).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0f64, 4.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![3.0f64, 12.0]).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_operation_differentiation_homogeneous() {
        // Family dispatch projects multiplication while preserving the primal and tangent input order.
        let program = array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 3.0)]);
        let differentiated = program.jvp().unwrap();
        assert_eq!(
            differentiated.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const 3.0
                    %3:f64[] = mul %0 %2
                    %4:f64[] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            differentiated.interpret(vec![
                ArrayIrValue::Array(Array::scalar(2.0).unwrap()),
                ArrayIrValue::Array(Array::scalar(4.0).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(6.0).unwrap()),
                ArrayIrValue::Array(Array::scalar(12.0).unwrap())
            ]),
        );
    }

    #[test]
    fn test_array_ir_operation_differentiation_custom_jvp() {
        // The tangent scale of three distinguishes the attached JVP rule from the primal scale of two.
        let program = array_ir_custom_jvp_program();

        let differentiated = program.jvp().unwrap();
        assert_eq!(
            differentiated.to_string(),
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
        assert_eq!(
            differentiated.interpret(vec![
                ArrayIrValue::Array(Array::scalar(5.0).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(10.0).unwrap()),
                ArrayIrValue::Array(Array::scalar(3.0).unwrap())
            ]),
        );
    }

    #[test]
    fn test_array_ir_operation_differentiation_shared_extents() {
        let program = array_ir_shared_extent_program();
        let extent_type = program.input_types()[1].as_dimension_type().unwrap().clone();

        // The explicit dimensions have structural-zero tangents while both shape operations replay on live data.
        let differentiated = program.jvp().unwrap();
        assert_eq!(
            differentiated.to_string(),
            indoc! {r#"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 5)>, %2:f64[extent] .
                let %3:dimension<1> = const 1
                    %4:dimension<2> = dimension_add %3 %3
                    %5:f64[1, extent] = reshape [requires_runtime_assertion=false] %0 %3 %1
                    %6:f64[1, extent] = linear_call [residual_count=2] %3 %1 %2 [
                        forward={
                            lambda %0:dimension<1>, %1:dimension<extent ∈ [1, 5)>, %2:f64[extent] .
                            let %3:f64[1, extent] = reshape [requires_runtime_assertion=false] %2 %0 %1
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<1>, %1:dimension<extent ∈ [1, 5)>, %2:f64[1, extent] .
                            let %3:f64[extent] = reshape [requires_runtime_assertion=false] %2 %1
                            in (%3)
                        },
                    ]
                    %7:f64[2, extent] = broadcast [output_axes=[0, 1]] %5 %4 %1
                    %8:f64[2, extent] = linear_call [residual_count=2] %4 %1 %6 [
                        forward={
                            lambda %0:dimension<2>, %1:dimension<extent ∈ [1, 5)>, %2:f64[1, extent] .
                            let %3:f64[2, extent] = broadcast [output_axes=[0, 1]] %2 %0 %1
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<2>, %1:dimension<extent ∈ [1, 5)>, %2:f64[2, extent] .
                            let %3:f64[extent] = reduce [kind=sum, axes=[0]] %2
                                %4:dimension<1> = constant [value=1]
                                %5:f64[1, extent] = reshape %3 %4 %1
                            in (%5)
                        },
                    ]
                in (%7, %8)
            "#}
            .trim_end(),
        );
        assert_eq!(
            differentiated.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![4.0f64, 5.0, 6.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![4.0f64, 5.0, 6.0, 4.0, 5.0, 6.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_ir_operation_linearization_homogeneous() {
        // Reverse mode composes the projected JVP-for-transpose and transposition rules with a known literal factor.
        let program = array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 3.0)]);
        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 3.0
                    %2:f64[] = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 3.0
                    %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 3.0
                    %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(2.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6.0).unwrap())]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![ArrayIrValue::Array(Array::scalar(4.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(12.0).unwrap())]),
        );
        assert_eq!(
            pullback.program().interpret(vec![ArrayIrValue::Array(Array::scalar(5.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(15.0).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_operation_linearization_custom_jvp() {
        // Linearization keeps the custom derivative in the tangent program and its derived pullback.
        let program = array_ir_custom_jvp_program();
        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 2.0
                    %2:f64[] = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 3.0
                    %2:f64[] = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 3.0
                    %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(5.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(10.0).unwrap())]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(3.0).unwrap())]),
        );
        assert_eq!(
            pullback.program().interpret(vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(3.0).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_operation_transposition_homogeneous() {
        // Raw transposition projects the homogeneous multiply with its literal factor held known.
        let program = array_ir_scalar_program(1, |builder, inputs| vec![array_ir_scaled(builder, inputs[0], 3.0)]);
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 3.0
                    %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![ArrayIrValue::Array(Array::scalar(5.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(15.0).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_operation_transposition_custom_jvp_primal() {
        // Raw transposition reaches the primal region before linearization replaces the custom call.
        let program = array_ir_custom_jvp_program();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 2.0
                    %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(2.0).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_operations() {
        fn requires_array_ir_operations<V: ArrayIrOperations>() {}

        requires_array_ir_operations::<ArrayIrValue<Array>>();
        requires_array_ir_operations::<Tracer<TracingContext<TestValue, TestOperation>>>();
        requires_array_ir_operations::<LinearizationTracer<EagerContext<TestValue, TestOperation>>>();

        // The bundle's member profiles make one bound enough to reach ordinary array math through projection, checked
        // dimension arithmetic, and family-native reference capabilities, with no projection bounds restated here.
        fn square_and_element_count<V: ArrayIrOperations>(value: &V) -> Result<(V, V), ProgramError> {
            // The array member carries both the fallible capability and its panicking operator sugar, so the
            // capability is named explicitly here.
            let array = ValueProjection::<ArrayType>::into_projected(value.clone())?;
            let square = ValueProjection::<ArrayType>::from_projected(Mul::mul(&array, &array)?);
            let rows = ValueProjection::<DimensionType>::into_projected(value.dimension_size(0)?)?;
            let columns = ValueProjection::<DimensionType>::into_projected(value.dimension_size(1)?)?;
            let elements = ValueProjection::<DimensionType>::from_projected(rows.mul(&columns)?);
            Ok((square, elements))
        }

        fn reference_round_trip<V: ArrayIrOperations>(value: &V) -> Result<V, ProgramError> {
            value.reference_new()?.freeze()
        }

        let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let (square, elements) = square_and_element_count(&input).unwrap();
        assert_eq!(square, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f64, 4.0, 9.0, 16.0, 25.0, 36.0]).unwrap()));
        let ArrayIrValue::Dimension(elements) = elements else {
            panic!("expected a first-class dimension member");
        };
        assert_eq!(elements.extent(), 6);
        assert_eq!(reference_round_trip(&input), Ok(input));
    }

    #[test]
    fn test_array_ir_operation_from_array_operation_constructors() {
        let scalar = ArrayType::scalar(DataType::F32);
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap());
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));

        // Existing array zero instructions choose the mixed carrier only when their type names runtime identities.
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(scalar.clone()))),
            ArrayIrOperation::Array(ArrayOperation::Zero(operation)) if operation.r#type() == &scalar,
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(dynamic.clone()))),
            ArrayIrOperation::Zero(operation) if operation.r#type() == &dynamic,
        ));

        // Direct payload conversions apply the same static/dynamic distinction to one and iota.
        assert!(matches!(
            ArrayIrOperation::<Array>::from(OneOperation::new(scalar.clone())),
            ArrayIrOperation::Array(ArrayOperation::One(operation)) if operation.r#type() == &scalar,
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(OneOperation::new(dynamic.clone())),
            ArrayIrOperation::One(operation) if operation.r#type() == &dynamic,
        ));
        let static_iota = ArrayType::new_static(DataType::I32, [3]);
        assert!(matches!(
            ArrayIrOperation::<Array>::from(IotaOperation::new(static_iota.clone(), 0).unwrap()),
            ArrayIrOperation::Array(ArrayOperation::Iota(operation))
                if operation.r#type() == &static_iota && operation.dimension() == 0,
        ));
        let dynamic_iota =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(extent), Dimension::Static(2)]));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(IotaOperation::new(dynamic_iota.clone(), 0).unwrap()),
            ArrayIrOperation::Iota(operation)
                if operation.r#type() == &dynamic_iota && operation.dimension() == 0,
        ));
    }

    #[test]
    fn test_array_ir_operation_from_array_operation_homogeneous_inputs() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let scalar_type = ArrayType::scalar(DataType::F32);
        let output_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]));

        // Existing homogeneous instructions retain their original inputs. The mixed payload constructors require
        // additional extent inputs, so their direct `From` implementations cannot lift these instructions.
        let pad =
            ArrayIrOperation::<Array>::from(ArrayOperation::Pad(PadOperation::new(vec![1], vec![1], vec![0]).unwrap()));
        assert!(matches!(pad, ArrayIrOperation::Array(ArrayOperation::Pad(_))));
        assert_eq!(
            pad.infer_output_types(&[array_type.clone().into(), scalar_type.into()], &[]),
            Ok(vec![output_type.clone()]),
        );
        let concatenate = ArrayIrOperation::<Array>::from(ArrayOperation::Concatenate(
            ConcatenateOperation::<ArrayType>::new(0, 1).unwrap(),
        ));
        assert!(matches!(concatenate, ArrayIrOperation::Array(ArrayOperation::Concatenate(_))));
        assert_eq!(
            concatenate.infer_output_types(&[array_type.clone().into(), array_type.into()], &[]),
            Ok(vec![output_type]),
        );
    }

    #[test]
    fn test_array_ir_operation_from_array_operation_regions() {
        // Member control-flow operations promote to their direct array IR carriers, preserving every semantic and
        // lowering attribute.
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new())),
            ArrayIrOperation::Condition(_),
        ));
        let while_operation = WhileOperation::new().with_iteration_bound(7).unwrap();
        let promoted_while = ArrayIrOperation::<Array>::from(ArrayOperation::While(while_operation.clone()));
        assert!(matches!(
            promoted_while,
            ArrayIrOperation::While(operation)
                if operation.iteration_bound() == while_operation.iteration_bound(),
        ));
        let scan_operation = ScanOperation::<ArrayType>::new(1, 4).with_reverse(true).with_unroll(2).unwrap();
        let promoted_scan = ArrayIrOperation::<Array>::from(ArrayOperation::Scan(scan_operation));
        let ArrayIrOperation::Scan(promoted_scan) = promoted_scan else {
            panic!("expected a direct array IR scan operation");
        };
        assert_eq!(promoted_scan.carry_count(), 1);
        assert_eq!(promoted_scan.iteration_count(), &Dimension::Static(4));
        assert!(promoted_scan.reverse());
        assert_eq!(promoted_scan.unroll(), 2);

        // Every region-carrying array payload has an array IR carrier, so none of them reaches the region-free
        // projected `Array` variant. Composite calls preserve their rule layout and non-differentiated input split.
        // Rematerialization preserves its policy's identity, optimization barrier, and differentiation state.
        for jvp_rule in [CompositeJvpRule::Absent, CompositeJvpRule::Explicit, CompositeJvpRule::Primal] {
            for has_vjp_rule in [false, true] {
                let operation = CompositeOperation::from_rule_regions(jvp_rule, has_vjp_rule)
                    .with_non_differentiated_count(2)
                    .unwrap();
                assert!(matches!(
                    TestOperation::from(ArrayOperation::Composite(operation)),
                    ArrayIrOperation::Composite(operation)
                        if operation.jvp_rule() == jvp_rule
                            && operation.has_vjp_rule() == has_vjp_rule
                            && operation.non_differentiated_count() == 2,
                ));
            }
        }
        let policy = ResidualPolicyReference::<ArrayType>::new(DotsSavable);
        assert!(matches!(
            TestOperation::from(ArrayOperation::Rematerialize(
                RematerializeOperation::new(policy.clone())
                    .with_optimization_barrier(RematerializationOptimizationBarrier::None)
                    .with_differentiated(true),
            )),
            ArrayIrOperation::Rematerialize(operation)
                if operation.policy().id() == policy.id()
                    && operation.policy().name() == DOTS_SAVABLE_POLICY_NAME
                    && operation.optimization_barrier() == &RematerializationOptimizationBarrier::None
                    && operation.differentiated(),
        ));
        assert!(matches!(
            TestOperation::from(ArrayOperation::LinearCall(LinearCallOperation::new(1))),
            ArrayIrOperation::LinearCall(operation) if operation.residual_count() == 1,
        ));

        // A carrier with an attached backward rule maps its stored tangent types into the array IR type universe and
        // follows its region into the native carrier variant.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let promoted = TestOperation::from(ArrayOperation::CompositeTranspose(
            CompositeTransposeOperation::from_backward_region(1, vec![scalar_type.clone()], vec![scalar_type.clone()]),
        ));
        let ArrayIrOperation::CompositeTranspose(promoted) = promoted else {
            panic!("expected a promoted array IR composite carrier");
        };
        assert_eq!(
            promoted,
            CompositeTransposeOperation::from_backward_region(
                1,
                vec![ArrayIrType::Array(scalar_type.clone())],
                vec![ArrayIrType::Array(scalar_type)],
            ),
        );
    }

    #[test]
    fn test_array_ir_operation_from_reference_operations() {
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceNewOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceNew(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(
                ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new(),
            ),
            ArrayIrOperation::ReferenceRead(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(
                ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
            ),
            ArrayIrOperation::ReferenceWrite(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(
                ReferenceSwapOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new(),
            ),
            ArrayIrOperation::ReferenceSwap(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceAddUpdateOperation::<
                ArrayType,
                ArrayIrType,
                ArrayReferenceTransform,
            >::new()),
            ArrayIrOperation::ReferenceAddUpdate(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceAtomicAddUpdateOperation::<
                ArrayType,
                ArrayIrType,
                ArrayReferenceTransform,
            >::new()),
            ArrayIrOperation::ReferenceAtomicAddUpdate(_),
        ));
        assert!(matches!(
            ArrayIrOperation::<Array>::from(ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new()),
            ArrayIrOperation::ReferenceFreeze(_),
        ));
    }
}
