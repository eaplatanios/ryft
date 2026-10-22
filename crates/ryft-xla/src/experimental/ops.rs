use std::borrow::Cow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ryft_core::kernels::KernelReferenceOperation;
use ryft_core::operations::attention::{DotProductAttentionBackwardOperation, DotProductAttentionOperation};
use ryft_core::{
    AbsOperation, AddOperation, AndOperation, ArgMaxOperation, ArgMinOperation, Array, ArrayIrOperation, ArrayIrType,
    ArrayOperation, ArrayReferenceTransform, ArrayType, AssertOperation, AssertionValue, Atan2Operation,
    AxisIndexOperation, BroadcastOperation, CaptureConstant, CaptureReference, CastOperation, CeilOperation,
    ClampOperation, CompareOperation, ComplexOperation, ConcatenateOperation, Concretizable, ConditionOperation,
    ConjugateOperation, ConstantOperation, ConstrainShardingOperation, ConvertElementTypeOperation, CosOperation,
    CumulativeOperation, CustomCallOperation, CustomFunctionOperation, CustomFunctionTransposeOperation, DataType,
    DimensionAddOperation, DimensionDivOperation, DimensionFromScalarOperation, DimensionMaxOperation,
    DimensionMinOperation, DimensionMulOperation, DimensionOperation, DimensionPowOperation, DimensionRemOperation,
    DimensionSaturatingSubOperation, DimensionSizeOperation, DimensionSubOperation, DimensionToScalarOperation,
    DimensionType, DimensionValue, DivOperation, DotOperation, DynamicBroadcastOperation, DynamicReshapeOperation,
    DynamicSliceOperation, DynamicUpdateSliceOperation, EagerContext, ErfOperation, ExpOperation, FloorOperation,
    GatherOperation, GuardOperation, ImaginaryOperation, IotaOperation, LiftedCustomRules, LinearCallOperation,
    LiteralIdentity, Ln1pOperation, LogAddExpOperation, LogOperation, LogisticOperation, MaxOperation, MinOperation,
    MulOperation, NegOperation, NotOperation, OneLikeOperation, OneOperation, Operation, OperationProvider,
    OrOperation, PadOperation, ParallelAllGatherOperation, ParallelAllToAllOperation, ParallelPermuteOperation,
    ParallelRaggedAllToAllOperation, ParallelReduceOperation, ParallelSumScatterOperation, ParallelVaryOperation,
    Parameter, PowOperation, PrintOperation, Program, ProgramBuilder, ProgramError, ProjectedValue, RaggedAddOperation,
    RaggedDotOperation, RaggedExtentsOperation, RaggedNewOperation, RaggedStorageOperation, RaggedZeroOperation,
    RealOperation, ReduceOperation, ReducePrecisionOperation, ReferenceAccessDescriptor, ReferenceAccessOperation,
    ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation,
    ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation, RemOperation, RematerializeOperation,
    ReshapeOperation, ReshardOperation, RngBitGeneratorOperation, RoundOperation, RsqrtOperation, ScaledDotOperation,
    ScanOperation, ScatterOperation, SelectOperation, ShardMapOperation, SignOperation, SinOperation, SliceOperation,
    SortOperation, SqrtOperation, StopGradientOperation, SubOperation, TagOperation, TanOperation, TanhOperation,
    TransferToMemoryOperation, TransposeOperation, Type, TypeError, TypeIdentityRenaming, Typed,
    UnavailableCustomRules, UpdateSliceOperation, Value, ValueDirectDispatch, ValueProjection, WhileOperation,
    XorOperation, ZeroLikeOperation, ZeroOperation,
};
use ryft_macros::Parameter;

use crate::kernels::XlaKernelOperation;
use crate::operations::jit_call::JitCallOperation;

/// Exact host array literal embedded by backend legalization, shared without device allocation.
#[derive(Clone, Debug, Parameter)]
pub struct XlaArrayLiteral(Arc<Array>);

impl XlaArrayLiteral {
    /// Returns the complete host literal, including its physical layout and exact element encodings.
    pub fn array(&self) -> &Array {
        &self.0
    }
}
impl PartialEq for XlaArrayLiteral {
    fn eq(&self, other: &Self) -> bool {
        self.0.literal_eq(&other.0)
    }
}
impl Eq for XlaArrayLiteral {}
impl Hash for XlaArrayLiteral {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.literal_hash(state);
    }
}
impl Display for XlaArrayLiteral {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "literal<{}>[", self.array().r#type())?;
        for byte in self.array().storage_bytes() {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str("]")
    }
}

/// Array constants retained by staged XLA programs. Captures refer to the enclosing capture table; Boolean
/// literals preserve dimension-comparison proofs without allocating device storage or introducing a capture.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum XlaArrayConstant {
    /// A reference to array data in the enclosing capture table.
    Captured(CaptureReference<ArrayType>),

    /// An immediate scalar Boolean predicate.
    Boolean(bool),

    /// Exact ordinary host array data introduced by backend legalization.
    Literal(XlaArrayLiteral),
}

impl From<CaptureReference<ArrayType>> for XlaArrayConstant {
    fn from(value: CaptureReference<ArrayType>) -> Self {
        Self::Captured(value)
    }
}

impl TryFrom<bool> for XlaArrayConstant {
    type Error = ProgramError;

    fn try_from(value: bool) -> Result<Self, ProgramError> {
        Ok(Self::Boolean(value))
    }
}

impl Display for XlaArrayConstant {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Captured(value) => Display::fmt(value, formatter),
            Self::Boolean(value) => Display::fmt(value, formatter),
            Self::Literal(value) => Display::fmt(value, formatter),
        }
    }
}

impl Typed for XlaArrayConstant {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        match self {
            Self::Captured(value) => value.r#type(),
            Self::Boolean(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean)),
            Self::Literal(value) => value.array().r#type(),
        }
    }
}

impl Value for XlaArrayConstant {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self>;

    fn domain(&self) -> Self::Domain {
        EagerContext::new()
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Captured(value) => Ok(Self::Captured(value.rename_type_identities(renaming)?)),
            Self::Boolean(value) => Ok(Self::Boolean(*value)),
            Self::Literal(value) => {
                Ok(Self::Literal(XlaArrayLiteral(Arc::new(value.array().rename_type_identities(renaming)?))))
            }
        }
    }

    fn capture_index(&self) -> Option<usize> {
        match self {
            Self::Captured(value) => Some(value.index()),
            Self::Boolean(_) | Self::Literal(_) => None,
        }
    }
}

impl CaptureConstant for XlaArrayConstant {
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
        match self {
            Self::Captured(value) => Self::Captured(value.map_capture_index(map)),
            Self::Boolean(value) => Self::Boolean(*value),
            Self::Literal(value) => Self::Literal(value.clone()),
        }
    }
}

impl Concretizable<bool> for XlaArrayConstant {
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Captured(value) => value.concretize(),
            Self::Boolean(value) => Ok(*value),
            Self::Literal(value) => value.array().concretize(),
        }
    }
}

impl AssertionValue for XlaArrayConstant {
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        match self {
            Self::Captured(value) => value.assertion_array(),
            Self::Boolean(value) => Ok(Some(Array::scalar(*value)?)),
            Self::Literal(value) => Ok(Some(value.array().clone())),
        }
    }

    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self {
            Self::Captured(value) => value.assertion_observation(),
            Self::Boolean(value) => Ok(value.to_string()),
            Self::Literal(value) => value.array().assertion_observation(),
        }
    }
}

/// Constant payload stored in the atom table of a staged XLA [`Program`].
///
/// Staged XLA programs keep the following kinds of constants apart, and this sum is the staged counterpart of the eager
/// [`ArrayIrValue`](ryft_core::ArrayIrValue) universe:
///
///   - **Captured runtime values:** array buffers and external references stay in the surrounding
///     compiled function's capture table, while the program stores only a lifetime-free [`CaptureReference`] carrying
///     its index and structural [`ArrayIrType`]. This keeps runtime storage out of literal IR payloads. A captured
///     reference remains metadata only and must be validated and discharged before ordinary XLA lowering.
///   - **Immediate first-class dimensions:** a [`DimensionValue`] is a checked host integer, so embedding it costs
///     nothing and it lowers to a scalar `stablehlo.constant`. Unlike a capture reference it also stays usable inside
///     a nested region — most importantly a `shard_map` manual computation, which owns no capture table of its own —
///     and that is what lets shape arithmetic such as the explicit result extents of the collectives be staged
///     against a manual region's shard-local values.
///   - **Immediate Boolean predicates:** static dimension proofs remain concrete through array projection, logical
///     composition, and assertion folding without requiring a captured device buffer.
///   - **Exact ordinary host literals:** constants introduced by backend checking retain their complete bytes and
///     physical type, shared by `Arc` until dense literal lowering. Ragged aggregates remain excluded from literals.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum XlaConstant {
    /// Reference to a value held in the surrounding compiled function's capture table.
    Captured(CaptureReference<ArrayIrType>),

    /// Immediate checked host-side first-class dimension extent.
    Dimension(DimensionValue),

    /// Immediate scalar Boolean predicate.
    Boolean(bool),

    /// Exact ordinary host array data introduced by backend legalization.
    Literal(XlaArrayLiteral),
}

impl Display for XlaConstant {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Captured(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
            Self::Boolean(value) => Display::fmt(value, formatter),
            Self::Literal(value) => Display::fmt(value, formatter),
        }
    }
}

// Literal wrappers use exact bytes, so ordinary equality is already literal identity for this complete sum.
impl LiteralIdentity for XlaConstant {
    #[inline]
    fn literal_eq(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn literal_hash<H: Hasher>(&self, state: &mut H) {
        self.hash(state);
    }
}

impl Typed for XlaConstant {
    type Type = ArrayIrType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        match self {
            Self::Captured(value) => value.r#type(),
            Self::Dimension(value) => Cow::Owned(ArrayIrType::Dimension(value.r#type().into_owned())),
            Self::Boolean(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean).into()),
            Self::Literal(value) => Cow::Owned(value.array().r#type().into_owned().into()),
        }
    }
}

impl Value for XlaConstant {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self>;

    #[inline]
    fn domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Captured(value) => Ok(Self::Captured(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
            Self::Boolean(value) => Ok(Self::Boolean(*value)),
            Self::Literal(value) => {
                Ok(Self::Literal(XlaArrayLiteral(Arc::new(value.array().rename_type_identities(renaming)?))))
            }
        }
    }

    #[inline]
    fn capture_index(&self) -> Option<usize> {
        match self {
            Self::Captured(value) => Some(value.index()),
            Self::Dimension(_) | Self::Boolean(_) | Self::Literal(_) => None,
        }
    }
}

impl CaptureConstant for XlaConstant {
    #[inline]
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
        match self {
            Self::Captured(value) => Self::Captured(value.map_capture_index(map)),
            Self::Dimension(value) => Self::Dimension(value.clone()),
            Self::Boolean(value) => Self::Boolean(*value),
            Self::Literal(value) => Self::Literal(value.clone()),
        }
    }
}

impl ValueProjection<ArrayType> for XlaConstant {
    type Projected = XlaArrayConstant;
    type ProjectedRef<'v> = ProjectedValue<ArrayType, &'v Self>;

    #[inline]
    fn from_projected(value: XlaArrayConstant) -> Self {
        match value {
            XlaArrayConstant::Captured(value) => Self::Captured(ValueProjection::<ArrayType>::from_projected(value)),
            XlaArrayConstant::Boolean(value) => Self::Boolean(value),
            XlaArrayConstant::Literal(value) => Self::Literal(value),
        }
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        ArrayType: 'v,
    {
        Ok(ProjectedValue::new(self, <&ArrayType>::try_from(self.r#type().as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<XlaArrayConstant, TypeError> {
        match self {
            Self::Captured(value) => value.into_projected().map(XlaArrayConstant::Captured),
            Self::Boolean(value) => Ok(XlaArrayConstant::Boolean(value)),
            Self::Literal(value) => Ok(XlaArrayConstant::Literal(value)),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }
}

impl ValueProjection<DimensionType> for XlaConstant {
    type Projected = DimensionValue;
    type ProjectedRef<'v> = &'v DimensionValue;

    #[inline]
    fn from_projected(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v DimensionValue, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Captured(_) => Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
            Self::Boolean(_) => Err(TypeError::invalid("expected a dimension but got a Boolean array")),
            Self::Literal(_) => Err(TypeError::invalid("expected a dimension but got an array literal")),
            Self::Dimension(value) => Ok(value),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Captured(_) => Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
            Self::Boolean(_) => Err(TypeError::invalid("expected a dimension but got a Boolean array")),
            Self::Literal(_) => Err(TypeError::invalid("expected a dimension but got an array literal")),
            Self::Dimension(value) => Ok(value),
        }
    }
}

impl From<CaptureReference<ArrayIrType>> for XlaConstant {
    #[inline]
    fn from(value: CaptureReference<ArrayIrType>) -> Self {
        Self::Captured(value)
    }
}

impl From<DimensionValue> for XlaConstant {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

impl TryFrom<bool> for XlaConstant {
    type Error = ProgramError;

    fn try_from(value: bool) -> Result<Self, ProgramError> {
        Ok(Self::Boolean(value))
    }
}

impl From<Array> for XlaConstant {
    fn from(value: Array) -> Self {
        Self::Literal(XlaArrayLiteral(Arc::new(value)))
    }
}
impl From<Array> for XlaArrayConstant {
    fn from(value: Array) -> Self {
        Self::Literal(XlaArrayLiteral(Arc::new(value)))
    }
}

impl Concretizable<bool> for XlaConstant {
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Boolean(value) => Ok(*value),
            Self::Literal(value) => value.array().concretize(),
            _ => Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from the staged xla constant `{self}`"),
            }),
        }
    }
}

impl AssertionValue for XlaConstant {
    fn assertion_array(&self) -> Result<Option<Array>, ProgramError> {
        match self {
            Self::Dimension(_) => Ok(None),
            Self::Boolean(value) => Ok(Some(Array::scalar(*value)?)),
            Self::Literal(value) => Ok(Some(value.array().clone())),
            Self::Captured(_) => Err(ProgramError::Concretization {
                message: "cannot read an assertion array from a captured constant reference".to_owned(),
            }),
        }
    }

    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self {
            Self::Dimension(value) => Ok(value.extent().to_string()),
            Self::Boolean(value) => Ok(value.to_string()),
            Self::Literal(value) => value.array().assertion_observation(),
            Self::Captured(_) => Err(ProgramError::Concretization {
                message: "cannot read an assertion observation from a captured constant reference".to_owned(),
            }),
        }
    }
}

/// Ordinary staged-operation universe owned by the XLA backend.
///
/// This enum flattens the core array operation payloads directly into the backend-owned operation family. Higher-order
/// instructions attach their nested computations as regions of the containing XLA program, so those regions can
/// contain backend-specific operations such as [`jit_call`](JitCallOperation) and
/// [`shard_map`](ShardMapOperation).
///
/// The [`Operation`] contract, the interpretation and partial-evaluation dispatchers, the forward-mode and
/// transposition dispatchers, the member operation-family projections, and the payload conversions are all derived.
/// The variant classes mirror [`ArrayIrOperation`] exactly for the payloads the two families share, so both
/// dispatchers report identical semantics for every shared member and mixed payload. Only the backend-owned surfaces
/// stay handwritten: the normalizing conversions that select between a member and a mixed carrier, the zero and
/// residual-zero providers, the canonical core-operation view used by lowering, and the MLIR lowering dispatch.
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(
    crate = "ryft_core",
    type(ArrayIrType),
    constant(Constant),
    members(ArrayType, structural(DimensionType)),
    dispatch(identity, discharge, batching, differentiation, transposition)
)]
pub enum XlaOperation<Constant = XlaConstant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Portable kernel instruction whose actual body remains attached until explicit XLA compiler selection.
    Kernel(XlaKernelOperation),

    /// Mixed zero constructor whose explicit first-class dimension operands provide its dynamic result extents.
    /// This variant cannot be represented by the homogeneous array member because its signature crosses member
    /// kinds: it consumes dimension members and produces an array member.
    #[ryft(mixed(structural), skip_from)]
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor with explicit dynamic-extent operands.
    #[ryft(mixed(structural), skip_from)]
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor with explicit dynamic-extent operands.
    #[ryft(mixed(structural), skip_from)]
    DynamicIota(IotaOperation<ArrayType>),

    /// Homogeneous array operation. Member zero constructors are promoted to their mixed composite carrier when an
    /// array operation is lifted into this family.
    #[ryft(projected(ArrayType), skip_from)]
    Array(ArrayOperation<<Constant as ValueProjection<ArrayType>>::Projected>),

    /// Homogeneous first-class-dimension operation.
    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two dimensions producing Boolean array data.
    Compare(CompareOperation<ArrayIrType>),

    /// Ordered assertion with optional scalar diagnostic inputs.
    Assert(AssertOperation<ArrayIrType>),

    /// Runtime check of the identity relationships of an entry that the actual input types do not prove, which
    /// returns its inputs narrowed by the facts that those relationships prove.
    Guard(GuardOperation<ArrayIrType>),

    /// Conversion of a value to a compatible array or first-class dimension type.
    Cast(CastOperation<ArrayIrType>),

    /// Reads an array extent as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Unresolved whole-array reference allocation retained until reference discharge.
    ReferenceNew(ReferenceNewOperation<ArrayType, ArrayIrType>),

    /// Unresolved read through an optional folded transform path retained until reference discharge.
    ReferenceRead(ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Unresolved write-only replacement through an optional folded transform path retained until reference discharge.
    ReferenceWrite(ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Unresolved replacement through an optional folded transform path retained until reference discharge.
    ReferenceSwap(ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Unresolved additive update through an optional folded transform path retained until reference discharge.
    ReferenceAddUpdate(ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Unresolved atomic additive update retained until sequential reference discharge or a supporting kernel lowering.
    ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),

    /// Unresolved consuming whole-array reference freeze retained until reference discharge.
    ReferenceFreeze(ReferenceFreezeOperation<ArrayType, ArrayIrType>),

    /// Converts scalar array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Converts a first-class dimension into scalar array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Checked first-class ragged construction, legalized into guarded dense payloads before lowering.
    RaggedNew(RaggedNewOperation),

    /// Explicit ragged storage projection, resolved through the canonical grouped value layout.
    RaggedStorage(RaggedStorageOperation),

    /// Explicit ragged extent projection, resolved through the canonical grouped value layout.
    RaggedExtents(RaggedExtentsOperation),

    /// Shape-dependent ragged zero, expanded through the shared grouped replay before MLIR lowering.
    RaggedZero(RaggedZeroOperation),

    /// Ragged cotangent accumulation, retaining extent equality checks through grouped replay.
    RaggedAdd(RaggedAddOperation),

    /// Reshapes an array using explicit dimension operands.
    Reshape(DynamicReshapeOperation),

    /// Broadcasts an array using explicit dimension operands.
    Broadcast(DynamicBroadcastOperation),

    /// Concatenates arrays with an explicit result extent.
    Concatenate(ConcatenateOperation<ArrayIrType>),

    /// Calls a foreign kernel with explicit dynamic result extents.
    #[ryft(mixed)]
    CustomCall(CustomCallOperation),

    /// Pads an array with explicit result extents.
    Pad(PadOperation<ArrayIrType>),

    /// Slices an array using first-class start and size dimensions.
    DynamicSlice(DynamicSliceOperation<ArrayIrType>),

    /// Generates random bits with explicit dynamic result extents.
    RngBitGenerator(RngBitGeneratorOperation<ArrayIrType>),

    /// Gathers values with one explicit extent per result axis.
    #[ryft(mixed)]
    ParallelAllGather(ParallelAllGatherOperation),

    /// Scatters values with one explicit extent per result axis.
    #[ryft(mixed)]
    ParallelSumScatter(ParallelSumScatterOperation),

    /// Exchanges values with one explicit extent per result axis.
    #[ryft(mixed)]
    ParallelAllToAll(ParallelAllToAllOperation),

    /// Exchanges variable-length leading-axis slices through an XLA FFI custom call.
    #[ryft(mixed)]
    ParallelRaggedAllToAll(ParallelRaggedAllToAllOperation),

    /// Backend-owned condition whose attached branch regions can contain XLA operations.
    Condition(ConditionOperation<ArrayIrType>),

    /// Backend-owned loop whose attached condition and body regions can contain XLA operations.
    While(WhileOperation<ArrayIrType>),

    /// Backend-owned scan whose attached body region can contain XLA operations.
    Scan(ScanOperation<ArrayIrType>),

    /// Backend-owned custom function call whose primal region can contain XLA operations, with either attached rule
    /// regions that can contain XLA operations or retained rules registered in this family, which are traced lazily.
    CustomFunction(CustomFunctionOperation<Constant, XlaOperation<Constant>>),

    /// Reverse-mode carrier of a [`CustomFunction`](Self::CustomFunction) call, which transposition replaces with
    /// the call's backward rule.
    CustomFunctionTranspose(CustomFunctionTransposeOperation<Constant, XlaOperation<Constant>>),

    /// Differentiation-owned call to an explicitly transposable linear map with ordinary trailing residual operands,
    /// which lowers by inlining its forward region.
    LinearCall(LinearCallOperation<ArrayIrType>),

    /// Backend-owned rematerialized call whose body can contain XLA operations.
    Rematerialize(RematerializeOperation<ArrayIrType>),

    /// Custom function call whose retained rules are registered in the [`ArrayOperation`] member family, promoted
    /// from its member payload. Its derivative rules are specialized by the member definition and converted into this
    /// family (refer to [`LiftedCustomRules`]). Conversions never produce such a call with attached rules, which follow
    /// their call into the [`CustomFunction`](Self::CustomFunction) variant instead.
    LiftedCustomFunction(
        CustomFunctionOperation<
            Constant,
            XlaOperation<Constant>,
            LiftedCustomRules<
                <Constant as ValueProjection<ArrayType>>::Projected,
                ArrayOperation<<Constant as ValueProjection<ArrayType>>::Projected>,
            >,
        >,
    ),

    /// Reverse-mode carrier of a [`LiftedCustomFunction`](Self::LiftedCustomFunction) call, whose backward rule is
    /// specialized by the member definition and converted into this family.
    LiftedCustomFunctionTranspose(
        CustomFunctionTransposeOperation<
            Constant,
            XlaOperation<Constant>,
            LiftedCustomRules<
                <Constant as ValueProjection<ArrayType>>::Projected,
                ArrayOperation<<Constant as ValueProjection<ArrayType>>::Projected>,
            >,
        >,
    ),

    /// Call to a flat jitted XLA sub-program.
    JitCall(JitCallOperation<ArrayIrType>),

    /// Core `shard_map` operation (refer to [`ShardMapOperation`]), which lowering emits as a Shardy
    /// `sdy.manual_computation` through its dedicated path rather than through the core composite lowering.
    ShardMap(Box<ShardMapOperation>),
}

impl<Constant> ReferenceAccessOperation for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match self {
            Self::Kernel(operation) => operation.operation().base_input_count(),
            Self::ReferenceRead(operation) => operation.base_input_count(),
            Self::ReferenceWrite(operation) => operation.base_input_count(),
            Self::ReferenceSwap(operation) => operation.base_input_count(),
            Self::ReferenceAddUpdate(operation) => operation.base_input_count(),
            Self::ReferenceAtomicAddUpdate(operation) => operation.base_input_count(),
            Self::ReferenceFreeze(_) => 1,
            _ => 0,
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
        match self {
            Self::Kernel(operation) => operation.operation().reference_access_descriptor(input_index),
            Self::ReferenceRead(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceWrite(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceSwap(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceAddUpdate(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceAtomicAddUpdate(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceFreeze(_) if input_index == 0 => Some(ReferenceAccessDescriptor::new(&[], 1..1)),
            _ => None,
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<Self::Transform>,
    ) -> Result<Self, ProgramError> {
        match self {
            Self::Kernel(operation) => operation
                .operation()
                .with_reference_access_transforms(input_index, transforms)
                .map(|operation| Self::Kernel(operation.into())),
            Self::ReferenceRead(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceRead)
            }
            Self::ReferenceWrite(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceWrite)
            }
            Self::ReferenceSwap(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceSwap)
            }
            Self::ReferenceAddUpdate(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceAddUpdate)
            }
            Self::ReferenceAtomicAddUpdate(operation) => operation
                .with_reference_access_transforms(input_index, transforms)
                .map(Self::ReferenceAtomicAddUpdate),
            _ if self.reference_access_descriptor(input_index).is_some() && transforms.is_empty() => Ok(self.clone()),
            _ => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` cannot replace the reference transforms at input {input_index}", self.name()),
            }),
        }
    }
}

impl<Constant> KernelReferenceOperation for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn swap_output_index(&self) -> Option<usize> {
        match self {
            Self::Kernel(operation) => operation.operation().swap_output_index(),
            _ => matches!(self, Self::ReferenceSwap(_)).then_some(0),
        }
    }
}

impl<Constant> From<ArrayOperation<Constant::Projected>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ArrayOperation<Constant::Projected>) -> Self {
        // Delegating to the composite family's conversion keeps constructor normalization and member control-flow
        // promotion identical across both families: `Condition`, `While`, and `Scan` must become composite carriers
        // because the projected `Array` variant cannot own composite regions.
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

impl<Constant> From<ArrayIrOperation<Constant::Projected>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn from(operation: ArrayIrOperation<Constant::Projected>) -> Self {
        match operation {
            ArrayIrOperation::Zero(operation) => Self::Zero(operation),
            ArrayIrOperation::One(operation) => Self::DynamicOne(operation),
            ArrayIrOperation::Iota(operation) => Self::DynamicIota(operation),
            ArrayIrOperation::Array(operation) => Self::Array(operation),
            ArrayIrOperation::Dimension(operation) => Self::Dimension(operation),
            ArrayIrOperation::Compare(operation) => Self::Compare(operation),
            ArrayIrOperation::Assert(operation) => Self::Assert(operation),
            ArrayIrOperation::Guard(operation) => Self::Guard(operation),
            ArrayIrOperation::Cast(operation) => Self::Cast(operation),
            ArrayIrOperation::DimensionSize(operation) => Self::DimensionSize(operation),
            ArrayIrOperation::ReferenceNew(operation) => Self::ReferenceNew(operation),
            ArrayIrOperation::ReferenceRead(operation) => Self::ReferenceRead(operation),
            ArrayIrOperation::ReferenceWrite(operation) => Self::ReferenceWrite(operation),
            ArrayIrOperation::ReferenceSwap(operation) => Self::ReferenceSwap(operation),
            ArrayIrOperation::ReferenceAddUpdate(operation) => Self::ReferenceAddUpdate(operation),
            ArrayIrOperation::ReferenceAtomicAddUpdate(operation) => Self::ReferenceAtomicAddUpdate(operation),
            ArrayIrOperation::ReferenceFreeze(operation) => Self::ReferenceFreeze(operation),
            ArrayIrOperation::DimensionFromScalar(operation) => Self::DimensionFromScalar(operation),
            ArrayIrOperation::DimensionToScalar(operation) => Self::DimensionToScalar(operation),
            ArrayIrOperation::RaggedNew(operation) => Self::RaggedNew(operation),
            ArrayIrOperation::RaggedStorage(operation) => Self::RaggedStorage(operation),
            ArrayIrOperation::RaggedExtents(operation) => Self::RaggedExtents(operation),
            ArrayIrOperation::RaggedZero(operation) => Self::RaggedZero(operation),
            ArrayIrOperation::RaggedAdd(operation) => Self::RaggedAdd(operation),
            ArrayIrOperation::Reshape(operation) => Self::Reshape(operation),
            ArrayIrOperation::Broadcast(operation) => Self::Broadcast(operation),
            ArrayIrOperation::Concatenate(operation) => Self::Concatenate(operation),
            ArrayIrOperation::CustomCall(operation) => Self::CustomCall(operation),
            ArrayIrOperation::Pad(operation) => Self::Pad(operation),
            ArrayIrOperation::DynamicSlice(operation) => Self::DynamicSlice(operation),
            ArrayIrOperation::RngBitGenerator(operation) => Self::RngBitGenerator(operation),
            ArrayIrOperation::ParallelAllGather(operation) => Self::ParallelAllGather(operation),
            ArrayIrOperation::ParallelSumScatter(operation) => Self::ParallelSumScatter(operation),
            ArrayIrOperation::ParallelAllToAll(operation) => Self::ParallelAllToAll(operation),
            ArrayIrOperation::ParallelRaggedAllToAll(operation) => Self::ParallelRaggedAllToAll(operation),
            ArrayIrOperation::Condition(_) => Self::Condition(ConditionOperation::new()),
            ArrayIrOperation::While(operation) => Self::While(operation),
            ArrayIrOperation::Scan(operation) => Self::Scan(operation),
            ArrayIrOperation::LinearCall(operation) => Self::LinearCall(operation),
            ArrayIrOperation::Rematerialize(operation) => Self::Rematerialize(operation),
            ArrayIrOperation::ShardMap(operation) => Self::ShardMap(operation),
            // Custom function calls and carriers with attached rules follow their regions into the native variants.
            // Converted member calls keep their member definition. Calls registered in the array IR family itself
            // cannot follow their program, because constants of this family cannot represent array IR values, so their
            // rules become unavailable while their primal still executes.
            ArrayIrOperation::CustomFunction(operation) => {
                match operation.into_attached_family() {
                    Ok(operation) => Self::CustomFunction(operation),
                    Err(operation) => Self::LiftedCustomFunction(operation.into_family(|rules| {
                        LiftedCustomRules::Unavailable(UnavailableCustomRules::from_source(&rules))
                    })),
                }
            }
            ArrayIrOperation::CustomFunctionTranspose(operation) => {
                match operation.into_attached_family() {
                    Ok(operation) => Self::CustomFunctionTranspose(operation),
                    Err(operation) => Self::LiftedCustomFunctionTranspose(operation.into_family(|rules| {
                        LiftedCustomRules::Unavailable(UnavailableCustomRules::from_source(&rules))
                    })),
                }
            }
            ArrayIrOperation::LiftedCustomFunction(operation) => match operation.into_attached_family() {
                Ok(operation) => Self::CustomFunction(operation),
                Err(operation) => Self::LiftedCustomFunction(operation.into_family(|rules| rules)),
            },
            ArrayIrOperation::LiftedCustomFunctionTranspose(operation) => match operation.into_attached_family() {
                Ok(operation) => Self::CustomFunctionTranspose(operation),
                Err(operation) => Self::LiftedCustomFunctionTranspose(operation.into_family(|rules| rules)),
            },
        }
    }
}

// Dimension constants additionally lift directly, so that generic staging code (e.g., `ExactShape::dimensions`)
// can bound only `From<ConstantOperation<DimensionValue>>` without naming this family's dimension member.
impl<Constant> From<ConstantOperation<DimensionValue>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ConstantOperation<DimensionValue>) -> Self {
        Self::Dimension(DimensionOperation::Constant(operation))
    }
}

macro_rules! impl_composite_operation_conversion {
    // Generates the composite-operation conversions the derive cannot generate directly: the canonical constructors
    // whose lift selects between the homogeneous member carrier and the mixed dimension-operand carrier, and the
    // homogeneous-array payload forms of mixed operations whose canonical composite payload is type-promoted.
    ($($operation:ty),+ $(,)?) => {
        $(
            impl<Constant> From<$operation> for XlaOperation<Constant>
            where
                Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            {
                #[inline]
                fn from(operation: $operation) -> Self {
                    ArrayIrOperation::<Constant::Projected>::from(operation).into()
                }
            }
        )+
    };
}

// Every first-class dimension arithmetic operation lifts directly, mirroring the composite family, so that generic
// staging code can bound `From<DimensionMulOperation>`-style conversions without naming this family's dimension member.
impl_composite_operation_conversion!(
    DimensionAddOperation,
    DimensionSubOperation,
    DimensionSaturatingSubOperation,
    DimensionMulOperation,
    DimensionPowOperation,
    DimensionDivOperation,
    DimensionRemOperation,
    DimensionMinOperation,
    DimensionMaxOperation,
);

impl_composite_operation_conversion!(
    ZeroLikeOperation<ArrayIrType>,
    OneLikeOperation<ArrayIrType>,
    ZeroOperation<ArrayType>,
    OneOperation<ArrayType>,
    IotaOperation<ArrayType>,
    ConcatenateOperation<ArrayType>,
    PadOperation<ArrayType>,
);

macro_rules! impl_array_operation_conversion {
    // Generates homogeneous array-operation conversions through the canonical projected member family.
    ($($operation:ty),+ $(,)?) => {
        $(
            impl<Constant> From<$operation> for XlaOperation<Constant>
            where
                Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            {
                #[inline]
                fn from(operation: $operation) -> Self {
                    ArrayOperation::<Constant::Projected>::from(operation).into()
                }
            }
        )+
    };
}

impl_array_operation_conversion!(
    ZeroLikeOperation<ArrayType>,
    OneLikeOperation<ArrayType>,
    ConstantOperation<Array>,
    AbsOperation<ArrayType>,
    NegOperation<ArrayType>,
    AddOperation<ArrayType>,
    SubOperation<ArrayType>,
    MulOperation<ArrayType>,
    DivOperation<ArrayType>,
    SinOperation<ArrayType>,
    CosOperation<ArrayType>,
    TanOperation<ArrayType>,
    Atan2Operation<ArrayType>,
    ExpOperation<ArrayType>,
    LogOperation<ArrayType>,
    Ln1pOperation<ArrayType>,
    LogAddExpOperation<ArrayType>,
    SqrtOperation<ArrayType>,
    RsqrtOperation<ArrayType>,
    TanhOperation<ArrayType>,
    LogisticOperation<ArrayType>,
    ErfOperation<ArrayType>,
    PowOperation<ArrayType>,
    SignOperation<ArrayType>,
    FloorOperation<ArrayType>,
    CeilOperation<ArrayType>,
    RoundOperation<ArrayType>,
    MaxOperation<ArrayType>,
    MinOperation<ArrayType>,
    ClampOperation<ArrayType>,
    RemOperation<ArrayType>,
    NotOperation<ArrayType>,
    AndOperation<ArrayType>,
    OrOperation<ArrayType>,
    XorOperation<ArrayType>,
    ComplexOperation<ArrayType>,
    ConjugateOperation<ArrayType>,
    RealOperation<ArrayType>,
    ImaginaryOperation<ArrayType>,
    DotOperation,
    RaggedDotOperation,
    ScaledDotOperation,
    DotProductAttentionOperation,
    DotProductAttentionBackwardOperation,
    ReduceOperation,
    ArgMaxOperation,
    ArgMinOperation,
    CumulativeOperation,
    SortOperation,
    ParallelReduceOperation,
    ParallelVaryOperation,
    ParallelPermuteOperation,
    AxisIndexOperation,
    TransposeOperation,
    ReshapeOperation,
    BroadcastOperation,
    GatherOperation,
    ScatterOperation,
    SliceOperation,
    UpdateSliceOperation,
    DynamicSliceOperation,
    DynamicUpdateSliceOperation,
    SelectOperation<ArrayType>,
    ConvertElementTypeOperation<ArrayType>,
    ReducePrecisionOperation<ArrayType>,
    TransferToMemoryOperation,
    ReshardOperation,
    ConstrainShardingOperation,
    StopGradientOperation<ArrayType>,
    TagOperation<ArrayType>,
    PrintOperation<ArrayType>,
);

impl_array_operation_conversion!(AssertOperation<ArrayType>);
impl_array_operation_conversion!(GuardOperation<ArrayType>);
impl_array_operation_conversion!(CastOperation<ArrayType>);

impl<Constant> OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Operation = Self;

    fn provide(request: ZeroOperation<ArrayIrType>, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Constant::Projected>::provide(request, input_types)?.into())
    }
}

impl<Constant> OperationProvider<ArrayIrType, OneOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Operation = Self;

    fn provide(request: OneOperation<ArrayIrType>, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Constant::Projected>::provide(request, input_types)?.into())
    }
}

impl<Constant> OperationProvider<ArrayIrType, AddOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Operation = Self;

    fn provide(request: AddOperation<ArrayIrType>, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Constant::Projected>::provide(request, input_types)?.into())
    }
}

// Composite operations whose generic capabilities bind them through any context lift into the array member exactly as
// they do in the core composite family.
impl<Constant> From<SelectOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: SelectOperation<ArrayIrType>) -> Self {
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

impl<Constant> From<StopGradientOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: StopGradientOperation<ArrayIrType>) -> Self {
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

impl<Constant> From<TagOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: TagOperation<ArrayIrType>) -> Self {
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

impl<Constant> From<PrintOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: PrintOperation<ArrayIrType>) -> Self {
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

/// Staged XLA program over an operation family `O`, which defaults to the backend-owned XLA operation universe.
pub type XlaProgram<Input, Output, O = XlaOperation> = Program<XlaConstant, O, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<XlaConstant, XlaOperation>;

/// Flat XLA program over an operation family `O`, which defaults to the backend-owned XLA operation universe, used for
/// materialized regions and shared callees.
pub type FlatXlaProgram<O = XlaOperation> = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>, O>;

#[cfg(test)]
pub(crate) mod tests {

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrOperation, ArrayIrOperations, ArrayIrType, ArrayOperation, ArrayOperations,
        ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, Assert, AssertOperation, CaptureReference,
        CapturingContext, Compare, CompareOperation, ComparisonDirection, ConditionOperation, Context,
        CustomFunctionJvpRule, CustomFunctionOperation, CustomFunctionTransposeOperation, DataType, DifferentiableType,
        Dimension, DimensionBounds, DimensionFromScalarOperation, DimensionType, DimensionValue, DimensionVariable,
        DomainTracingContext, DynamicBroadcastOperation, EffectClass, EffectClasses, ErasedOperation,
        ExternalReferenceBinding, MaybeZero, Memory, MulOperation, Operation, OperationPayloadProjection,
        OperationProvider, PartialValue, Placeholder, ProgramBuilder, ProgramError, RaggedExtentsOperation,
        RaggedNewOperation, RaggedStorageOperation, RaggedType, RaggedValueKind, ReferenceAccessOperation,
        ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceDischargeResult,
        ReferenceDischargeTarget, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceSource, ReferenceSwapOperation, ReferenceType, ReferenceWriteOperation,
        RematerializationOptimizationBarrier, RematerializeOperation, ResidualZeroProvider, ScanOperation, Shape,
        StagingContext, TagOperation, Tracer, TracingContext, TransferToMemoryOperation, TransposedProgramCotangent,
        TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection, ValueResolution, WhileOperation, ZeroOperation,
    };

    use crate::XlaArray;
    use crate::experimental::domains::{XlaDomain, XlaTracer};
    use crate::experimental::tracing::XlaArrayTracer;

    use crate::operations::jit_call::JitCallOperation;

    use super::{CaptureConstant, XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder};

    pub(crate) fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))
    }

    #[test]
    fn test_array_literals_keep_exact_bits_and_strict_projection() {
        let positive = XlaConstant::try_from(ryft_core::Array::scalar(0.0f32).unwrap()).unwrap();
        let negative = XlaConstant::try_from(ryft_core::Array::scalar(-0.0f32).unwrap()).unwrap();
        let nan = XlaConstant::try_from(ryft_core::Array::scalar(f32::from_bits(0x7fc00001)).unwrap()).unwrap();
        let other_nan = XlaConstant::try_from(ryft_core::Array::scalar(f32::from_bits(0x7fc00002)).unwrap()).unwrap();
        assert_ne!(positive, negative);
        assert_ne!(positive.to_string(), negative.to_string());
        assert_eq!(nan, nan.clone());
        assert_ne!(nan, other_nan);
        assert_ne!(nan.to_string(), other_nan.to_string());
        let projected = <XlaConstant as ValueProjection<ArrayType>>::into_projected(nan.clone()).unwrap();
        assert_eq!(<XlaConstant as ValueProjection<ArrayType>>::from_projected(projected), nan);
        assert!(<XlaConstant as ValueProjection<DimensionType>>::into_projected(positive).is_err());
    }

    #[test]
    fn test_xla_operation_values_satisfy_the_core_capability_bundles() {
        // Each bundle is satisfied exactly when every one of its member capabilities is, so instantiating these
        // functions is a compile-time assertion that the XLA value families expose the complete value-level surface
        // of the operation families they pair with.
        fn requires_array_operations<V: ArrayOperations>() {}
        fn requires_array_ir_operations<V: ArrayIrOperations>() {}
        fn requires_dimension_operations<V: ryft_core::DimensionOperations>() {}

        requires_array_operations::<XlaArray<'static>>();
        requires_array_operations::<XlaArrayTracer>();
        requires_array_operations::<crate::XlaValue<'static>>();
        requires_array_ir_operations::<crate::XlaValue<'static>>();
        requires_dimension_operations::<crate::XlaDimension<'static>>();
        requires_array_operations::<XlaTracer<'static>>();
        requires_array_ir_operations::<XlaTracer<'static>>();
        requires_array_ir_operations::<Tracer<TracingContext<XlaConstant, XlaOperation>>>();
    }

    #[test]
    fn test_xla_operation_payload_projection_and_construction() {
        // Array payloads are reached through the projected array member in both directions, including tags, while
        // native payloads construct their own variants.
        let add = ErasedOperation::new(AddOperation::<ArrayType>::new());
        let add = XlaOperation::<XlaConstant>::from_payload(add).unwrap();
        assert!(matches!(add, XlaOperation::Array(ArrayOperation::Add(_))));
        assert!(add.projected_payload::<AddOperation<ArrayType>>().is_some());
        let reference = ErasedOperation::new(ReferenceNewOperation::<ArrayType, ArrayIrType>::new());
        let reference = XlaOperation::<XlaConstant>::from_payload(reference).unwrap();
        assert!(matches!(reference, XlaOperation::ReferenceNew(_)));
        assert!(reference.projected_payload::<ReferenceNewOperation<ArrayType, ArrayIrType>>().is_some());
        assert!(reference.projected_payload::<AddOperation<ArrayType>>().is_none());
        let tag = ErasedOperation::new(TagOperation::<ArrayType>::new("residual"));
        let tag = XlaOperation::<XlaConstant>::from_payload(tag).unwrap();
        assert_eq!(TagOperation::<ArrayIrType>::key_of(&tag), Some("residual"));
    }

    #[test]
    fn test_xla_operation_provides_memory_transfers() {
        // The XLA family provides memory transfers through its projected array member.
        let destination = Memory::Host { pinned: true };
        let input_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let operation =
            XlaOperation::<XlaConstant>::provide(TransferToMemoryOperation::new(destination), &[&input_type]).unwrap();
        assert!(matches!(
            &operation,
            XlaOperation::Array(ArrayOperation::TransferToMemory(operation)) if operation.destination() == destination,
        ));
        assert_eq!(operation.to_string(), "transfer_to_memory [destination=Host[Pinned]]");
        assert_eq!(
            XlaOperation::<XlaConstant>::provide(TransferToMemoryOperation::new(destination), &[]).unwrap_err(),
            ProgramError::InvalidInputCount { expected: 1, actual: 0 },
        );
    }

    #[test]
    fn test_xla_dimension_assertion_proofs() {
        let dimension_type = DimensionType::new("extent", DimensionBounds::unbounded());
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let dimension = context.input(dimension_type.clone().into());
        let dimension = ValueProjection::<DimensionType>::into_projected(dimension).unwrap();
        let predicate = dimension.equal(&dimension).unwrap();
        assert!(matches!(context.resolve(&predicate), ValueResolution::Constant(XlaConstant::Boolean(true))));
        predicate.assert("identity must agree", &[]).unwrap();
        let predicate = dimension.not_equal(&dimension).unwrap();
        predicate.assert("identity must differ", &[]).unwrap();
        assert!(matches!(
            context.builder().borrow().instructions(),
            [instruction] if matches!(
                instruction.operation(),
                XlaOperation::Assert(operation) if operation.message() == "identity must differ",
            ),
        ));

        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let mut builder = XlaProgramBuilder::new();
        let dimension = builder.add_input(dimension_type.clone().into());
        let predicate = builder
            .add_instruction(
                CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                vec![],
                vec![dimension, dimension],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(AssertOperation::<ArrayIrType>::new("identity must agree"), vec![], vec![predicate], None)
            .unwrap();
        let program = builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder], vec![]).unwrap();
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let evaluation = program
            .partially_evaluate_in_context(&context, &[PartialValue::Unknown(dimension_type.into())])
            .unwrap();
        assert_eq!(evaluation.program().effects().classes(), EffectClasses::NONE);
        assert!(evaluation.program().instructions().is_empty());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_xla_operation_batch_dimension_member() {
        use ryft_core::{
            ArrayIrBatch, ArrayIrBatchingPolicy, BatchableOperation, BatchingContext, DimensionDivOperation,
            DimensionOperation, EmptyRegionDriver,
        };

        // The first-class-dimension member family has no batching rules of its own, so the derived dispatcher binds a
        // checked dimension division once in the parent trace under the replicated dimension policy. The staged
        // division keeps its replicated inputs and its ordered zero-divisor assertion.
        let dividend_type = DimensionType::new("dividend", DimensionBounds::new(0, Some(9)).unwrap());
        let divisor_type = DimensionType::new("divisor", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<XlaConstant, XlaOperation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(XlaConstant::Dimension(DimensionValue::constant(3).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let operation = XlaOperation::from(DimensionOperation::Div(DimensionDivOperation::new(
                    &dividend_type,
                    &divisor_type,
                )?));
                let (outputs, evidence) = operation
                    .batch(
                        &context,
                        &EmptyRegionDriver,
                        &[ArrayIrBatch::replicated(inputs[0].clone()), ArrayIrBatch::replicated(inputs[1].clone())],
                    )?
                    .into_parts();
                assert!(outputs.iter().all(|output| output.batch_axis().is_replicated()));
                assert!(evidence.is_empty());
                Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
            },
            vec![ArrayIrType::Dimension(dividend_type.clone()), ArrayIrType::Dimension(divisor_type.clone())],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<dividend ∈ [0, 9)>, %1:dimension<divisor ∈ [0, 5)> .
                let %2:dimension<3> = const 3
                    %3:dimension<dividend / divisor ∈ [0, 9)> = dimension_div [requires_runtime_assertion=true] %0 %1
                in (%3)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_xla_constant() {
        let array_type = vector_type();
        let capture = CaptureReference::new(2, ArrayIrType::Array(array_type.clone()));
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let extent = DimensionValue::new(extent_type, 4).unwrap();

        // Captures and dimensions are reachable through their payload conversions, report their member type, and render
        // exactly like the payload they wrap.
        let captured = XlaConstant::from(capture.clone());
        let immediate = XlaConstant::from(extent.clone());
        assert_eq!(captured, XlaConstant::Captured(capture));
        assert_eq!(immediate, XlaConstant::Dimension(extent.clone()));
        assert_eq!(captured.r#type().into_owned(), ArrayIrType::Array(array_type.clone()));
        assert_eq!(immediate.r#type().into_owned(), ArrayIrType::Dimension(extent.r#type().into_owned()));
        assert_eq!(captured.to_string(), "capture#2:f64[4]");
        assert_eq!(immediate.to_string(), extent.to_string());

        let predicate = XlaConstant::try_from(true).unwrap();
        assert_eq!(predicate, XlaConstant::Boolean(true));
        assert_eq!(predicate.r#type().into_owned(), ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        assert_eq!(predicate.to_string(), "true");
        assert_eq!(ryft_core::Concretizable::<bool>::concretize(&predicate), Ok(true));
        assert_eq!(predicate.capture_index(), None);
        assert_eq!(predicate.map_capture_index(|_| 99), predicate);
        let projected = ValueProjection::<ArrayType>::into_projected(predicate.clone()).unwrap();
        assert_eq!(projected, XlaArrayConstant::Boolean(true));
        assert_eq!(ryft_core::Concretizable::<bool>::concretize(&projected), Ok(true));
        assert_eq!(projected.capture_index(), None);
        assert_eq!(projected.map_capture_index(|_| 99), projected);
        assert_eq!(<XlaConstant as ValueProjection<ArrayType>>::from_projected(projected), predicate);

        // Only the captured variant names a capture-table slot, and so only it is renumbered by capture bookkeeping.
        // An immediate is index-free and passes through every remapping unchanged.
        assert_eq!(captured.capture_index(), Some(2));
        assert_eq!(immediate.capture_index(), None);
        assert_eq!(
            captured.map_capture_index(|index| index + 3),
            XlaConstant::Captured(CaptureReference::new(5, ArrayIrType::Array(array_type.clone()))),
        );
        assert_eq!(immediate.map_capture_index(|_| 7), immediate);

        // The array-typed captured case and immediate-dimension case project to their corresponding member universes
        // and reject the other one.
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::into_projected(captured.clone()),
            Ok(XlaArrayConstant::Captured(CaptureReference::new(2, array_type.clone()))),
        );
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::into_projected(immediate.clone()),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::from_projected(XlaArrayConstant::Captured(
                CaptureReference::new(2, array_type)
            )),
            captured,
        );
        assert_eq!(<XlaConstant as ValueProjection<DimensionType>>::projected(&immediate), Ok(&extent));
        assert_eq!(<XlaConstant as ValueProjection<DimensionType>>::into_projected(immediate), Ok(extent),);
        assert_eq!(
            <XlaConstant as ValueProjection<DimensionType>>::projected(&captured),
            Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
        );

        // A reference-typed capture carries only its capture-table index and structural type metadata. Identity
        // renaming preserves that index, while ordinary array/dimension projections remain unavailable. (No
        // reference-typed `ValueProjection` exists: capture lifting is its intended consumer and has not landed.)
        let source = DimensionVariable::new("source", DimensionBounds::positive(Some(8)).unwrap());
        let target = DimensionVariable::new("target", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())])));
        let reference = XlaConstant::Captured(CaptureReference::new(7, ArrayIrType::Reference(reference_type.clone())));
        assert_eq!(reference.capture_index(), Some(7));
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::projected(&reference).map(|value| value.r#type().into_owned()),
            Err(TypeError::invalid("expected array type but got reference type")),
        );
        assert_eq!(
            <XlaConstant as ValueProjection<DimensionType>>::projected(&reference),
            Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
        );
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();
        let renamed_reference = reference.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed_reference.capture_index(), Some(7));
        assert_eq!(
            renamed_reference.r#type().into_owned(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target)]),
            ))),
        );
    }

    #[test]
    fn test_core_reference_operations_promote_to_xla_reference_operations() {
        let write: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceWrite(ReferenceWriteOperation::new()).into();
        assert!(matches!(&write, XlaOperation::ReferenceWrite(_)));

        let swap: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceSwap(ReferenceSwapOperation::new()).into();
        assert!(matches!(&swap, XlaOperation::ReferenceSwap(_)));

        let add_update: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()).into();
        assert!(matches!(&add_update, XlaOperation::ReferenceAddUpdate(_)));

        let atomic_update: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation::new())
                .into();
        assert!(matches!(&atomic_update, XlaOperation::ReferenceAtomicAddUpdate(_)));

        let freeze: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceFreeze(ReferenceFreezeOperation::new()).into();
        assert!(matches!(&freeze, XlaOperation::ReferenceFreeze(_)));
    }

    #[test]
    fn test_core_ragged_operations_promote_to_xla_ragged_operations() {
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2]),
            RaggedValueKind::Dimension(Dimension::Static(3).into()),
            Vec::new(),
        )
        .unwrap();
        let construct: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::RaggedNew(RaggedNewOperation::new(r#type.clone())).into();
        assert!(matches!(&construct, XlaOperation::RaggedNew(operation) if operation.output_type() == &r#type));

        let storage: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::RaggedStorage(RaggedStorageOperation).into();
        assert!(matches!(&storage, XlaOperation::RaggedStorage(_)));

        let extents: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::RaggedExtents(RaggedExtentsOperation::new(-1)).into();
        assert!(matches!(&extents, XlaOperation::RaggedExtents(operation) if operation.axis() == (-1).into()));
    }

    #[test]
    fn test_xla_reference_discharge_rewrites_the_flat_reference_language() {
        // The backend operation family participates in reference discharge through its generated `discharge`
        // dispatcher, so a flat backend program rewrites into explicit immutable state with its external root
        // entering at its own boundary position and publishing one hidden final-state output.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let external = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type.clone())));
        let initial = builder.add_input(ArrayIrType::Array(scalar_type.clone()));
        let update = builder.add_input(ArrayIrType::Array(scalar_type));
        let root = builder
            .add_instruction(XlaOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
            .unwrap()[0];
        builder
            .add_instruction(
                XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                Vec::new(),
                vec![root, update],
                None,
            )
            .unwrap();
        let local = builder
            .add_instruction(
                XlaOperation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let previous = builder
            .add_instruction(
                XlaOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                Vec::new(),
                vec![external, local],
                None,
            )
            .unwrap()[0];
        let source: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>> =
            builder.build(vec![previous], vec![Placeholder; 3], vec![Placeholder]).unwrap();

        let discharged = source.discharge_references(0).unwrap();

        // The one external root is written, so it enters as state at its own boundary position and publishes its
        // final state as the single hidden output after the public prefix, while the local root's allocation,
        // accumulation, and consumption become ordinary array SSA.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[] = add %1 %2
                in (%0, %3)"},
        );
    }

    #[test]
    fn test_core_composite_control_flow_promotes_to_xla_control_flow() {
        let condition: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, XlaOperation::Condition(_)));
        assert_eq!(condition.region_slots(), ConditionOperation::<ArrayIrType>::new().region_slots());

        let r#while: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap()).into();
        assert!(matches!(r#while, XlaOperation::While(operation) if operation.iteration_bound() == Some(3)));

        let scan: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
        assert!(matches!(
            scan,
            XlaOperation::Scan(operation)
                if operation.carry_count() == 2
                    && operation.length() == &Dimension::Static(5)
                    && operation.reverse()
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let zero: XlaOperation<XlaConstant> = ArrayOperation::Zero(ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )))
        .into();
        assert!(matches!(zero, XlaOperation::Zero(_)));

        // Member control flow must also promote to the composite carriers when it enters through the member family
        // conversion: the projected `Array` variant cannot own composite regions, so landing there would make the
        // staged regions unprojectable.
        let condition: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, XlaOperation::Condition(_)));
        let r#while: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap()).into();
        assert!(matches!(r#while, XlaOperation::While(operation) if operation.iteration_bound() == Some(3)));
        let scan: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
        assert!(matches!(
            scan,
            XlaOperation::Scan(operation)
                if operation.carry_count() == 2
                    && operation.length() == &Dimension::Static(5)
                    && operation.reverse()
        ));
    }

    #[test]
    fn test_xla_operation_reference_access_descriptor() {
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let operation = XlaOperation::<XlaConstant>::ReferenceRead(
            ReferenceReadOperation::new().with_transforms(vec![transform.clone()]),
        );
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), &[transform]);
        assert_eq!(descriptor.bindings(), 1..2);
        assert_eq!(operation.reference_access_descriptor(1), None);
        let root = ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 2])).into();
        let index = ArrayType::scalar(DataType::I64).into();
        assert_eq!(
            operation.infer_output_types(&[root, index], &[]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2]).into()])
        );
    }

    #[test]
    fn test_core_custom_function_and_rematerialization_promotions_preserve_metadata() {
        use ryft_core::{DotsSavable, ResidualPolicyReference};

        // These payloads are promoted by move rather than reconstructed, so their complete stored surface must
        // survive: the rule layout and the non-differentiated operand split of custom-function calls, and the policy
        // and flags of rematerialized calls. The promoted carrier must also keep
        // contributing the payload's own operation name and region slots, because the attached regions are matched
        // against those slots by name.
        for (jvp_rule, has_vjp_rule, non_differentiated_count) in [
            (CustomFunctionJvpRule::Explicit, false, 2),
            (CustomFunctionJvpRule::Absent, true, 3),
            (CustomFunctionJvpRule::Primal, true, 1),
        ] {
            let custom_function = CustomFunctionOperation::from_rule_regions(jvp_rule, has_vjp_rule)
                .with_non_differentiated_count(non_differentiated_count)
                .unwrap();
            let promoted: XlaOperation<XlaConstant> =
                ArrayIrOperation::<XlaArrayConstant>::CustomFunction(custom_function.clone()).into();
            assert!(matches!(
                &promoted,
                XlaOperation::CustomFunction(operation)
                    if operation.rules().is_none()
                        && operation.jvp_rule() == jvp_rule
                        && operation.has_vjp_rule() == has_vjp_rule
                        && operation.non_differentiated_count() == non_differentiated_count,
            ));
            assert_eq!(promoted.name(), custom_function.name());
            assert_eq!(promoted.region_slots(), custom_function.region_slots());

            // A call with attached rules that reaches this family from the array member family follows its regions
            // into the native variant as well.
            let promoted: XlaOperation<XlaConstant> = ArrayOperation::<XlaArrayConstant>::CustomFunction(
                CustomFunctionOperation::from_rule_regions(jvp_rule, has_vjp_rule),
            )
            .into();
            assert!(matches!(
                &promoted,
                XlaOperation::CustomFunction(operation)
                    if operation.rules().is_none()
                        && operation.jvp_rule() == jvp_rule
                        && operation.has_vjp_rule() == has_vjp_rule,
            ));
        }

        // A carrier with an attached backward rule keeps its tangent interface and follows its region into the native
        // carrier variant.
        let tangent_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let carrier = CustomFunctionTransposeOperation::from_backward_region(
            1,
            vec![tangent_type.clone()],
            vec![tangent_type.clone()],
        );
        let promoted: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::CustomFunctionTranspose(carrier).into();
        assert!(matches!(
            &promoted,
            XlaOperation::CustomFunctionTranspose(operation)
                if operation.rules().is_none()
                    && operation.leading_input_count() == 1
                    && operation.input_tangent_types() == std::slice::from_ref(&tangent_type)
                    && operation.output_tangent_types() == std::slice::from_ref(&tangent_type),
        ));

        let rematerialize = RematerializeOperation::<ArrayIrType>::new(ResidualPolicyReference::new(DotsSavable))
            .with_optimization_barrier(RematerializationOptimizationBarrier::None)
            .with_differentiated(true);
        let promoted: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::Rematerialize(rematerialize.clone()).into();
        assert!(matches!(&promoted, XlaOperation::Rematerialize(operation) if operation == &rematerialize));
        assert_eq!(promoted.name(), rematerialize.name());
        assert_eq!(promoted.region_slots(), rematerialize.region_slots());
    }

    #[test]
    fn test_core_custom_function_retained_rule_promotions() {
        use ryft_core::{ArrayIrValue, CustomRuleDefinition, CustomRuleRegistration, LiftedCustomRules};

        // Converted member calls and carriers keep their member definition, which specializes their rules for this
        // family on demand.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let member = CustomRuleRegistration::new(CustomRuleDefinition::<
            XlaArrayConstant,
            ArrayOperation<XlaArrayConstant>,
        >::new("member"));
        let promoted: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::CustomFunction(CustomFunctionOperation::new(member.reference())).into();
        assert!(matches!(
            &promoted,
            XlaOperation::LiftedCustomFunction(operation)
                if operation.rules() == Some(&LiftedCustomRules::Member(member.reference())),
        ));
        assert_eq!(promoted.to_string(), "custom_function [name=\"member\"]");
        let promoted: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::CustomFunctionTranspose(CustomFunctionTransposeOperation::new(
                member.reference(),
                0,
                vec![scalar_type.clone()],
                vec![scalar_type.clone()],
            ))
            .into();
        assert!(matches!(&promoted, XlaOperation::LiftedCustomFunctionTranspose(_)));

        // Calls registered in the array IR family itself cannot follow their program, because constants of this family
        // cannot represent array IR values. Their primal still executes, but their explicit rules are unavailable.
        let composite = CustomRuleRegistration::new(
            CustomRuleDefinition::<ArrayIrValue<XlaArrayConstant>, ArrayIrOperation<XlaArrayConstant>>::new(
                "composite",
            )
            .with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec()))),
        );
        let promoted: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::CustomFunction(CustomFunctionOperation::new(composite.reference()))
                .into();
        assert!(matches!(
            &promoted,
            XlaOperation::LiftedCustomFunction(operation)
                if matches!(operation.rules(), Some(LiftedCustomRules::Unavailable(_))),
        ));
        let mut primal = XlaProgramBuilder::new();
        let input = primal.add_input(scalar_type.clone().into());
        let primal = primal
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let primal = builder.import_program(primal);
        let input = builder.add_input(scalar_type.into());
        let output = builder.add_instruction(promoted, vec![primal], vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap_err().to_string(),
            "`custom_function` `composite` was converted from a family whose values its current family cannot \
             represent, so its derivative rules are unavailable",
        );
    }

    #[test]
    fn test_rematerialize_recomputes_known_jit_calls_in_its_body() {
        use ryft_core::{NothingSavable, rematerialize};

        // A rematerialized body squares its input inside a jitted call and squares the result of the call outside of
        // it. Linearizing it with the default policy, which saves nothing, saves only the input. Its pure all-known
        // call is inlined into the policy walk; the differentiated call recomputes that multiplication.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let callee = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![squared], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let function = rematerialize(move |x: XlaTracer<'static>| -> Result<XlaTracer<'static>, ProgramError> {
            let context = x.context().clone();
            let called = context.bind(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee.clone()], &[x])?;
            let mut outputs = context.bind(MulOperation::new(), Vec::new(), &[called[0].clone(), called[0].clone()])?;
            Ok(outputs.remove(0))
        })
        .with_policy(NothingSavable);
        let (_, program) =
            DomainTracingContext::<XlaDomain<'static>>::trace(|x: XlaTracer<'static>| function.call(x), scalar_type)
                .unwrap();
        let linearization = program.into_flat_program().linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = mul %1 %1
                in (%2, %0)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = rematerialize [differentiated=true] %0 %1 [
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %1
                            %3:f64[] = jit_call %0 %1 [
                                callee={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = mul %1 %0
                                        %3:f64[] = mul %1 %0
                                        %4:f64[] = add %2 %3
                                    in (%4)
                                },
                            ]
                            %4:f64[] = mul %2 %3
                            %5:f64[] = mul %2 %3
                            %6:f64[] = add %4 %5
                        in (%6)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_rematerialize_rejects_captures_registered_in_its_body() {
        use ryft_core::rematerialize;

        // The rematerialized body is traced through a fresh-root context whose capture table is discarded, so a
        // capturing body is rejected at trace time; refer to `TracingContext::trace_with_named_axes` for the full
        // silent-aliasing rationale.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let captured_value = XlaConstant::Captured(CaptureReference::new(0, scalar_type.clone()));
        let function = rematerialize(move |x: XlaTracer<'static>| -> Result<XlaTracer<'static>, ProgramError> {
            let context = x.context().clone();
            let reference = context.capture(captured_value.clone())?;
            let captured = StagingContext::constant(&context, reference);
            let mut outputs = context.bind(AddOperation::new(), Vec::new(), &[x, captured])?;
            Ok(outputs.remove(0))
        });
        let root = DomainTracingContext::<XlaDomain<'static>>::new();
        let input = root.input(scalar_type);
        let result = function.call(input);
        assert!(matches!(result, Err(ProgramError::DiscardedCaptures { count: 1 })));
    }

    #[test]
    fn test_xla_dynamic_disconnected_pullback_omits_structural_zero_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = XlaProgramBuilder::new();
        builder.add_input(dynamic_type.into());
        let scalar = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![scalar],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // A disconnected cotangent remains structural and spends no extent residual or executable output.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert!(linearization.primal().instructions().is_empty());
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
            lambda %0:f64[] .
            in (%0)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_xla_residual_zero_provider_materializes_dynamic_zero_from_array_source() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let tangent_type = primal_type.tangent().unwrap();
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let primal = context.input(ArrayIrType::Array(primal_type));

        // The shared array-IR blanket captures the named extent even when the source's element representation differs
        // from the dynamic tangent zero's. The XLA family uses the same identity-directed rule as the core family.
        let zero = XlaOperation::<XlaConstant>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(ArrayIrType::Array(tangent_type.clone())),
            std::slice::from_ref(&primal),
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &ArrayIrType::Array(tangent_type));

        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one dynamic zero instruction");
        };
        assert!(matches!(dimension_size.operation(), XlaOperation::DimensionSize(operation) if operation.axis() == 0));
        assert_eq!(dimension_size.inputs(), &[primal.atom_id().unwrap()]);
        assert!(matches!(zero.operation(), XlaOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0]]);
    }

    #[test]
    fn test_xla_residual_zero_provider_assembles_dynamic_zero_from_several_sources() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(7)).unwrap());
        let zero_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
        );
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let sources = [
            context.input(ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])))),
            context.input(ArrayIrType::Array(ArrayType::new(
                DataType::F8E8M0FNU,
                Shape::new(vec![Dimension::Dynamic(rows)]),
            ))),
            context.input(ArrayIrType::Dimension(DimensionType::from(columns))),
        ];

        // No single source has the zero's type, so its shape is assembled per declared residual: the statically shaped
        // candidate names neither identity and is skipped without staging anything, the dynamic array supplies the row
        // extent through a `dimension_size` read, and the first-class dimension supplies the column extent as itself.
        // The shared array-IR implementation emits XLA operations directly for the selected sources.
        let zero = XlaOperation::<XlaConstant>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(ArrayIrType::Array(zero_type.clone())),
            &sources,
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &ArrayIrType::Array(zero_type));

        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one dynamic zero instruction");
        };
        assert!(matches!(dimension_size.operation(), XlaOperation::DimensionSize(operation) if operation.axis() == 0));
        assert_eq!(dimension_size.inputs(), &[sources[1].atom_id().unwrap()]);
        assert!(matches!(zero.operation(), XlaOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0], sources[2].atom_id().unwrap()]);
    }

    #[test]
    fn test_gateway_region_program_imports_with_fresh_alpha_equivalent_identities() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        let branch = || {
            let mut builder = XlaProgramBuilder::new();
            let extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let integer_type = ArrayType::scalar(DataType::I32);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut source_builder = XlaProgramBuilder::new();
        let true_region = source_builder.import_region(branch().entry_region_ref());
        let false_region = source_builder.import_region(branch().entry_region_ref());
        let integer = source_builder.add_input(integer_type.clone().into());
        let predicate = source_builder.add_input(predicate_type.clone().into());
        let scalar = source_builder.add_input(scalar_type.clone().into());
        let gateway = source_builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![integer], None)
            .unwrap()[0];
        let output = source_builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, gateway, scalar],
                None,
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(source.output_types(), vec![ArrayIrType::Array(output_type)]);
        assert!(source.type_identity_signature().input_identities().is_empty());
        assert_eq!(
            source.type_identity_signature().internal_identities(),
            std::slice::from_ref(extent_type.variable().unwrap())
        );

        let mut destination = XlaProgramBuilder::new();
        let integer = destination.add_input(integer_type.into());
        let predicate = destination.add_input(predicate_type.into());
        let scalar = destination.add_input(scalar_type.into());
        let first = destination.splice_program(&source, &[integer, predicate, scalar]).unwrap()[0];
        let second = destination.splice_program(&source, &[integer, predicate, scalar]).unwrap()[0];

        let [first_gateway, first_condition, second_gateway, second_condition] = destination.instructions() else {
            panic!("expected two imported gateway-condition pairs");
        };
        assert_eq!(first_condition.inputs(), &[predicate, first_gateway.outputs()[0], scalar]);
        assert_eq!(second_condition.inputs(), &[predicate, second_gateway.outputs()[0], scalar]);
        assert_eq!(first_condition.outputs(), &[first]);
        assert_eq!(second_condition.outputs(), &[second]);
        assert_eq!(first_condition.regions().len(), 2);
        assert_eq!(second_condition.regions().len(), 2);

        let first_type = destination.atoms()[first_gateway.outputs()[0].index()].r#type().into_owned();
        let second_type = destination.atoms()[second_gateway.outputs()[0].index()].r#type().into_owned();
        let first_dimension = <&DimensionType>::try_from(&first_type).unwrap();
        let second_dimension = <&DimensionType>::try_from(&second_type).unwrap();
        assert_ne!(first_dimension.variable(), second_dimension.variable());
        assert_eq!(first_dimension.bounds(), second_dimension.bounds());
        for (condition, dimension) in [(first_condition, first_dimension), (second_condition, second_dimension)] {
            for region in condition.regions() {
                let interface = destination.region_ref(*region).unwrap().interface();
                assert_eq!(interface.input_types()[0], ArrayIrType::Dimension(dimension.clone()));
                let output = <&ArrayType>::try_from(&interface.output_types()[0]).unwrap();
                assert_eq!(output.shape().dimensions(), &[dimension.dimension().clone()],);
            }
        }

        let destination = destination
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![first, second],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let signature = destination.type_identity_signature();
        assert!(signature.input_identities().is_empty());
        assert_eq!(
            signature.internal_identities(),
            &[first_dimension.variable().unwrap().clone(), second_dimension.variable().unwrap().clone()],
        );
    }

    #[test]
    fn test_scan_reference_discharge_widens_carries_and_preserves_scan_metadata() {
        let reference_type = ReferenceType::new(vector_type());
        let mut body_builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let body_carry = body_builder.add_input(vector_type().into());
        let reference =
            body_builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, reference_type.clone().into())));
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![body_carry, value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        builder.add_input(reference_type.into());
        let carry = builder.add_input(vector_type().into());
        let operation = ScanOperation::new(1, 3).with_reverse(true).with_unroll(3).unwrap();
        let outputs = builder
            .add_instruction(XlaOperation::Scan(operation), vec![body], vec![carry], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let targets = program.reference_discharge_targets(1).unwrap();
        assert_eq!(targets, vec![ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })],);

        // The partial capture-aware entry point resolves the body's capture constant even when its allocation is
        // preserved. The scan retains its reference access and gains a reference-typed carry that binds the rebuilt
        // body's capture without turning it into state.
        let preserved = program.clone().partially_discharge_references_in_capture_lifted_program(1, &[]).unwrap();
        assert!(preserved.external_reference_bindings().is_empty());
        assert!(preserved.program().entry_region_ref().contains_references_in_closure());
        let XlaOperation::Scan(scan) = preserved.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected a partially discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);

        // Selecting the captured allocation through the partial entry point agrees exactly with full capture-aware
        // discharge. Threading its state widens the carry list while preserving the scan's direction and unroll factor.
        let selected = ReferenceDischargeResult::try_from(
            program
                .clone()
                .partially_discharge_references_in_capture_lifted_program(1, targets.as_slice())
                .unwrap(),
        )
        .unwrap();
        let discharged = program.discharge_references_in_capture_lifted_program(1).unwrap();
        assert_eq!(selected.program().to_string(), discharged.program().to_string());
        assert_eq!(selected.external_reference_bindings(), discharged.external_reference_bindings());
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        let XlaOperation::Scan(scan) = discharged.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected a discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.unroll(), 3);
        assert!(scan.reverse());
    }

    #[test]
    fn test_rematerialization_policies_are_available_for_the_xla_operation_family() {
        use ryft_core::{
            DotsSavable, DotsWithNoBatchDimensionsSavable, EverythingSavable, Memory, NothingSavable,
            OffloadDotsWithNoBatchDimensions, ResidualPolicy, SaveAndOffloadOnlyTheseNames, SaveFromBothPolicies,
            SaveOnlyTheseNames,
        };

        // The built-in rematerialization policies, including the offloading ones, classify candidates of the
        // `ArrayIrType` universe of `XlaOperation`, whose producers they recognize through the derive-generated
        // payload projection. This is a compile-time capability check: the assertions below fail to compile if a
        // policy is not available for the universe.
        fn assert_policy<P: ResidualPolicy<ArrayIrType>>(_policy: P) {}
        assert_policy(NothingSavable);
        assert_policy(EverythingSavable);
        assert_policy(DotsSavable);
        assert_policy(DotsWithNoBatchDimensionsSavable);
        assert_policy(SaveOnlyTheseNames::new(["u"]));
        assert_policy(SaveAndOffloadOnlyTheseNames::new(["u"], ["v"], Memory::Host { pinned: true }).unwrap());
        assert_policy(OffloadDotsWithNoBatchDimensions::new(Memory::Host { pinned: true }));
        assert_policy(SaveFromBothPolicies::new(DotsSavable, SaveOnlyTheseNames::new(["u"])));
    }
}
