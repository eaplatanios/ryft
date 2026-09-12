// Derive macros emitted by `ryft-macros` use the public `ryft::...` facade path by default. This `self`-alias
// lets those same generated paths resolve when those macros are used inside `ryft-core` itself.
extern crate self as ryft;

pub mod arrays;
pub mod axes;
pub mod batching;
pub mod captures;
pub mod compilation;
pub mod contexts;
pub mod differentiation;
pub mod errors;
pub mod interpretation;
pub mod macros;
pub mod operations;
pub mod parameters;
pub mod partial;
pub mod programs;
pub mod specialization;
pub mod tracing;
pub mod tracing_v2;
pub mod utilities;

pub use arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayExtentBatchingPolicy, ArrayIndexRange,
    ArrayIndexRanges, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrOperations, ArrayIrType,
    ArrayIrTypeRefinements, ArrayIrValue, ArrayOperation, ArrayOperations, ArrayReference, ArrayReferenceAnalysis,
    ArrayReferenceDischarge, ArrayReferenceView, ArrayReferenceViewError, ArrayReferenceViewIndex,
    ArrayReferenceViewOperation, ArrayReferenceViewPath, ArraySliceAxis, ArrayTracingContext, ArrayType,
    ArrayTypeRefinements, Broadcastable, BroadcastingError, Complex, DataType, DataTypeError, Device, DeviceId,
    DeviceMesh, Dimension, DimensionBounds, DimensionError, DimensionOperation, DimensionOperations, DimensionSource,
    DimensionTracingContext, DimensionType, DimensionValue, DimensionVariable, ExactShape, ExactShapeDimension, Layout,
    LayoutError, LinearResiduals, LogicalMesh, MAX_DIMENSION_EXTENT, Memory, MeshAxis, MeshAxisType, ProcessIndex,
    REFERENCE_INDEX_OPERATION_NAME, REFERENCE_SLICE_OPERATION_NAME, RaggedArrayExtentBatchingPolicy, RaggedAxis,
    RaggedMaskIdentity, ReferenceDynamicIndex, ReferenceDynamicIndexOperation, ReferenceIndex, ReferenceIndexOperation,
    ReferenceSlice, ReferenceSliceOperation, ReplicatedDimensionBatchingPolicy, Shape, Sharding, ShardingDimension,
    ShardingError, ShardingVisualization, StaticArrayExtentBatchingPolicy, StaticShape, StridedLayout, Tile,
    TileDimension, TiledLayout, bf16, decode_elements, decode_logical_bytes, encode_elements, encode_logical_bytes,
    f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu,
    f16, i1, i2, i4, materialize_array_tangent, reapply_array_reference_view, u1, u2, u4,
    validate_array_reference_view,
};
pub use axes::{AXIS_INDEX_OPERATION_NAME, Axes, Axis, AxisError, AxisIndex, AxisIndexOperation, NamedAxes, NamedAxis};
pub use batching::{
    Batch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableType, BatchedOutputs, BatchedProgram,
    BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError, BatchingPolicy, BatchingPolicyProjection,
    BatchingTracer, BoundaryPreservingBatchedProgram, InterpretableBatchableOperation, MemberBatchableOperation,
    ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy, batch, batch_projected_operation,
};
pub use captures::{CaptureConstant, CaptureReference, CapturingContext, ClosedProgram};
pub use compilation::{
    AnalyzableCompilationDomain, CallRequest, CompilationArtifactExchange, CompilationArtifactExchangePolicy,
    CompilationCacheDomain, CompilationCacheLevel, CompilationCacheOutcome, CompilationCacheStatistics,
    CompilationCall, CompilationContext, CompilationDomain, CompilationEvent, CompilationExchangeError,
    CompilationMissReason, CompilationStagingRequest, CompilationTracer, CompileRequest, CompiledCallOperation,
    CompiledFunction, CompiledFunctionDispatcher, DiskCache, ExecutableFunction, FlatCompilationProgram,
    FunctionSpecializationKey, JitCacheStatistics, LoweredFunction, LoweringRequest, ReferenceExecution, StageRequest,
    StagedFunction, StatefulCompilationDomain, call_function, call_function_statefully, call_function_statefully_async,
    jit, jit_with_options, stage_function, try_jit, try_jit_with_options, try_jit_with_options_and_capacity,
};
pub use contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext, ValueResolution};
pub use differentiation::{
    BinaryElementwiseJvpOperands, BroadcastDerivativeAlignment, CotangentAccumulator, CotangentBatchingPolicy,
    CotangentDestination, CotangentDestinationKind, CotangentDestinations, CotangentReferenceAccumulator,
    CotangentSeed, DenseDifferentiableType, DerivativeTransform, DifferentiableOperation, DifferentiableType,
    Differentiate, DifferentiationBoundaryPosition, DifferentiationBuilder, DifferentiationBuilderContext,
    DifferentiationBuilderLinearityMode, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationParameterRole, DifferentiationPolicy, DifferentiationTracer,
    ElementwiseDerivativeAlignment, ForwardModeDifferentiate, FusedDifferentiationPolicy, Hessian, HessianBlock,
    HolomorphicLinearity, Jacobian, JacobianBlock, Linearization, LinearizationContext, LinearizationTracer,
    MemberDifferentiableOperation, MemberTransposableOperation, PartitionedDifferentiationPolicy, Pullback,
    Pushforward, RealLinearity, ResidualZeroProvider, ReverseModeDifferentiate, TransposableOperation,
    TranspositionContext, TranspositionDriver, UnaryElementwiseJvpOperands, WithAuxiliaryOutput, WithCapture,
    WithContext, WithoutAuxiliaryOutput, WithoutCapture, WithoutContext, binary_elementwise_jvp, differentiate_at,
    jvp_projected_operation, transpose_mixed_operation, transpose_projected_operation, unary_elementwise_jvp,
};
pub use errors::{CustomError, Error, MaybeFallible};
pub use interpretation::{
    InterpretableOperation, InterpretationDriver, MemberInterpretableOperation, interpret_projected_operation,
};
pub use operations::{
    ABS_OPERATION_NAME, ADD_OPERATION_NAME, AND_OPERATION_NAME, ATAN2_OPERATION_NAME, Abs, AbsOperation, Add,
    AddOperation, And, AndOperation, ArithmeticDimensionOperation, Atan2, Atan2Operation, BROADCAST_OPERATION_NAME,
    BlockQuantize, Broadcast, BroadcastOperation, CEIL_OPERATION_NAME, COMPARE_OPERATION_NAME,
    CONCATENATE_OPERATION_NAME, CONDITION_OPERATION_NAME, CONSTANT_OPERATION_NAME, CONVERT_ELEMENT_TYPE_OPERATION_NAME,
    COS_OPERATION_NAME, CUMULATIVE_LOG_SUM_EXP_OPERATION_NAME, CUMULATIVE_MAX_OPERATION_NAME,
    CUMULATIVE_MIN_OPERATION_NAME, CUMULATIVE_PRODUCT_OPERATION_NAME, CUMULATIVE_SUM_OPERATION_NAME,
    CUSTOM_JVP_OPERATION_NAME, CUSTOM_VJP_OPERATION_NAME, Ceil, CeilOperation, Clamp, Compare, CompareOperation,
    ComparisonDirection, Concatenate, ConcatenateOperation, ConditionOperation, Constant, ConstantOperation,
    ConstrainSharding, ConvertElementType, ConvertElementTypeOperation, Cos, CosOperation, CumulativeLogSumExp,
    CumulativeLogSumExpOperation, CumulativeMax, CumulativeMaxOperation, CumulativeMin, CumulativeMinOperation,
    CumulativeProduct, CumulativeProductOperation, CumulativeSum, CumulativeSumOperation, CustomJvp,
    CustomJvpOperation, CustomVjp, CustomVjpOperation, DIMENSION_ADD_OPERATION_NAME,
    DIMENSION_DIV_FLOOR_OPERATION_NAME, DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_MAX_OPERATION_NAME,
    DIMENSION_MIN_OPERATION_NAME, DIMENSION_MUL_OPERATION_NAME, DIMENSION_POW_OPERATION_NAME,
    DIMENSION_REM_OPERATION_NAME, DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME,
    DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME, DIMENSION_REQUIRE_EQUAL_OPERATION_NAME,
    DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME, DIMENSION_SATURATING_SUB_OPERATION_NAME,
    DIMENSION_SIZE_OPERATION_NAME, DIMENSION_SUB_OPERATION_NAME, DIMENSION_TO_SCALAR_OPERATION_NAME,
    DIV_OPERATION_NAME, DOT_OPERATION_NAME, DYNAMIC_SHAPE_SLICE_OPERATION_NAME, DYNAMIC_SLICE_OPERATION_NAME,
    DYNAMIC_UPDATE_SLICE_OPERATION_NAME, DimensionAddOperation, DimensionArithmetic, DimensionDivFloorOperation,
    DimensionFromScalar, DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin,
    DimensionMinOperation, DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation,
    DimensionRequirement, DimensionRequirementOperation, DimensionRequirementPredicate, DimensionSaturatingSub,
    DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation, DimensionSubOperation, DimensionToScalar,
    DimensionToScalarOperation, Div, DivOperation, Dot, DotDimensionNumbers, DotOperation, DotOps, DynamicBroadcast,
    DynamicBroadcastOperation, DynamicFill, DynamicIota, DynamicOne, DynamicReshape, DynamicReshapeOperation,
    DynamicShapeSliceOperation, DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation,
    DynamicZero, ERF_OPERATION_NAME, EXP_OPERATION_NAME, ElementType, ElementwiseOperation, Erf, ErfOperation, Exp,
    ExpOperation, FLOOR_OPERATION_NAME, Fill, Floor, FloorOperation, GATHER_OPERATION_NAME, Gather,
    GatherDimensionNumbers, GatherOperation, GatherScatterMode, IOTA_OPERATION_NAME, Iota, IotaOperation,
    LOG_ADD_EXP_OPERATION_NAME, LOG_OPERATION_NAME, LOG_SUM_EXP_OPERATION_NAME, LOG1P_OPERATION_NAME,
    LOGISTIC_OPERATION_NAME, LinearCallOperation, Log, Log1p, Log1pOperation, LogAddExp, LogAddExpOperation,
    LogOperation, LogSumExp, LogSumExpOperation, Logistic, LogisticOperation, MAX_OPERATION_NAME, MIN_OPERATION_NAME,
    MUL_OPERATION_NAME, Max, MaxOperation, Min, MinOperation, Mul, MulOperation, NEG_OPERATION_NAME,
    NOT_OPERATION_NAME, Neg, NegOperation, Not, NotOperation, ONE_LIKE_OPERATION_NAME, ONE_OPERATION_NAME,
    OR_OPERATION_NAME, One, OneLike, OneLikeOperation, OneOperation, Or, OrOperation, PAD_OPERATION_NAME,
    POW_OPERATION_NAME, PRINT_OPERATION_NAME, Pad, PadOperation, ParallelReduce, ParallelReduceOperation,
    ParallelReductionKind, Permutation, Pow, PowOperation, Print, PrintOperation, RAGGED_DOT_OPERATION_NAME,
    REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME, REFERENCE_NEW_OPERATION_NAME,
    REFERENCE_READ_OPERATION_NAME, REFERENCE_SWAP_OPERATION_NAME, REFERENCE_WRITE_OPERATION_NAME, REM_OPERATION_NAME,
    RESHAPE_OPERATION_NAME, RESHARD_OPERATION_NAME, ROUND_OPERATION_NAME, RSQRT_OPERATION_NAME,
    RUNTIME_DIMENSION_DATA_TYPE, RaggedDot, RaggedDotDimensionNumbers, RaggedDotMode, RaggedDotOperation, Reduce,
    ReduceOperation, ReductionKind, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceFreeze,
    ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead, ReferenceReadOperation,
    ReferenceSwap, ReferenceSwapOperation, ReferenceWrite, ReferenceWriteOperation, Rem, RemOperation, Reshape,
    ReshapeOperation, ReshapeParameters, Reshard, ReshardOperation, Round, RoundOperation, Rsqrt, RsqrtOperation,
    SCALED_DOT_OPERATION_NAME, SCAN_OPERATION_NAME, SCATTER_OPERATION_NAME, SELECT_OPERATION_NAME,
    SHARDING_CONSTRAINT_OPERATION_NAME, SIGN_OPERATION_NAME, SIN_OPERATION_NAME, SLICE_OPERATION_NAME,
    SQRT_OPERATION_NAME, STOP_GRADIENT_OPERATION_NAME, SUB_OPERATION_NAME, ScaledDot, ScaledDotOperation,
    ScanOperation, ScanReferenceDischarge, Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind,
    Select, SelectOperation, ShardingConstraintOperation, Sign, SignOperation, Sin, SinOperation, Slice,
    SliceOperation, Sqrt, SqrtOperation, StopGradient, StopGradientOperation, StopGradients, Sub, SubOperation,
    TAG_OPERATION_NAME, TANH_OPERATION_NAME, TRANSFER_TO_MEMORY_OPERATION_NAME, TRANSPOSE_OPERATION_NAME, Tag,
    TagOperation, Tanh, TanhOperation, TransferToMemory, TransferToMemoryOperation, Transpose, TransposeOperation,
    UPDATE_SLICE_OPERATION_NAME, UpdateSlice, UpdateSliceOperation, WHILE_OPERATION_NAME, WhileOperation,
    WhilePredicate, WhileTypeSemantics, XOR_OPERATION_NAME, Xor, XorOperation, ZERO_LIKE_OPERATION_NAME,
    ZERO_OPERATION_NAME, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation, custom_jvp, custom_vjp,
    forward_collective_to_parent, transpose_primal_condition, transpose_primal_scan,
};
pub use parameters::{
    ArrayParameterizedFamily, BTreeMapParameterizedFamily, HashMapParameterizedFamily, Parameter, ParameterError,
    ParameterParameterizedFamily, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
    PathPrefixedParameterIterator, PhantomDataParameterizedFamily, Placeholder, VecParameterizedFamily,
};
pub use partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialTracer, PartialValue, PartialValueMaterialization,
    PartiallyEvaluatableOperation, PartitionedProgram,
};
pub use programs::{
    Atom, AtomId, AttachedRegionStatistics, BatchableReferenceView, BindingRegionDriver, CalleeRegionDriver,
    Concretizable, DestinationRegionMapping, EffectClass, EffectClassOccurrence, EffectClasses, Effects,
    EffectsSummary, EmptyRegionDriver, ExternalReferenceBinding, FlatProgram, Instruction, InstructionId, MaybeZero,
    MemberOperation, NoIdentity, NoReferenceViewBinding, Operation, OperationFormatter, OperationProjection,
    OperationProvider, OutputRegionProvenance, ParameterProjection, PartialReferenceDischargeResult,
    PreparedReferenceReplacement, Program, ProgramBuilder, ProgramBuilderId, ProgramError, ProgramLiveSets,
    ProgramRenderingMode, ProgramStatistics, ProjectedValue, Provenance, ProvenanceScope, ProvenanceState,
    ReadyOrPendingReferenceGuard, ReadyReferenceGuard, RecursiveReferenceDischargeDriver, Reference, ReferenceAccess,
    ReferenceAccessMode, ReferenceAccumulationPolicy, ReferenceAlias, ReferenceAliasEdge, ReferenceAliasKind,
    ReferenceAnalysis, ReferenceAnalysisError, ReferenceBoundary, ReferenceBoundaryError, ReferenceBoundaryPosition,
    ReferenceCompletion, ReferenceCompletionBackend, ReferenceDischargeAllocationId,
    ReferenceDischargeBoundaryWidening, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeReference, ReferenceDischargeRegionBoundary, ReferenceDischargeRegionBoundaryInsertion,
    ReferenceDischargeRegionInput, ReferenceDischargeRegionOutput, ReferenceDischargeRegionResult,
    ReferenceDischargeRegionSummary, ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceDischargeableType, ReferenceEffect, ReferenceError, ReferenceGeneration,
    ReferenceId, ReferenceIdentity, ReferenceObservation, ReferenceRegionInputBinding, ReferenceReplacementPreparation,
    ReferenceReplacementTransaction, ReferenceRoot, ReferenceSource, ReferenceTransitiveAccess, ReferenceType,
    ReferenceTypeRefinements, ReferenceView, ReferenceViewAnalysis, ReferenceViewAnalysisError, ReferenceViewOperation,
    ReferenceViewOverlap, ReferenceViewPath, ReferenceViewStep, ReferenceViewValidationError, Region, RegionArena,
    RegionArenaIterator, RegionDriver, RegionId, RegionInterface, RegionRef, RegionReplayMappings, RegionRole,
    RegionSlot, RegionStatistics, RegionWithMetadata, ReplayRegionDriver, TakenReferenceGuard, Transform,
    TransformArtifact, TransformCache, Type, TypeError, TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming,
    TypeIdentitySignature, TypeRefinements, Typed, ValidatedPendingReplacementTransaction, Value, ValueId,
    ValueProjection, discharge_local_reference_operation, discharge_positional_region_operation,
    discharge_reference_free_operation, infer_projected_operation_output_types,
    infer_projected_operation_region_input_types, validate_reference_boundary,
};
pub use specialization::{
    ReentrantSpecializationError, SpecializationCache, SpecializationCacheEntry, SpecializationCacheError,
    SpecializationCacheProducer, SpecializationCacheStatistics,
};
pub use tracing::{
    DomainTracer, DomainTracingContext, NestedTracer, NestedTracingContext, Trace, Tracer, TracerState, TracingContext,
    infer_output_type, trace,
};
pub use tracing_v2::rematerialization::{REMATERIALIZE_OPERATION_NAME, RematerializeOperation};

#[cfg(test)]
pub(crate) mod tests {
    use std::any::TypeId;
    use std::borrow::Cow;
    use std::convert::Infallible;
    use std::fmt::{Debug, Display};
    use std::sync::{Arc, Weak};

    use ryft_macros::Operation;

    use crate::arrays::{Array, ArrayIrType, ArrayIrValue, ArrayType};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext,
        DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy, TransposableOperation,
        TranspositionContext, TranspositionDriver,
    };
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::{
        AddOperation, BroadcastOperation, CompareOperation, ConstantOperation, ConvertElementTypeOperation,
        DivOperation, MulOperation, NegOperation, OneLikeOperation, OneOperation, ReduceOperation,
        ReferenceReadOperation, ReferenceWriteOperation, ReshapeOperation, ReshardOperation, TransposeOperation,
        ZeroLikeOperation, ZeroOperation,
    };
    use crate::parameters::Parameter;
    use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
    use crate::programs::transforms::{RegionTransformCache, RegionTransformRegistry};
    use crate::programs::{
        EffectClass, EffectClasses, Effects, MaybeZero, NoIdentity, Operation, OperationProjection, Program,
        ProgramError, Region, RegionInterface, RegionRef, RegionSlot, Transform, TransformArtifact, Type, TypeError,
        Typed, Value, ValueProjection,
    };
    use crate::specialization::SpecializationCacheStatistics;
    use crate::tracing::{Tracer, TracingContext};

    /// Small ordinary array operation family for core tests. Each payload is a real operation, while malformed rules
    /// and arbitrary region interfaces remain explicit, separate protocol fixtures. Captures and interpretation need
    /// constant and arithmetic operations, and tracing's static constructors additionally need the zero and one
    /// operation types. Broadcast and transpose support staged batching alignment. Differentiation's shared alignment
    /// rules also require conversion, reduction, reshape, and reshard. Reduction derivatives require compare and
    /// divide. These dependencies apply to the operation family even when a particular scalar test emits none of
    /// those operations. This family has no region-bearing payloads and therefore needs no value type parameter.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type = ArrayType, constant = Array, dispatch(batching, differentiation, transposition))]
    pub(crate) enum TestArrayOperation {
        Constant(ConstantOperation<Array>),
        Zero(ZeroOperation<ArrayType>),
        One(OneOperation<ArrayType>),
        ZeroLike(ZeroLikeOperation<ArrayType>),
        OneLike(OneLikeOperation<ArrayType>),
        Neg(NegOperation<ArrayType>),
        Add(AddOperation<ArrayType>),
        Mul(MulOperation<ArrayType>),
        ConvertElementType(ConvertElementTypeOperation<ArrayType>),
        Broadcast(BroadcastOperation),
        Transpose(TransposeOperation),
        Reshape(ReshapeOperation),
        Reduce(ReduceOperation),
        Reshard(ReshardOperation),
        Compare(CompareOperation<ArrayType>),
        Div(DivOperation<ArrayType>),
    }

    /// Eager array context with only the operations required by ordinary core protocol tests.
    pub(crate) type TestArrayContext = EagerContext<Array, TestArrayOperation>;

    /// Staging context for the same ordinary array family, used to replay residual programs into an outer trace.
    pub(crate) type TestArrayTracingContext = TracingContext<Array, TestArrayOperation>;

    /// Small reference-capable family for mixed Intermediate Representation (IR) boundary tests. Keeping its type
    /// universe explicit avoids treating a reference as an ordinary array or importing the complete production mixed
    /// operation catalog.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type = ArrayIrType, constant = ArrayIrValue<Array>, members(ArrayType))]
    pub(crate) enum TestArrayIrOperation {
        Constant(Box<ConstantOperation<ArrayIrValue<Array>>>),
        #[ryft(projected(ArrayType))]
        Array(Box<TestArrayOperation>),
        ReferenceRead(ReferenceReadOperation<ArrayType, ArrayIrType>),
        ReferenceWrite(ReferenceWriteOperation<ArrayType, ArrayIrType>),
    }

    /// Eager context for mixed array/reference boundary tests using the small mixed operation family.
    pub(crate) type TestArrayIrContext = EagerContext<ArrayIrValue<Array>, TestArrayIrOperation>;

    /// Test-only homogeneous member type used by the generic projected-context fixtures.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct ProjectedMemberType<const MEMBER: u8>;

    impl<const MEMBER: u8> Display for ProjectedMemberType<MEMBER> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "member_{MEMBER}")
        }
    }

    impl<const MEMBER: u8> Parameter for ProjectedMemberType<MEMBER> {}

    impl<const MEMBER: u8> Type for ProjectedMemberType<MEMBER> {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl<const MEMBER: u8> DifferentiableType for ProjectedMemberType<MEMBER> {
        fn is_zero_space(&self) -> bool {
            false
        }

        fn tangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }

        fn cotangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }
    }

    /// Test-only concrete value for one homogeneous projected member.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct ProjectedMemberValue<const MEMBER: u8>(pub(crate) usize);

    impl<const MEMBER: u8> Display for ProjectedMemberValue<MEMBER> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.0)
        }
    }

    impl<const MEMBER: u8> Parameter for ProjectedMemberValue<MEMBER> {}

    impl<const MEMBER: u8> Typed for ProjectedMemberValue<MEMBER> {
        type Type = ProjectedMemberType<MEMBER>;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            Cow::Owned(ProjectedMemberType)
        }
    }

    impl<const MEMBER: u8> Value for ProjectedMemberValue<MEMBER> {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    /// Test-only composite storage type with three distinct member kinds.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedProgramType {
        /// First member kind, used by the ordinary projection tests.
        First(ProjectedMemberType<0>),

        /// Second member kind, which exercises the additional-member extensibility gate.
        Second(ProjectedMemberType<1>),

        /// Third member kind, used by transform tests to prove that generic machinery is member-kind-agnostic.
        Third(ProjectedMemberType<2>),
    }

    impl Display for ProjectedProgramType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::First(r#type) => Display::fmt(r#type, formatter),
                Self::Second(r#type) => Display::fmt(r#type, formatter),
                Self::Third(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for ProjectedProgramType {}

    impl Type for ProjectedProgramType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl DifferentiableType for ProjectedProgramType {
        fn is_zero_space(&self) -> bool {
            false
        }

        fn tangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }

        fn cotangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }
    }

    /// Test-only composite storage value mirroring [`ProjectedProgramType`].
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedProgramValue {
        /// First member value, used by the ordinary projection tests.
        First(ProjectedMemberValue<0>),

        /// Second member value, which exercises the additional-member extensibility gate.
        Second(ProjectedMemberValue<1>),

        /// Third member value, used by transform tests to prove that generic machinery is member-kind-agnostic.
        Third(ProjectedMemberValue<2>),
    }

    impl Display for ProjectedProgramValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::First(value) => Display::fmt(value, formatter),
                Self::Second(value) => Display::fmt(value, formatter),
                Self::Third(value) => Display::fmt(value, formatter),
            }
        }
    }

    impl Parameter for ProjectedProgramValue {}

    impl Typed for ProjectedProgramValue {
        type Type = ProjectedProgramType;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            Cow::Owned(match self {
                Self::First(_) => ProjectedProgramType::First(ProjectedMemberType),
                Self::Second(_) => ProjectedProgramType::Second(ProjectedMemberType),
                Self::Third(_) => ProjectedProgramType::Third(ProjectedMemberType),
            })
        }
    }

    impl Value for ProjectedProgramValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    /// Implements type/value conversion and projection for one named member of the test composite family.
    macro_rules! impl_projected_test_member {
        ($member:literal, $variant:ident) => {
            impl From<ProjectedMemberType<$member>> for ProjectedProgramType {
                fn from(r#type: ProjectedMemberType<$member>) -> Self {
                    Self::$variant(r#type)
                }
            }

            impl<'t> TryFrom<&'t ProjectedProgramType> for &'t ProjectedMemberType<$member> {
                type Error = TypeError;

                fn try_from(r#type: &'t ProjectedProgramType) -> Result<Self, Self::Error> {
                    match r#type {
                        ProjectedProgramType::$variant(r#type) => Ok(r#type),
                        _ => Err(TypeError::invalid(format!("expected member {} but got {}", $member, r#type))),
                    }
                }
            }

            impl ValueProjection<ProjectedMemberType<$member>> for ProjectedProgramValue {
                type Projected = ProjectedMemberValue<$member>;
                type ProjectedRef<'v>
                    = &'v ProjectedMemberValue<$member>
                where
                    Self: 'v;

                fn from_projected(value: Self::Projected) -> Self {
                    Self::$variant(value)
                }

                fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
                where
                    ProjectedMemberType<$member>: 'v,
                {
                    match self {
                        Self::$variant(value) => Ok(value),
                        _ => Err(TypeError::invalid(format!("expected member {} but got {}", $member, self.r#type()))),
                    }
                }

                fn into_projected(self) -> Result<Self::Projected, TypeError> {
                    match self {
                        Self::$variant(value) => Ok(value),
                        _ => Err(TypeError::invalid(format!("expected member {} but got {}", $member, self.r#type()))),
                    }
                }
            }

            impl From<ProjectedMemberOperation<$member>> for ProjectedProgramOperation {
                fn from(operation: ProjectedMemberOperation<$member>) -> Self {
                    Self::$variant(operation)
                }
            }

            impl OperationProjection<ProjectedMemberType<$member>> for ProjectedProgramOperation {
                type Projected = ProjectedMemberOperation<$member>;
            }
        };
    }

    impl_projected_test_member!(0, First);
    impl_projected_test_member!(1, Second);
    impl_projected_test_member!(2, Third);

    /// Test-only homogeneous identity and addition operations for one projected member kind.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedMemberOperation<const MEMBER: u8> {
        /// Preserves one value.
        Identity,

        /// Adds two values in the same member family.
        Add,
    }

    impl<const MEMBER: u8> From<AddOperation<ProjectedMemberType<MEMBER>>> for ProjectedMemberOperation<MEMBER> {
        fn from(_: AddOperation<ProjectedMemberType<MEMBER>>) -> Self {
            Self::Add
        }
    }

    impl<const MEMBER: u8> Operation for ProjectedMemberOperation<MEMBER> {
        type Type = ProjectedMemberType<MEMBER>;

        fn name(&self) -> &'static str {
            match self {
                Self::Identity => "projected_member",
                Self::Add => "add",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ProjectedMemberType<MEMBER>],
            _region_interfaces: &[RegionInterface<ProjectedMemberType<MEMBER>>],
        ) -> Result<Vec<ProjectedMemberType<MEMBER>>, TypeError> {
            check_count!("input", input_types, if matches!(self, Self::Identity) { 1 } else { 2 }, TypeError);
            Ok(vec![ProjectedMemberType])
        }
    }

    impl<const MEMBER: u8, C: Context<Type = ProjectedMemberType<MEMBER>, Operation: From<Self>>>
        DifferentiableOperation<C> for ProjectedMemberOperation<MEMBER>
    {
        fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
            &self,
            context: &DifferentiationContext<C, P>,
            _driver: &D,
            inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
            check_count!("input", inputs, if matches!(self, Self::Identity) { 1 } else { 2 }, ProgramError);
            let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.primal().bind(self.clone(), Vec::new(), &primals)?.remove(0);
            let tangent = if matches!(self, Self::Identity) {
                inputs[0].tangent().clone()
            } else {
                match (inputs[0].tangent(), inputs[1].tangent()) {
                    (MaybeZero::Zero(_), tangent) | (tangent, MaybeZero::Zero(_)) => tangent.clone(),
                    (MaybeZero::Value(left), MaybeZero::Value(right)) => MaybeZero::Value(
                        context.tangent().bind(self.clone(), Vec::new(), &[left.clone(), right.clone()])?.remove(0),
                    ),
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    }

    impl<
        const MEMBER: u8,
        V: Value<Type = ProjectedMemberType<MEMBER>>,
        O: Operation<Type = ProjectedMemberType<MEMBER>> + From<AddOperation<ProjectedMemberType<MEMBER>>>,
    > TransposableOperation<V, O> for ProjectedMemberOperation<MEMBER>
    {
        fn transpose<D: TranspositionDriver<V, O>>(
            &self,
            context: &mut TranspositionContext<V, O>,
            _driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            check_count!("input", inputs, if matches!(self, Self::Identity) { 1 } else { 2 }, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            for accumulator in accumulators {
                accumulator.accumulate(context, outputs[0].clone())?;
            }
            Ok(())
        }
    }

    /// Test-only composite operation family embedding all three member families.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedProgramOperation {
        /// First member operation, used by the ordinary projection tests.
        First(ProjectedMemberOperation<0>),

        /// Second member operation, which exercises the additional-member extensibility gate.
        Second(ProjectedMemberOperation<1>),

        /// Third member operation, used by transform tests to prove that generic machinery is member-kind-agnostic.
        Third(ProjectedMemberOperation<2>),

        /// Adds two values of the same composite member type.
        Add,
    }

    impl From<AddOperation<ProjectedProgramType>> for ProjectedProgramOperation {
        fn from(_: AddOperation<ProjectedProgramType>) -> Self {
            Self::Add
        }
    }

    impl ProjectedProgramOperation {
        /// Delegates composite inference to one member operation. Fixture operations declare no region slots,
        /// so staging rejects attached regions before inference and the member sees none.
        fn infer_member<const MEMBER: u8>(
            operation: &ProjectedMemberOperation<MEMBER>,
            input_types: &[ProjectedProgramType],
        ) -> Result<Vec<ProjectedProgramType>, TypeError>
        where
            for<'t> &'t ProjectedMemberType<MEMBER>: TryFrom<&'t ProjectedProgramType, Error = TypeError>,
            ProjectedProgramType: From<ProjectedMemberType<MEMBER>>,
        {
            let input_types = input_types
                .iter()
                .map(|r#type| <&ProjectedMemberType<MEMBER>>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            operation
                .infer_output_types(input_types.as_slice(), &[])
                .map(|types| types.into_iter().map(Into::into).collect())
        }
    }

    impl Operation for ProjectedProgramOperation {
        type Type = ProjectedProgramType;

        fn name(&self) -> &'static str {
            match self {
                Self::First(operation) => operation.name(),
                Self::Second(operation) => operation.name(),
                Self::Third(operation) => operation.name(),
                Self::Add => "add",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ProjectedProgramType],
            _region_interfaces: &[RegionInterface<ProjectedProgramType>],
        ) -> Result<Vec<ProjectedProgramType>, TypeError> {
            match self {
                Self::First(operation) => Self::infer_member(operation, input_types),
                Self::Second(operation) => Self::infer_member(operation, input_types),
                Self::Third(operation) => Self::infer_member(operation, input_types),
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    if input_types[0] != input_types[1] {
                        return Err(TypeError::invalid("addition requires matching member types"));
                    }
                    Ok(vec![input_types[0].clone()])
                }
            }
        }
    }

    impl InterpretableOperation<EagerContext<ProjectedProgramValue, Self>> for ProjectedProgramOperation {
        fn interpret<D: InterpretationDriver<EagerContext<ProjectedProgramValue, Self>>>(
            &self,
            _context: &EagerContext<ProjectedProgramValue, Self>,
            _driver: &D,
            inputs: &[ProjectedProgramValue],
        ) -> Result<Vec<ProjectedProgramValue>, ProgramError> {
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            self.infer_output_types(input_types.as_slice(), &[])?;
            if matches!(
                self,
                Self::Add
                    | Self::First(ProjectedMemberOperation::Add)
                    | Self::Second(ProjectedMemberOperation::Add)
                    | Self::Third(ProjectedMemberOperation::Add)
            ) {
                let output = match (&inputs[0], &inputs[1]) {
                    (ProjectedProgramValue::First(left), ProjectedProgramValue::First(right)) => {
                        ProjectedProgramValue::First(ProjectedMemberValue(left.0 + right.0))
                    }
                    (ProjectedProgramValue::Second(left), ProjectedProgramValue::Second(right)) => {
                        ProjectedProgramValue::Second(ProjectedMemberValue(left.0 + right.0))
                    }
                    (ProjectedProgramValue::Third(left), ProjectedProgramValue::Third(right)) => {
                        ProjectedProgramValue::Third(ProjectedMemberValue(left.0 + right.0))
                    }
                    _ => unreachable!(),
                };
                Ok(vec![output])
            } else {
                Ok(inputs.to_vec())
            }
        }
    }

    /// Test [`Operation`] with declared attached-region slots, used to exercise the [`Region`](crate::Region) machinery
    /// (i.e., construction, interning and sharing, interface derivation, validation, effects propagation, rendering,
    /// splicing, and rebuild paths) in isolation. Production region-carrying operations (e.g., the control-flow family)
    /// impose their own type invariants such as Boolean predicates and congruent branch outputs, whereas this fixture
    /// declares arbitrary region slot names with a trivial inference rule, so machinery tests stay three-line fixtures
    /// whose failures cannot be masked by control-flow inference.
    #[derive(Clone, Debug, PartialEq)]
    pub enum TestRegionOperation {
        /// Region-free binary addition stand-in used inside region bodies.
        Add,

        /// Region-free unary identity stand-in with the declared observable effect.
        Effectful(EffectClass),

        /// Region-carrying operation declaring its region slots. Its inferred output types are the first attached
        /// region's output types, which pins that region interfaces are derived and delivered during inference.
        WithRegions(&'static [RegionSlot]),

        /// Pure call whose body input types must match its operands, used by builder identity-instantiation tests.
        Call,
    }

    impl Operation for TestRegionOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "add",
                Self::Effectful(_) => "effectful",
                Self::WithRegions(_) => "with_regions",
                Self::Call => "array_identity",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Add | Self::Effectful(_) => &[],
                Self::WithRegions(slots) => slots,
                Self::Call => const { &[RegionSlot::computation("body")] },
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Effectful(_) => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::WithRegions(names) => {
                    check_count!("input", input_types, 1, TypeError);
                    if region_interfaces.len() != names.len() {
                        return Err(TypeError::invalid(format!(
                            "expected {} region interfaces but got {}",
                            names.len(),
                            region_interfaces.len(),
                        )));
                    }
                    Ok(region_interfaces[0].output_types().to_vec())
                }
                Self::Call => {
                    let [region_interface] = region_interfaces else {
                        return Err(TypeError::invalid(format!(
                            "array identity expects 1 attached region but got {}",
                            region_interfaces.len(),
                        )));
                    };
                    if region_interface.input_types() != input_types {
                        return Err(TypeError::invalid(
                            "array identity region input types do not match its operand types",
                        ));
                    }
                    Ok(region_interface.output_types().to_vec())
                }
            }
        }

        fn effects(&self) -> Cow<'_, Effects> {
            Cow::Owned(Effects::explicit(match self {
                Self::Add | Self::WithRegions(_) | Self::Call => EffectClasses::NONE,
                Self::Effectful(effect) => EffectClasses::single(*effect),
            }))
        }
    }

    /// Region-free test [`Operation`] family isolating ordered-state effect handling (i.e., simplification liveness
    /// and ordering, rematerialization boundaries) from reference-operation semantics.
    #[derive(Clone, Debug, PartialEq)]
    pub enum TestOrderedStateOperation {
        /// Pure unary work that transforms may remove when its result is dead.
        Pure,

        /// Unary ordered-state access carrying a stable ordinal used to assert relative order.
        State(u8),
    }

    impl Operation for TestOrderedStateOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            match self {
                Self::Pure => "pure",
                Self::State(_) => "state",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            _region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(input_types.to_vec())
        }

        fn effects(&self) -> Cow<'_, Effects> {
            Cow::Owned(Effects::explicit(if matches!(self, Self::State(_)) {
                EffectClasses::single(EffectClass::OrderedState)
            } else {
                EffectClasses::NONE
            }))
        }
    }

    impl InterpretableOperation<EagerContext<Array, Self>> for TestOrderedStateOperation {
        fn interpret<D: InterpretationDriver<EagerContext<Array, Self>>>(
            &self,
            _context: &EagerContext<Array, Self>,
            _driver: &D,
            inputs: &[Array],
        ) -> Result<Vec<Array>, ProgramError> {
            Ok(inputs.to_vec())
        }
    }

    impl PartiallyEvaluatableOperation<EagerContext<Array, Self>> for TestOrderedStateOperation {}

    /// Transform marker used to observe generic region-cache preservation without depending on a built-in transform.
    pub(crate) struct IdentityTransform;

    impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for IdentityTransform {
        type Arguments = ();
        type Artifact = TransformArtifact<V, O, ()>;

        const DEFAULT_CACHE_CAPACITY: usize = 1;
    }

    /// Weak handle used to test [`RegionTransformCache`] ownership and cycle behavior.
    pub(crate) struct WeakRegionTransformCache<V: Typed + Parameter, O> {
        /// Weak reference to the cache state under test.
        state: Weak<RegionTransformRegistry<V, O>>,
    }

    impl<V: Typed + Parameter, O> WeakRegionTransformCache<V, O> {
        /// Returns whether the source cache state is still retained.
        #[inline]
        pub(crate) fn is_alive(&self) -> bool {
            self.state.upgrade().is_some()
        }
    }

    impl<V: Typed + Parameter, O> RegionTransformCache<V, O> {
        /// Returns a weak handle to this cache state for ownership tests.
        #[inline]
        pub(crate) fn downgrade(&self) -> WeakRegionTransformCache<V, O> {
            WeakRegionTransformCache { state: Arc::downgrade(&self.state) }
        }

        /// Returns whether any transform namespace has been created on this cache.
        #[inline]
        pub(crate) fn has_namespaces(&self) -> bool {
            !self.state.lock().expect("region transform registry mutex is poisoned").is_empty()
        }

        /// Returns statistics for `T` when its namespace has been initialized.
        pub(crate) fn statistics<T: 'static>(&self) -> Option<SpecializationCacheStatistics> {
            self.state
                .lock()
                .expect("region transform registry mutex is poisoned")
                .get(&TypeId::of::<T>())
                .map(|namespace| namespace.statistics())
        }
    }

    impl<'r, V: Value, O: Operation<Type = V::Type>> RegionRef<'r, V, O> {
        /// Returns statistics for transform `T` when its namespace has been initialized.
        #[inline]
        pub(crate) fn transform_statistics<T: 'static>(self) -> Option<SpecializationCacheStatistics> {
            self.transform_cache().statistics::<T>()
        }

        /// Inserts a purpose-built artifact into `T`'s namespace for diagnostic-corruption and provenance tests.
        #[cfg(debug_assertions)]
        pub(crate) fn insert_transform_artifact_for_testing<
            T: 'static
                + Transform<
                    Region<V, O>,
                    Arguments: 'static + Debug + Send + Sync,
                    Artifact = TransformArtifact<V, O, Metadata>,
                >,
            Metadata: 'static + Clone + Debug + PartialEq + Send + Sync,
        >(
            self,
            arguments: T::Arguments,
            artifact: TransformArtifact<V, O, Metadata>,
        ) {
            let previous_productions = self.transform_statistics::<T>().map_or(0, |statistics| statistics.productions);
            let retained = self.transform::<T, _, Infallible>(arguments, move |_, _| Ok(artifact)).unwrap();
            drop(retained);
            let statistics = self.transform_statistics::<T>().unwrap();
            assert_eq!(
                statistics.productions,
                previous_productions + 1,
                "test transform namespace entry must be vacant",
            );
        }

        /// Returns this region materialized through the test-only [`IdentityTransform`] namespace.
        pub(crate) fn retained_identity_transform(self) -> Arc<Program<V, O, Vec<V>, Vec<V>>> {
            let artifact = self
                .transform::<IdentityTransform, _, Infallible>((), |region, _| {
                    Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], ()))
                })
                .unwrap();
            let (mut programs, ()) = artifact.into_parts();
            assert_eq!(programs.len(), 1);
            programs.pop().unwrap()
        }
    }
}
