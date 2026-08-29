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
    Array, ArrayAddressing, ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayElement, ArrayIndexRange,
    ArrayIndexRanges, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrOperations, ArrayIrType,
    ArrayIrTypeRefinements, ArrayIrValue, ArrayOperation, ArrayOperations, ArrayReference, ArrayReferenceDischarge,
    ArrayReferenceView, ArrayReferenceViewError, ArrayReferenceViewOperation, ArrayReferenceViewTransform,
    ArraySliceAxis, ArrayTracingContext, ArrayType, ArrayTypeRefinements, Broadcastable, BroadcastingError, Complex,
    DataType, DataTypeError, Device, DeviceId, DeviceMesh, Dimension, DimensionBounds, DimensionError,
    DimensionOperation, DimensionOperations, DimensionSource, DimensionTracingContext, DimensionType, DimensionValue,
    DimensionVariable, ExactShape, ExactShapeDimension, Layout, LayoutError, LinearResiduals, LogicalMesh,
    MAX_DIMENSION_EXTENT, Memory, MeshAxis, MeshAxisType, ProcessIndex, REFERENCE_INDEX_OPERATION_NAME,
    REFERENCE_SLICE_OPERATION_NAME, RaggedArrayBatchingPolicy, RaggedAxis, RaggedMaskIdentity, ReferenceIndex,
    ReferenceIndexOperation, ReferenceSlice, ReferenceSliceOperation, ReplicatedDimensionBatchingPolicy, Shape,
    Sharding, ShardingDimension, ShardingError, ShardingVisualization, StaticArrayBatchingPolicy, StaticShape,
    StridedLayout, Tile, TileDimension, TiledLayout, bf16, decode_elements, decode_logical_bytes, encode_elements,
    encode_logical_bytes, f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2,
    f8e5m2fnuz, f8e8m0fnu, f16, i1, i2, i4, materialize_array_tangent, u1, u2, u4, validate_storage_bytes,
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
    BinaryElementwiseJvpOperands, BroadcastDerivativeAlignment, CotangentBatchingPolicy, CustomJvp, CustomJvpOperation,
    CustomVjp, CustomVjpOperation, DenseDifferentiableType, DerivativeTransform, DifferentiableOperation,
    DifferentiableType, Differentiate, DifferentiationBuilder, DifferentiationBuilderContext,
    DifferentiationBuilderExecutionContext, DifferentiationBuilderLinearityMode, DifferentiationContext,
    DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationParameterRole,
    DifferentiationTracer, ElementwiseDerivativeAlignment, ForwardModeDifferentiate, Hessian, HessianBlock,
    HolomorphicLinearity, Jacobian, JacobianBlock, LinearCallOperation, Linearization, LinearizationTracer,
    MemberDifferentiableOperation, Pullback, Pushforward, RealLinearity, ResidualZeroProvider,
    ReverseModeDifferentiate, STOP_GRADIENT_OPERATION_NAME, StopGradient, StopGradientOperation, StopGradients,
    TransposableOperation, TranspositionDriver, UnaryElementwiseJvpOperands, WithAuxiliaryOutput, WithCapture,
    WithContext, WithoutAuxiliaryOutput, WithoutCapture, WithoutContext, binary_elementwise_jvp, custom_jvp,
    custom_vjp, differentiate_at, jvp_projected_operation, transpose_mixed_operation, transpose_projected_operation,
    unary_elementwise_jvp,
};
pub use errors::{CustomError, Error, MaybeFallible};
pub use interpretation::{
    InterpretableOperation, InterpretationDriver, MemberInterpretableOperation, interpret_projected_operation,
};
// TODO(eaplatanios): We should be importing directly from `operations` and not from nested modules.
pub use operations::compare::{COMPARE_OPERATION_NAME, Compare, CompareOperation, ComparisonDirection};
pub use operations::constants::{
    CONSTANT_OPERATION_NAME, Constant, ConstantOperation, Fill, IOTA_OPERATION_NAME, Iota, IotaOperation,
    ONE_LIKE_OPERATION_NAME, ONE_OPERATION_NAME, One, OneLike, OneLikeOperation, OneOperation, OneOperationProvider,
    ZERO_LIKE_OPERATION_NAME, ZERO_OPERATION_NAME, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
    ZeroOperationProvider,
};
pub use operations::control_flow::{
    CONDITION_OPERATION_NAME, ConditionOperation, SCAN_OPERATION_NAME, SELECT_OPERATION_NAME, ScanOperation, Select,
    SelectOperation, WHILE_OPERATION_NAME, WhileOperation, WhilePredicate, WhileTypeSemantics,
    transpose_primal_condition, transpose_primal_scan,
};
pub use operations::cumulative::{
    CUMULATIVE_LOG_SUM_EXP_OPERATION_NAME, CUMULATIVE_MAX_OPERATION_NAME, CUMULATIVE_MIN_OPERATION_NAME,
    CUMULATIVE_PRODUCT_OPERATION_NAME, CUMULATIVE_SUM_OPERATION_NAME, CumulativeLogSumExp,
    CumulativeLogSumExpOperation, CumulativeMax, CumulativeMaxOperation, CumulativeMin, CumulativeMinOperation,
    CumulativeProduct, CumulativeProductOperation, CumulativeSum, CumulativeSumOperation,
};
pub use operations::dot::{
    DOT_OPERATION_NAME, Dot, DotDimensionNumbers, DotOperation, DotOps, RAGGED_DOT_OPERATION_NAME, RaggedDot,
    RaggedDotDimensionNumbers, RaggedDotMode, RaggedDotOperation,
};
pub use operations::logical::{
    AND_OPERATION_NAME, And, AndOperation, NOT_OPERATION_NAME, Not, NotOperation, OR_OPERATION_NAME, Or, OrOperation,
    XOR_OPERATION_NAME, Xor, XorOperation,
};
pub use operations::manipulation::{
    BROADCAST_OPERATION_NAME, Broadcast, BroadcastOperation, CONCATENATE_OPERATION_NAME,
    CONVERT_ELEMENT_TYPE_OPERATION_NAME, Concatenate, ConcatenateOperation, ConvertElementType,
    ConvertElementTypeOperation, DYNAMIC_SHAPE_SLICE_OPERATION_NAME, DYNAMIC_SLICE_OPERATION_NAME,
    DYNAMIC_UPDATE_SLICE_OPERATION_NAME, DynamicBroadcast, DynamicBroadcastOperation, DynamicReshape,
    DynamicReshapeOperation, DynamicShapeSliceOperation, DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice,
    DynamicUpdateSliceOperation, ElementType, GATHER_OPERATION_NAME, Gather, GatherDimensionNumbers, GatherOperation,
    GatherScatterMode, PAD_OPERATION_NAME, Pad, PadOperation, Permutation, RESHAPE_OPERATION_NAME, Reshape,
    ReshapeOperation, ReshapeParameters, SCATTER_OPERATION_NAME, SLICE_OPERATION_NAME, Scatter,
    ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, Slice, SliceOperation, TRANSPOSE_OPERATION_NAME,
    Transpose, TransposeOperation, UPDATE_SLICE_OPERATION_NAME, UpdateSlice, UpdateSliceOperation,
};
pub use operations::math::{
    ABS_OPERATION_NAME, ADD_OPERATION_NAME, ATAN2_OPERATION_NAME, Abs, AbsOperation, Add, AddOperation, Atan2,
    Atan2Operation, CEIL_OPERATION_NAME, COS_OPERATION_NAME, Ceil, CeilOperation, Clamp, Cos, CosOperation,
    DIV_OPERATION_NAME, Div, DivOperation, ERF_OPERATION_NAME, EXP_OPERATION_NAME, Erf, ErfOperation, Exp,
    ExpOperation, FLOOR_OPERATION_NAME, Floor, FloorOperation, LOG_ADD_EXP_OPERATION_NAME, LOG_OPERATION_NAME,
    LOG_SUM_EXP_OPERATION_NAME, LOG1P_OPERATION_NAME, LOGISTIC_OPERATION_NAME, Log, Log1p, Log1pOperation, LogAddExp,
    LogAddExpOperation, LogOperation, LogSumExp, LogSumExpOperation, Logistic, LogisticOperation, MAX_OPERATION_NAME,
    MIN_OPERATION_NAME, MUL_OPERATION_NAME, Max, MaxOperation, Min, MinOperation, Mul, MulOperation,
    NEG_OPERATION_NAME, Neg, NegOperation, POW_OPERATION_NAME, Pow, PowOperation, REM_OPERATION_NAME,
    ROUND_OPERATION_NAME, RSQRT_OPERATION_NAME, Reduce, ReduceOperation, ReductionKind, Rem, RemOperation, Round,
    RoundOperation, Rsqrt, RsqrtOperation, SIGN_OPERATION_NAME, SIN_OPERATION_NAME, SQRT_OPERATION_NAME,
    SUB_OPERATION_NAME, Sign, SignOperation, Sin, SinOperation, Sqrt, SqrtOperation, Sub, SubOperation,
    TANH_OPERATION_NAME, Tanh, TanhOperation,
};
pub use operations::quantization::{BlockQuantize, SCALED_DOT_OPERATION_NAME, ScaledDot, ScaledDotOperation};
pub use operations::sharding::{
    ConstrainSharding, RESHARD_OPERATION_NAME, Reshard, ReshardOperation, SHARDING_CONSTRAINT_OPERATION_NAME,
    ShardingConstraintOperation,
};
pub use operations::{
    ArithmeticDimensionOperation, DIMENSION_ADD_OPERATION_NAME, DIMENSION_DIV_FLOOR_OPERATION_NAME,
    DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_MAX_OPERATION_NAME, DIMENSION_MIN_OPERATION_NAME,
    DIMENSION_MUL_OPERATION_NAME, DIMENSION_POW_OPERATION_NAME, DIMENSION_REM_OPERATION_NAME,
    DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME, DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME,
    DIMENSION_REQUIRE_EQUAL_OPERATION_NAME, DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME,
    DIMENSION_SATURATING_SUB_OPERATION_NAME, DIMENSION_SIZE_OPERATION_NAME, DIMENSION_SUB_OPERATION_NAME,
    DIMENSION_TO_SCALAR_OPERATION_NAME, DimensionAddOperation, DimensionArithmetic, DimensionDivFloorOperation,
    DimensionFromScalar, DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin,
    DimensionMinOperation, DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation,
    DimensionRequirement, DimensionRequirementOperation, DimensionRequirementPredicate, DimensionSaturatingSub,
    DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation, DimensionSubOperation, DimensionToScalar,
    DimensionToScalarOperation, ElementwiseOperation, PRINT_OPERATION_NAME, ParallelReduce, ParallelReduceOperation,
    ParallelReductionKind, Print, PrintOperation, RUNTIME_DIMENSION_DATA_TYPE, TAG_OPERATION_NAME,
    TRANSFER_TO_MEMORY_OPERATION_NAME, Tag, TagOperation, TransferToMemory, TransferToMemoryOperation,
    forward_collective_to_parent,
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
    Atom, AtomId, AttachedRegionStatistics, BindingRegionDriver, CalleeRegionDriver, Concretizable,
    DestinationRegionMapping, EagerInterpretationValidation, Effect, EffectOccurrence, Effects, EmptyRegionDriver,
    ExternalReferenceBinding, FlatProgram, Instruction, InstructionId, MaybeZero, MemberOperation, NoIdentity,
    Operation, OperationFormatter, OperationProjection, OperationProvider, OutputRegionProvenance, ParameterProjection,
    PartialReferenceDischargeResult, PreparedReferenceReplacement, Program, ProgramBuilder, ProgramError,
    ProgramLiveSets, ProgramRenderingMode, ProgramStatistics, ProjectedValue, Provenance, ProvenanceScope,
    ProvenanceState, REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME,
    REFERENCE_NEW_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME, REFERENCE_SWAP_OPERATION_NAME,
    REFERENCE_WRITE_OPERATION_NAME, ReadyOrPendingReferenceGuard, ReadyReferenceGuard,
    RecursiveReferenceDischargeDriver, Reference, ReferenceAccessMode, ReferenceAccumulationPolicy, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceAliasKind, ReferenceAllocationHandle, ReferenceCompletion,
    ReferenceCompletionBackend, ReferenceDischarge, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeReference, ReferenceDischargeRegionDestination,
    ReferenceDischargeResult, ReferenceDischargeSite, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceError, ReferenceFreeze, ReferenceFreezeOperation, ReferenceGeneration, ReferenceId, ReferenceInput,
    ReferenceNew, ReferenceNewOperation, ReferenceObservation, ReferenceOperationSemantics, ReferenceOutput,
    ReferenceRead, ReferenceReadOperation, ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork,
    ReferenceRegionStateInsertion, ReferenceRegionSummary, ReferenceReplacementPreparation,
    ReferenceReplacementTransaction, ReferenceSource, ReferenceStateWidening, ReferenceSwap, ReferenceSwapOperation,
    ReferenceType, ReferenceTypeRefinements, ReferenceWrite, ReferenceWriteOperation, Region, RegionArena,
    RegionArenaIterator, RegionDriver, RegionId, RegionInterface, RegionRef, RegionReplayMappings, RegionRole,
    RegionSlot, RegionStatistics, RegionWithMetadata, ReplayRegionDriver, TakenReferenceGuard, Transform,
    TransformArtifact, TransformCache, Type, TypeError, TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming,
    TypeIdentitySignature, TypeRefinements, Typed, ValidatedPendingReplacementTransaction, Value, ValueId,
    ValueProjection, discharge_positional_region_operation, discharge_preserved_access,
    discharge_reference_free_operation, infer_projected_operation_output_types,
    infer_projected_operation_region_input_types,
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
    use std::cell::Cell;
    use std::convert::Infallible;
    use std::fmt::Debug;
    use std::sync::{Arc, Weak};

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType};
    use crate::batching::{
        BatchAxis, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
        RecursiveBatchingDriver, RecursiveBatchingPolicy,
    };
    use crate::contexts::Context;
    use crate::macros::check_count;
    use crate::operations::ConditionOperation;
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::transforms::{RegionTransformCache, RegionTransformRegistry};
    use crate::programs::{
        Effect, Effects, Operation, Program, ProgramBuilder, ReferenceAddUpdateOperation, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceSwapOperation, ReferenceType, Region, RegionDriver,
        RegionInterface, RegionRef, RegionSlot, Transform, TransformArtifact, TypeError, Typed, Value,
    };
    use crate::specialization::SpecializationCacheStatistics;

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
        Effectful(Effect),

        /// Region-carrying operation declaring its region slots. Its inferred output types are the first attached
        /// region's output types, which pins that region interfaces are derived and delivered during inference.
        WithRegions(&'static [RegionSlot]),
    }

    impl Operation for TestRegionOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "add",
                Self::Effectful(_) => "effectful",
                Self::WithRegions(_) => "with_regions",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Add | Self::Effectful(_) => &[],
                Self::WithRegions(slots) => slots,
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
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Add | Self::WithRegions(_) => Effects::PURE,
                Self::Effectful(effect) => Effects::single(*effect),
            }
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

        fn effects(&self) -> Effects {
            if matches!(self, Self::State(_)) { Effects::single(Effect::OrderedState) } else { Effects::PURE }
        }
    }

    /// [`BatchingDriver`] that counts the structural [`batch_program`](BatchingDriver::batch_program) requests a
    /// region-carrying batching rule makes, delegating every request to the ordinary [`RecursiveBatchingDriver`] over
    /// the same regions. Region-carrying rules discover their nested programs' natural output axes before instantiating
    /// them at reconciled targets, and reuse a discovery program when its axes already match. This fixture lets rule
    /// tests pin how many structural passes such a rule actually performs, which a program rendering alone cannot
    /// observe.
    pub(crate) struct CountingBatchingDriver<'r, V: Value, O: Operation<Type = V::Type>> {
        /// [`Region`]s attached to the operation application under test, in operation-defined order.
        regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>,

        /// Number of structural program-batching requests observed so far.
        batch_program_calls: Cell<usize>,
    }

    impl<'r, V: Value, O: Operation<Type = V::Type>> CountingBatchingDriver<'r, V, O> {
        /// Creates a new [`CountingBatchingDriver`] over the provided attached regions.
        pub(crate) fn new(regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>) -> Self {
            Self { regions, batch_program_calls: Cell::new(0) }
        }

        /// Returns the number of structural program-batching requests observed so far.
        pub(crate) fn batch_program_calls(&self) -> usize {
            self.batch_program_calls.get()
        }
    }

    impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for CountingBatchingDriver<'_, V, O> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
        where
            V: 'r,
            O: 'r,
        {
            self.regions.regions()
        }
    }

    impl<C: Context, P: RecursiveBatchingPolicy<C>> BatchingDriver<C, P>
        for CountingBatchingDriver<'_, C::Constant, C::Operation>
    {
        fn batch_region(
            &self,
            context: &BatchingContext<C, P>,
            index: usize,
            inputs: Vec<P::Batch>,
        ) -> Result<Vec<P::Batch>, BatchingError> {
            RecursiveBatchingDriver::new(self.regions).batch_region(context, index, inputs)
        }

        fn batch_program(
            &self,
            context: &BatchingContext<C, P>,
            region: RegionRef<'_, C::Constant, C::Operation>,
            input_axes: &[BatchAxis],
            output_axes_policy: ProgramBatchingOutputAxesPolicy,
        ) -> Result<P::BatchedProgram, BatchingError> {
            self.batch_program_calls.set(self.batch_program_calls.get() + 1);
            RecursiveBatchingDriver::new(self.regions).batch_program(context, region, input_axes, output_axes_policy)
        }

        fn restore_batch(
            &self,
            value: C::Value,
            batch_axis: BatchAxis,
            r#type: &C::Type,
            inputs: &[P::Batch],
        ) -> Result<P::Batch, BatchingError> {
            P::restore_batch(value, batch_axis, r#type, inputs)
        }
    }

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

    /// Builds the canonical array IR test program whose whole-array state crosses a [`ConditionOperation`] boundary,
    /// shared by the transform adapters that must discharge local references before transforming. The program takes
    /// a Boolean predicate and an `f32[]` initial value, allocates one local reference from that initial value, and
    /// passes the reference into a condition whose branches access it with unequal modes. The `true` branch accumulates
    /// `1.0` and reads the reference, while the `false` branch swaps in `9.0` and yields the replaced value. Its two
    /// outputs are the condition's snapshot followed by the frozen final state, so a discharged program must thread
    /// identical state through both branches and keep both public outputs interpretable. On `[true, 4.0]` the outputs
    /// are `[5.0, 5.0]`, and on `[false, 4.0]` they are `[4.0, 9.0]`.
    pub(crate) fn test_condition_program()
    -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        type TestValue = ArrayIrValue<Array>;
        type TestOperation = ArrayIrOperation<Array>;

        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(TestValue::Array(Array::scalar(1.0_f32)));
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let replacement = false_builder.add_constant(TestValue::Array(Array::scalar(9.0_f32)));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let initial = builder.add_input(ArrayIrType::Array(scalar_type));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }
}
