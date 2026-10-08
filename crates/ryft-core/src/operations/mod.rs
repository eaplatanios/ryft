use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

use crate::arrays::{
    ArrayType, Broadcastable, bf16, f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz,
    f8e5m2, f8e5m2fnuz, f8e8m0fnu, f16, i1, i2, i4, u1, u2, u4,
};
use crate::macros::check_count;
use crate::parameters::{Parameter, Parameterized, Parameterwise};
use crate::programs::{Operation, ProgramError, TypeError, Typed};

pub mod arithmetic;
pub mod assertions;
pub mod attention;
pub mod collectives;
pub mod comparisons;
pub mod complex;
pub mod constants;
pub mod control_flow;
pub mod cumulative;
pub mod custom_call;
pub mod custom_functions;
pub mod debugging;
pub mod differentiation;
pub mod dimensions;
pub mod dot;
pub mod exponential;
pub mod extrema;
pub mod logical;
pub mod manipulation;
pub mod quantization;
pub mod random;
pub mod reductions;
pub mod references;
pub mod rounding;
pub mod sharding;
pub mod sorting;
pub mod special;
pub mod tagging;
pub mod trigonometric;

pub use arithmetic::{
    ABS_OPERATION_NAME, ADD_OPERATION_NAME, Abs, AbsOperation, Add, AddOperation, ArithmeticOperations,
    DIV_OPERATION_NAME, Div, DivOperation, MUL_OPERATION_NAME, Mul, MulOperation, NEG_OPERATION_NAME, Neg,
    NegOperation, POW_OPERATION_NAME, Pow, PowOperation, REM_OPERATION_NAME, RSQRT_OPERATION_NAME, Rem, RemOperation,
    Rsqrt, RsqrtOperation, SIGN_OPERATION_NAME, SQRT_OPERATION_NAME, SUB_OPERATION_NAME, Sign, SignOperation, Sqrt,
    SqrtOperation, Sub, SubOperation,
};
pub use assertions::{
    ASSERT_OPERATION_NAME, Assert, AssertOperation, AssertionError, AssertionFailure, AssertionValue,
};
pub use collectives::{
    AXIS_INDEX_OPERATION_NAME, AxisIndex, AxisIndexOperation, CollectiveMode, CollectiveOperations, CollectiveOptions,
    ManualVariationAlignment, PARALLEL_ALL_GATHER_OPERATION_NAME, PARALLEL_ALL_TO_ALL_OPERATION_NAME,
    PARALLEL_PERMUTE_OPERATION_NAME, PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, PARALLEL_REDUCE_OPERATION_NAME,
    PARALLEL_SUM_SCATTER_OPERATION_NAME, PARALLEL_VARY_OPERATION_NAME, ParallelAllGather, ParallelAllGatherOperation,
    ParallelAllGatherOutputVariance, ParallelAllToAll, ParallelAllToAllOperation, ParallelPermute,
    ParallelPermuteOperation, ParallelRaggedAllToAll, ParallelRaggedAllToAllOperation, ParallelReduce,
    ParallelReduceOperation, ParallelSumScatter, ParallelSumScatterOperation, ParallelVary, ParallelVaryOperation,
    ShapeChangingCollectiveValue,
};
pub use comparisons::{COMPARE_OPERATION_NAME, Compare, CompareOperation, ComparisonDirection, ComparisonType};
pub use complex::{
    COMPLEX_OPERATION_NAME, CONJUGATE_OPERATION_NAME, Complex, ComplexOperation, ComplexOperations, Conjugate,
    ConjugateOperation, IMAGINARY_OPERATION_NAME, Imaginary, ImaginaryOperation, REAL_OPERATION_NAME, Real,
    RealOperation,
};
pub use constants::{
    CONSTANT_OPERATION_NAME, Constant, ConstantOperation, ConstantOperations, DimensionConstant, DynamicFill,
    DynamicIota, DynamicOne, DynamicZero, Fill, IOTA_OPERATION_NAME, Iota, IotaOperation, ONE_LIKE_OPERATION_NAME,
    ONE_OPERATION_NAME, One, OneLike, OneLikeOperation, OneOperation, ZERO_LIKE_OPERATION_NAME, ZERO_OPERATION_NAME,
    Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
pub use control_flow::{
    CONDITION_OPERATION_NAME, Condition, ConditionOperation, ConditionType, SCAN_OPERATION_NAME, SELECT_OPERATION_NAME,
    ScanOperation, ScanType, Select, SelectOperation, WHILE_OPERATION_NAME, WhileOperation, WhilePredicate, WhileType,
    transpose_primal_condition,
};
pub use cumulative::{CUMULATIVE_OPERATION_NAME, Cumulative, CumulativeKind, CumulativeOperation, associative_scan};
pub use custom_call::{
    CUSTOM_CALL_OPERATION_NAME, CustomCall, CustomCallArrayAttribute, CustomCallAttribute, CustomCallBatching,
    CustomCallInputOutputAlias, CustomCallOperation, CustomCallRaggedContract, CustomCallRaggedInputBinding,
    CustomCallRaggedOutputBinding,
};
pub use custom_functions::{
    BatchingRuleConsistencyError, CUSTOM_FUNCTION_OPERATION_NAME, CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME,
    CustomCallPrimal, CustomFunction, CustomFunctionBatching, CustomFunctionJvp, CustomFunctionJvpRule,
    CustomFunctionOperation, CustomFunctionTransposeOperation, CustomFunctionVjp, CustomRuleDefinition,
    CustomRuleReference, CustomRuleRegistration, CustomRuleSource, CustomRuleSpecializer, CustomRuleTracer,
    DefaultBatching, DefaultJvp, DefaultVjp, JvpFromPrimal, LiftedCustomRules, UnavailableCustomRules,
    WeakCustomRuleRegistration, WithAccumulatingVjp, WithAxisDependentBatching, WithBatching, WithJvp,
    WithSymbolicZeroJvp, WithSymbolicZeroVjp, WithVjp, check_batching_rule_consistency, custom_function,
};
pub use debugging::{PRINT_OPERATION_NAME, Print, PrintOperation};
pub use differentiation::{
    LINEAR_CALL_OPERATION_NAME, LinearCallOperation, REMATERIALIZE_OPERATION_NAME,
    RematerializationOptimizationBarrier, RematerializeOperation, STOP_GRADIENT_OPERATION_NAME, StopGradient,
    StopGradientOperation, StopGradients,
};
pub use dimensions::{
    ArithmeticDimensionOperation, DIMENSION_ADD_OPERATION_NAME, DIMENSION_DATA_TYPE, DIMENSION_DIV_OPERATION_NAME,
    DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_MAX_OPERATION_NAME, DIMENSION_MIN_OPERATION_NAME,
    DIMENSION_MUL_OPERATION_NAME, DIMENSION_POW_OPERATION_NAME, DIMENSION_REM_OPERATION_NAME,
    DIMENSION_SATURATING_SUB_OPERATION_NAME, DIMENSION_SIZE_OPERATION_NAME, DIMENSION_SUB_OPERATION_NAME,
    DIMENSION_TO_SCALAR_OPERATION_NAME, DimensionAddOperation, DimensionDivOperation, DimensionFromScalar,
    DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin, DimensionMinOperation,
    DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation, DimensionSaturatingSub,
    DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation, DimensionSubOperation, DimensionToScalar,
    DimensionToScalarOperation,
};
pub use dot::{
    DOT_OPERATION_NAME, Dot, DotDimensionNumbers, DotOperation, DotOperations, RAGGED_DOT_OPERATION_NAME, RaggedDot,
    RaggedDotDimensionNumbers, RaggedDotMode, RaggedDotOperation,
};
pub use exponential::{
    EXP_OPERATION_NAME, Exp, ExpOperation, ExponentialOperations, LN_1P_OPERATION_NAME, LOG_ADD_EXP_OPERATION_NAME,
    LOG_OPERATION_NAME, LOGISTIC_OPERATION_NAME, Ln1p, Ln1pOperation, Log, LogAddExp, LogAddExpOperation, LogOperation,
    Logistic, LogisticOperation,
};
pub use extrema::{
    CLAMP_OPERATION_NAME, Clamp, ClampOperation, ExtremaOperations, MAX_OPERATION_NAME, MIN_OPERATION_NAME, Max,
    MaxOperation, Min, MinOperation,
};
pub use logical::{
    AND_OPERATION_NAME, And, AndOperation, LogicalOperations, NOT_OPERATION_NAME, Not, NotOperation, OR_OPERATION_NAME,
    Or, OrOperation, XOR_OPERATION_NAME, Xor, XorOperation,
};
pub use manipulation::{
    BROADCAST_OPERATION_NAME, BasicIndex, Broadcast, BroadcastOperation, CONCATENATE_OPERATION_NAME,
    CONVERT_ELEMENT_TYPE_OPERATION_NAME, Concatenate, ConcatenateOperation, ConvertElementType,
    ConvertElementTypeOperation, DYNAMIC_SLICE_OPERATION_NAME, DYNAMIC_UPDATE_SLICE_OPERATION_NAME, DynamicBroadcast,
    DynamicBroadcastOperation, DynamicConcatenate, DynamicGather, DynamicManipulationOperations, DynamicPad,
    DynamicReshape, DynamicReshapeOperation, DynamicScatter, DynamicSlice, DynamicSliceBounds, DynamicSliceOperation,
    DynamicSliceWithDimensions, DynamicUpdateSlice, DynamicUpdateSliceOperation, ElementType, GATHER_OPERATION_NAME,
    Gather, GatherDimensionNumbers, GatherMode, GatherOperation, GatherOptions, IndexInteger, IndexMask, IndexSelector,
    IndexSlice, Indexed, Indexing, ManipulationOperations, PAD_OPERATION_NAME, Pad, PadOperation, Permutation,
    REDUCE_PRECISION_OPERATION_NAME, RESHAPE_OPERATION_NAME, REVERSE_OPERATION_NAME, ReducePrecision,
    ReducePrecisionOperation, Reshape, ReshapeOperation, Reverse, ReverseOperation, SCATTER_OPERATION_NAME,
    SLICE_OPERATION_NAME, Scatter, ScatterDimensionNumbers, ScatterMode, ScatterOperation, ScatterOptions,
    ScatterReductionKind, Slice, SliceOperation, TRANSFER_TO_MEMORY_OPERATION_NAME, TRANSPOSE_OPERATION_NAME,
    TransferToMemory, TransferToMemoryOperation, Transpose, TransposeOperation, UPDATE_SLICE_OPERATION_NAME,
    UpdateSlice, UpdateSliceOperation,
};
pub use quantization::{BlockQuantize, SCALED_DOT_OPERATION_NAME, ScaledDot, ScaledDotOperation};
pub use random::{
    CategoricalSamplingMode, DynamicRngBitGenerator, RNG_BIT_GENERATOR_OPERATION_NAME, Random, RandomAlgorithm,
    RngBitGenerator, RngBitGeneratorOperation,
};
pub use reductions::{
    ARG_MAX_OPERATION_NAME, ARG_MIN_OPERATION_NAME, ArgMax, ArgMaxOperation, ArgMin, ArgMinOperation,
    REDUCE_OPERATION_NAME, Reduce, ReduceOperation, ReductionKind, ReductionOperations,
};
pub use references::{
    REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME,
    REFERENCE_NEW_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME, REFERENCE_SWAP_OPERATION_NAME,
    REFERENCE_WRITE_OPERATION_NAME, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceAtomicAddUpdate,
    ReferenceAtomicAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
    ReferenceOperations, ReferenceRead, ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation, ReferenceWrite,
    ReferenceWriteOperation,
};
pub use rounding::{
    CEIL_OPERATION_NAME, Ceil, CeilOperation, FLOOR_OPERATION_NAME, Floor, FloorOperation, ROUND_OPERATION_NAME, Round,
    RoundOperation, RoundingOperations,
};
pub use sharding::{
    CONSTRAIN_SHARDING_OPERATION_NAME, ConstrainSharding, ConstrainShardingDispatch, ConstrainShardingOperation,
    RESHARD_OPERATION_NAME, Reshard, ReshardDispatch, ReshardOperation, SHARD_MAP_OPERATION_NAME, ShardMap,
    ShardMapContext, ShardMapError, ShardMapOperation, ShardMapTracer, ShardingOperations, TracedShardMap, shard_map,
    shard_map_in_context, shard_map_with_options, trace_shard_map, trace_shard_map_with_named_axes,
    trace_shard_map_with_options,
};
pub use sorting::{SORT_OPERATION_NAME, Sort, SortDirection, SortOperation, SortOrdering, TopK};
pub use special::{ERF_OPERATION_NAME, Erf, ErfOperation};
pub use tagging::{TAG_OPERATION_NAME, Tag, TagOperation};
pub use trigonometric::{
    ATAN2_OPERATION_NAME, Atan2, Atan2Operation, COS_OPERATION_NAME, Cos, CosOperation, SIN_OPERATION_NAME, Sin,
    SinOperation, TAN_OPERATION_NAME, TANH_OPERATION_NAME, Tan, TanOperation, Tanh, TanhOperation,
    TrigonometricOperations,
};

/// Universe membership of a capability implementor. Every capability (e.g., [`Add`]) is parameterized by the universe
/// that it operates in, and that parameter defaults to its implementor's [`Universe`](Self::Universe), so that ordinary
/// bounds and calls such as `A: Add` and `a.add(&b)` refer to the implementor's own universe. Values belong to the
/// universe of their [`Typed::Type`] through a blanket implementation, host types that are not [`Typed`] (e.g.,
/// primitive integers and floating-point numbers) are their own universe, and [`Parameterwise`] structures belong
/// to the universe of their parameters.
///
/// A universe is a type-level marker only, and it does not need to be a [`Type`](crate::Type). Capabilities only use
/// it to tell implementations apart, and so host types need no artificial [`Type`](crate::Type) to belong to one. The
/// universe parameter lets one capability have separate implementations for different universes (e.g., a homogeneous
/// array implementation and a composite array IR implementation that projects onto it), which coherence could not
/// otherwise tell apart, because it cannot use associated types to prove two implementations disjoint. Capabilities
/// require [`Capability`] without pinning its universe to their parameter, because pinning it overflows the trait
/// solver. An implementor must therefore only implement capabilities for its own universe.
///
/// In generic code, prefer stating capability bounds with their default universe (e.g., `V: Add + Broadcast`) or a
/// bundle such as [`ArrayOperations`](crate::ArrayOperations), and avoid mixing them with explicitly named universes
/// for the same value (e.g., `V: Broadcast` in one bound and `V: Broadcast<ArrayType>` in another). The trait solver
/// cannot always prove that `<V as Capability>::Universe` and `<V as Typed>::Type` are the same type when `V` is a
/// generic or projected type, and so such mixed bounds may fail to unify. Likewise, traits must not state
/// `Typed<Type = T>` alongside a universe-parameterized capability bound at `T`, because the resulting
/// equality makes bounds on `V::Type` unusable.
///
/// Provided functions that inspect the type of their receiver therefore do not pin the universe
/// of their capability. Instead, they bound a view of that type, such as [`AsArrayType`](crate::AsArrayType) or
/// [`AsDimensionType`](crate::AsDimensionType) (e.g., `Self: Typed<Type: AsArrayType>`), so that every universe whose
/// types offer that view inherits them. Implementations name their universe explicitly (e.g., `Add<ArrayType>`), and a
/// blanket implementation over values of several universes introduces that universe as its own type parameter (e.g.,
/// the staging implementation of [`ReferenceAddUpdate`], which serves every universe that has reference members).
///
/// # The `capability` Attribute
///
/// Capability traits are declared with the [`capability`](macro@ryft_macros::capability) attribute, which checks their
/// conventions at compile time (i.e., exactly one type parameter defaults to `<Self as Capability>::Universe`, that
/// parameter is unbounded because host types are their own universes, and [`Capability`] is a direct super-trait).
/// Its `projection(Composite => Member)` argument additionally implements the capability for every value of the
/// `Composite` universe whose [`ValueProjection`](crate::ValueProjection) onto `Member` implements it. The
/// generated functions project the receiver and every `&Self`, `&[Self]`, and `Option<&Self>` input, apply the member
/// implementation, and lift `Self` outputs (also inside `Vec`s and tuples) back, while passing other inputs and outputs
/// through unchanged. Projecting a member of another kind fails with the projection's error (e.g., a [`TypeError`] for
/// a first-class dimension). This is how composite array IR values (i.e., [`ArrayIrValue`](crate::ArrayIrValue) and the
/// tracers over [`ArrayIrType`](crate::ArrayIrType)) implement most array capabilities, through
/// `#[capability(projection(ArrayIrType => ArrayType))]`, so that their staged programs contain
/// the same array instructions as homogeneous programs.
///
/// The projection covers the required functions of a capability, and provided functions keep their default bodies.
/// A default must therefore be correct for every implementor, typically by composing required functions. A function
/// whose behavior depends on the implementor (e.g., one whose configuration staging values record in the staged
/// operation while concrete values only validate it) must be required instead, so that composite values project it too.
/// Type parameters that precede the universe must default to `Self` (e.g., a right input declared as `Rhs = Self`), and
/// they resolve to the composite value in the implemented capability and to its projection in the delegated one. The
/// generated code refers to the `ryft` crate, and a `crate = "path"` argument overrides that path (e.g., for crates
/// that depend on `ryft-core` directly). Capabilities whose composite implementations are irregular (e.g., [`Sort`] or
/// [`Gather`]) keep handwritten implementations and use the bare attribute.
///
/// ```rust
/// # extern crate ryft_core as ryft;
/// # use ryft_core::{Array, ArrayIrType, ArrayIrValue, ArrayType, Capability, ProgramError};
/// # use ryft_macros::capability;
///
/// #[capability(projection(ArrayIrType => ArrayType))]
/// trait Double<T = <Self as Capability>::Universe>: Capability + Sized {
///     fn double(&self) -> Result<Self, ProgramError>;
/// }
///
/// impl Double<ArrayType> for Array {
///     fn double(&self) -> Result<Self, ProgramError> {
///         Ok(self.clone() + self.clone())
///     }
/// }
///
/// let value = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0])?);
/// assert_eq!(value.double()?, ArrayIrValue::Array(Array::vector(vec![2.0f32, 4.0])?));
/// # Ok::<(), ProgramError>(())
/// ```
pub trait Capability {
    /// Universe that this implementor belongs to (e.g., the [`Typed::Type`] of a value, or a host type itself).
    type Universe;
}

impl<V: Typed> Capability for V {
    type Universe = V::Type;
}

// A parameterwise structure belongs to the universe of its parameters, so that default universe parameters resolve
// for it exactly as they do for its parameters.
impl<P: Parameter + Capability, S: Parameterized<P>> Capability for Parameterwise<P, S> {
    type Universe = P::Universe;
}

/// Implements [`Capability`] for a host type that is not [`Typed`], making the host type its own universe.
macro_rules! impl_host_capability {
    ($type:ty) => {
        impl Capability for $type {
            type Universe = $type;
        }
    };
}

impl_host_capability!(bool);
impl_host_capability!(i8);
impl_host_capability!(i16);
impl_host_capability!(i32);
impl_host_capability!(i64);
impl_host_capability!(i128);
impl_host_capability!(isize);
impl_host_capability!(u8);
impl_host_capability!(u16);
impl_host_capability!(u32);
impl_host_capability!(u64);
impl_host_capability!(u128);
impl_host_capability!(usize);
impl_host_capability!(f32);
impl_host_capability!(f64);
impl_host_capability!(i1);
impl_host_capability!(i2);
impl_host_capability!(i4);
impl_host_capability!(u1);
impl_host_capability!(u2);
impl_host_capability!(u4);
impl_host_capability!(f4e2m1fn);
impl_host_capability!(f6e2m3fn);
impl_host_capability!(f6e3m2fn);
impl_host_capability!(f8e3m4);
impl_host_capability!(f8e4m3);
impl_host_capability!(f8e4m3fn);
impl_host_capability!(f8e4m3fnuz);
impl_host_capability!(f8e4m3b11fnuz);
impl_host_capability!(f8e5m2);
impl_host_capability!(f8e5m2fnuz);
impl_host_capability!(f8e8m0fnu);
impl_host_capability!(bf16);
impl_host_capability!(f16);
impl_host_capability!(num_complex::Complex<f32>);
impl_host_capability!(num_complex::Complex<f64>);

/// Represents [`Operation`]s that operate elementwise on arrays and that support _broadcasting_ semantics.
/// [`ElementwiseOperation`] captures the shared type inference behavior of elementwise array operations.
/// Implementations declare their fixed input count, while the default type inference implementation checks
/// the input count and matching manual variation, then broadcasts all input [`ArrayType`]s. Binding must
/// insert explicit variation transitions before inference when invariant and varying values are combined.
pub trait ElementwiseOperation: Operation<Type = ArrayType> {
    /// Returns the number of input arrays consumed by this elementwise [`Operation`].
    fn input_count(&self) -> usize;

    /// Infers the broadcasted output [`ArrayType`] for this elementwise [`Operation`]. Operations whose output
    /// [`Sharding`](crate::Sharding) does not follow plain broadcasting semantics (e.g., [`MulOperation`], which is
    /// bilinear in its inputs and combines their reduction state accordingly) must override this function, typically
    /// using [`infer_elementwise_broadcast_type`](Self::infer_elementwise_broadcast_type) for the data type, shapes,
    /// and placement, and layering their own sharding rule on top.
    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, self.input_count(), TypeError);
        Ok(vec![self.infer_elementwise_broadcast_type(input_types)?])
    }

    /// Broadcasts input geometry and placement after validating matching manual variation. Operations with specialized
    /// reduction-state rules may normalize those states before calling this function and restore their output state.
    fn infer_elementwise_broadcast_type(&self, input_types: &[ArrayType]) -> Result<ArrayType, TypeError> {
        ArrayType::check_matching_manual_variation(self.name(), &input_types.iter().collect::<Vec<_>>())?;
        ArrayType::broadcasted(input_types)
            .map_err(|_| TypeError::invalid(format!("`{}` input types are not broadcast-compatible", self.name())))
    }
}

/// Result accuracy requested from an elementwise transcendental [`Operation`] (e.g., a [`SinOperation`] or a
/// [`ExpOperation`]), mirroring the StableHLO [`result_accuracy`](https://openxla.org/stablehlo/spec#exponential)
/// attribute. The accuracy only selects among the implementations that a backend provides for the operation, and so
/// it never changes the operation's type or its mathematical definition. Backends without alternative implementations,
/// including the eager reference [`Array`](crate::Array) kernels, evaluate their only implementation for every
/// accuracy, and a backend compiler reports an error when it cannot satisfy a requested [`Tolerance`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Accuracy {
    /// Backend-selected default implementation.
    #[default]
    Default,

    /// Most accurate implementation that the backend provides.
    Highest,

    /// Implementation whose error stays within the provided [`Tolerance`].
    Tolerance(Tolerance),
}

impl Display for Accuracy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => formatter.write_str("default"),
            Self::Highest => formatter.write_str("highest"),
            Self::Tolerance(tolerance) => write!(
                formatter,
                "tolerance(absolute={}, relative={}, units_of_least_precision={})",
                tolerance.absolute(),
                tolerance.relative(),
                tolerance.units_of_least_precision(),
            ),
        }
    }
}

/// Error tolerance of an [`Accuracy::Tolerance`] request, mirroring JAX's
/// [`jax.lax.Tolerance`](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Tolerance). An implementation satisfies
/// the tolerance when its error stays within the absolute tolerance, the relative tolerance, or the provided number of
/// units in the last place of the exact result. At most two of the three tolerances are typically combined (i.e., an
/// absolute tolerance with either a relative tolerance or a number of units in the last place).
///
/// Equality and hashing compare the floating-point tolerances bitwise, so `-0.0` and `+0.0` tolerances are distinct.
/// This makes [`Tolerance`] (and the operations that carry it) a faithful key of the attribute that backends receive.
#[derive(Copy, Clone, Debug)]
pub struct Tolerance {
    /// Absolute error tolerance.
    absolute: f64,

    /// Relative error tolerance.
    relative: f64,

    /// Error tolerance measured in Units in the Last Place (ULPs) of the exact result.
    units_of_least_precision: usize,
}

impl Tolerance {
    /// Creates a new [`Tolerance`], applying the same validation as JAX's
    /// [`Tolerance`](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Tolerance).
    ///
    /// # Parameters
    ///
    ///   - `absolute`: Absolute error tolerance, which must be finite and non-negative.
    ///   - `relative`: Relative error tolerance, which must be finite and non-negative.
    ///   - `units_of_least_precision`: Error tolerance in units in the last place of the exact result.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] if either floating-point tolerance is negative or not finite,
    /// or if all three tolerances are zero.
    pub fn new(absolute: f64, relative: f64, units_of_least_precision: usize) -> Result<Self, ProgramError> {
        if !absolute.is_finite() || !relative.is_finite() || absolute < 0.0 || relative < 0.0 {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "accuracy tolerances must be finite and non-negative but got absolute tolerance {absolute} and \
                     relative tolerance {relative}",
                ),
            });
        }

        if absolute == 0.0 && relative == 0.0 && units_of_least_precision == 0 {
            return Err(ProgramError::InvalidArgument {
                message: "at least one accuracy tolerance must be non-zero".to_string(),
            });
        }

        Ok(Self { absolute, relative, units_of_least_precision })
    }

    /// Returns the absolute error tolerance of this [`Tolerance`] instance.
    #[inline]
    pub fn absolute(&self) -> f64 {
        self.absolute
    }

    /// Returns the relative error tolerance of this [`Tolerance`] instance.
    #[inline]
    pub fn relative(&self) -> f64 {
        self.relative
    }

    /// Returns the error tolerance in Units in the Last Place (ULPs) of the exact result
    /// for this [`Tolerance`] instance.
    #[inline]
    pub fn units_of_least_precision(&self) -> usize {
        self.units_of_least_precision
    }
}

impl PartialEq for Tolerance {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.absolute.to_bits() == other.absolute.to_bits()
            && self.relative.to_bits() == other.relative.to_bits()
            && self.units_of_least_precision == other.units_of_least_precision
    }
}

impl Eq for Tolerance {}

impl Hash for Tolerance {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.absolute.to_bits().hash(state);
        self.relative.to_bits().hash(state);
        self.units_of_least_precision.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_macros::capability;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionValue, DimensionVariable, Layout, LogicalMesh, MeshAxis, MeshAxisType, Shape,
        Sharding, ShardingDimension, StridedLayout,
    };
    use crate::programs::{RegionInterface, Value};
    use crate::tests::hash_of;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    #[test]
    fn test_capability() {
        // Values belong to the universe of their type, so universe-defaulted capabilities refer to their own universe.
        fn assert_value_universe<V: Typed + Capability<Universe = <V as Typed>::Type>>() {}

        assert_value_universe::<Array>();
        assert_value_universe::<ArrayType>();
        assert_value_universe::<DimensionValue>();
        assert_value_universe::<ArrayIrValue<Array>>();
        assert_value_universe::<Tracer<TracingContext<Array, ArrayOperation<Array>>>>();
        assert_value_universe::<Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>();

        // Host types that are not typed belong to the universe of their own host type, including integers that are
        // wider than every array data type.
        fn assert_host_universe<T: Capability<Universe = T>>() {}

        assert_host_universe::<bool>();
        assert_host_universe::<i8>();
        assert_host_universe::<i128>();
        assert_host_universe::<isize>();
        assert_host_universe::<u128>();
        assert_host_universe::<usize>();
        assert_host_universe::<f32>();
        assert_host_universe::<f64>();

        // Host types satisfy the capability bundles whose members they implement, in their own universes.
        fn assert_floating_point_bundles<
            V: ArithmeticOperations
                + ExponentialOperations
                + TrigonometricOperations
                + RoundingOperations
                + ExtremaOperations,
        >() {
        }

        assert_floating_point_bundles::<f32>();
        assert_floating_point_bundles::<f64>();

        fn assert_integer_bundles<V: ExtremaOperations + LogicalOperations>() {}

        assert_integer_bundles::<bool>();
        assert_integer_bundles::<i8>();
        assert_integer_bundles::<i128>();
        assert_integer_bundles::<usize>();
    }

    #[test]
    fn test_capability_attribute() {
        // Two projected capabilities cover projected `&Self` inputs, passed-through inputs, functions without inputs,
        // several functions per capability, a provided function that keeps its default body, and a right-input type
        // parameter that precedes the universe.
        #[capability(projection(ArrayIrType => ArrayType))]
        trait Scale<T = <Self as Capability>::Universe>: Capability + Sized {
            fn scale(&self, right: &Self, count: usize) -> Result<Self, ProgramError>;

            fn double(&self) -> Result<Self, ProgramError>;

            fn triple(&self) -> Result<Self, ProgramError> {
                self.scale(self, 2)
            }
        }

        #[capability(projection(ArrayIrType => ArrayType))]
        trait Combine<Rhs = Self, T = <Self as Capability>::Universe>: Capability + Sized {
            fn combine(&self, right: &Rhs) -> Result<Self, ProgramError>;
        }

        impl<V: Value<Type = ArrayType> + Add<ArrayType>> Scale<ArrayType> for V {
            fn scale(&self, right: &Self, count: usize) -> Result<Self, ProgramError> {
                (0..count).try_fold(self.clone(), |sum, _| sum.add(right))
            }

            fn double(&self) -> Result<Self, ProgramError> {
                self.add(self)
            }
        }

        impl<V: Value<Type = ArrayType> + Add<ArrayType>> Combine<V, ArrayType> for V {
            fn combine(&self, right: &V) -> Result<Self, ProgramError> {
                self.add(right)
            }
        }

        // Composite tracers stage the homogeneous array instructions of every generated function.
        type CompositeContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = CompositeContext::trace(
            |inputs: Vec<Tracer<CompositeContext>>| inputs[0].scale(&inputs[1], 2)?.double()?.combine(&inputs[1]),
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = add %0 %1
                    %3:f32[2] = add %2 %1
                    %4:f32[2] = add %3 %3
                    %5:f32[2] = add %4 %1
                in (%5)"
            },
        );

        // Concrete composite values apply the capabilities to their array members, and the provided
        // function composes them, while first-class dimension members are rejected by the projection.
        let left = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let right = ArrayIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap());
        assert_eq!(left.scale(&right, 2), Ok(ArrayIrValue::Array(Array::vector(vec![7.0f32, 10.0]).unwrap())));
        assert_eq!(left.double(), Ok(ArrayIrValue::Array(Array::vector(vec![2.0f32, 4.0]).unwrap())));
        assert_eq!(left.triple(), Ok(ArrayIrValue::Array(Array::vector(vec![3.0f32, 6.0]).unwrap())));
        assert_eq!(left.combine(&right), Ok(ArrayIrValue::Array(Array::vector(vec![4.0f32, 6.0]).unwrap())));
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            left.scale(&dimension, 1),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
        assert_eq!(
            dimension.combine(&left),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_elementwise_operation_type_inference() {
        #[derive(Clone, Debug)]
        struct TestElementwiseArrayOperation {
            input_count: usize,
        }

        impl Operation for TestElementwiseArrayOperation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                "elementwise_test"
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl ElementwiseOperation for TestElementwiseArrayOperation {
            #[inline]
            fn input_count(&self) -> usize {
                self.input_count
            }
        }

        let operation = TestElementwiseArrayOperation { input_count: 1 };
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(Operation::infer_output_types(&operation, &[input_type.clone()], &[]), Ok(vec![input_type]));
        assert_eq!(
            Operation::infer_output_types(&operation, &[], &[]),
            Err(TypeError::invalid("expected 1 input but got 0".to_string())),
        );

        let operation = TestElementwiseArrayOperation { input_count: 2 };
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[
                    ArrayType::scalar(DataType::F32).with_layout(Layout::Strided(StridedLayout::new(Vec::new()))),
                    ArrayType::scalar(DataType::F32),
                ],
                &[],
            ),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])),
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError::invalid("`elementwise_test` input types are not broadcast-compatible".to_string())),
        );

        let operation = TestElementwiseArrayOperation { input_count: 3 };
        let output = Operation::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)])),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(
            output,
            vec![ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))],
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let first = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let second = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let third = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["z"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::infer_output_types(&operation, &[first.clone(), second, third], &[]),
            Err(TypeError::invalid(
                "`elementwise_test` inputs must have matching varying manual axes; insert `parallel_vary` on the \
                 inputs that lack an axis, as `align_manual_variation` does",
            )),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[first.clone(), first.clone(), ArrayType::scalar(DataType::F32)],
                &[],
            ),
            Err(TypeError::invalid(
                "`elementwise_test` inputs must have matching varying manual axes; insert `parallel_vary` on the \
                 inputs that lack an axis, as `align_manual_variation` does",
            )),
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[first.clone(), first.clone(), first.clone()], &[]),
            Ok(vec![first]),
        );

        // Dynamic dimensions flow through elementwise congruence when they match exactly, while static-vs-dynamic
        // mismatches are rejected.
        let operation = TestElementwiseArrayOperation { input_count: 2 };
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[dynamic_type.clone(), dynamic_type.clone()], &[]),
            Ok(vec![dynamic_type.clone()]),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[
                    dynamic_type,
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError::invalid("`elementwise_test` input types are not broadcast-compatible".to_string())),
        );
    }

    #[test]
    fn test_accuracy() {
        let tolerance = Tolerance::new(1e-6, 0.0, 2).unwrap();
        assert_eq!(Accuracy::default(), Accuracy::Default);
        assert_eq!(Accuracy::Default.to_string(), "default");
        assert_eq!(Accuracy::Highest.to_string(), "highest");
        assert_eq!(
            Accuracy::Tolerance(tolerance).to_string(),
            "tolerance(absolute=0.000001, relative=0, units_of_least_precision=2)",
        );
        assert_eq!(format!("{:?}", Accuracy::Highest), "Highest");
    }

    #[test]
    fn test_tolerance() {
        let tolerance = Tolerance::new(1e-6, 1e-3, 2).unwrap();
        assert_eq!(tolerance.absolute(), 1e-6);
        assert_eq!(tolerance.relative(), 1e-3);
        assert_eq!(tolerance.units_of_least_precision(), 2);
        assert_eq!(Tolerance::new(0.0, 0.0, 1).map(|tolerance| tolerance.units_of_least_precision()), Ok(1));
    }

    #[test]
    fn test_tolerance_new_validation() {
        // Tolerances follow JAX's validation: they must be non-negative, finite, and not all zero.
        assert_eq!(
            Tolerance::new(-1.0, 0.0, 0),
            Err(ProgramError::InvalidArgument {
                message: "accuracy tolerances must be finite and non-negative but got absolute tolerance -1 and \
                          relative tolerance 0"
                    .to_string(),
            }),
        );
        assert_eq!(
            Tolerance::new(0.0, f64::NAN, 0),
            Err(ProgramError::InvalidArgument {
                message: "accuracy tolerances must be finite and non-negative but got absolute tolerance 0 and \
                          relative tolerance NaN"
                    .to_string(),
            }),
        );
        assert_eq!(
            Tolerance::new(0.0, 0.0, 0),
            Err(ProgramError::InvalidArgument {
                message: "at least one accuracy tolerance must be non-zero".to_string()
            }),
        );
    }

    #[test]
    fn test_tolerance_identity() {
        // Equal tolerances are equal and hash identically, so they work as map keys.
        let tolerance = Tolerance::new(1e-6, 1e-3, 2).unwrap();
        let same_tolerance = Tolerance::new(1e-6, 1e-3, 2).unwrap();
        assert_eq!(tolerance, tolerance);
        assert_eq!(tolerance, same_tolerance);
        assert_eq!(hash_of(&tolerance), hash_of(&same_tolerance));
        let tolerances = HashMap::from([(tolerance, "tolerance")]);
        assert_eq!(tolerances.get(&same_tolerance), Some(&"tolerance"));
        assert_ne!(tolerance, Tolerance::new(1e-6, 1e-3, 3).unwrap());

        // Floating-point tolerances compare bitwise, so signed zeros are distinct even though `-0.0 == 0.0`.
        let negative_zero = Tolerance::new(-0.0, 0.0, 1).unwrap();
        let positive_zero = Tolerance::new(0.0, 0.0, 1).unwrap();
        assert_ne!(negative_zero, positive_zero);
        assert_ne!(Tolerance::new(0.0, -0.0, 1).unwrap(), positive_zero);
        assert_eq!(tolerances.get(&negative_zero), None);

        // Accuracy requests inherit the bitwise tolerance identity and distinguish their variants.
        assert_eq!(Accuracy::Tolerance(tolerance), Accuracy::Tolerance(same_tolerance));
        assert_eq!(hash_of(&Accuracy::Tolerance(tolerance)), hash_of(&Accuracy::Tolerance(same_tolerance)));
        assert_ne!(Accuracy::Tolerance(negative_zero), Accuracy::Tolerance(positive_zero));
        assert_ne!(Accuracy::Tolerance(tolerance), Accuracy::Highest);
        assert_ne!(Accuracy::Highest, Accuracy::Default);
    }
}
