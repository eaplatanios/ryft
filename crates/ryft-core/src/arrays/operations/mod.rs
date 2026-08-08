//! Closed operation families and array-owned operation implementations.
//!
//! [`ArrayOperation`], [`DimensionOperation`], and [`ArrayIrOperation`] are the staged operation families for the
//! array universe. The private child modules group concrete reference-array capabilities and mixed array-IR machinery
//! by the same semantic families used under [`crate::operations`]; generic operation contracts and payload types
//! remain owned there.

// TODO(eaplatanios): Review this module.

use ryft_macros::Operation;

use crate::arrays::arrays::Array;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::DimensionType;
use crate::arrays::types::ir::ArrayIrType;
use crate::axes::AxisIndexOperation;
use crate::contexts::{Context, ProjectedContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    LinearCallOperation, MemberDifferentiableOperation, jvp_projected_operation,
};
use crate::operations::attention::{DotProductAttentionBackwardOperation, DotProductAttentionOperation};
use crate::operations::collectives::{AllGatherOperation, AllToAllOperation, PSumScatterOperation, PpermuteOperation};
use crate::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use crate::operations::custom_call::CustomCallOperation;
use crate::operations::random::RngBitGeneratorOperation;
use crate::operations::sort::SortOperation;
use crate::operations::{
    AbsOperation, AddOperation, AndOperation, Atan2Operation, BroadcastOperation, CeilOperation, CollectiveOperation,
    CompareOperation, ConcatenateOperation, ConditionOperation, ConstantOperation, ConvertElementTypeOperation,
    CoordinateBasisOperation, CosOperation, DimensionAddOperation, DimensionDivFloorOperation,
    DimensionFromScalarOperation, DimensionMaxOperation, DimensionMinOperation, DimensionMulOperation,
    DimensionPowOperation, DimensionRemOperation, DimensionRequirementOperation, DimensionSaturatingSubOperation,
    DimensionSizeOperation, DimensionSubOperation, DimensionToScalarOperation, DivOperation, DotOperation,
    DynamicShapeSliceOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, ErfOperation, ExpOperation,
    FloorOperation, GatherOperation, IotaOperation, LegacyBroadcastOperation, LegacyReshapeOperation, LogOperation,
    LogisticOperation, MaxOperation, MinOperation, MulOperation, NegOperation, NotOperation, OneLikeOperation,
    OneOperation, OrOperation, PadOperation, PowOperation, PrintOperation, ReduceOperation, RemOperation,
    ReshapeOperation, ReshardOperation, RoundOperation, RsqrtOperation, ScaledDotOperation, ScanOperation,
    ScatterOperation, SelectOperation, ShardingConstraintOperation, SignOperation, SinOperation, SliceOperation,
    SqrtOperation, StopGradientOperation, SubOperation, TagOperation, TanhOperation, TransferToMemoryOperation,
    TransposeOperation, UpdateSliceOperation, WhileOperation, XorOperation, Zero, ZeroLikeOperation, ZeroOperation,
};
use crate::programs::{
    MaybeZero, Operation, OperationProjection, Type, TypeIdentityPosition, Typed, Value, ValueProjection,
};
use crate::tracing::TracingContext;
use crate::tracing_v2::RematerializeOperation;
use crate::tracing_v2::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};

mod attention;
mod collectives;
mod compare;
mod complex;
mod constants;
mod control_flow;
mod custom_call;
mod differentiation;
mod dimensions;
mod logical;
mod manipulation;
mod math;
mod memory;
mod random;
mod sharding;
mod sort;
mod tag;

/// Reusable [`Operation`] enum for ordinary staged programs over arrays.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates, pairing with [`Array`]. Most
/// variants are thin tags around one semantic primitive defined in [`crate::operations`] or
/// [`crate::tracing_v2::custom_derivatives`].
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
    CoordinateBasis(CoordinateBasisOperation<ArrayType>),
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
    Reshape(LegacyReshapeOperation),
    Broadcast(LegacyBroadcastOperation),
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

/// [`Operation`](crate::Operation) family used for staged [`DimensionValue`] [`Program`](crate::Program)s.
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

/// Closed [`Operation`](crate::Operation) family for Ryft's array IR, whose values include ordinary arrays and
/// first-class runtime dimensions. This dispatcher preserves the homogeneous contracts of [`ArrayOperation`] and
/// [`DimensionOperation`]: it selects the member family, projects the composite type boundary once, delegates to that
/// family, and lifts the inferred result types back into [`ArrayIrType`].
///
/// Operations whose signatures mix arrays and dimensions are represented as explicit variants because no homogeneous
/// member family can express such a signature. For example, [`DimensionSizeOperation`] consumes an array and produces
/// a first-class dimension without changing either homogeneous family.
#[derive(Clone, Debug, Operation)]
#[ryft(crate = "crate", type = ArrayIrType, constant = ArrayIrValue<A>)]
#[ryft(members(ArrayType, structural(DimensionType)))]
#[ryft(dispatch(batching, differentiation, transposition))]
pub enum ArrayIrOperation<A: Value<Type = ArrayType>> {
    /// Mixed zero constructor whose stored [`ArrayType`] defines the array result and whose dynamic dimensions are
    /// consumed as explicit first-class dimension operands, one per dynamic axis in axis order. This constructor
    /// lives at the composite-family level because its signature crosses member kinds: a homogeneous
    /// [`ArrayOperation`] cannot consume dimension operands, while the stored structural type carries identities and
    /// bounds but not the concrete runtime extents required to materialize the result.
    #[ryft(mixed(structural), skip_from)]
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor whose stored [`ArrayType`] fully defines the output type and whose dynamic dimensions
    /// are consumed as explicit first-class dimension operands, one per dynamic axis in axis order.
    #[ryft(mixed(structural), skip_from)]
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor whose stored [`ArrayType`] and iota axis define the complete output, and whose dynamic
    /// dimensions are consumed as explicit first-class dimension operands in axis order.
    #[ryft(mixed(structural), skip_from)]
    DynamicIota(IotaOperation<ArrayType>),

    /// Region-free homogeneous array operation. Member control-flow operations are promoted to their direct
    /// composite carriers when an array-only operation is lifted into the array IR.
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

    /// Mixed operation that converts ordinary scalar-array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Mixed operation that converts a first-class dimension into ordinary scalar-array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Mixed operation that reshapes an array using one first-class dimension operand per output axis.
    Reshape(ReshapeOperation),

    /// Mixed operation that broadcasts an array using one first-class dimension operand per output axis.
    Broadcast(BroadcastOperation),

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

    /// Composite condition whose attached branches may carry arrays and first-class dimensions.
    Condition(ConditionOperation<ArrayIrValue<A>>),

    /// Composite while loop whose condition and body may carry arrays and first-class dimensions.
    While(WhileOperation<ArrayIrType>),

    /// Composite scan whose body may carry arrays and first-class dimensions.
    Scan(ScanOperation<ArrayIrValue<A>>),

    /// Differentiation-owned executable linear call with ordinary trailing residual operands.
    LinearCall(LinearCallOperation<ArrayIrType>),
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

impl<A: Value<Type = ArrayType>> From<AddOperation<ArrayIrType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(_operation: AddOperation<ArrayIrType>) -> Self {
        Self::Array(ArrayOperation::Add(AddOperation::new()))
    }
}

impl<A, C> MemberDifferentiableOperation<C> for ArrayOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayIrType,
            Constant: ValueProjection<ArrayType, Projected = A>,
            Operation: From<ArrayIrOperation<A>>
                           + From<BroadcastOperation>
                           + From<DimensionSizeOperation>
                           + From<DimensionToScalarOperation>
                           + From<LinearCallOperation<ArrayIrType>>
                           + From<ZeroOperation<ArrayType>>
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
            Self::Reduce(operation) => operation.jvp_in_parent(context, driver, inputs)?,
            operation => jvp_projected_operation(context, operation, inputs)?,
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

    use crate::arrays::arrays::Array;
    use crate::arrays::batching::{ArrayIrBatch, ArrayIrBatching};
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
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableType, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::{
        AddOperation, BroadcastOperation, ConcatenateOperation, ConditionOperation, DimensionAddOperation,
        DimensionMulOperation, DimensionRequirementOperation, DimensionSizeOperation, MulOperation, ReduceOperation,
        ReductionKind, ReshapeOperation, ScanOperation, WhileOperation, ZeroOperation, ZeroOperationProvider,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{
        Effect, Effects, EmptyRegionDriver, OperationProjection, ProgramBuilder, ProgramError, RegionInterface, Type,
        TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;

    #[test]
    fn test_composite_pullback_materializes_a_dynamic_zero_space_input_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let context = TracingContext::<TestValue, TestOperation>::new();
        let key = context.input(key_type.clone().into());
        let accumulator = context.input(ArrayType::scalar(DataType::F64).into());
        let (_, pullback) = context.vjp(|inputs: Vec<_>| Ok(inputs[1].clone()), vec![key, accumulator]).unwrap();
        let cotangent = context.input(ArrayType::scalar(DataType::F64).into());

        // The compact pullback has no result slot for the key's zero differential space. Rebuilding the public result
        // must use the key extent captured at linearization time rather than attempt a nullary dynamic zero.
        let cotangents = pullback.apply(cotangent).unwrap();
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayIrType::Array(key_type.tangent()));
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
        let (_, pushforward) = context.linearize(|inputs: Vec<_>| Ok(inputs[1].clone()), vec![extent, key]).unwrap();
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let key_tangent = context.input(key_type.tangent().into());

        // The compact pushforward has no output slot for the key's zero differential space. Rebuilding its public
        // result must consume the key extent captured at linearization time.
        let tangent = pushforward.apply(vec![extent_tangent, key_tangent]).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayIrType::Array(key_type.tangent()));
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

        // Identity-bearing zeros promote to the mixed constructor and retain ordinary identity renaming.
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

        let static_zero = ArrayIrOperation::<Array>::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(static_zero, ArrayIrOperation::Array(ArrayOperation::Zero(_))));

        // Identity-free ones remain homogeneous, while identity-bearing ones use the explicit mixed constructor.
        let static_one = ArrayIrOperation::<Array>::from(OneOperation::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(static_one, ArrayIrOperation::Array(ArrayOperation::One(_))));
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
                "'one' expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
        let other = DimensionVariable::new("other", bounds);
        assert_eq!(
            dynamic_one.infer_output_types(&[DimensionType::new(other).into()], &[]),
            Err(TypeError::invalid(
                "'one' operand 0 has type dimension<other ∈ [1, 9)> but the output shape requires \
                 dimension<source ∈ [1, 9)>",
            )),
        );
        assert_eq!(
            dynamic_one.infer_output_types(
                &[DimensionType::new(source.clone()).into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("'one' expects no regions but got 1")),
        );

        // Iota follows the same static-versus-dynamic routing while retaining and validating its varying axis.
        let static_iota = ArrayIrOperation::<Array>::from(
            IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])), 0).unwrap(),
        );
        assert!(matches!(static_iota, ArrayIrOperation::Array(ArrayOperation::Iota(_))));
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
            TypeError::invalid("'iota' dimension 2 is out of bounds for rank 2"),
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
        let reshape = ArrayIrOperation::<Array>::from(ReshapeOperation::new());
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
        let permuted = ArrayIrOperation::<Array>::from(ReshapeOperation::new().with_dimensions([1, 0]));
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
                ReshapeOperation::new(),
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
        assert_eq!(
            context.bind(
                ArrayIrOperation::DynamicOne(OneOperation::new(ArrayType::scalar(DataType::F32))),
                Vec::new(),
                &[],
            ),
            Err(TypeError::invalid(
                "'one' with static output type f32[] has no dynamic dimensions; use the homogeneous nullary \
                 constructor instead",
            )
            .into()),
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
        // axis's bounds can catch it. Identity equality is deliberately not required (inputs may be alpha-renamed).
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
            .bind(ReshapeOperation::new(), Vec::new(), &[reshape_input, first_extent, second_extent])
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
                let %1:dimension<2> = const
                    %2:dimension<3> = const
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
                let %0:dimension<2> = const
                    %1:dimension<3> = const
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
                let %0:dimension<3> = const
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
                let %0:dimension<3> = const
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
                let %2:dimension<3> = const
                    %3:f32[3] = concatenate [axis=0] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_homogeneous_differentiation_dispatch() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = TestContext::new();
        let (primal, tangent) = context
            .jvp(
                |input| {
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
            )
            .unwrap();
        assert_eq!(primal, ArrayIrValue::Array(Array::scalar(6.0_f64)));
        assert_eq!(tangent, ArrayIrValue::Array(Array::scalar(12.0_f64)));

        // Reverse mode composes the same projected JVP with projected transposition. The constant factor is a known
        // replay input to the homogeneous multiply transpose rule.
        let (primal, pullback) = context
            .vjp(
                |input| {
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
            .add_instruction(ReshapeOperation::new(), Vec::new(), vec![input, one, repeated_extent])
            .unwrap()[0];
        let output = builder
            .add_instruction(BroadcastOperation::new(vec![0, 1]), Vec::new(), vec![reshaped, two, repeated_extent])
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
                let %2:dimension<1> = const
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
}
