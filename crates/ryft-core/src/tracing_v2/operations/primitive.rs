//! Reusable staged operation enums for built-in primitives and backend extensions.
//!
//! [`ArrayOperation`] and [`LinearArrayOperation`] contain the core operations implemented by `ryft-core` plus an
//! optional statically typed backend extension slot. A backend that needs additional operations should define an
//! ordinary extension enum, define a linear extension enum when it has linear-only operations, implement the standard
//! operation traits for those enums, and select `ArrayOperation<Type, Value, Extension>` and
//! `LinearArrayOperation<LinearExtension, Tangent, Type>` as its tracing operation types.
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
use crate::contexts::{EagerContext, StagingContext};
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, Scale, ScaleOperation, SubOperation,
};
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::constants::{
    ConstantOperation, Fill, FillOperation, One, OneLike, OneLikeOperation, OneOperation, Zero, ZeroLike,
    ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::scan::{interpret_scan_lanes, read_scan_lane};
use crate::operations::control_flow::{
    ConditionOperation, ScanOperation, Select, SelectCondition, SelectOperation, WhileOperation,
};
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, ConcatenateOperation, DYNAMIC_SLICE_OPERATION_NAME,
    DYNAMIC_UPDATE_SLICE_OPERATION_NAME, DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice,
    DynamicUpdateSliceOperation, GATHER_OPERATION_NAME, Gather, GatherDimensionNumbers, GatherOperation,
    LinearDynamicSliceOperation, LinearDynamicUpdateSliceOperation, LinearGatherOperation, LinearScatterAddOperation,
    PadOperation, ReshapeOperation, SCATTER_OPERATION_NAME, Scatter, ScatterDimensionNumbers, ScatterOperation,
    ScatterReductionKind, Slice, SliceOperation, Transpose, TransposeOperation, UpdateSliceOperation,
    inverse_permutation,
};
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::operations::sharding::{ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation};
use crate::operations::trigonometric::{CosOperation, SinOperation};
use crate::operations::{BooleanLike, InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::sharding::Sharding;
use crate::tracing::{AbstractTracingContext, Tracer, TracingContext};
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingContext, ProgramBatchableOperation, ProgramBatchingContext,
    ProgramBatchingOutputAxes,
};
use crate::tracing_v2::differentiation::{
    DifferentiationContext, FactorParameterizedOperation, JvpTracer, LinearOperationOf, LinearizationContextOf,
    NestedLinearization, ProgramLinearizableOperation, TangentContext,
};
use crate::tracing_v2::operations::collective::CollectiveOperation;
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation, CustomVjpResidual,
};
use crate::tracing_v2::operations::dot::{LeftDot, LeftDotOperation, MaybeDot, RightDot, RightDotOperation};
use crate::tracing_v2::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::tracing_v2::operations::reduce::ReduceOperation;
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::tracing_v2::operations::{DotDimensionNumbers, DotOperation};
use crate::tracing_v2::rematerialization::{MaybeRematerializationName, RematerializationNameOperation};
use crate::tracing_v2::{CapturedFactor, DifferentiableOperation};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::bounds::{
    SupportsArithmeticOperations, SupportsComparisonOperations, SupportsConstantOperations,
    SupportsLinearAlgebraOperations, SupportsLinearArithmeticOperations, SupportsLinearArrayOperation,
    SupportsLinearScalarOperation, SupportsManipulationOperations, SupportsTrigonometricOperations,
};
use super::captures::MaterializeCapturedFactorOperation;
use super::control_flow::{
    DefactorizedOperation, LinearConditionOperation, LinearOperandConditionOperation, SupportsLinearCondition,
    SupportsLinearWhile, batch_condition_with_interpreter, batch_while_with_interpreter,
};
use super::dot::DotOps;
use super::scan::{LinearScanOperation, SupportsLinearScan};
use super::slicing::static_update_sizes;
use crate::operations::manipulation::Reshape;

/// Reusable operation enum for ordinary staged programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates. Most variants are thin tags
/// around one semantic primitive defined elsewhere in [`super`]. The [`Extension`](Self::Extension) variant lets
/// backends statically compose their own operation enum into the same operation type without dynamic
/// custom-operation registries. Backends that only need built-in operations can omit the `Extension` parameter and
/// use the uninhabited [`Infallible`] default.
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`],
/// [`Scale`](Self::Scale) a [`ScaleOperation`], and [`Dot`](Self::Dot) a [`DotOperation`]. The
/// [`Operation`]/[`Display`] and per-variant [`From`]/[`TryFrom`] impls are pinned to [`ArrayType`] (the enum stays
/// generic over its type parameter `T`, but its operations are only meaningful for array metadata).
#[derive(Clone, Debug)]
pub enum ArrayOperation<T: Type, V: Value<T>, Extension = Infallible> {
    /// Typed array zero.
    Zero(ZeroOperation<T>),

    /// Exemplar-derived array zero.
    ZeroLike(ZeroLikeOperation),

    /// Typed array one.
    One(OneOperation<T>),

    /// Exemplar-derived array one.
    OneLike(OneLikeOperation),

    /// Typed array constant.
    Constant(ConstantOperation<T, V>),

    /// Typed array filled with a scalar.
    Fill(FillOperation<T, f64>),

    /// Array negation.
    Neg(NegOperation),

    /// Array addition.
    Add(AddOperation),

    /// Array subtraction.
    Sub(SubOperation),

    /// Scaling by a captured scalar factor.
    Scale(ScaleOperation<T, V>),

    /// Array multiplication.
    Mul(MulOperation),

    /// Array division.
    Div(DivOperation),

    /// Array sine.
    Sin(SinOperation),

    /// Array cosine.
    Cos(CosOperation),

    /// Gradient barrier that passes its primal through unchanged.
    StopGradient(StopGradientOperation),

    /// Rematerialization tag attached to an array value.
    RematerializationName(RematerializationNameOperation),

    /// Transfer of an array value to a target memory space.
    TransferToMemory(TransferToMemoryOperation),

    /// General contraction (dot) of two arrays.
    Dot(DotOperation),

    /// Axis permutation.
    Transpose(TransposeOperation),

    /// Shape change preserving element count.
    Reshape(ReshapeOperation),

    /// Tracked resharding across a device mesh.
    Reshard(ReshardOperation),

    /// Sharding hint that constrains the layout of an array value.
    ShardingConstraint(ShardingConstraintOperation),

    /// Broadcast of an array to additional output axes.
    Broadcast(BroadcastOperation),

    /// Static slice of an array.
    Slice(SliceOperation),

    /// In-place static slice update.
    UpdateSlice(UpdateSliceOperation),

    /// Dynamic slice with runtime start indices.
    DynamicSlice(DynamicSliceOperation),

    /// Dynamic slice update with runtime start indices.
    DynamicUpdateSlice(DynamicUpdateSliceOperation),

    /// Array padding.
    Pad(PadOperation),

    /// Array concatenation.
    Concatenate(ConcatenateOperation),

    /// Gather of array elements by index.
    Gather(GatherOperation),

    /// Scatter of array elements by index.
    Scatter(ScatterOperation),

    /// Reduction over array axes.
    Reduce(ReduceOperation),

    /// Array comparison.
    Compare(CompareOperation),

    /// Logical negation.
    Not(NotOperation),

    /// Logical conjunction.
    And(AndOperation),

    /// Logical disjunction.
    Or(OrOperation),

    /// Logical exclusive disjunction.
    Xor(XorOperation),

    /// Collective communication across a device mesh.
    Collective(CollectiveOperation),

    /// Array selection on a Boolean condition.
    Select(SelectOperation),

    /// Conditional with two branch programs.
    Condition(Box<ConditionOperation<T, V, Self>>),

    /// While loop with condition and body programs.
    While(Box<WhileOperation<T, V, Self>>),

    /// Scan over a leading axis with a body program.
    Scan(Box<ScanOperation<T, V, Self>>),

    /// User-supplied `custom_jvp` operation with a closed array body.
    CustomJvp(Box<CustomJvpOperation<T, V, Self>>),

    /// User-supplied `custom_vjp` operation with a closed array body.
    CustomVjp(Box<CustomVjpOperation<T, V, Self>>),

    /// Backend extension operation.
    Extension(Extension),
}

impl<V: Value<ArrayType>, Extension> Operation<ArrayType> for ArrayOperation<ArrayType, V, Extension>
where
    Extension: Operation<ArrayType>,
{
    // Several variant payloads (the elementwise arithmetic and trigonometric operations) implement
    // [`Operation`](crate::operations::Operation) for both [`DataType`] and [`ArrayType`], so plain method-call syntax
    // cannot infer the type parameter here. The arms therefore disambiguate to `Operation<ArrayType>` explicitly.
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(operation) => <ZeroOperation<ArrayType> as Operation<ArrayType>>::name(operation),
            Self::ZeroLike(operation) => <ZeroLikeOperation as Operation<ArrayType>>::name(operation),
            Self::One(operation) => <OneOperation<ArrayType> as Operation<ArrayType>>::name(operation),
            Self::OneLike(operation) => <OneLikeOperation as Operation<ArrayType>>::name(operation),
            Self::Constant(operation) => <ConstantOperation<ArrayType, V> as Operation<ArrayType>>::name(operation),
            Self::Fill(operation) => <FillOperation<ArrayType, f64> as Operation<ArrayType>>::name(operation),
            Self::Neg(operation) => <NegOperation as Operation<ArrayType>>::name(operation),
            Self::Add(operation) => <AddOperation as Operation<ArrayType>>::name(operation),
            Self::Sub(operation) => <SubOperation as Operation<ArrayType>>::name(operation),
            Self::Scale(operation) => <ScaleOperation<ArrayType, V> as Operation<ArrayType>>::name(operation),
            Self::Mul(operation) => <MulOperation as Operation<ArrayType>>::name(operation),
            Self::Div(operation) => <DivOperation as Operation<ArrayType>>::name(operation),
            Self::Sin(operation) => <SinOperation as Operation<ArrayType>>::name(operation),
            Self::Cos(operation) => <CosOperation as Operation<ArrayType>>::name(operation),
            Self::StopGradient(operation) => <StopGradientOperation as Operation<ArrayType>>::name(operation),
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as Operation<ArrayType>>::name(operation)
            }
            Self::TransferToMemory(operation) => <TransferToMemoryOperation as Operation<ArrayType>>::name(operation),
            Self::Dot(operation) => <DotOperation as Operation<ArrayType>>::name(operation),
            Self::Transpose(operation) => <TransposeOperation as Operation<ArrayType>>::name(operation),
            Self::Reshape(operation) => <ReshapeOperation as Operation<ArrayType>>::name(operation),
            Self::Reshard(operation) => <ReshardOperation as Operation<ArrayType>>::name(operation),
            Self::ShardingConstraint(operation) => {
                <ShardingConstraintOperation as Operation<ArrayType>>::name(operation)
            }
            Self::Broadcast(operation) => <BroadcastOperation as Operation<ArrayType>>::name(operation),
            Self::Slice(operation) => <SliceOperation as Operation<ArrayType>>::name(operation),
            Self::UpdateSlice(operation) => <UpdateSliceOperation as Operation<ArrayType>>::name(operation),
            Self::DynamicSlice(operation) => <DynamicSliceOperation as Operation<ArrayType>>::name(operation),
            Self::DynamicUpdateSlice(operation) => {
                <DynamicUpdateSliceOperation as Operation<ArrayType>>::name(operation)
            }
            Self::Pad(operation) => <PadOperation as Operation<ArrayType>>::name(operation),
            Self::Concatenate(operation) => <ConcatenateOperation as Operation<ArrayType>>::name(operation),
            Self::Gather(operation) => <GatherOperation as Operation<ArrayType>>::name(operation),
            Self::Scatter(operation) => <ScatterOperation as Operation<ArrayType>>::name(operation),
            Self::Reduce(operation) => <ReduceOperation as Operation<ArrayType>>::name(operation),
            Self::Compare(operation) => <CompareOperation as Operation<ArrayType>>::name(operation),
            Self::Not(operation) => <NotOperation as Operation<ArrayType>>::name(operation),
            Self::And(operation) => <AndOperation as Operation<ArrayType>>::name(operation),
            Self::Or(operation) => <OrOperation as Operation<ArrayType>>::name(operation),
            Self::Xor(operation) => <XorOperation as Operation<ArrayType>>::name(operation),
            Self::Collective(operation) => <CollectiveOperation as Operation<ArrayType>>::name(operation),
            Self::Select(operation) => <SelectOperation as Operation<ArrayType>>::name(operation),
            Self::Condition(operation) => {
                <ConditionOperation<ArrayType, V, Self> as Operation<ArrayType>>::name(&**operation)
            }
            Self::While(operation) => <WhileOperation<ArrayType, V, Self> as Operation<ArrayType>>::name(&**operation),
            Self::Scan(operation) => <ScanOperation<ArrayType, V, Self> as Operation<ArrayType>>::name(&**operation),
            Self::CustomJvp(operation) => {
                <CustomJvpOperation<ArrayType, V, Self> as Operation<ArrayType>>::name(&**operation)
            }
            Self::CustomVjp(operation) => {
                <CustomVjpOperation<ArrayType, V, Self> as Operation<ArrayType>>::name(&**operation)
            }
            Self::Extension(operation) => <Extension as Operation<ArrayType>>::name(operation),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<ArrayType> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::One(operation) => {
                <OneOperation<ArrayType> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Constant(operation) => {
                <ConstantOperation<ArrayType, V> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Fill(operation) => {
                <FillOperation<ArrayType, f64> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Neg(operation) => <NegOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Add(operation) => <AddOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Sub(operation) => <SubOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Scale(operation) => {
                <ScaleOperation<ArrayType, V> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Mul(operation) => <MulOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Div(operation) => <DivOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Sin(operation) => <SinOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Cos(operation) => <CosOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::StopGradient(operation) => {
                <StopGradientOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::TransferToMemory(operation) => {
                <TransferToMemoryOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Dot(operation) => <DotOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Transpose(operation) => {
                <TransposeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Reshape(operation) => {
                <ReshapeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Reshard(operation) => {
                <ReshardOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::ShardingConstraint(operation) => {
                <ShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Broadcast(operation) => {
                <BroadcastOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Slice(operation) => {
                <SliceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::UpdateSlice(operation) => {
                <UpdateSliceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::DynamicSlice(operation) => {
                <DynamicSliceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::DynamicUpdateSlice(operation) => {
                <DynamicUpdateSliceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Pad(operation) => <PadOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Concatenate(operation) => {
                <ConcatenateOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Gather(operation) => {
                <GatherOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Scatter(operation) => {
                <ScatterOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Reduce(operation) => {
                <ReduceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Compare(operation) => {
                <CompareOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Not(operation) => <NotOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::And(operation) => <AndOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Or(operation) => <OrOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Xor(operation) => <XorOperation as Operation<ArrayType>>::infer_output_types(operation, input_types),
            Self::Collective(operation) => {
                <CollectiveOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Select(operation) => {
                <SelectOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Condition(operation) => {
                <ConditionOperation<ArrayType, V, Self> as Operation<ArrayType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
            Self::While(operation) => <WhileOperation<ArrayType, V, Self> as Operation<ArrayType>>::infer_output_types(
                &**operation,
                input_types,
            ),
            Self::Scan(operation) => <ScanOperation<ArrayType, V, Self> as Operation<ArrayType>>::infer_output_types(
                &**operation,
                input_types,
            ),
            Self::CustomJvp(operation) => {
                <CustomJvpOperation<ArrayType, V, Self> as Operation<ArrayType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
            Self::CustomVjp(operation) => {
                <CustomVjpOperation<ArrayType, V, Self> as Operation<ArrayType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
            Self::Extension(operation) => {
                <Extension as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<ArrayType> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::One(operation) => {
                <OneOperation<ArrayType> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Constant(operation) => {
                <ConstantOperation<ArrayType, V> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Fill(operation) => {
                <FillOperation<ArrayType, f64> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Neg(operation) => <NegOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Add(operation) => <AddOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Sub(operation) => <SubOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Scale(operation) => {
                <ScaleOperation<ArrayType, V> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Mul(operation) => <MulOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Div(operation) => <DivOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Sin(operation) => <SinOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Cos(operation) => <CosOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::StopGradient(operation) => {
                <StopGradientOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::TransferToMemory(operation) => {
                <TransferToMemoryOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Dot(operation) => <DotOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Transpose(operation) => {
                <TransposeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Reshape(operation) => {
                <ReshapeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Reshard(operation) => {
                <ReshardOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::ShardingConstraint(operation) => {
                <ShardingConstraintOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Broadcast(operation) => {
                <BroadcastOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Slice(operation) => {
                <SliceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::UpdateSlice(operation) => {
                <UpdateSliceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::DynamicSlice(operation) => {
                <DynamicSliceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::DynamicUpdateSlice(operation) => {
                <DynamicUpdateSliceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Pad(operation) => <PadOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Concatenate(operation) => {
                <ConcatenateOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Gather(operation) => {
                <GatherOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Scatter(operation) => {
                <ScatterOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Reduce(operation) => {
                <ReduceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Compare(operation) => {
                <CompareOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Not(operation) => <NotOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::And(operation) => <AndOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Or(operation) => <OrOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Xor(operation) => <XorOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Collective(operation) => {
                <CollectiveOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Select(operation) => {
                <SelectOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Condition(operation) => <ConditionOperation<ArrayType, V, Self> as Operation<ArrayType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::While(operation) => <WhileOperation<ArrayType, V, Self> as Operation<ArrayType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::Scan(operation) => <ScanOperation<ArrayType, V, Self> as Operation<ArrayType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::CustomJvp(operation) => <CustomJvpOperation<ArrayType, V, Self> as Operation<ArrayType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::CustomVjp(operation) => <CustomVjpOperation<ArrayType, V, Self> as Operation<ArrayType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::Extension(operation) => {
                <Extension as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
        }
    }
}

impl<V: Value<ArrayType>, Extension> Display for ArrayOperation<ArrayType, V, Extension>
where
    Extension: Operation<ArrayType>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Value<T>, Extension> From<ZeroOperation<T>> for ArrayOperation<T, V, Extension> {
    fn from(operation: ZeroOperation<T>) -> Self {
        Self::Zero(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ZeroOperation<T> {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Zero(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ZeroLikeOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ZeroLikeOperation) -> Self {
        Self::ZeroLike(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ZeroLikeOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::ZeroLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<OneOperation<T>> for ArrayOperation<T, V, Extension> {
    fn from(operation: OneOperation<T>) -> Self {
        Self::One(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a OneOperation<T> {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::One(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<OneLikeOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: OneLikeOperation) -> Self {
        Self::OneLike(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a OneLikeOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::OneLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ConstantOperation<T, V>> for ArrayOperation<T, V, Extension> {
    fn from(operation: ConstantOperation<T, V>) -> Self {
        Self::Constant(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ConstantOperation<T, V> {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Constant(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<FillOperation<T, f64>> for ArrayOperation<T, V, Extension> {
    fn from(operation: FillOperation<T, f64>) -> Self {
        Self::Fill(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a FillOperation<T, f64> {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Fill(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<NegOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: NegOperation) -> Self {
        Self::Neg(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a NegOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Neg(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<AddOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: AddOperation) -> Self {
        Self::Add(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a AddOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Add(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<SubOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: SubOperation) -> Self {
        Self::Sub(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a SubOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Sub(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ScaleOperation<T, V>> for ArrayOperation<T, V, Extension> {
    fn from(operation: ScaleOperation<T, V>) -> Self {
        Self::Scale(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ScaleOperation<T, V> {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Scale(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<MulOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: MulOperation) -> Self {
        Self::Mul(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a MulOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Mul(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<DivOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: DivOperation) -> Self {
        Self::Div(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a DivOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Div(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<SinOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: SinOperation) -> Self {
        Self::Sin(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a SinOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Sin(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<CosOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: CosOperation) -> Self {
        Self::Cos(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a CosOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Cos(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<StopGradientOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: StopGradientOperation) -> Self {
        Self::StopGradient(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a StopGradientOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::StopGradient(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<RematerializationNameOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: RematerializationNameOperation) -> Self {
        Self::RematerializationName(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a RematerializationNameOperation
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::RematerializationName(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<TransferToMemoryOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: TransferToMemoryOperation) -> Self {
        Self::TransferToMemory(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a TransferToMemoryOperation
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::TransferToMemory(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<DotOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: DotOperation) -> Self {
        Self::Dot(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a DotOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Dot(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<TransposeOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: TransposeOperation) -> Self {
        Self::Transpose(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a TransposeOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Transpose(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ReshapeOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ReshapeOperation) -> Self {
        Self::Reshape(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ReshapeOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Reshape(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ReshardOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ReshardOperation) -> Self {
        Self::Reshard(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ReshardOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Reshard(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ShardingConstraintOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ShardingConstraintOperation) -> Self {
        Self::ShardingConstraint(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a ShardingConstraintOperation
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::ShardingConstraint(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<BroadcastOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: BroadcastOperation) -> Self {
        Self::Broadcast(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a BroadcastOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Broadcast(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<SliceOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: SliceOperation) -> Self {
        Self::Slice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a SliceOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Slice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<UpdateSliceOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: UpdateSliceOperation) -> Self {
        Self::UpdateSlice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a UpdateSliceOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::UpdateSlice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<DynamicSliceOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: DynamicSliceOperation) -> Self {
        Self::DynamicSlice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a DynamicSliceOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::DynamicSlice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<DynamicUpdateSliceOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: DynamicUpdateSliceOperation) -> Self {
        Self::DynamicUpdateSlice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a DynamicUpdateSliceOperation
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::DynamicUpdateSlice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<PadOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: PadOperation) -> Self {
        Self::Pad(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a PadOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Pad(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ConcatenateOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ConcatenateOperation) -> Self {
        Self::Concatenate(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ConcatenateOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Concatenate(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<GatherOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: GatherOperation) -> Self {
        Self::Gather(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a GatherOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Gather(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ScatterOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ScatterOperation) -> Self {
        Self::Scatter(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ScatterOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Scatter(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ReduceOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: ReduceOperation) -> Self {
        Self::Reduce(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a ReduceOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Reduce(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<CompareOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: CompareOperation) -> Self {
        Self::Compare(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a CompareOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Compare(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<NotOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: NotOperation) -> Self {
        Self::Not(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a NotOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Not(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<AndOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: AndOperation) -> Self {
        Self::And(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a AndOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::And(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<OrOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: OrOperation) -> Self {
        Self::Or(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a OrOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Or(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<XorOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: XorOperation) -> Self {
        Self::Xor(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a XorOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Xor(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<CollectiveOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: CollectiveOperation) -> Self {
        Self::Collective(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a CollectiveOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Collective(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<SelectOperation> for ArrayOperation<T, V, Extension> {
    fn from(operation: SelectOperation) -> Self {
        Self::Select(operation)
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>> for &'a SelectOperation {
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Select(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ConditionOperation<T, V, ArrayOperation<T, V, Extension>>>
    for ArrayOperation<T, V, Extension>
{
    fn from(operation: ConditionOperation<T, V, ArrayOperation<T, V, Extension>>) -> Self {
        Self::Condition(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a ConditionOperation<T, V, ArrayOperation<T, V, Extension>>
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Condition(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<WhileOperation<T, V, ArrayOperation<T, V, Extension>>>
    for ArrayOperation<T, V, Extension>
{
    fn from(operation: WhileOperation<T, V, ArrayOperation<T, V, Extension>>) -> Self {
        Self::While(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a WhileOperation<T, V, ArrayOperation<T, V, Extension>>
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::While(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<ScanOperation<T, V, ArrayOperation<T, V, Extension>>>
    for ArrayOperation<T, V, Extension>
{
    fn from(operation: ScanOperation<T, V, ArrayOperation<T, V, Extension>>) -> Self {
        Self::Scan(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a ScanOperation<T, V, ArrayOperation<T, V, Extension>>
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::Scan(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<CustomJvpOperation<T, V, ArrayOperation<T, V, Extension>>>
    for ArrayOperation<T, V, Extension>
{
    fn from(operation: CustomJvpOperation<T, V, ArrayOperation<T, V, Extension>>) -> Self {
        Self::CustomJvp(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a CustomJvpOperation<T, V, ArrayOperation<T, V, Extension>>
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::CustomJvp(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, Extension> From<CustomVjpOperation<T, V, ArrayOperation<T, V, Extension>>>
    for ArrayOperation<T, V, Extension>
{
    fn from(operation: CustomVjpOperation<T, V, ArrayOperation<T, V, Extension>>) -> Self {
        Self::CustomVjp(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, Extension> TryFrom<&'a ArrayOperation<T, V, Extension>>
    for &'a CustomVjpOperation<T, V, ArrayOperation<T, V, Extension>>
{
    type Error = ();

    fn try_from(value: &'a ArrayOperation<T, V, Extension>) -> Result<Self, ()> {
        match value {
            ArrayOperation::CustomVjp(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

// Differentiation (JVP) for the `ArrayOperation` sum type: each variant delegates to its backing operation's own
// `DifferentiableOperation` rule. The per-variant `<Payload>: DifferentiableOperation<D>` bounds cover the
// non-self-referential variants (including the `Extension` slot); the self-referential higher-order
// `Condition`/`While`/`Scan` and `CustomJvp`/`CustomVjp` arms resolve against this impl's assumed
// `Self: DifferentiableOperation<D>`. The remaining where-clause spells the leaf closure of value and
// linear-operation capabilities those per-variant rules require.
impl<V: Value<ArrayType>, Extension, D> DifferentiableOperation<D> for ArrayOperation<ArrayType, V, Extension>
where
    Extension: Operation<ArrayType>,
    ZeroOperation<ArrayType>: DifferentiableOperation<D>,
    ZeroLikeOperation: DifferentiableOperation<D>,
    OneOperation<ArrayType>: DifferentiableOperation<D>,
    OneLikeOperation: DifferentiableOperation<D>,
    ConstantOperation<ArrayType, V>: DifferentiableOperation<D>,
    FillOperation<ArrayType, f64>: DifferentiableOperation<D>,
    NegOperation: DifferentiableOperation<D>,
    AddOperation: DifferentiableOperation<D>,
    SubOperation: DifferentiableOperation<D>,
    ScaleOperation<ArrayType, V>: DifferentiableOperation<D>,
    MulOperation: DifferentiableOperation<D>,
    DivOperation: DifferentiableOperation<D>,
    SinOperation: DifferentiableOperation<D>,
    CosOperation: DifferentiableOperation<D>,
    StopGradientOperation: DifferentiableOperation<D>,
    RematerializationNameOperation: DifferentiableOperation<D>,
    TransferToMemoryOperation: DifferentiableOperation<D>,
    DotOperation: DifferentiableOperation<D>,
    TransposeOperation: DifferentiableOperation<D>,
    ReshapeOperation: DifferentiableOperation<D>,
    ReshardOperation: DifferentiableOperation<D>,
    ShardingConstraintOperation: DifferentiableOperation<D>,
    BroadcastOperation: DifferentiableOperation<D>,
    SliceOperation: DifferentiableOperation<D>,
    UpdateSliceOperation: DifferentiableOperation<D>,
    DynamicSliceOperation: DifferentiableOperation<D>,
    DynamicUpdateSliceOperation: DifferentiableOperation<D>,
    PadOperation: DifferentiableOperation<D>,
    ConcatenateOperation: DifferentiableOperation<D>,
    GatherOperation: DifferentiableOperation<D>,
    ScatterOperation: DifferentiableOperation<D>,
    ReduceOperation: DifferentiableOperation<D>,
    CompareOperation: DifferentiableOperation<D>,
    NotOperation: DifferentiableOperation<D>,
    AndOperation: DifferentiableOperation<D>,
    OrOperation: DifferentiableOperation<D>,
    XorOperation: DifferentiableOperation<D>,
    CollectiveOperation: DifferentiableOperation<D>,
    SelectOperation: DifferentiableOperation<D>,
    Extension: DifferentiableOperation<D>,
    D: DifferentiationContext<Type = ArrayType, Constant = V>
        + Domain<Operation = ArrayOperation<ArrayType, V, Extension>>,
    D::Operation: From<ZeroOperation<ArrayType>> + From<OneOperation<ArrayType>> + From<FillOperation<ArrayType, f64>>,
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
        + SelectCondition<Condition = D::Value>
        + BooleanLike
        + Parameterized<D::Value>,
    D::Tangent: Transpose + Broadcast + super::reduce::Reduce + Slice + Reshard + ConstrainSharding,
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
    LinearOperationOf<D>: SupportsLinearArrayOperation<ArrayType, CapturedFactor<ArrayType, D::Value>>
        + crate::tracing_v2::ResidualizedOperation<D>
        + From<
            CustomVjpCallOperation<
                ArrayType,
                V,
                ArrayOperation<ArrayType, V, Extension>,
                CapturedFactor<ArrayType, D::Value>,
            >,
        > + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<CapturedFactor<ArrayType, D::Value>>>
        + From<LinearDynamicSliceOperation<CapturedFactor<ArrayType, D::Value>>>
        + From<LinearDynamicUpdateSliceOperation<CapturedFactor<ArrayType, D::Value>>>
        + From<LinearGatherOperation<CapturedFactor<ArrayType, D::Value>>>
        + From<LinearScatterAddOperation<CapturedFactor<ArrayType, D::Value>>>
        + SupportsLinearCondition<ArrayType, D::Tangent, CapturedFactor<ArrayType, D::Value>>
        + SupportsLinearWhile<
            ArrayType,
            D::Tangent,
            CapturedFactor<ArrayType, D::Value>,
            ArrayOperation<ArrayType, V, Extension>,
        > + SupportsLinearScan<ArrayType, D::Tangent, CapturedFactor<ArrayType, D::Value>>,
    ArrayOperation<ArrayType, V, Extension>: Clone + ProgramLinearizableOperation<D>,
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
            Self::Zero(operation) => operation.jvp(context, inputs),
            Self::ZeroLike(operation) => operation.jvp(context, inputs),
            Self::One(operation) => operation.jvp(context, inputs),
            Self::OneLike(operation) => operation.jvp(context, inputs),
            Self::Constant(operation) => operation.jvp(context, inputs),
            Self::Fill(operation) => operation.jvp(context, inputs),
            Self::Neg(operation) => operation.jvp(context, inputs),
            Self::Add(operation) => operation.jvp(context, inputs),
            Self::Sub(operation) => operation.jvp(context, inputs),
            Self::Scale(operation) => operation.jvp(context, inputs),
            Self::Mul(operation) => operation.jvp(context, inputs),
            Self::Div(operation) => operation.jvp(context, inputs),
            Self::Sin(operation) => operation.jvp(context, inputs),
            Self::Cos(operation) => operation.jvp(context, inputs),
            Self::StopGradient(operation) => operation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::TransferToMemory(operation) => operation.jvp(context, inputs),
            Self::Dot(operation) => operation.jvp(context, inputs),
            Self::Transpose(operation) => operation.jvp(context, inputs),
            Self::Reshape(operation) => operation.jvp(context, inputs),
            Self::Reshard(operation) => operation.jvp(context, inputs),
            Self::ShardingConstraint(operation) => operation.jvp(context, inputs),
            Self::Broadcast(operation) => operation.jvp(context, inputs),
            Self::Slice(operation) => operation.jvp(context, inputs),
            Self::UpdateSlice(operation) => operation.jvp(context, inputs),
            Self::DynamicSlice(operation) => operation.jvp(context, inputs),
            Self::DynamicUpdateSlice(operation) => operation.jvp(context, inputs),
            Self::Pad(operation) => operation.jvp(context, inputs),
            Self::Concatenate(operation) => operation.jvp(context, inputs),
            Self::Gather(operation) => operation.jvp(context, inputs),
            Self::Scatter(operation) => operation.jvp(context, inputs),
            Self::Reduce(operation) => operation.jvp(context, inputs),
            Self::Compare(operation) => operation.jvp(context, inputs),
            Self::Not(operation) => operation.jvp(context, inputs),
            Self::And(operation) => operation.jvp(context, inputs),
            Self::Or(operation) => operation.jvp(context, inputs),
            Self::Xor(operation) => operation.jvp(context, inputs),
            Self::Collective(operation) => operation.jvp(context, inputs),
            Self::Select(operation) => operation.jvp(context, inputs),
            Self::Condition(operation) => <ConditionOperation<ArrayType, V, Self> as DifferentiableOperation<D>>::jvp(
                &**operation,
                context,
                inputs,
            ),
            Self::While(operation) => {
                <WhileOperation<ArrayType, V, Self> as DifferentiableOperation<D>>::jvp(&**operation, context, inputs)
            }
            Self::Scan(operation) => {
                <ScanOperation<ArrayType, V, Self> as DifferentiableOperation<D>>::jvp(&**operation, context, inputs)
            }
            Self::CustomJvp(operation) => <CustomJvpOperation<ArrayType, V, Self> as DifferentiableOperation<D>>::jvp(
                &**operation,
                context,
                inputs,
            ),
            Self::CustomVjp(operation) => <CustomVjpOperation<ArrayType, V, Self> as DifferentiableOperation<D>>::jvp(
                &**operation,
                context,
                inputs,
            ),
            Self::Extension(extension) => extension.jvp(context, inputs),
        }
    }
}

/// Reusable operation enum for staged linear programs.
///
/// [`LinearArrayOperation`] is the linear-program sibling of [`ArrayOperation`]. It contains
/// operations that can appear in tangent and cotangent programs, including captured-factor linear
/// maps such as [`LeftDot`](Self::LeftDot) and [`RightDot`](Self::RightDot), and the
/// linearized higher-order operations needed by rematerialization and control flow. The
/// [`Extension`](Self::Extension) variant lets backends statically compose linear backend operations into the same
/// operation type. Backends that only need built-in linear operations can omit the `Extension` parameter and use
/// the uninhabited [`Infallible`] default.
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation). The [`Operation`]/[`Display`] and per-variant [`From`]/[`TryFrom`] impls are
/// pinned to [`ArrayType`] (the enum stays generic over its type parameter `T`, but its operations are only
/// meaningful for array metadata).
///
/// The `C` parameter is the constant type of the [`DifferentiationContext`]
/// that stages the linear program: every context pins `C` to its [`Domain::Constant`](crate::domains::Domain) in
/// its `LinearOperation` associated-type definition. It types the user-supplied programs captured by
/// [`CustomVjpCall`](Self::CustomVjpCall), which are written over context constants rather than over the linear
/// value type `V` (`V` instantiates to tracers inside transform contexts, while captured programs always hold
/// concrete constants).
#[derive(Clone, Debug)]
pub enum LinearArrayOperation<
    T: Type,
    V: Value<T>,
    C: Value<T>,
    Extension = Infallible,
    F: Value<T> = V,
    O = ArrayOperation<T, C, Extension>,
> {
    /// Typed array zero.
    Zero(ZeroOperation<T>),

    /// Exemplar-derived array zero.
    ZeroLike(ZeroLikeOperation),

    /// Typed array one.
    One(OneOperation<T>),

    /// Exemplar-derived array one.
    OneLike(OneLikeOperation),

    /// Typed array constant.
    Constant(ConstantOperation<T, V>),

    /// Typed array filled with a scalar.
    Fill(FillOperation<T, f64>),

    /// Array negation.
    Neg(NegOperation),

    /// Array addition.
    Add(AddOperation),

    /// Array subtraction.
    Sub(SubOperation),

    /// Scaling by a captured factor.
    Scale(ScaleOperation<T, F>),

    /// Array multiplication.
    Mul(MulOperation),

    /// Transfer of an array value to a target memory space.
    TransferToMemory(TransferToMemoryOperation),

    /// Axis permutation.
    Transpose(TransposeOperation),

    /// Contraction against a captured left factor.
    LeftDot(LeftDotOperation<F>),

    /// Contraction against a captured right factor.
    RightDot(RightDotOperation<F>),

    /// Shape change preserving element count.
    Reshape(ReshapeOperation),

    /// Tracked resharding across a device mesh.
    Reshard(ReshardOperation),

    /// Sharding hint that constrains the layout of an array value.
    ShardingConstraint(ShardingConstraintOperation),

    /// Broadcast of an array to additional output axes.
    Broadcast(BroadcastOperation),

    /// Static slice of an array.
    Slice(SliceOperation),

    /// In-place static slice update.
    UpdateSlice(UpdateSliceOperation),

    /// Dynamic slice with captured start indices.
    DynamicSlice(LinearDynamicSliceOperation<F>),

    /// Dynamic slice update with captured start indices.
    DynamicUpdateSlice(LinearDynamicUpdateSliceOperation<F>),

    /// Gather of array elements by captured index.
    Gather(LinearGatherOperation<F>),

    /// Scatter-add of array elements by captured index.
    ScatterAdd(LinearScatterAddOperation<F>),

    /// Array padding.
    Pad(PadOperation),

    /// Array concatenation.
    Concatenate(ConcatenateOperation),

    /// Reduction over array axes.
    Reduce(ReduceOperation),

    /// Selection on a captured Boolean condition.
    Select(LinearSelectOperation<F>),

    /// Residual reference into the linearization environment.
    Residual(MaterializeCapturedFactorOperation<F>),

    /// Recomputed primal operation.
    Recompute(O),

    /// Linear conditional with two captured-factor branch programs.
    Condition(LinearConditionOperation<T, V, C, Extension, F, O>),

    /// Linear conditional whose predicate is an operand rather than a captured factor.
    OperandCondition(LinearOperandConditionOperation<T, V, C, Extension, F, O>),

    /// While loop with condition and body programs.
    While(Box<WhileOperation<T, V, Self>>),

    /// Linear scan over a leading axis with a body program.
    Scan(LinearScanOperation<T, V, C, Extension, F, O>),

    /// Opaque `custom_vjp` call whose transpose replays the user's backward program.
    CustomVjpCall(Box<CustomVjpCallOperation<T, C, O, F>>),

    /// Backend extension operation.
    Extension(Extension),
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> Operation<ArrayType>
    for LinearArrayOperation<ArrayType, V, C, Extension, F, O>
where
    Extension: Operation<ArrayType>,
    O: Operation<ArrayType>,
{
    // Several variant payloads (the elementwise arithmetic operations) implement
    // [`Operation`](crate::operations::Operation) for both [`DataType`] and [`ArrayType`], so plain method-call
    // syntax cannot infer the type parameter here. The arms therefore disambiguate to `Operation<ArrayType>`
    // explicitly.
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(operation) => <ZeroOperation<ArrayType> as Operation<ArrayType>>::name(operation),
            Self::ZeroLike(operation) => <ZeroLikeOperation as Operation<ArrayType>>::name(operation),
            Self::One(operation) => <OneOperation<ArrayType> as Operation<ArrayType>>::name(operation),
            Self::OneLike(operation) => <OneLikeOperation as Operation<ArrayType>>::name(operation),
            Self::Constant(operation) => <ConstantOperation<ArrayType, V> as Operation<ArrayType>>::name(operation),
            Self::Fill(operation) => <FillOperation<ArrayType, f64> as Operation<ArrayType>>::name(operation),
            Self::Neg(operation) => <NegOperation as Operation<ArrayType>>::name(operation),
            Self::Add(operation) => <AddOperation as Operation<ArrayType>>::name(operation),
            Self::Sub(operation) => <SubOperation as Operation<ArrayType>>::name(operation),
            Self::Scale(operation) => <ScaleOperation<ArrayType, F> as Operation<ArrayType>>::name(operation),
            Self::Mul(operation) => <MulOperation as Operation<ArrayType>>::name(operation),
            Self::TransferToMemory(operation) => <TransferToMemoryOperation as Operation<ArrayType>>::name(operation),
            Self::Transpose(operation) => <TransposeOperation as Operation<ArrayType>>::name(operation),
            Self::LeftDot(operation) => <LeftDotOperation<F> as Operation<ArrayType>>::name(operation),
            Self::RightDot(operation) => <RightDotOperation<F> as Operation<ArrayType>>::name(operation),
            Self::Reshape(operation) => <ReshapeOperation as Operation<ArrayType>>::name(operation),
            Self::Reshard(operation) => <ReshardOperation as Operation<ArrayType>>::name(operation),
            Self::ShardingConstraint(operation) => {
                <ShardingConstraintOperation as Operation<ArrayType>>::name(operation)
            }
            Self::Broadcast(operation) => <BroadcastOperation as Operation<ArrayType>>::name(operation),
            Self::Slice(operation) => <SliceOperation as Operation<ArrayType>>::name(operation),
            Self::UpdateSlice(operation) => <UpdateSliceOperation as Operation<ArrayType>>::name(operation),
            Self::DynamicSlice(operation) => <LinearDynamicSliceOperation<F> as Operation<ArrayType>>::name(operation),
            Self::DynamicUpdateSlice(operation) => {
                <LinearDynamicUpdateSliceOperation<F> as Operation<ArrayType>>::name(operation)
            }
            Self::Gather(operation) => <LinearGatherOperation<F> as Operation<ArrayType>>::name(operation),
            Self::ScatterAdd(operation) => <LinearScatterAddOperation<F> as Operation<ArrayType>>::name(operation),
            Self::Pad(operation) => <PadOperation as Operation<ArrayType>>::name(operation),
            Self::Concatenate(operation) => <ConcatenateOperation as Operation<ArrayType>>::name(operation),
            Self::Reduce(operation) => <ReduceOperation as Operation<ArrayType>>::name(operation),
            Self::Select(operation) => <LinearSelectOperation<F> as Operation<ArrayType>>::name(operation),
            Self::Residual(operation) => {
                <MaterializeCapturedFactorOperation<F> as Operation<ArrayType>>::name(operation)
            }
            Self::Recompute(operation) => <O as Operation<ArrayType>>::name(operation),
            Self::Condition(operation) => {
                <LinearConditionOperation<ArrayType, V, C, Extension, F, O> as Operation<ArrayType>>::name(operation)
            }
            Self::OperandCondition(operation) => {
                <LinearOperandConditionOperation<ArrayType, V, C, Extension, F, O> as Operation<ArrayType>>::name(
                    operation,
                )
            }
            Self::While(operation) => <WhileOperation<ArrayType, V, Self> as Operation<ArrayType>>::name(&**operation),
            Self::Scan(operation) => {
                <LinearScanOperation<ArrayType, V, C, Extension, F, O> as Operation<ArrayType>>::name(operation)
            }
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<ArrayType, C, O, F> as Operation<ArrayType>>::name(&**operation)
            }
            Self::Extension(operation) => <Extension as Operation<ArrayType>>::name(operation),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<ArrayType> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::One(operation) => {
                <OneOperation<ArrayType> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Constant(operation) => {
                <ConstantOperation<ArrayType, V> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Fill(operation) => {
                <FillOperation<ArrayType, f64> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Neg(operation) => {
                <NegOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Add(operation) => {
                <AddOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Sub(operation) => {
                <SubOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Scale(operation) => {
                <ScaleOperation<ArrayType, F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Mul(operation) => {
                <MulOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::TransferToMemory(operation) => {
                <TransferToMemoryOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Transpose(operation) => {
                <TransposeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::LeftDot(operation) => {
                <LeftDotOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::RightDot(operation) => {
                <RightDotOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Reshape(operation) => {
                <ReshapeOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Reshard(operation) => {
                <ReshardOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::ShardingConstraint(operation) => {
                <ShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Broadcast(operation) => {
                <BroadcastOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Slice(operation) => {
                <SliceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::UpdateSlice(operation) => {
                <UpdateSliceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::DynamicSlice(operation) => {
                <LinearDynamicSliceOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::DynamicUpdateSlice(operation) => {
                <LinearDynamicUpdateSliceOperation<F> as Operation<ArrayType>>::infer_output_types(
                    operation,
                    input_types,
                )
            }
            Self::Gather(operation) => {
                <LinearGatherOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::ScatterAdd(operation) => {
                <LinearScatterAddOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Pad(operation) => {
                <PadOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Concatenate(operation) => {
                <ConcatenateOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Reduce(operation) => {
                <ReduceOperation as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Select(operation) => {
                <LinearSelectOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Residual(operation) => {
                <MaterializeCapturedFactorOperation<F> as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Recompute(operation) => {
                <O as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
            Self::Condition(operation) => {
                <LinearConditionOperation<ArrayType, V, C, Extension, F, O> as Operation<
                    ArrayType,
                >>::infer_output_types(operation, input_types)
            }
            Self::OperandCondition(operation) => {
                <LinearOperandConditionOperation<ArrayType, V, C, Extension, F, O> as Operation<
                    ArrayType,
                >>::infer_output_types(operation, input_types)
            }
            Self::While(operation) => {
                <WhileOperation<ArrayType, V, Self> as Operation<ArrayType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
            Self::Scan(operation) => {
                <LinearScanOperation<ArrayType, V, C, Extension, F, O> as Operation<ArrayType>>::infer_output_types(
                    operation,
                    input_types,
                )
            }
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<ArrayType, C, O, F> as Operation<ArrayType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
            Self::Extension(operation) => {
                <Extension as Operation<ArrayType>>::infer_output_types(operation, input_types)
            }
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<ArrayType> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::One(operation) => {
                <OneOperation<ArrayType> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Constant(operation) => {
                <ConstantOperation<ArrayType, V> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Fill(operation) => {
                <FillOperation<ArrayType, f64> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Neg(operation) => <NegOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Add(operation) => <AddOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Sub(operation) => <SubOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Scale(operation) => {
                <ScaleOperation<ArrayType, F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Mul(operation) => <MulOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::TransferToMemory(operation) => {
                <TransferToMemoryOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Transpose(operation) => {
                <TransposeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::LeftDot(operation) => {
                <LeftDotOperation<F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::RightDot(operation) => {
                <RightDotOperation<F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Reshape(operation) => {
                <ReshapeOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Reshard(operation) => {
                <ReshardOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::ShardingConstraint(operation) => {
                <ShardingConstraintOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Broadcast(operation) => {
                <BroadcastOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Slice(operation) => {
                <SliceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::UpdateSlice(operation) => {
                <UpdateSliceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::DynamicSlice(operation) => {
                <LinearDynamicSliceOperation<F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::DynamicUpdateSlice(operation) => {
                <LinearDynamicUpdateSliceOperation<F> as Operation<ArrayType>>::render(
                    operation,
                    formatter,
                    indentation,
                )
            }
            Self::Gather(operation) => {
                <LinearGatherOperation<F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::ScatterAdd(operation) => {
                <LinearScatterAddOperation<F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Pad(operation) => <PadOperation as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Concatenate(operation) => {
                <ConcatenateOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Reduce(operation) => {
                <ReduceOperation as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Select(operation) => {
                <LinearSelectOperation<F> as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
            Self::Residual(operation) => <MaterializeCapturedFactorOperation<F> as Operation<ArrayType>>::render(
                operation,
                formatter,
                indentation,
            ),
            Self::Recompute(operation) => <O as Operation<ArrayType>>::render(operation, formatter, indentation),
            Self::Condition(operation) => <LinearConditionOperation<ArrayType, V, C, Extension, F, O> as Operation<
                ArrayType,
            >>::render(operation, formatter, indentation),
            Self::OperandCondition(operation) => {
                <LinearOperandConditionOperation<ArrayType, V, C, Extension, F, O> as Operation<ArrayType>>::render(
                    operation,
                    formatter,
                    indentation,
                )
            }
            Self::While(operation) => <WhileOperation<ArrayType, V, Self> as Operation<ArrayType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::Scan(operation) => {
                <LinearScanOperation<ArrayType, V, C, Extension, F, O> as Operation<ArrayType>>::render(
                    operation,
                    formatter,
                    indentation,
                )
            }
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<ArrayType, C, O, F> as Operation<ArrayType>>::render(
                    &**operation,
                    formatter,
                    indentation,
                )
            }
            Self::Extension(operation) => {
                <Extension as Operation<ArrayType>>::render(operation, formatter, indentation)
            }
        }
    }
}

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O> Display
    for LinearArrayOperation<ArrayType, V, C, Extension, F, O>
where
    Extension: Operation<ArrayType>,
    O: Operation<ArrayType>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ZeroOperation<T>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ZeroOperation<T>) -> Self {
        Self::Zero(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ZeroOperation<T>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Zero(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ZeroLikeOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ZeroLikeOperation) -> Self {
        Self::ZeroLike(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ZeroLikeOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::ZeroLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<OneOperation<T>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: OneOperation<T>) -> Self {
        Self::One(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a OneOperation<T>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::One(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<OneLikeOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: OneLikeOperation) -> Self {
        Self::OneLike(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a OneLikeOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::OneLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ConstantOperation<T, V>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ConstantOperation<T, V>) -> Self {
        Self::Constant(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ConstantOperation<T, V>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Constant(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<FillOperation<T, f64>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: FillOperation<T, f64>) -> Self {
        Self::Fill(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a FillOperation<T, f64>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Fill(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<NegOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: NegOperation) -> Self {
        Self::Neg(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a NegOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Neg(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<AddOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: AddOperation) -> Self {
        Self::Add(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a AddOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Add(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<SubOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: SubOperation) -> Self {
        Self::Sub(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a SubOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Sub(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ScaleOperation<T, F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ScaleOperation<T, F>) -> Self {
        Self::Scale(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ScaleOperation<T, F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Scale(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<MulOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: MulOperation) -> Self {
        Self::Mul(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a MulOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Mul(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<TransferToMemoryOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: TransferToMemoryOperation) -> Self {
        Self::TransferToMemory(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a TransferToMemoryOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::TransferToMemory(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<TransposeOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: TransposeOperation) -> Self {
        Self::Transpose(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a TransposeOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Transpose(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LeftDotOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LeftDotOperation<F>) -> Self {
        Self::LeftDot(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LeftDotOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::LeftDot(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<RightDotOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: RightDotOperation<F>) -> Self {
        Self::RightDot(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a RightDotOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::RightDot(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ReshapeOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ReshapeOperation) -> Self {
        Self::Reshape(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ReshapeOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Reshape(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ReshardOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ReshardOperation) -> Self {
        Self::Reshard(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ReshardOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Reshard(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ShardingConstraintOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ShardingConstraintOperation) -> Self {
        Self::ShardingConstraint(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ShardingConstraintOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::ShardingConstraint(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<BroadcastOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: BroadcastOperation) -> Self {
        Self::Broadcast(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a BroadcastOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Broadcast(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<SliceOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: SliceOperation) -> Self {
        Self::Slice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a SliceOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Slice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<UpdateSliceOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: UpdateSliceOperation) -> Self {
        Self::UpdateSlice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a UpdateSliceOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::UpdateSlice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LinearDynamicSliceOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearDynamicSliceOperation<F>) -> Self {
        Self::DynamicSlice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LinearDynamicSliceOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::DynamicSlice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LinearDynamicUpdateSliceOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearDynamicUpdateSliceOperation<F>) -> Self {
        Self::DynamicUpdateSlice(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LinearDynamicUpdateSliceOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::DynamicUpdateSlice(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LinearGatherOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearGatherOperation<F>) -> Self {
        Self::Gather(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LinearGatherOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Gather(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LinearScatterAddOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearScatterAddOperation<F>) -> Self {
        Self::ScatterAdd(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LinearScatterAddOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::ScatterAdd(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<PadOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: PadOperation) -> Self {
        Self::Pad(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a PadOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Pad(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ConcatenateOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ConcatenateOperation) -> Self {
        Self::Concatenate(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ConcatenateOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Concatenate(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<ReduceOperation>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: ReduceOperation) -> Self {
        Self::Reduce(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a ReduceOperation
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Reduce(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LinearSelectOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearSelectOperation<F>) -> Self {
        Self::Select(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LinearSelectOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Select(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<MaterializeCapturedFactorOperation<F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: MaterializeCapturedFactorOperation<F>) -> Self {
        Self::Residual(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a MaterializeCapturedFactorOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Residual(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    From<LinearConditionOperation<T, V, C, Extension, F, O>> for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearConditionOperation<T, V, C, Extension, F, O>) -> Self {
        Self::Condition(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>>
    for &'a LinearConditionOperation<T, V, C, Extension, F, O>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Condition(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    From<LinearOperandConditionOperation<T, V, C, Extension, F, O>> for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearOperandConditionOperation<T, V, C, Extension, F, O>) -> Self {
        Self::OperandCondition(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>>
    for &'a LinearOperandConditionOperation<T, V, C, Extension, F, O>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::OperandCondition(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    From<WhileOperation<T, V, LinearArrayOperation<T, V, C, Extension, F, O>>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: WhileOperation<T, V, LinearArrayOperation<T, V, C, Extension, F, O>>) -> Self {
        Self::While(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>>
    for &'a WhileOperation<T, V, LinearArrayOperation<T, V, C, Extension, F, O>>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::While(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<LinearScanOperation<T, V, C, Extension, F, O>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: LinearScanOperation<T, V, C, Extension, F, O>) -> Self {
        Self::Scan(operation)
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a LinearScanOperation<T, V, C, Extension, F, O>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::Scan(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O> From<CustomVjpCallOperation<T, C, O, F>>
    for LinearArrayOperation<T, V, C, Extension, F, O>
{
    fn from(operation: CustomVjpCallOperation<T, C, O, F>) -> Self {
        Self::CustomVjpCall(Box::new(operation))
    }
}

impl<'a, T: Type, V: Value<T>, C: Value<T>, Extension, F: Value<T>, O>
    TryFrom<&'a LinearArrayOperation<T, V, C, Extension, F, O>> for &'a CustomVjpCallOperation<T, C, O, F>
{
    type Error = ();

    fn try_from(value: &'a LinearArrayOperation<T, V, C, Extension, F, O>) -> Result<Self, ()> {
        match value {
            LinearArrayOperation::CustomVjpCall(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

// Transposition of the `LinearArrayOperation` sum type: each variant delegates to its backing operation's own
// `TransposableOperation` rule. The per-variant bounds cover every factor-independent variant at the outer factor
// `F`; the `Scan` and `Condition` recursions instead re-instantiate this impl at the scan-local factor
// `CapturedFactor<ArrayType, V>`, and the `While` recursion resolves against this impl's assumed
// `Self: TransposableOperation<ArrayType, V, Self>`. The remaining where-clause spells the leaf value capabilities
// the per-variant rules read off `V`, plus the recompute-primal, custom-VJP, and extension obligations at the
// scan-local fixed point.
impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>
    for LinearArrayOperation<ArrayType, V, C, Extension, F, O>
where
    Extension: Operation<ArrayType>,
    O: Operation<ArrayType>,
    ZeroOperation<ArrayType>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ZeroLikeOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    OneOperation<ArrayType>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    OneLikeOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ConstantOperation<ArrayType, V>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    FillOperation<ArrayType, f64>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    NegOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    AddOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    SubOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ScaleOperation<ArrayType, F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    MulOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    TransferToMemoryOperation:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    TransposeOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LeftDotOperation<F>: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    RightDotOperation<F>: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ReshapeOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ReshardOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ShardingConstraintOperation:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    BroadcastOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    SliceOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    UpdateSliceOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LinearDynamicSliceOperation<F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LinearDynamicUpdateSliceOperation<F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LinearGatherOperation<F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LinearScatterAddOperation<F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    PadOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ConcatenateOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    ReduceOperation: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LinearSelectOperation<F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    MaterializeCapturedFactorOperation<F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    O: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    LinearOperandConditionOperation<ArrayType, V, C, Extension, F, O>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    CustomVjpCallOperation<ArrayType, C, O, F>:
        TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    Extension: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>,
    V: Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + BooleanLike,
    O: Clone
        + TransposableOperation<
            ArrayType,
            V,
            LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, V>, O>,
        >,
    Extension: TransposableOperation<
            ArrayType,
            V,
            LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, V>, O>,
        >,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            V,
            LinearArrayOperation<ArrayType, V, C, Extension, F, O>,
        >,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            V,
            LinearArrayOperation<ArrayType, V, C, Extension, F, O>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>>,
        ProgramError,
    > {
        match self {
            Self::Zero(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::ZeroLike(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::One(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::OneLike(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Constant(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Fill(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Neg(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Add(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Sub(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Scale(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Mul(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::TransferToMemory(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Transpose(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::LeftDot(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::RightDot(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Reshape(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Reshard(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::ShardingConstraint(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Broadcast(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Slice(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::UpdateSlice(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::DynamicSlice(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::DynamicUpdateSlice(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Gather(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::ScatterAdd(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Pad(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Concatenate(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Reduce(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Select(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Residual(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Recompute(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Condition(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::OperandCondition(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::While(operation) => <WhileOperation<ArrayType, V, Self> as TransposableOperation<
                ArrayType,
                V,
                LinearArrayOperation<ArrayType, V, C, Extension, F, O>,
            >>::transpose(&**operation, context, input_types, output_cotangents),
            Self::Scan(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<ArrayType, C, O, F> as TransposableOperation<
                    ArrayType,
                    V,
                    LinearArrayOperation<ArrayType, V, C, Extension, F, O>,
                >>::transpose(&**operation, context, input_types, output_cotangents)
            }
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

impl<T: Type> Operation<T> for Infallible {
    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        match *self {}
    }
}

impl<T: Type, V: Value<T>> InterpretableOperation<T, V> for Infallible {
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        _inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match *self {}
    }
}

impl<T, V, O> TransposableOperation<T, V, O> for Infallible
where
    T: Type,
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

impl<T, V, Extension> MaybeRematerializationName for ArrayOperation<T, V, Extension>
where
    T: Type,
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

impl<T, V, Extension> MaybeDot for ArrayOperation<T, V, Extension>
where
    T: Type,
    V: Value<T>,
    Extension: MaybeDot,
{
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        match self {
            Self::Dot(operation) => Some(operation.dimensions()),
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

impl<V: Value<ArrayType>, C: Value<ArrayType>, Extension, F: Value<ArrayType>, O>
    SupportsLinearCondition<ArrayType, V, F> for LinearArrayOperation<ArrayType, V, C, Extension, F, O>
{
    #[inline]
    fn linear_condition_operation(
        predicate: F,
        true_branch: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        false_branch: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Self {
        LinearArrayOperation::Condition(LinearConditionOperation::new(
            predicate,
            Box::new(true_branch),
            Box::new(false_branch),
        ))
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
        LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, R>, O>,
        Vec<V>,
        Vec<V>,
    >,
    dispositions: &[Option<NestedResidualDisposition>],
    forwarded_input_types: &[ArrayType],
) -> Result<
    Program<
        ArrayType,
        V,
        LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, R>, O>,
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
        + From<MulOperation>
        + From<DotOperation>
        + From<SelectOperation>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>
        + From<ConcatenateOperation>,
{
    let mut builder = ProgramBuilder::<
        ArrayType,
        V,
        LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, R>, O>,
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
        instruction.operation().try_map_factors_preserving_extensions(&mut |factor: &CapturedFactor<
            ArrayType,
            R,
        >| {
            if let CapturedFactor::Reference { index, .. } = factor {
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
            CapturedFactor::Reference { index, r#type } => {
                let position = match resolve_disposition(*index)? {
                    NestedResidualDisposition::Operand(position) => position,
                    NestedResidualDisposition::Factor(position) => position,
                };
                Ok(CapturedFactor::Reference { index: position, r#type: r#type.clone() })
            }
            CapturedFactor::Constant(value) => Ok(CapturedFactor::Constant(value.clone())),
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

impl<V, C, Extension, R, O> SupportsLinearWhile<ArrayType, V, CapturedFactor<ArrayType, R>, O>
    for LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, R>, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    R: Value<ArrayType>,
    O: Clone
        + Operation<ArrayType>
        + From<MulOperation>
        + From<DotOperation>
        + From<SelectOperation>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>
        + From<ConcatenateOperation>,
{
    #[inline]
    fn recompute_operation(operation: O) -> Self {
        LinearArrayOperation::Recompute(operation)
    }

    #[inline]
    fn residual_operation(factor: CapturedFactor<ArrayType, R>) -> Self {
        LinearArrayOperation::Residual(MaterializeCapturedFactorOperation::new(factor))
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
            Self::Scale(operation) if matches!(operation.factor(), CapturedFactor::Reference { .. }) => {
                let CapturedFactor::Reference { index, .. } = operation.factor() else { unreachable!() };
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::from(MulOperation)),
                    inputs,
                })
            }
            Self::LeftDot(operation) if matches!(operation.factor(), CapturedFactor::Reference { .. }) => {
                let CapturedFactor::Reference { index, .. } = operation.factor() else { unreachable!() };
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::from(
                        DotOperation::new(operation.dimensions().clone())
                            .with_output_sharding(operation.output_sharding().cloned()),
                    )),
                    inputs,
                })
            }
            Self::RightDot(operation) if matches!(operation.factor(), CapturedFactor::Reference { .. }) => {
                let CapturedFactor::Reference { index, .. } = operation.factor() else { unreachable!() };
                inputs.push(resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::from(
                        DotOperation::new(operation.dimensions().clone())
                            .with_output_sharding(operation.output_sharding().cloned()),
                    )),
                    inputs,
                })
            }
            // `DynamicSlice` / `DynamicUpdateSlice` over loop-varying residual start indices become the recomputed
            // operand-form primal operations with the residual atoms spliced in as index operands. Mixed
            // constant/reference index lists are rejected because defactorization stages exactly one instruction,
            // while constant indices would need their own materializing instructions.
            Self::DynamicSlice(operation)
                if operation.start_indices().iter().any(|index| matches!(index, CapturedFactor::Reference { .. })) =>
            {
                for start_index in operation.start_indices() {
                    let CapturedFactor::Reference { index, .. } = start_index else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "jvp of a while loop whose body captures a mix of loop-varying and constant \
                                      dynamic_slice start indices is not supported"
                                .to_string(),
                        });
                    };
                    inputs.push(resolve_residual_atom(*index)?);
                }
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::from(DynamicSliceOperation::new(
                        operation.sizes().to_vec(),
                    ))),
                    inputs,
                })
            }
            Self::DynamicUpdateSlice(operation)
                if operation.start_indices().iter().any(|index| matches!(index, CapturedFactor::Reference { .. })) =>
            {
                for start_index in operation.start_indices() {
                    let CapturedFactor::Reference { index, .. } = start_index else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "jvp of a while loop whose body captures a mix of loop-varying and constant \
                                      dynamic_update_slice start indices is not supported"
                                .to_string(),
                        });
                    };
                    inputs.push(resolve_residual_atom(*index)?);
                }
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::from(DynamicUpdateSliceOperation)),
                    inputs,
                })
            }
            // A nested loop's residual injection materializes a value the fused body already recomputes, so the
            // instruction collapses to forwarding the residual atom.
            Self::Residual(operation) if matches!(operation.factor(), CapturedFactor::Reference { .. }) => {
                let CapturedFactor::Reference { index, .. } = operation.factor() else { unreachable!() };
                Ok(DefactorizedOperation::Forward { atom: resolve_residual_atom(*index)? })
            }
            // `Select` over a loop-varying residual condition becomes the recomputed operand-form primal select
            // with the residual atom spliced in as the condition operand.
            Self::Select(operation) if matches!(operation.condition(), CapturedFactor::Reference { .. }) => {
                let CapturedFactor::Reference { index, .. } = operation.condition() else { unreachable!() };
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(O::from(SelectOperation)),
                    inputs,
                })
            }
            // A loop-varying condition predicate becomes operand 0 of an operand-form condition
            // (`OperandCondition`). The branch programs may carry their own references into the same while-body
            // residual table (the condition JVP rule remapped them onto the enclosing linearization environment), so
            // the union of the residual indices referenced by both branches is forwarded as additional trailing
            // operands — both branches receive the full union because their signatures must agree — and each branch
            // is recursively defactorized against the new trailing branch inputs.
            Self::Condition(operation) if matches!(operation.predicate(), CapturedFactor::Reference { .. }) => {
                let CapturedFactor::Reference { index, .. } = operation.predicate() else { unreachable!() };
                let (true_branch, false_branch) = (operation.true_branch(), operation.false_branch());
                let predicate_atom = resolve_residual_atom(*index)?;
                let mut forwarded_residuals = BTreeMap::new();
                for branch in [true_branch, false_branch] {
                    for instruction in branch.instructions() {
                        instruction.operation().try_map_factors_preserving_extensions(
                            &mut |factor: &CapturedFactor<ArrayType, R>| {
                                if let CapturedFactor::Reference { index, r#type } = factor {
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
                    operation: LinearArrayOperation::OperandCondition(LinearOperandConditionOperation::new(
                        Box::new(true_branch),
                        Box::new(false_branch),
                    )),
                    inputs: condition_inputs,
                })
            }
            // A linear scan whose residual stacks reference loop-varying residuals moves those stacks into operand
            // position: each referenced stack becomes one extra scanned input, the body gains one trailing lane
            // input per moved stack (the stack type minus its leading length axis), and the body's scan-local
            // references to moved stacks are rewritten into operand form against those inputs. Constant stacks stay
            // factor payloads, with the surviving body references re-indexed against the compacted constant-only
            // stack list.
            Self::Scan(operation)
                if operation
                    .residual_stacks()
                    .iter()
                    .any(|stack| matches!(stack, CapturedFactor::Reference { .. })) =>
            {
                let residual_stacks = operation.residual_stacks();
                let mut dispositions = Vec::with_capacity(residual_stacks.len());
                let mut lane_types = Vec::new();
                let mut moved_stack_atoms = Vec::new();
                let mut surviving_stacks = Vec::new();
                for stack in residual_stacks {
                    match stack {
                        CapturedFactor::Reference { index, r#type } => {
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
                let body = defactorize_nested_linear_program(
                    operation.body(),
                    dispositions.as_slice(),
                    lane_types.as_slice(),
                )?;
                inputs.extend(moved_stack_atoms);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Scan(LinearScanOperation::new(
                        Box::new(body),
                        surviving_stacks,
                        operation.carry_count(),
                        operation.length(),
                        operation.reverse(),
                        operation.unroll(),
                    )),
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
                operation.try_map_factors_preserving_extensions(&mut |factor: &CapturedFactor<ArrayType, R>| {
                    if matches!(factor, CapturedFactor::Reference { .. }) {
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

impl<V, C, Extension, R, O> SupportsLinearScan<ArrayType, V, CapturedFactor<ArrayType, R>>
    for LinearArrayOperation<ArrayType, V, C, Extension, CapturedFactor<ArrayType, R>, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: Clone + Operation<ArrayType>,
    R: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    fn linear_scan_operation(
        body: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        residual_stacks: Vec<CapturedFactor<ArrayType, R>>,
        carry_count: usize,
        length: usize,
        reverse: bool,
        unroll: usize,
    ) -> Result<Self, ProgramError> {
        // Rebind the body's factor payloads into the scan-local residual-reference namespace pinned at
        // `CapturedFactor<ArrayType, V>`: references carry over index-for-index against `residual_stacks`, while
        // closed constants are rejected because their payloads live in the enclosing context's value family (the
        // scan JVP rule broadcasts every captured constant into a lane-uniform residual stack before staging, so
        // the rejection is unreachable from the rule).
        let body = body.map_operations(|operation| {
            operation.try_map_factors_preserving_extensions(&mut |factor| match factor {
                CapturedFactor::Reference { index, r#type } => {
                    Ok(CapturedFactor::Reference { index: *index, r#type: r#type.clone() })
                }
                CapturedFactor::Constant(_) => Err(ProgramError::UnsupportedOperation {
                    message: "scan body pushforwards must reference residual stacks instead of carrying closed \
                                  constant factors"
                        .to_string(),
                }),
            })
        })?;
        Ok(LinearArrayOperation::Scan(LinearScanOperation::new(
            Box::new(body),
            residual_stacks,
            carry_count,
            length,
            reverse,
            unroll,
        )))
    }
}

/// Renders a captured factor list as a bracketed, comma-separated sequence of `Display` renderings, for use in the
/// bracketed-attribute rendering of captured-index linear operations.
pub(crate) fn render_factor_list<F: Display>(factors: &[F]) -> String {
    format!("[{}]", factors.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}

fn symbolic_zero_one_error<T: Type>(r#type: &T) -> TypeError {
    TypeError { message: format!("zero tangent space has no one value for {type}", type = r#type) }
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
    context: &<V as Value<T>>::InterpretationContext,
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
    Ok(operation
        .interpret(context, materialized_inputs.as_slice())?
        .into_iter()
        .map(Tangent::Value)
        .collect())
}

fn tangent_value_type_matches<T: Type, V: Value<T>>(value: &V, output_type: &T) -> bool {
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

fn interpret_tangent_value_add<T: Type, V: Value<T> + Add<Output = V> + Zero<T>>(
    context: &<V as Value<T>>::InterpretationContext,
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
        _ => interpret_materialized_tangent_value_operation(context, &AddOperation, inputs),
    }
}

fn interpret_tangent_value_mul<T: Type, V: Value<T> + Mul<Output = V> + Zero<T>>(
    context: &<V as Value<T>>::InterpretationContext,
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
    interpret_materialized_tangent_value_operation(context, &MulOperation, inputs)
}

fn interpret_tangent_value_sub<T: Type, V: Value<T> + Neg<Output = V> + Sub<Output = V> + Zero<T>>(
    context: &<V as Value<T>>::InterpretationContext,
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
        _ => interpret_materialized_tangent_value_operation(context, &SubOperation, inputs),
    }
}

fn interpret_tangent_value_neg<T: Type, V: Value<T> + Neg<Output = V>>(
    context: &<V as Value<T>>::InterpretationContext,
    inputs: &[Tangent<T, V>],
) -> Result<Vec<Tangent<T, V>>, ProgramError>
where
    NegOperation: InterpretableOperation<T, V>,
{
    let output_types = infer_tangent_value_output_types(&NegOperation, inputs)?;
    check_count!("output", output_types, 1, ProgramError);
    match inputs {
        [Tangent::Zero(_)] => Ok(symbolic_zero_tangent_value_outputs(output_types)),
        [Tangent::Value(value)] => Ok(NegOperation
            .interpret(context, std::slice::from_ref(value))?
            .into_iter()
            .map(Tangent::Value)
            .collect()),
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
    T: Type,
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
    context: &<V as Value<T>>::InterpretationContext,
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
                .interpret(context, std::slice::from_ref(input))?
                .into_iter()
                .map(Tangent::Value)
                .collect())
        }
        _ => unreachable!("scale output type inference validates the input count"),
    }
}

/// Transposes a captured-condition select (the `Select` variant of [`LinearArrayOperation`] and the scalar
/// [`LinearSelectOperation`](crate::tracing_v2::operations::select::LinearSelectOperation)).
///
/// The forward linear map is `(t, f) ↦ select(condition, t, f)`. Its transpose routes the output cotangent into the
/// branch that the condition selected: the `on_true` cotangent is `select(condition, cotangent, 0)` and the
/// `on_false` cotangent is `select(condition, 0, cotangent)`. The zero operand is staged as a typed `Zero` operation
/// via [`stage_cotangent`](crate::tracing_v2::operations::control_flow::stage_cotangent), and `make_operation`
/// rebuilds the captured-condition select for staging into the transpose builder.
pub(crate) fn transpose_captured_condition_select<'transpose, T, V, O, MakeOperationFn>(
    make_operation: MakeOperationFn,
    context: &mut AbstractTracingContext<'transpose, T, V, O>,
    input_types: &[&T],
    output_cotangents: &[Cotangent<'transpose, T, V, O>],
) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError>
where
    T: Type,
    V: Value<T>,
    O: Operation<T> + From<ZeroOperation<T>>,
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
pub(crate) fn transpose_linear_condition<'transpose, V, C, Extension, F, O>(
    predicate: &F,
    true_branch: &Program<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>, Vec<V>, Vec<V>>,
    false_branch: &Program<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>, Vec<V>, Vec<V>>,
    context: &mut AbstractTracingContext<
        'transpose,
        ArrayType,
        V,
        LinearArrayOperation<ArrayType, V, C, Extension, F, O>,
    >,
    output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>],
) -> Result<
    Vec<Cotangent<'transpose, ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>>,
    ProgramError,
>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    LinearArrayOperation<ArrayType, V, C, Extension, F, O>: TransposableOperation<ArrayType, V, LinearArrayOperation<ArrayType, V, C, Extension, F, O>>
        + From<ZeroOperation<ArrayType>>
        + From<AddOperation>,
{
    // A condition with no outputs (or only zero output cotangents) is a zero linear map, so every input
    // cotangent is zero. Note that `all` is trivially true for an empty cotangent slice.
    if output_cotangents.iter().all(Cotangent::is_zero) {
        return Ok(vec![Cotangent::Zero; true_branch.input_types().len()]);
    }
    let transposed_condition = LinearArrayOperation::Condition(LinearConditionOperation::new(
        predicate.clone(),
        Box::new(true_branch.transpose()?),
        Box::new(false_branch.transpose()?),
    ));
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
pub(crate) fn transpose_captured_index_dynamic_slice<'transpose, T, V, O, MakeOperationFn>(
    make_dynamic_update_slice: MakeOperationFn,
    context: &mut AbstractTracingContext<'transpose, T, V, O>,
    input_types: &[&T],
    output_cotangents: &[Cotangent<'transpose, T, V, O>],
) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError>
where
    T: Type,
    V: Value<T>,
    O: Operation<T> + From<ZeroOperation<T>>,
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
pub(crate) fn transpose_captured_index_dynamic_update_slice<
    'transpose,
    V,
    O,
    MakeUpdateOperationFn,
    MakeSliceOperationFn,
>(
    make_dynamic_update_slice: MakeUpdateOperationFn,
    make_dynamic_slice: MakeSliceOperationFn,
    context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
    input_types: &[&ArrayType],
    output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>>,
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

/// Builds the scatter-add operation that is the transpose dual of a captured-index gather. The forward gather
/// `t ↦ gather(t, indices, ...)` has adjoint `cotangent ↦ scatter_add(zeros, indices, cotangent, ...)`, so the scatter
/// dimension numbers mirror the gather's axis-for-axis (offset↔update-window, collapsed↔inserted-window,
/// `start_index_map`↔`scatter_dimensions_to_operand_dimensions`, the batching pairs carried over), the combiner is
/// [`ScatterReductionKind::Add`], and the mode/flags carry through unchanged. The scatter writes into a zero operand of
/// the gather operand's type, so no `output_sharding` is requested (that zero operand already carries it).
pub(crate) fn gather_to_scatter_operation(operation: &GatherOperation) -> ScatterOperation {
    let dimensions = operation.dimensions();
    let scatter_dimensions = ScatterDimensionNumbers::new(
        dimensions.offset_dimensions().to_vec(),
        dimensions.collapsed_slice_dimensions().to_vec(),
        dimensions.start_index_map().to_vec(),
    )
    .with_batching_dimensions(
        dimensions.operand_batching_dimensions().to_vec(),
        dimensions.start_indices_batching_dimensions().to_vec(),
    );
    ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add)
        .with_mode(operation.mode())
        .with_indices_are_sorted(operation.indices_are_sorted())
        .with_unique_indices(operation.unique_indices())
}

/// Builds the gather operation that recovers the update cotangent in the transpose of a captured-index scatter-add.
/// The forward scatter-add `(t, u) ↦ scatter_add(t, indices, u, ...)` has, for its update operand, adjoint
/// `cotangent ↦ gather(cotangent, indices, ...)`: the gather dimension numbers mirror the scatter's axis-for-axis and
/// the slice sizes recover the per-axis update window extent (size 1 on the inserted-window and operand-batching axes,
/// the update window extent elsewhere). The update shape must be static so the slice sizes are known.
pub(crate) fn scatter_add_to_gather_operation(
    operation: &ScatterOperation,
    operand_type: &ArrayType,
    updates_type: &ArrayType,
) -> Result<GatherOperation, ProgramError> {
    let dimensions = operation.dimensions();
    let operand_rank = operand_type.rank();
    let update_window_dimensions = dimensions.update_window_dimensions();
    let inserted_window_dimensions = dimensions.inserted_window_dimensions();
    let operand_batching_dimensions = dimensions.operand_batching_dimensions();
    let mut slice_sizes = Vec::with_capacity(operand_rank);
    let mut window_position = 0;
    for operand_axis in 0..operand_rank {
        if inserted_window_dimensions.contains(&operand_axis) || operand_batching_dimensions.contains(&operand_axis) {
            slice_sizes.push(1);
        } else {
            let update_axis = update_window_dimensions[window_position];
            let extent = updates_type.dimension(update_axis as isize).value().ok_or_else(|| {
                ProgramError::from(TypeError {
                    message: format!(
                        "{SCATTER_OPERATION_NAME} transpose requires a static update shape but update axis \
                         {update_axis} has a dynamic size",
                    ),
                })
            })?;
            slice_sizes.push(extent);
            window_position += 1;
        }
    }
    let gather_dimensions = GatherDimensionNumbers::new(
        update_window_dimensions.to_vec(),
        inserted_window_dimensions.to_vec(),
        dimensions.scatter_dimensions_to_operand_dimensions().to_vec(),
    )
    .with_batching_dimensions(
        operand_batching_dimensions.to_vec(),
        dimensions.scatter_indices_batching_dimensions().to_vec(),
    );
    Ok(GatherOperation::new(gather_dimensions, slice_sizes)
        .with_mode(operation.mode())
        .with_indices_are_sorted(operation.indices_are_sorted())
        .with_unique_indices(operation.unique_indices())
        .with_output_sharding(updates_type.sharding().cloned()))
}

/// Transposes a captured-index scatter-add (the `ScatterAdd` variant of [`LinearArrayOperation`]).
///
/// The forward linear map is `(t, u) ↦ scatter_add(t, indices, u, ...)`. Because scatter-add accumulates into its
/// operand (`output = operand + scattered(updates)`, so `∂output/∂operand = I`), the operand cotangent is the output
/// cotangent unchanged; the update cotangent gathers the output cotangent at the scattered windows via the mirrored
/// gather rebuilt by `make_gather`. Symbolic-zero cotangents propagate unchanged.
pub(crate) fn transpose_captured_index_scatter_add<'transpose, V, O, MakeGatherOperationFn>(
    make_gather: MakeGatherOperationFn,
    context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
    input_types: &[&ArrayType],
    output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType>,
    MakeGatherOperationFn: Fn() -> O,
{
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_cotangents, 1, ProgramError);
    match &output_cotangents[0] {
        Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
        Cotangent::Staged(cotangent) => {
            let update_cotangents = context.stage_operation(make_gather(), std::slice::from_ref(cotangent))?;
            check_count!("output", update_cotangents, 1, ProgramError);
            Ok(vec![
                Cotangent::Staged(cotangent.clone()),
                Cotangent::Staged(update_cotangents.into_iter().next().unwrap()),
            ])
        }
    }
}

fn interpret_tangent_value_unary_value_or_zero<T, V, MetadataOperation, ConcreteOperation>(
    context: &<V as Value<T>>::InterpretationContext,
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
        [Tangent::Value(input)] => Ok(concrete_operation
            .interpret(context, std::slice::from_ref(input))?
            .into_iter()
            .map(Tangent::Value)
            .collect()),
        _ => unreachable!("unary output type inference validates the input count"),
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
    operation: &LinearArrayOperation<ArrayType, V, C, Extension, F, O>,
    map_factor: &mut MapFactorFn,
) -> Result<LinearArrayOperation<ArrayType, V, C, Strategy::MappedExtension, MappedFactor, O>, ProgramError>
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
            LinearArrayOperation::ZeroLike(operation) => Ok(LinearArrayOperation::ZeroLike(operation.clone())),
            LinearArrayOperation::OneLike(operation) => Ok(LinearArrayOperation::OneLike(operation.clone())),
            LinearArrayOperation::Add(operation) => Ok(LinearArrayOperation::Add(operation.clone())),
            LinearArrayOperation::Sub(operation) => Ok(LinearArrayOperation::Sub(operation.clone())),
            LinearArrayOperation::Neg(operation) => Ok(LinearArrayOperation::Neg(operation.clone())),
            LinearArrayOperation::Mul(operation) => Ok(LinearArrayOperation::Mul(operation.clone())),
            LinearArrayOperation::TransferToMemory(operation) => {
                Ok(LinearArrayOperation::TransferToMemory(operation.clone()))
            }
            LinearArrayOperation::Transpose(operation) => Ok(LinearArrayOperation::Transpose(operation.clone())),
            LinearArrayOperation::Scale(operation) => {
                Ok(LinearArrayOperation::Scale(ScaleOperation::new(map_factor(operation.factor())?)))
            }
            LinearArrayOperation::LeftDot(operation) => Ok(LinearArrayOperation::LeftDot(
                LeftDotOperation::new(map_factor(operation.factor())?, operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            )),
            LinearArrayOperation::RightDot(operation) => Ok(LinearArrayOperation::RightDot(
                RightDotOperation::new(map_factor(operation.factor())?, operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            )),
            LinearArrayOperation::Reshape(operation) => Ok(LinearArrayOperation::Reshape(operation.clone())),
            LinearArrayOperation::Reshard(operation) => Ok(LinearArrayOperation::Reshard(operation.clone())),
            LinearArrayOperation::ShardingConstraint(operation) => {
                Ok(LinearArrayOperation::ShardingConstraint(operation.clone()))
            }
            LinearArrayOperation::Broadcast(operation) => Ok(LinearArrayOperation::Broadcast(operation.clone())),
            LinearArrayOperation::Slice(operation) => Ok(LinearArrayOperation::Slice(operation.clone())),
            LinearArrayOperation::UpdateSlice(operation) => Ok(LinearArrayOperation::UpdateSlice(operation.clone())),
            LinearArrayOperation::Pad(operation) => Ok(LinearArrayOperation::Pad(operation.clone())),
            LinearArrayOperation::Concatenate(operation) => Ok(LinearArrayOperation::Concatenate(operation.clone())),
            LinearArrayOperation::DynamicSlice(operation) => {
                Ok(LinearArrayOperation::DynamicSlice(LinearDynamicSliceOperation::new(
                    operation.start_indices().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                    operation.sizes().to_vec(),
                )))
            }
            LinearArrayOperation::DynamicUpdateSlice(operation) => {
                Ok(LinearArrayOperation::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(
                    operation.start_indices().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                )))
            }
            LinearArrayOperation::Gather(operation) => Ok(LinearArrayOperation::Gather(LinearGatherOperation::new(
                operation.operation().clone(),
                map_factor(operation.indices())?,
            ))),
            LinearArrayOperation::ScatterAdd(operation) => Ok(LinearArrayOperation::ScatterAdd(
                LinearScatterAddOperation::new(operation.operation().clone(), map_factor(operation.indices())?),
            )),
            LinearArrayOperation::Reduce(operation) => Ok(LinearArrayOperation::Reduce(operation.clone())),
            LinearArrayOperation::Select(operation) => {
                Ok(LinearArrayOperation::Select(LinearSelectOperation::new(map_factor(operation.condition())?)))
            }
            LinearArrayOperation::Residual(operation) => Ok(LinearArrayOperation::Residual(
                MaterializeCapturedFactorOperation::new(map_factor(operation.factor())?),
            )),
            LinearArrayOperation::Recompute(operation) => Ok(LinearArrayOperation::Recompute(operation.clone())),
            LinearArrayOperation::Condition(operation) => {
                Ok(LinearArrayOperation::Condition(LinearConditionOperation::new(
                    map_factor(operation.predicate())?,
                    Box::new(operation.true_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                    Box::new(operation.false_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                )))
            }
            // Operand-form condition branches carry only closed constant factors after defactorization, but the
            // traversal stays total over them like the factor-form variant's.
            LinearArrayOperation::OperandCondition(operation) => {
                Ok(LinearArrayOperation::OperandCondition(LinearOperandConditionOperation::new(
                    Box::new(operation.true_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                    Box::new(operation.false_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _, _, Strategy>(operation, map_factor)
                    })?),
                )))
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
            LinearArrayOperation::Scan(operation) => {
                // The factor-cloning function is passed as a `fn` pointer (not a closure) so the recursive
                // monomorphization below reaches a fixed point: nested scans reuse the exact same
                // `(map_factor, Strategy)` instantiation instead of minting a fresh closure type per level.
                let mut clone_scan_local_factor = clone_factor::<CapturedFactor<ArrayType, V>>
                    as fn(&CapturedFactor<ArrayType, V>) -> Result<CapturedFactor<ArrayType, V>, ProgramError>;
                Ok(LinearArrayOperation::Scan(LinearScanOperation::new(
                    Box::new(operation.body().map_operations(|operation| {
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
                    operation.residual_stacks().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                    operation.carry_count(),
                    operation.length(),
                    operation.reverse(),
                    operation.unroll(),
                )))
            }
            LinearArrayOperation::Extension(extension) => {
                Ok(LinearArrayOperation::Extension(Strategy::map_extension(extension, map_factor)?))
            }
        }
    }
}

impl<V, C, Extension, F, O> FactorParameterizedOperation<ArrayType, F>
    for LinearArrayOperation<ArrayType, V, C, Extension, F, O>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension: FactorParameterizedOperation<ArrayType, F>,
    F: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    type WithFactor<MappedFactor: Value<ArrayType>> =
        LinearArrayOperation<ArrayType, V, C, Extension::WithFactor<MappedFactor>, MappedFactor, O>;

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

impl<V, C, Extension, F, O> LinearArrayOperation<ArrayType, V, C, Extension, F, O>
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
    ) -> Result<LinearArrayOperation<ArrayType, V, C, Extension, MappedFactor, O>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        map_linear_array_operation_factors::<_, _, _, _, _, _, _, PreserveExtensionFactors>(self, map_factor)
    }
}

impl<V: Value<DataType>> InterpretableOperation<DataType, Tangent<DataType, V>>
    for LinearScalarOperation<V, Tangent<DataType, V>>
where
    V: SupportsLinearArithmeticOperations + BooleanLike + Zero<DataType> + One<DataType> + OneLike,
    V: Select<Condition = bool>,
{
    fn interpret(
        &self,
        context: &<Tangent<DataType, V> as Value<DataType>>::InterpretationContext,
        inputs: &[Tangent<DataType, V>],
    ) -> Result<Vec<Tangent<DataType, V>>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback interpretation over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::Zero(zero) => Ok(vec![Tangent::Zero(*zero.r#type())]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(one.r#type())?)]),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::ZeroLike(_) => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike(_) => interpret_tangent_value_one_like(inputs),
            Self::Add(_) => interpret_tangent_value_add(context, inputs),
            Self::Sub(_) => interpret_tangent_value_sub(context, inputs),
            Self::Neg(_) => interpret_tangent_value_neg(context, inputs),
            Self::Scale(operation) => interpret_tangent_value_scale(context, self, operation.factor(), inputs),
            // The captured-condition select over tangents decodes the in-band Boolean condition and selects between
            // the two branch tangents, materializing symbolic zeros to concrete values first (and collapsing to a
            // symbolic zero when both branches are zero).
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                let boolean = operation.condition().select_condition()?;
                if inputs[0].is_zero() && inputs[1].is_zero() {
                    return Ok(vec![Tangent::Zero(output_types.into_iter().next().unwrap())]);
                }
                let materialize = |tangent: &Tangent<DataType, V>| match tangent {
                    Tangent::Zero(r#type) => V::zero(r#type),
                    Tangent::Value(value) => Ok(value.clone()),
                };
                Ok(vec![Tangent::Value(V::select(&boolean, &materialize(&inputs[0])?, &materialize(&inputs[1])?)?)])
            }
        }
    }
}

impl<V: Value<DataType>, F> InterpretableOperation<DataType, V> for LinearScalarOperation<V, F>
where
    V: SupportsLinearArithmeticOperations
        + SupportsConstantOperations<DataType>
        + Scale<F, Output = V>
        + Select<Condition = bool>,
    F: CustomVjpResidual<DataType, V> + SelectCondition<Condition = bool>,
    ScalarOperation<V>: InterpretableOperation<DataType, V>,
    ScaleOperation<DataType, F>: InterpretableOperation<DataType, V>,
    ConstantOperation<DataType, F>: InterpretableOperation<DataType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<DataType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret(context, inputs),
            Self::Zero(zero) => zero.interpret(context, inputs),
            Self::One(one) => one.interpret(context, inputs),
            Self::Constant(constant) => constant.interpret(context, inputs),
            Self::ZeroLike(_) => ZeroLikeOperation.interpret(context, inputs),
            Self::OneLike(_) => OneLikeOperation.interpret(context, inputs),
            Self::Add(_) => {
                <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, context, inputs)
            }
            Self::Sub(_) => {
                <SubOperation as InterpretableOperation<DataType, V>>::interpret(&SubOperation, context, inputs)
            }
            Self::Neg(_) => {
                <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, context, inputs)
            }
            Self::Scale(operation) => ScaleOperation::new(operation.factor().clone()).interpret(context, inputs),
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![V::select(&operation.condition().select_condition()?, &inputs[0], &inputs[1])?])
            }
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
        + OneLike
        + Select<Condition = Tracer<S>>
        + SelectCondition<Condition = Tracer<S>>,
    Vec<Tracer<S>>: Parameterized<Tracer<S>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, context: &S, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret_over_tracers(inputs),
            Self::Zero(zero) => context.stage_operation(ZeroOperation::new(zero.r#type().clone()), &[] as &[Tracer<S>]),
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
            Self::ZeroLike(_) => ZeroLikeOperation.interpret(context, inputs),
            Self::OneLike(_) => OneLikeOperation.interpret(context, inputs),
            Self::Add(_) => {
                <AddOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&AddOperation, context, inputs)
            }
            Self::Sub(_) => {
                <SubOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&SubOperation, context, inputs)
            }
            Self::Neg(_) => {
                <NegOperation as InterpretableOperation<DataType, Tracer<S>>>::interpret(&NegOperation, context, inputs)
            }
            Self::Scale(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![operation.factor().clone() * inputs[0].clone()])
            }
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![Tracer::select(&operation.condition().select_condition()?, &inputs[0], &inputs[1])?])
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
impl<V: Value<ArrayType>, Extension> InterpretableOperation<ArrayType, V> for ArrayOperation<ArrayType, V, Extension>
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
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomJvp(operation) => operation.interpret(context, inputs),
            Self::CustomVjp(operation) => operation.interpret(context, inputs),
            Self::Zero(operation) => operation.interpret(context, inputs),
            Self::One(operation) => operation.interpret(context, inputs),
            Self::Constant(operation) => operation.interpret(context, inputs),
            Self::Fill(operation) => operation.interpret(context, inputs),
            Self::ZeroLike(operation) => operation.interpret(context, inputs),
            Self::OneLike(operation) => operation.interpret(context, inputs),
            Self::Add(operation) => operation.interpret(context, inputs),
            Self::Sub(operation) => operation.interpret(context, inputs),
            Self::Mul(operation) => operation.interpret(context, inputs),
            Self::Div(operation) => operation.interpret(context, inputs),
            Self::Neg(operation) => operation.interpret(context, inputs),
            Self::Sin(operation) => operation.interpret(context, inputs),
            Self::Cos(operation) => operation.interpret(context, inputs),
            Self::StopGradient(operation) => operation.interpret(context, inputs),
            Self::RematerializationName(operation) => operation.interpret(context, inputs),
            Self::TransferToMemory(operation) => operation.interpret(context, inputs),
            Self::Dot(operation) => operation.interpret(context, inputs),
            Self::Transpose(operation) => operation.interpret(context, inputs),
            Self::Scale(operation) => operation.interpret(context, inputs),
            Self::Reshape(operation) => operation.interpret(context, inputs),
            Self::Reshard(operation) => operation.interpret(context, inputs),
            Self::ShardingConstraint(operation) => operation.interpret(context, inputs),
            Self::Broadcast(operation) => operation.interpret(context, inputs),
            Self::Slice(operation) => operation.interpret(context, inputs),
            Self::UpdateSlice(operation) => operation.interpret(context, inputs),
            Self::DynamicSlice(operation) => operation.interpret(context, inputs),
            Self::DynamicUpdateSlice(operation) => operation.interpret(context, inputs),
            Self::Pad(operation) => operation.interpret(context, inputs),
            Self::Concatenate(operation) => operation.interpret(context, inputs),
            Self::Gather(operation) => operation.interpret(context, inputs),
            Self::Scatter(operation) => operation.interpret(context, inputs),
            Self::Reduce(operation) => operation.interpret(context, inputs),
            Self::Compare(operation) => operation.interpret(context, inputs),
            Self::Not(operation) => operation.interpret(context, inputs),
            Self::And(operation) => operation.interpret(context, inputs),
            Self::Or(operation) => operation.interpret(context, inputs),
            Self::Xor(operation) => operation.interpret(context, inputs),
            Self::Collective(operation) => operation.interpret(context, inputs),
            Self::Select(operation) => operation.interpret(context, inputs),
            Self::Condition(condition) => condition.interpret(context, inputs),
            Self::While(while_operation) => while_operation.interpret(context, inputs),
            Self::Scan(scan) => scan.interpret(context, inputs),
            Self::Extension(extension) => extension.interpret(context, inputs),
        }
    }
}

impl<'domain, D, C, V, Extension> InterpretableOperation<ArrayType, Tracer<TracingContext<'domain, D, C>>>
    for ArrayOperation<ArrayType, V, Extension>
where
    D: Domain<Type = ArrayType, Value = V, Operation = ArrayOperation<ArrayType, V, Extension>>,
    V: Value<ArrayType>,
    Extension: Clone + InterpretableOperation<ArrayType, Tracer<TracingContext<'domain, D, C>>>,
{
    fn interpret(
        &self,
        context: &TracingContext<'domain, D, C>,
        inputs: &[Tracer<TracingContext<'domain, D, C>>],
    ) -> Result<Vec<Tracer<TracingContext<'domain, D, C>>>, ProgramError> {
        match self {
            Self::Extension(extension) => extension.interpret(context, inputs),
            _ => context.stage_operation(self.clone(), inputs),
        }
    }
}

impl<V: Value<ArrayType>, Extension, O> InterpretableOperation<ArrayType, Tangent<ArrayType, V>>
    for LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>
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
    fn interpret(
        &self,
        context: &<Tangent<ArrayType, V> as Value<ArrayType>>::InterpretationContext,
        inputs: &[Tangent<ArrayType, V>],
    ) -> Result<Vec<Tangent<ArrayType, V>>, ProgramError> {
        match self {
            Self::CustomVjpCall(_) => Err(crate::types::TypeError {
                message: "custom_vjp pullback interpretation over tangent-wrapped values is not supported".to_string(),
            }
            .into()),
            Self::TransferToMemory(operation) => {
                let destination = operation.destination();
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![match &inputs[0] {
                    Tangent::Zero(r#type) => Tangent::Zero(r#type.clone().with_memory(destination)),
                    Tangent::Value(value) => Tangent::Value(value.transfer_to_memory(destination)),
                }])
            }
            Self::Zero(zero) => Ok(vec![Tangent::Zero(zero.r#type().clone())]),
            Self::One(one) => Ok(vec![Tangent::Value(V::one(one.r#type())?)]),
            Self::Constant(constant) => interpret_tangent_value_constant(constant, inputs),
            Self::Fill(fill) if *fill.value() == 0.0 => Ok(vec![Tangent::Zero(fill.r#type().clone())]),
            Self::Fill(fill) => Ok(vec![Tangent::Value(V::fill(fill.r#type(), *fill.value())?)]),
            Self::ZeroLike(_) => interpret_tangent_value_zero_like(&ZeroLikeOperation, inputs),
            Self::OneLike(_) => interpret_tangent_value_one_like(inputs),
            Self::Add(_) => interpret_tangent_value_add(context, inputs),
            Self::Sub(_) => interpret_tangent_value_sub(context, inputs),
            Self::Mul(_) => interpret_tangent_value_mul(context, inputs),
            Self::Neg(_) => interpret_tangent_value_neg(context, inputs),
            Self::Transpose(operation) => {
                let op = TransposeOperation::new(operation.permutation().to_vec());
                interpret_tangent_value_unary_value_or_zero(context, &op, &op, inputs)
            }
            Self::Scale(operation) => interpret_tangent_value_scale(context, self, operation.factor(), inputs),
            Self::Broadcast(operation) => {
                let op = BroadcastOperation::new(operation.output_type().clone(), operation.output_axes().to_vec());
                interpret_tangent_value_unary_value_or_zero(context, &op, &op, inputs)
            }
            Self::LeftDot(operation) => {
                let factor = operation.factor();
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
                        Ok(super::dot::LeftDotOperation::new(factor.clone(), operation.dimensions().clone())
                            .interpret(context, std::slice::from_ref(input))?
                            .into_iter()
                            .map(Tangent::Value)
                            .collect())
                    }
                    _ => unreachable!("left_dot output type inference validates the input count"),
                }
            }
            Self::RightDot(operation) => {
                let factor = operation.factor();
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
                        Ok(super::dot::RightDotOperation::new(factor.clone(), operation.dimensions().clone())
                            .interpret(context, std::slice::from_ref(input))?
                            .into_iter()
                            .map(Tangent::Value)
                            .collect())
                    }
                    _ => unreachable!("right_dot output type inference validates the input count"),
                }
            }
            Self::Reshape(operation) => interpret_tangent_value_unary_value_or_zero(
                context,
                &ReshapeOperation::new(operation.output_shape().clone()),
                &ReshapeOperation::new(operation.output_shape().clone()),
                inputs,
            ),
            Self::Reshard(operation) => interpret_tangent_value_unary_value_or_zero(
                context,
                &ReshardOperation::new(operation.sharding().clone()),
                &ReshardOperation::new(operation.sharding().clone()),
                inputs,
            ),
            Self::ShardingConstraint(operation) => interpret_tangent_value_unary_value_or_zero(
                context,
                &ShardingConstraintOperation::new(operation.sharding().clone()),
                &ShardingConstraintOperation::new(operation.sharding().clone()),
                inputs,
            ),
            Self::Reduce(operation) => {
                let op = ReduceOperation::new(operation.axes().to_vec(), operation.kind())
                    .with_output_sharding(operation.output_sharding().cloned());
                interpret_tangent_value_unary_value_or_zero(context, &op, &op, inputs)
            }
            Self::Slice(operation) => {
                let op = SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                    .with_strides(operation.strides().to_vec())?;
                interpret_tangent_value_unary_value_or_zero(context, &op, &op, inputs)
            }
            Self::UpdateSlice(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                interpret_materialized_tangent_value_operation(
                    context,
                    &UpdateSliceOperation::new(operation.start_indices().to_vec()),
                    inputs,
                )
            }
            Self::Pad(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                interpret_materialized_tangent_value_operation(
                    context,
                    &PadOperation::new(
                        operation.edge_padding_low().to_vec(),
                        operation.edge_padding_high().to_vec(),
                        operation.interior_padding().to_vec(),
                    )?,
                    inputs,
                )
            }
            Self::Concatenate(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                interpret_materialized_tangent_value_operation(
                    context,
                    &ConcatenateOperation::new(operation.axis()),
                    inputs,
                )
            }
            Self::DynamicSlice(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                match inputs {
                    [input] if input.is_zero() => Ok(symbolic_zero_tangent_value_outputs(output_types)),
                    [Tangent::Value(input)] => {
                        let index_values =
                            concrete_tangent_factor_indices(DYNAMIC_SLICE_OPERATION_NAME, operation.start_indices())?;
                        Ok(vec![Tangent::Value(input.dynamic_slice(&index_values, operation.sizes())?)])
                    }
                    _ => unreachable!("dynamic_slice output type inference validates the input count"),
                }
            }
            Self::DynamicUpdateSlice(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                check_count!("input", inputs, 2, ProgramError);
                let index_values =
                    concrete_tangent_factor_indices(DYNAMIC_UPDATE_SLICE_OPERATION_NAME, operation.start_indices())?;
                let materialize = |tangent: &Tangent<ArrayType, V>| match tangent {
                    Tangent::Zero(r#type) => V::zero(r#type),
                    Tangent::Value(value) => Ok(value.clone()),
                };
                Ok(vec![Tangent::Value(
                    materialize(&inputs[0])?.dynamic_update_slice(&materialize(&inputs[1])?, &index_values)?,
                )])
            }
            Self::Gather(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                match inputs {
                    [input] if input.is_zero() => Ok(symbolic_zero_tangent_value_outputs(output_types)),
                    [Tangent::Value(operand)] => {
                        let index_value = concrete_tangent_factor_indices(
                            GATHER_OPERATION_NAME,
                            std::slice::from_ref(operation.indices()),
                        )?
                        .into_iter()
                        .next()
                        .expect("gather captures exactly one index factor");
                        Ok(vec![Tangent::Value(operand.gather(&index_value, operation.operation())?)])
                    }
                    _ => unreachable!("gather output type inference validates the input count"),
                }
            }
            Self::ScatterAdd(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                if inputs.iter().all(Tangent::is_zero) {
                    return Ok(symbolic_zero_tangent_value_outputs(output_types));
                }
                check_count!("input", inputs, 2, ProgramError);
                let index_value =
                    concrete_tangent_factor_indices(SCATTER_OPERATION_NAME, std::slice::from_ref(operation.indices()))?
                        .into_iter()
                        .next()
                        .expect("scatter captures exactly one index factor");
                let materialize = |tangent: &Tangent<ArrayType, V>| match tangent {
                    Tangent::Zero(r#type) => V::zero(r#type),
                    Tangent::Value(value) => Ok(value.clone()),
                };
                Ok(vec![Tangent::Value(materialize(&inputs[0])?.scatter(
                    &index_value,
                    &materialize(&inputs[1])?,
                    operation.operation(),
                )?)])
            }
            Self::Select(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                check_count!("output", output_types, 1, ProgramError);
                let Tangent::Value(condition) = operation.condition() else {
                    return Err(TypeError {
                        message: format!("captured select condition for {} must be a concrete value", output_types[0],),
                    }
                    .into());
                };
                Ok(vec![Tangent::select(condition, &inputs[0], &inputs[1])?])
            }
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![operation.factor().clone()])
            }
            Self::Recompute(_) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "recomputed primal operation {} does not support tangent-wrapped interpretation",
                    self.name(),
                ),
            }),
            Self::Condition(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let Tangent::Value(predicate) = operation.predicate() else {
                    return Err(TypeError {
                        message: "captured condition predicate must be a concrete value".to_string(),
                    }
                    .into());
                };
                let branch = if predicate.boolean()? { operation.true_branch() } else { operation.false_branch() };
                let outputs = branch.interpret_in_context(context, inputs.to_vec())?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::OperandCondition(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let Tangent::Value(predicate) = &inputs[0] else {
                    return Err(TypeError {
                        message: "operand-form condition predicate must be a concrete value".to_string(),
                    }
                    .into());
                };
                let branch = if predicate.boolean()? { operation.true_branch() } else { operation.false_branch() };
                let outputs = branch.interpret_in_context(context, inputs[1..].to_vec())?;
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
                    let condition_outputs = operation.condition().interpret_in_context(context, state.clone())?;
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
                    state = operation.body().interpret_in_context(context, state)?;
                    check_count!("output", state, operation.state_types().len(), ProgramError);
                    completed_iterations += 1;
                }
            }
            Self::Scan(operation) => {
                let output_types = infer_tangent_value_output_types(self, inputs)?;
                let body = operation.body();
                let carry_count = operation.carry_count();
                let y_slice_types = body.output_types().split_off(carry_count);
                let residual_stacks = operation.residual_stacks();
                let outputs = interpret_scan_lanes(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
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
                        lane_body.interpret_in_context(context, lane_inputs)
                    },
                )?;
                check_count!("output", outputs, output_types.len(), ProgramError);
                Ok(outputs)
            }
            Self::Extension(extension) => extension.interpret(context, inputs),
        }
    }
}

impl<V: Value<ArrayType>, Extension, F: Value<ArrayType>, O> InterpretableOperation<ArrayType, V>
    for LinearArrayOperation<ArrayType, V, V, Extension, F, O>
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
    ArrayOperation<ArrayType, V, Extension>: InterpretableOperation<ArrayType, V>,
    F: CustomVjpResidual<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone + InterpretableOperation<ArrayType, V>,
{
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret(context, inputs),
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone()])
            }
            Self::Zero(zero) => zero.interpret(context, inputs),
            Self::One(one) => one.interpret(context, inputs),
            Self::Constant(constant) => constant.interpret(context, inputs),
            Self::Fill(fill) => fill.interpret(context, inputs),
            Self::ZeroLike(_) => ZeroLikeOperation.interpret(context, inputs),
            Self::OneLike(_) => OneLikeOperation.interpret(context, inputs),
            Self::Add(_) => AddOperation.interpret(context, inputs),
            Self::Sub(_) => SubOperation.interpret(context, inputs),
            Self::Mul(_) => MulOperation.interpret(context, inputs),
            Self::Neg(_) => NegOperation.interpret(context, inputs),
            Self::Transpose(operation) => {
                TransposeOperation::new(operation.permutation().to_vec()).interpret(context, inputs)
            }
            Self::Scale(operation) => ScaleOperation::new(operation.factor().clone()).interpret(context, inputs),
            Self::LeftDot(operation) => {
                super::dot::LeftDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned())
                    .interpret(context, inputs)
            }
            Self::RightDot(operation) => {
                super::dot::RightDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned())
                    .interpret(context, inputs)
            }
            Self::Reshape(operation) => {
                ReshapeOperation::new(operation.output_shape().clone()).interpret(context, inputs)
            }
            Self::Reshard(operation) => ReshardOperation::new(operation.sharding().clone()).interpret(context, inputs),
            Self::ShardingConstraint(operation) => {
                ShardingConstraintOperation::new(operation.sharding().clone()).interpret(context, inputs)
            }
            Self::Broadcast(operation) => {
                BroadcastOperation::new(operation.output_type().clone(), operation.output_axes().to_vec())
                    .interpret(context, inputs)
            }
            Self::Reduce(operation) => ReduceOperation::new(operation.axes().to_vec(), operation.kind())
                .with_output_sharding(operation.output_sharding().cloned())
                .interpret(context, inputs),
            Self::Slice(operation) => {
                SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                    .with_strides(operation.strides().to_vec())?
                    .interpret(context, inputs)
            }
            Self::UpdateSlice(operation) => {
                UpdateSliceOperation::new(operation.start_indices().to_vec()).interpret(context, inputs)
            }
            Self::Pad(operation) => PadOperation::new(
                operation.edge_padding_low().to_vec(),
                operation.edge_padding_high().to_vec(),
                operation.interior_padding().to_vec(),
            )?
            .interpret(context, inputs),
            Self::Concatenate(operation) => ConcatenateOperation::new(operation.axis()).interpret(context, inputs),
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let index_values = operation
                    .start_indices()
                    .iter()
                    .map(|index| index.residual_value())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![inputs[0].dynamic_slice(&index_values, operation.sizes())?])
            }
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let index_values = operation
                    .start_indices()
                    .iter()
                    .map(|index| index.residual_value())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![inputs[0].dynamic_update_slice(&inputs[1], &index_values)?])
            }
            Self::Gather(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].gather(&operation.indices().residual_value()?, operation.operation())?])
            }
            Self::ScatterAdd(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].scatter(
                    &operation.indices().residual_value()?,
                    &inputs[1],
                    operation.operation(),
                )?])
            }
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![V::select(&operation.condition().residual_value()?, &inputs[0], &inputs[1])?])
            }
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![operation.factor().residual_value()?])
            }
            Self::Recompute(operation) => operation.interpret(context, inputs),
            Self::Condition(operation) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if operation.predicate().residual_value()?.boolean()? {
                    operation.true_branch()
                } else {
                    operation.false_branch()
                };
                branch.interpret_in_context(context, inputs.to_vec())
            }
            Self::OperandCondition(operation) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if inputs[0].boolean()? { operation.true_branch() } else { operation.false_branch() };
                branch.interpret_in_context(context, inputs[1..].to_vec())
            }
            Self::While(operation) => operation.interpret(context, inputs),
            Self::Scan(operation) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let stack_values = operation
                    .residual_stacks()
                    .iter()
                    .map(|stack| stack.residual_value())
                    .collect::<Result<Vec<_>, _>>()?;
                let body = operation.body();
                let carry_count = operation.carry_count();
                let y_slice_types = body.output_types().split_off(carry_count);
                interpret_scan_lanes(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
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
                        lane_body.interpret_in_context(context, lane_inputs)
                    },
                )
            }
            Self::Extension(extension) => extension.interpret(context, inputs),
        }
    }
}

impl<C, S, Extension, O> InterpretableOperation<ArrayType, Tracer<S>>
    for LinearArrayOperation<ArrayType, Tracer<S>, C, Extension, Tracer<S>, O>
where
    C: Value<ArrayType>,
    S: StagingContext<Type = ArrayType, Constant = C, Operation = O>,
    S::Operation: From<DotOperation>,
    Extension: Clone + InterpretableOperation<ArrayType, Tracer<S>>,
    Tracer<S>: Add<Output = Tracer<S>>
        + Sub<Output = Tracer<S>>
        + Neg<Output = Tracer<S>>
        + Mul<Output = Tracer<S>>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::dot::DotOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + Broadcast
        + crate::tracing_v2::operations::reduce::Reduce
        + BooleanLike,
    Vec<Tracer<S>>:
        Parameterized<Tracer<S>, To<Tracer<S>> = Vec<Tracer<S>>, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone
        + Operation<ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<TransferToMemoryOperation>
        + From<SelectOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<PadOperation>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>
        + From<GatherOperation>
        + From<ScatterOperation>
        + From<ConcatenateOperation>
        + From<ReshardOperation>
        + From<ShardingConstraintOperation>,
{
    fn interpret(&self, context: &S, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        match self {
            Self::CustomVjpCall(call) => call.interpret_over_tracers(inputs),
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].transfer_to_memory(operation.destination())])
            }
            Self::Zero(zero) => context.stage_operation(ZeroOperation::new(zero.r#type().clone()), &[] as &[Tracer<S>]),
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
            Self::ZeroLike(_) => ZeroLikeOperation.interpret(context, inputs),
            Self::OneLike(_) => OneLikeOperation.interpret(context, inputs),
            Self::Add(_) => <AddOperation as InterpretableOperation<ArrayType, Tracer<S>>>::interpret(
                &AddOperation,
                context,
                inputs,
            ),
            Self::Sub(_) => <SubOperation as InterpretableOperation<ArrayType, Tracer<S>>>::interpret(
                &SubOperation,
                context,
                inputs,
            ),
            Self::Mul(_) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].clone() * inputs[1].clone()])
            }
            Self::Neg(_) => <NegOperation as InterpretableOperation<ArrayType, Tracer<S>>>::interpret(
                &NegOperation,
                context,
                inputs,
            ),
            Self::Transpose(operation) => {
                TransposeOperation::new(operation.permutation().to_vec()).interpret(context, inputs)
            }
            Self::Scale(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![operation.factor().clone() * inputs[0].clone()])
            }
            Self::LeftDot(operation) => {
                use crate::tracing_v2::operations::dot::Dot;
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![operation.factor().dot(&inputs[0], operation.dimensions())])
            }
            Self::RightDot(operation) => {
                use crate::tracing_v2::operations::dot::Dot;
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].dot(operation.factor(), operation.dimensions())])
            }
            Self::Reshape(operation) => {
                ReshapeOperation::new(operation.output_shape().clone()).interpret(context, inputs)
            }
            Self::Reshard(operation) => ReshardOperation::new(operation.sharding().clone()).interpret(context, inputs),
            Self::ShardingConstraint(operation) => {
                ShardingConstraintOperation::new(operation.sharding().clone()).interpret(context, inputs)
            }
            Self::Broadcast(operation) => {
                BroadcastOperation::new(operation.output_type().clone(), operation.output_axes().to_vec())
                    .interpret(context, inputs)
            }
            Self::Reduce(operation) => ReduceOperation::new(operation.axes().to_vec(), operation.kind())
                .with_output_sharding(operation.output_sharding().cloned())
                .interpret(context, inputs),
            Self::Slice(operation) => {
                SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                    .with_strides(operation.strides().to_vec())?
                    .interpret(context, inputs)
            }
            Self::UpdateSlice(operation) => {
                UpdateSliceOperation::new(operation.start_indices().to_vec()).interpret(context, inputs)
            }
            Self::Pad(operation) => PadOperation::new(
                operation.edge_padding_low().to_vec(),
                operation.edge_padding_high().to_vec(),
                operation.interior_padding().to_vec(),
            )?
            .interpret(context, inputs),
            Self::Concatenate(operation) => ConcatenateOperation::new(operation.axis()).interpret(context, inputs),
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].dynamic_slice(operation.start_indices(), operation.sizes())?])
            }
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].dynamic_update_slice(&inputs[1], operation.start_indices())?])
            }
            Self::Gather(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].gather(operation.indices(), operation.operation())?])
            }
            Self::ScatterAdd(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].scatter(operation.indices(), &inputs[1], operation.operation())?])
            }
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![Tracer::select(operation.condition(), &inputs[0], &inputs[1])?])
            }
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![operation.factor().clone()])
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
            Self::Condition(operation) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch =
                    if operation.predicate().boolean()? { operation.true_branch() } else { operation.false_branch() };
                branch.interpret_in_context(context, inputs.to_vec())
            }
            Self::OperandCondition(operation) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                let branch = if inputs[0].boolean()? { operation.true_branch() } else { operation.false_branch() };
                branch.interpret_in_context(context, inputs[1..].to_vec())
            }
            Self::While(operation) => operation.interpret(context, inputs),
            Self::Scan(operation) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(input_types.as_slice())?;
                // Replaying a linear scan over tracers unrolls the statically counted loop: each lane's body
                // pushforward is bound against that lane's residual slices and inlined into the tracers' staging
                // context, mirroring how the linear condition inlines its captured branch. Stacked output
                // accumulators are staged as typed zero operations because tracer values cannot materialize
                // constants directly.
                let body = operation.body();
                let residual_stacks = operation.residual_stacks();
                let carry_count = operation.carry_count();
                let Some(exemplar) = inputs.first().or_else(|| residual_stacks.first()) else {
                    return Err(ProgramError::UnsupportedOperation {
                        message: "cannot replay a linear scan with no inputs and no residual stacks over tracer \
                                  values"
                            .to_string(),
                    });
                };
                let staging_context = exemplar.context();
                let y_slice_types = body.output_types().split_off(carry_count);
                interpret_scan_lanes(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| {
                        let mut outputs = staging_context
                            .stage_operation(O::from(ZeroOperation::new(stacked_type.clone())), &[] as &[Tracer<S>])?;
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
                        lane_body.interpret_in_context(context, lane_inputs)
                    },
                )
            }
            Self::Extension(extension) => extension.interpret(context, inputs),
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
            Self::ZeroLike(_) => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike(_) => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Neg(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale(operation) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::Scale(ScaleOperation::new(operation.factor().clone())),
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Select(operation) => transpose_captured_condition_select(
                || Self::Select(LinearSelectOperation::new(operation.condition().clone())),
                context,
                input_types,
                output_cotangents,
            ),
        }
    }
}

impl<V: Value<ArrayType>, Extension, O>
    TransposableOperation<
        ArrayType,
        Tangent<ArrayType, V>,
        LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>,
    > for LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>
where
    V: crate::tracing_v2::operations::dot::DotOps + Scale<f64, Output = V> + Reshard + ConstrainSharding,
    Extension: TransposableOperation<
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>,
        >,
    O: Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>,
        >,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            Tangent<ArrayType, V>,
            LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>,
        >],
    ) -> Result<
        Vec<
            Cotangent<
                'transpose,
                ArrayType,
                Tangent<ArrayType, V>,
                LinearArrayOperation<ArrayType, Tangent<ArrayType, V>, V, Extension, Tangent<ArrayType, V>, O>,
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
            Self::ZeroLike(_) => ZeroLikeOperation.transpose(context, input_types, output_cotangents),
            Self::OneLike(_) => OneLikeOperation.transpose(context, input_types, output_cotangents),
            Self::Add(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
            }
            Self::Sub(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
                }
            }
            Self::Mul(_) => {
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
            Self::Neg(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::TransferToMemory(_) => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::TransferToMemory(TransferToMemoryOperation::new(input_types[0].memory())),
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Transpose(operation) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                let inverse = inverse_permutation(operation.permutation());
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.transpose(inverse)?)]),
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Scale(operation) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let outputs = context.stage_operation(
                            Self::Scale(ScaleOperation::new(operation.factor().clone())),
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Fill(fill) => fill.transpose(context, input_types, output_cotangents),
            Self::LeftDot(operation) => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                let factor = operation.factor();
                let Tangent::Value(_) = factor else {
                    return Ok(vec![Cotangent::Zero]);
                };
                let factor_rank = factor.r#type().as_ref().rank();
                let adjoint = crate::tracing_v2::operations::dot::adjoint_dimensions_for_left_dot(
                    operation.dimensions(),
                    factor_rank,
                );
                // The adjoint's output *is* the input's cotangent, so its sharding is pinned to the cotangent dual
                // of the input's sharding instead of being re-derived.
                let adjoint_output_sharding = input_types[0].sharding().map(Sharding::cotangent);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let contribution = match &adjoint_output_sharding {
                            Some(output_sharding) => {
                                cotangent.left_dot_with_output_sharding(factor.clone(), &adjoint, output_sharding)
                            }
                            None => cotangent.left_dot(factor.clone(), &adjoint),
                        };
                        Ok(vec![Cotangent::Staged(contribution)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::RightDot(operation) => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                let factor = operation.factor();
                let dimensions = operation.dimensions();
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
                let adjoint_output_sharding = input_types[0].sharding().map(Sharding::cotangent);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        let contribution = match &adjoint_output_sharding {
                            Some(output_sharding) => {
                                cotangent.right_dot_with_output_sharding(factor.clone(), &adjoint, output_sharding)
                            }
                            None => cotangent.right_dot(factor.clone(), &adjoint),
                        };
                        Ok(vec![Cotangent::Staged(contribution)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Reshape(_) => {
                check_count!("input", input_types, 1, ProgramError);
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Staged(cotangent) => {
                        Ok(vec![Cotangent::Staged(cotangent.reshape(input_types[0].shape().clone())?)])
                    }
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                }
            }
            Self::Reshard(operation) => {
                ReshardOperation::new(operation.sharding().clone()).transpose(context, input_types, output_cotangents)
            }
            Self::ShardingConstraint(operation) => ShardingConstraintOperation::new(operation.sharding().clone())
                .transpose(context, input_types, output_cotangents),
            Self::Broadcast(_) => {
                check_count!("output", output_cotangents, 1, ProgramError);
                match &output_cotangents[0] {
                    Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    Cotangent::Staged(_) => Err(ProgramError::UnsupportedOperation {
                        message: "broadcast transpose is not supported (would need reduce-sum)".to_string(),
                    }),
                }
            }
            Self::Reduce(_) => {
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
            Self::Slice(operation) => {
                SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                    .with_strides(operation.strides().to_vec())?
                    .transpose(context, input_types, output_cotangents)
            }
            Self::UpdateSlice(operation) => UpdateSliceOperation::new(operation.start_indices().to_vec()).transpose(
                context,
                input_types,
                output_cotangents,
            ),
            Self::Pad(operation) => PadOperation::new(
                operation.edge_padding_low().to_vec(),
                operation.edge_padding_high().to_vec(),
                operation.interior_padding().to_vec(),
            )?
            .transpose(context, input_types, output_cotangents),
            Self::Concatenate(operation) => {
                ConcatenateOperation::new(operation.axis()).transpose(context, input_types, output_cotangents)
            }
            Self::DynamicSlice(operation) => transpose_captured_index_dynamic_slice(
                || Self::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(operation.start_indices().to_vec())),
                context,
                input_types,
                output_cotangents,
            ),
            Self::DynamicUpdateSlice(operation) => transpose_captured_index_dynamic_update_slice(
                || Self::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(operation.start_indices().to_vec())),
                |sizes| Self::DynamicSlice(LinearDynamicSliceOperation::new(operation.start_indices().to_vec(), sizes)),
                context,
                input_types,
                output_cotangents,
            ),
            Self::Gather(operation) => {
                let scatter_operation = gather_to_scatter_operation(operation.operation());
                let indices = operation.indices().clone();
                transpose_captured_index_dynamic_slice(
                    move || {
                        Self::ScatterAdd(LinearScatterAddOperation::new(scatter_operation.clone(), indices.clone()))
                    },
                    context,
                    input_types,
                    output_cotangents,
                )
            }
            Self::ScatterAdd(operation) => {
                check_count!("input", input_types, 2, ProgramError);
                let gather_operation =
                    scatter_add_to_gather_operation(operation.operation(), input_types[0], input_types[1])?;
                let indices = operation.indices().clone();
                transpose_captured_index_scatter_add(
                    move || Self::Gather(LinearGatherOperation::new(gather_operation.clone(), indices.clone())),
                    context,
                    input_types,
                    output_cotangents,
                )
            }
            Self::Select(operation) => {
                let condition = operation.condition().clone();
                transpose_captured_condition_select(
                    || Self::Select(LinearSelectOperation::new(condition.clone())),
                    context,
                    input_types,
                    output_cotangents,
                )
            }
            Self::Residual(_) => Err(ProgramError::UnsupportedOperation {
                message: "residual is not a linear map and does not support transposition".to_string(),
            }),
            Self::Recompute(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "recomputed primal operation {} is not a linear map and does not support transposition",
                    operation.name(),
                ),
            }),
            Self::Condition(operation) => transpose_linear_condition(
                operation.predicate(),
                operation.true_branch(),
                operation.false_branch(),
                context,
                output_cotangents,
            ),
            Self::OperandCondition(_) => Err(ProgramError::UnsupportedOperation {
                message: "operand-form condition inside a fused while body does not support transposition".to_string(),
            }),
            Self::While(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Scan(_) => Err(ProgramError::UnsupportedOperation {
                message: "scan transposition over tangent-wrapped values is not supported".to_string(),
            }),
            Self::Extension(extension) => extension.transpose(context, input_types, output_cotangents),
        }
    }
}

/// Transpose rule for a recomputed primal operation (the `Recompute` variant of [`LinearArrayOperation`], whose payload
/// is the primal [`ArrayOperation`]). A recomputed primal is replayed forward to reconstruct a residual rather than
/// participating in the linear map, so it is not a linear map and rejects transposition. The cotangent value type
/// (`CotangentValue`) is independent of the primal operation's own value type, and the staged linear operation type
/// (`LinearOperation`) is generic, so this single impl backs the `Recompute` delegation for every linear operation set.
impl<CotangentValue, PrimalValue, Extension, LinearOperation>
    TransposableOperation<ArrayType, CotangentValue, LinearOperation>
    for ArrayOperation<ArrayType, PrimalValue, Extension>
where
    CotangentValue: Value<ArrayType>,
    PrimalValue: Value<ArrayType>,
    Extension: Operation<ArrayType>,
    LinearOperation: Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, CotangentValue, LinearOperation>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, CotangentValue, LinearOperation>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, CotangentValue, LinearOperation>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "recomputed primal operation {} is not a linear map and does not support transposition",
                self.name(),
            ),
        })
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
    operation: &ArrayOperation<ArrayType, F, E>,
    context: &V::InterpretationContext,
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
        ArrayOperation::Add(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Sub(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Mul(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Div(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Neg(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Sin(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Cos(operation) => operation.batch(context, inputs)?,
        ArrayOperation::StopGradient(operation) => operation.batch(context, inputs)?,
        ArrayOperation::RematerializationName(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Select(operation) => operation.batch(context, inputs)?,
        ArrayOperation::ZeroLike(operation) => operation.batch(context, inputs)?,
        ArrayOperation::OneLike(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Scale(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Dot(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Transpose(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reshape(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reshard(operation) => operation.batch(context, inputs)?,
        ArrayOperation::ShardingConstraint(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Broadcast(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Slice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::UpdateSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::DynamicSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::DynamicUpdateSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Pad(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Concatenate(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Gather(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Scatter(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reduce(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Compare(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Not(operation) => operation.batch(context, inputs)?,
        ArrayOperation::And(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Or(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Xor(operation) => operation.batch(context, inputs)?,
        ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Collective(_)
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

/// Blanket active batching impl for the [`ArrayOperation`] sum type over a staged tracer context: each non-control
/// variant delegates to its backing operation's batching rule (shared with the eager impl through
/// [`batch_array_non_control_operation`]), while the lane-uniform memory transfer, the named-axis collective, and the
/// higher-order control-flow variants are handled by their specialized recursive rules.
impl<C, V, E> BatchableOperation<Tracer<C>, BatchingContext<C>> for ArrayOperation<ArrayType, V, E>
where
    C: StagingContext<Type = ArrayType, Constant = V, Operation = ArrayOperation<ArrayType, V, E>>,
    V: Value<ArrayType> + BooleanLike,
    C::Operation: From<CollectiveOperation> + From<FillOperation<ArrayType, f64>>,
    Tracer<C>: SupportsArithmeticOperations<V>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + Broadcast
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
        if let Some(outputs) = batch_array_non_control_operation(self, context.parent_context(), inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so it is staged unchanged on
            // the physical batched value in the parent context and the lane axis is preserved.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The staged collective rule owns named-axis resolution: it consumes the lane axis when this context's
            // axis name matches and forwards the collective to the parent context otherwise.
            Self::Collective(operation) => operation.batch(context, inputs),
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

/// Blanket value-level batching impl for the [`ArrayOperation`] sum type.
impl<V, E> BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<ArrayType, V, E>>>
    for ArrayOperation<ArrayType, V, E>
where
    V::InterpretationContext: Default,
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
    E: BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<ArrayType, V, E>>>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, ArrayOperation<ArrayType, V, E>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let interpretation_context = V::InterpretationContext::default();
        if let Some(outputs) = batch_array_non_control_operation(self, &interpretation_context, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            Self::Collective(operation) => operation.batch(context, inputs),
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

/// Blanket active batching impl for the [`ArrayOperation`] sum type.
///
/// The `Operation = Self` projection equality and the
/// [`ProgramBatchableOperation`](crate::tracing_v2::batching::ProgramBatchableOperation) / lane-alignment bounds exist
/// for the custom-derivative arms: their re-wrapping batch rules batch the captured programs and stage a new
/// custom-derivative call into the parent context, which is only expressible when the staged operation type is this
/// enum itself. Both extra bounds are leaf obligations (a structural type equality and a closed-enum capability
/// whose impl carries no batching-context obligations of its own), so instantiating this impl never recurses into
/// another batching-context obligation.
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
impl<V, E> ProgramBatchableOperation<V> for ArrayOperation<ArrayType, V, E>
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
        + Broadcast
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
impl<V, E, Extension> ProgramLinearizableOperation<E> for ArrayOperation<ArrayType, V, Extension>
where
    V: Value<ArrayType>,
    E: DifferentiationContext<Type = ArrayType, Constant = V>,
    E::Tangent: Transpose + Broadcast + super::reduce::Reduce + Slice + Reshard + ConstrainSharding,
    E::LinearOperation<E::Tangent, V>:
        FactorParameterizedOperation<ArrayType, V, WithFactor<V> = E::LinearOperation<E::Tangent, V>>,
    Extension: Clone + Operation<ArrayType> + DifferentiableOperation<LinearizationContextOf<E, Self>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: SupportsLinearArrayOperation<ArrayType, CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + crate::tracing_v2::ResidualizedOperation<LinearizationContextOf<E, Self>>
        + From<ZeroOperation<ArrayType>>
        + From<
            CustomVjpCallOperation<
                ArrayType,
                V,
                Self,
                CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            >,
        > + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearDynamicSliceOperation<CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearDynamicUpdateSliceOperation<CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearGatherOperation<CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearScatterAddOperation<CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + SupportsLinearCondition<
            ArrayType,
            E::Tangent,
            CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
        > + SupportsLinearWhile<
            ArrayType,
            E::Tangent,
            CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            Self,
        > + SupportsLinearScan<ArrayType, E::Tangent, CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: FactorParameterizedOperation<
            ArrayType,
            CapturedFactor<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            WithFactor<CapturedFactor<ArrayType, E::Value>> = LinearOperationOf<E>,
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
    LinearOperationOf<LinearizationContextOf<E, Self>>: SupportsLinearScalarOperation<DataType, CapturedFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>>
        + crate::tracing_v2::ResidualizedOperation<LinearizationContextOf<E, Self>>
        + From<ZeroOperation<DataType>>
        + From<LinearSelectOperation<CapturedFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<
            CustomVjpCallOperation<
                DataType,
                F,
                Self,
                CapturedFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>,
            >,
        >,
    LinearOperationOf<LinearizationContextOf<E, Self>>: FactorParameterizedOperation<
            DataType,
            CapturedFactor<DataType, Tracer<LinearizationContextOf<E, Self>>>,
            WithFactor<CapturedFactor<DataType, E::Value>> = LinearOperationOf<E>,
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
    operation: &LinearArrayOperation<ArrayType, F, C, E>,
    context: &V::InterpretationContext,
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
        LinearArrayOperation::Add(_) => AddOperation.batch(context, inputs)?,
        LinearArrayOperation::Sub(_) => SubOperation.batch(context, inputs)?,
        LinearArrayOperation::Mul(_) => MulOperation.batch(context, inputs)?,
        LinearArrayOperation::Neg(_) => NegOperation.batch(context, inputs)?,
        LinearArrayOperation::ZeroLike(_) => ZeroLikeOperation.batch(context, inputs)?,
        LinearArrayOperation::OneLike(_) => OneLikeOperation.batch(context, inputs)?,
        LinearArrayOperation::Scale(operation) => {
            ScaleOperation::new(operation.factor().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Transpose(operation) => {
            TransposeOperation::new(operation.permutation().to_vec()).batch(context, inputs)?
        }
        LinearArrayOperation::LeftDot(operation) => {
            LeftDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                .with_output_sharding(operation.output_sharding().cloned())
                .batch(context, inputs)?
        }
        LinearArrayOperation::RightDot(operation) => {
            RightDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                .with_output_sharding(operation.output_sharding().cloned())
                .batch(context, inputs)?
        }
        LinearArrayOperation::Reshape(operation) => {
            ReshapeOperation::new(operation.output_shape().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Reshard(operation) => {
            ReshardOperation::new(operation.sharding().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::ShardingConstraint(operation) => {
            ShardingConstraintOperation::new(operation.sharding().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Broadcast(operation) => {
            BroadcastOperation::new(operation.output_type().clone(), operation.output_axes().to_vec())
                .batch(context, inputs)?
        }
        LinearArrayOperation::Reduce(operation) => ReduceOperation::new(operation.axes().to_vec(), operation.kind())
            .with_output_sharding(operation.output_sharding().cloned())
            .batch(context, inputs)?,
        LinearArrayOperation::Slice(operation) => {
            SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                .with_strides(operation.strides().to_vec())?
                .batch(context, inputs)?
        }
        LinearArrayOperation::UpdateSlice(operation) => {
            UpdateSliceOperation::new(operation.start_indices().to_vec()).batch(context, inputs)?
        }
        LinearArrayOperation::Pad(operation) => PadOperation::new(
            operation.edge_padding_low().to_vec(),
            operation.edge_padding_high().to_vec(),
            operation.interior_padding().to_vec(),
        )?
        .batch(context, inputs)?,
        LinearArrayOperation::Concatenate(operation) => {
            ConcatenateOperation::new(operation.axis()).batch(context, inputs)?
        }
        LinearArrayOperation::TransferToMemory(_)
        | LinearArrayOperation::DynamicSlice(_)
        | LinearArrayOperation::DynamicUpdateSlice(_)
        | LinearArrayOperation::Gather(_)
        | LinearArrayOperation::ScatterAdd(_)
        | LinearArrayOperation::Select(_)
        | LinearArrayOperation::Residual(_)
        | LinearArrayOperation::Recompute(_)
        | LinearArrayOperation::Condition(_)
        | LinearArrayOperation::OperandCondition(_)
        | LinearArrayOperation::While(_)
        | LinearArrayOperation::Scan(_)
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
impl<V, E> BatchableOperation<V, EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>>
    for LinearArrayOperation<ArrayType, V, V, E>
where
    ArrayOperation<ArrayType, V, E>: BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<ArrayType, V, E>>>,
    V::InterpretationContext: Default,
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
    E: Clone + BatchableOperation<V, EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let interpretation_context = V::InterpretationContext::default();
        if let Some(outputs) = batch_linear_non_control_operation(self, &interpretation_context, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            // The captured condition is lane-uniform: prepending it as an unbatched operand lets the elementwise
            // select batching rule broadcast it to the batched physical shape before selecting per lane.
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                SelectOperation.batch(
                    &interpretation_context,
                    &[ArrayBatch::unbatched(operation.condition().clone()), inputs[0].clone(), inputs[1].clone()],
                )
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs
                    .extend(operation.start_indices().iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicSliceOperation::new(operation.sizes().to_vec())
                    .batch(&interpretation_context, lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs
                    .extend(operation.start_indices().iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicUpdateSliceOperation.batch(&interpretation_context, lifted_inputs.as_slice())
            }
            // The captured index operand is lane-uniform by construction: inserting it as the second (unbatched)
            // operand lets the primal gather batching rule lift the lane axis.
            Self::Gather(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                operation.operation().batch(
                    &interpretation_context,
                    &[inputs[0].clone(), ArrayBatch::unbatched(operation.indices().clone())],
                )
            }
            // The captured index operand is lane-uniform by construction: inserting it between the operand and update
            // tangents (unbatched) lets the primal scatter batching rule lift the lane axis.
            Self::ScatterAdd(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                operation.operation().batch(
                    &interpretation_context,
                    &[inputs[0].clone(), ArrayBatch::unbatched(operation.indices().clone()), inputs[1].clone()],
                )
            }
            // The captured factor is lane-uniform by construction: the same residual value applies to every lane.
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(operation.factor().clone())])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => {
                let primal_context = EagerContext::<ArrayType, V, ArrayOperation<ArrayType, V, E>>::new();
                operation.batch(&primal_context, inputs)
            }
            // The captured predicate is lane-uniform: prepending it as an unbatched input lets the condition
            // batching helper read the branch choice from input 0, exactly like an ordinary runtime predicate.
            Self::Condition(operation) => {
                let mut condition_inputs = Vec::with_capacity(inputs.len() + 1);
                condition_inputs.push(ArrayBatch::unbatched(operation.predicate().clone()));
                condition_inputs.extend(inputs.iter().cloned());
                batch_condition_with_interpreter(
                    operation.true_branch(),
                    operation.false_branch(),
                    condition_inputs.as_slice(),
                    |program, program_inputs| {
                        program.interpret_with(
                            program_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| {
                                instruction.operation().batch(context, instruction_inputs)
                            },
                        )
                    },
                )
            }
            // The operand-form condition already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate.
            Self::OperandCondition(operation) => batch_condition_with_interpreter(
                operation.true_branch(),
                operation.false_branch(),
                inputs,
                |program, program_inputs| {
                    program.interpret_with(
                        program_inputs,
                        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                        |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
                    )
                },
            ),
            Self::While(operation) => operation.batch(context, inputs),
            // Each lane's body pushforward is bound against that lane's residual slices and batched through the
            // shared scan loop; the residual stacks are concrete values in the direct linear form.
            Self::Scan(operation) => {
                let body = operation.body();
                let carry_count = operation.carry_count();
                let residual_stacks = operation.residual_stacks();
                let y_slice_types = body.output_types().split_off(carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
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
                            |instruction, instruction_inputs| {
                                instruction.operation().batch(context, instruction_inputs)
                            },
                        )
                    },
                )
            }
            Self::CustomVjpCall(call) => {
                let primal_context = EagerContext::<ArrayType, V, ArrayOperation<ArrayType, V, E>>::new();
                call.batch(&primal_context, inputs)
            }
            Self::Extension(extension) => extension.batch(context, inputs),
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`LinearArrayOperation`] sum type.
impl<C, E> BatchableOperation<Tracer<C>, BatchingContext<C>>
    for LinearArrayOperation<ArrayType, C::Constant, C::Constant, E>
where
    ArrayOperation<ArrayType, C::Constant, E>: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType> + BooleanLike + Slice + Reshape,
    C::Operation: From<ZeroOperation<ArrayType>>,
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
        if let Some(outputs) = batch_linear_non_control_operation(self, context.parent_context(), inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so the transfer is
            // staged unchanged on the physical batched value (in its own parent context) and the lane axis is
            // preserved. The parent operation type is generic here, so the value-level capability stages it.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The captured condition is a lane-uniform parent-context constant: lift it into the parent trace and
            // let the elementwise select batching rule broadcast it to the batched physical shape.
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let condition = context.parent_context().constant(operation.condition().clone());
                SelectOperation.batch(
                    context.parent_context(),
                    &[ArrayBatch::unbatched(condition), inputs[0].clone(), inputs[1].clone()],
                )
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    operation
                        .start_indices()
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicSliceOperation::new(operation.sizes().to_vec())
                    .batch(context.parent_context(), lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    operation
                        .start_indices()
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicUpdateSliceOperation.batch(context.parent_context(), lifted_inputs.as_slice())
            }
            // The captured factor is a lane-uniform parent-context constant: lift it into the parent trace.
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(context.parent_context().constant(operation.factor().clone()))])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => operation.batch(context, inputs),
            // The captured predicate is a lane-uniform parent-context constant, so the branch choice is concrete:
            // extract it from the factor and batch only the selected branch. Prepending a lifted predicate tracer
            // would defeat the lane-uniform extraction because tracers cannot be concretized.
            Self::Condition(operation) => {
                let branch =
                    if operation.predicate().boolean()? { operation.true_branch() } else { operation.false_branch() };
                context.interpret_program(branch, inputs.to_vec())
            }
            // The operand-form condition already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate (lane-uniform predicates extract
            // concretely, lane-varying ones run both branches and select per lane).
            Self::OperandCondition(operation) => batch_condition_with_interpreter::<C::Constant, Tracer<C>, _, _>(
                operation.true_branch(),
                operation.false_branch(),
                inputs,
                |program, program_inputs| context.interpret_program(program, program_inputs),
            ),
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
            Self::Scan(operation) => {
                let body = operation.body();
                let carry_count = operation.carry_count();
                let residual_stacks = operation.residual_stacks();
                let y_slice_types = body.output_types().split_off(carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| {
                        let mut outputs = context.parent_context().stage_operation(
                            C::Operation::from(ZeroOperation::new(stacked_type.clone())),
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

impl<V, E>
    BatchableOperation<
        Tangent<ArrayType, V>,
        EagerContext<ArrayType, Tangent<ArrayType, V>, LinearArrayOperation<ArrayType, V, V, E>>,
    > for LinearArrayOperation<ArrayType, V, V, E>
where
    ArrayOperation<ArrayType, V, E>: BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<ArrayType, V, E>>>,
    LinearArrayOperation<ArrayType, V, V, E>:
        BatchableOperation<V, EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>>,
    V::InterpretationContext: Default,
    V: Value<ArrayType>
        + SupportsLinearArithmeticOperations
        + SupportsConstantOperations<ArrayType>
        + SupportsLinearAlgebraOperations
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    E: Clone
        + BatchableOperation<V, EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>>
        + BatchableOperation<
            Tangent<ArrayType, V>,
            EagerContext<ArrayType, Tangent<ArrayType, V>, LinearArrayOperation<ArrayType, V, V, E>>,
        >,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        _context: &EagerContext<ArrayType, Tangent<ArrayType, V>, LinearArrayOperation<ArrayType, V, V, E>>,
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
            LinearArrayOperation::ZeroLike(_)
                | LinearArrayOperation::OneLike(_)
                | LinearArrayOperation::Residual(_)
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
            let context = EagerContext::<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>::new();
            let v_outputs = <LinearArrayOperation<ArrayType, V, V, E> as BatchableOperation<
                V,
                EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>,
            >>::batch(self, &context, materialized_zero_inputs.as_slice())?;
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
        let context = EagerContext::<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>::new();
        let v_outputs = <LinearArrayOperation<ArrayType, V, V, E> as BatchableOperation<
            V,
            EagerContext<ArrayType, V, LinearArrayOperation<ArrayType, V, V, E>>,
        >>::batch(self, &context, materialized.as_slice())?;
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

    use crate::contexts::EagerContext;
    use crate::domains::AbstractDomain;
    use crate::operations::InterpretableOperation as _;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::types::{Shape, Size};

    use super::*;

    type MixedScalar = Tangent<DataType, f64>;
    type MixedScalarOperation = LinearScalarOperation<f64, MixedScalar>;
    type MixedArray = Tangent<ArrayType, TestArray>;
    type MixedArrayOperation = LinearArrayOperation<ArrayType, MixedArray, TestArray>;

    fn f64_array_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
    }

    #[test]
    fn test_linear_condition_transpose_supports_runtime_predicates() {
        // Linear-condition transposition is total: the captured predicate factor is a residual of the primal
        // computation rather than a linear operand, so it is carried verbatim into one staged transposed condition
        // over the transposed branch programs. Runtime (factor) predicates used to be rejected with an
        // `UnsupportedOperation` error.
        type TestLinearOperation = LinearArrayOperation<ArrayType, TestArray, TestArray>;
        let scale_branch = |factor: f64| {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let output = builder
                .add_instruction(
                    TestLinearOperation::Scale(ScaleOperation::new(TestArray::scalar(factor))),
                    vec![input],
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = TestLinearOperation::Condition(LinearConditionOperation::new(
            TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
            Box::new(scale_branch(2.0)),
            Box::new(scale_branch(3.0)),
        ));

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

        assert_eq!(
            MixedScalarOperation::Add(AddOperation)
                .interpret(&crate::EagerContext::new(), &[value.clone(), zero.clone()]),
            Ok(vec![value.clone()])
        );
        assert_eq!(
            MixedScalarOperation::Add(AddOperation)
                .interpret(&crate::EagerContext::new(), &[zero.clone(), value.clone()]),
            Ok(vec![value.clone()])
        );
        assert_eq!(
            MixedScalarOperation::Sub(SubOperation)
                .interpret(&crate::EagerContext::new(), &[zero.clone(), value.clone()]),
            Ok(vec![MixedScalar::value(-3.0)])
        );
        assert_eq!(
            (MixedScalarOperation::Scale(ScaleOperation::new(MixedScalar::zero(DataType::F64))))
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&value)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            (MixedScalarOperation::Scale(ScaleOperation::new(MixedScalar::value(2.0))))
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&zero)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            (MixedScalarOperation::Scale(ScaleOperation::new(MixedScalar::value(2.0))))
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&value)),
            Ok(vec![MixedScalar::value(6.0)])
        );
        assert_eq!(
            MixedScalarOperation::ZeroLike(ZeroLikeOperation)
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&value)),
            Ok(vec![zero.clone()])
        );
        assert_eq!(
            MixedScalarOperation::One(OneOperation::new(DataType::F64)).interpret(&crate::EagerContext::new(), &[]),
            Ok(vec![MixedScalar::value(1.0)])
        );
        assert_eq!(
            MixedScalarOperation::OneLike(OneLikeOperation)
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&zero))
                .unwrap_err()
                .to_string(),
            "zero tangent space has no one value for f64"
        );
    }

    #[test]
    fn test_linear_array_tangent_value_interpretation_preserves_symbolic_zero_metadata() {
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input_zero = MixedArray::zero(input.r#type().into_owned());

        assert_eq!(
            MixedArrayOperation::Add(AddOperation)
                .interpret(&crate::EagerContext::new(), &[MixedArray::value(input.clone()), input_zero.clone()]),
            Ok(vec![MixedArray::value(input.clone())])
        );
        assert_eq!(
            MixedArrayOperation::Neg(NegOperation)
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&input_zero)),
            Ok(vec![input_zero.clone()])
        );

        let reshaped_type = f64_array_type(&[3, 2]);
        assert_eq!(
            (MixedArrayOperation::Reshape(ReshapeOperation::new(reshaped_type.shape().clone())))
                .interpret(&crate::EagerContext::new(), std::slice::from_ref(&input_zero)),
            Ok(vec![MixedArray::zero(reshaped_type.clone())])
        );

        use crate::tracing_v2::operations::dot::DotDimensionNumbers;

        let left_factor_type = f64_array_type(&[4, 2]);
        assert_eq!(
            (MixedArrayOperation::LeftDot(LeftDotOperation::new(
                MixedArray::zero(left_factor_type),
                DotDimensionNumbers::matmul(),
            )))
            .interpret(&crate::EagerContext::new(), &[MixedArray::value(input)]),
            Ok(vec![MixedArray::zero(f64_array_type(&[4, 3]))])
        );

        let right_factor = TestArray::matrix(3, 4, vec![0.0; 12]);
        assert_eq!(
            (MixedArrayOperation::RightDot(RightDotOperation::new(
                MixedArray::value(right_factor),
                DotDimensionNumbers::matmul(),
            )))
            .interpret(&crate::EagerContext::new(), std::slice::from_ref(&input_zero)),
            Ok(vec![MixedArray::zero(f64_array_type(&[2, 4]))])
        );

        // A captured-predicate condition over the general tangent representation must reject a symbolic-zero
        // predicate: the predicate is a primal residual and can never be a symbolic zero at interpretation time.
        let state_type = f64_array_type(&[2, 3]);
        let identity_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, MixedArray, MixedArrayOperation>::new();
            let branch_input = builder.add_input(state_type.clone());
            builder
                .build::<Vec<MixedArray>, Vec<MixedArray>>(vec![branch_input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let condition = MixedArrayOperation::Condition(LinearConditionOperation::new(
            MixedArray::zero(ArrayType::scalar(DataType::Boolean)),
            Box::new(identity_branch()),
            Box::new(identity_branch()),
        ));
        assert_eq!(
            condition
                .interpret(&crate::EagerContext::new(), &[MixedArray::zero(state_type)])
                .unwrap_err()
                .to_string(),
            "captured condition predicate must be a concrete value"
        );
    }

    #[test]
    fn test_linear_scalar_tangent_value_program_supports_nested_structured_parameters() {
        let mut builder = ProgramBuilder::<DataType, MixedScalar, MixedScalarOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let sum = builder.add_instruction(MixedScalarOperation::Add(AddOperation), vec![left, right]).unwrap()[0];
        let difference =
            builder.add_instruction(MixedScalarOperation::Sub(SubOperation), vec![right, left]).unwrap()[0];
        let scaled = builder
            .add_instruction(
                MixedScalarOperation::Scale(ScaleOperation::new(MixedScalar::zero(DataType::F64))),
                vec![sum],
            )
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

        let op: LinearArrayOperation<ArrayType, TestArray, TestArray> = LinearArrayOperation::Add(AddOperation);
        let context = EagerContext::<
            ArrayType,
            Tangent<ArrayType, TestArray>,
            LinearArrayOperation<ArrayType, TestArray, TestArray>,
        >::new();
        let outputs = <LinearArrayOperation<ArrayType, TestArray, TestArray> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
            EagerContext<
                ArrayType,
                Tangent<ArrayType, TestArray>,
                LinearArrayOperation<ArrayType, TestArray, TestArray>,
            >,
        >>::batch(&op, &context, &[zero_input.clone(), zero_input])
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

        let operation: LinearArrayOperation<ArrayType, TestArray, TestArray> = LinearArrayOperation::Add(AddOperation);
        let context = EagerContext::<
            ArrayType,
            Tangent<ArrayType, TestArray>,
            LinearArrayOperation<ArrayType, TestArray, TestArray>,
        >::new();
        let outputs = <LinearArrayOperation<ArrayType, TestArray, TestArray> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
            EagerContext<
                ArrayType,
                Tangent<ArrayType, TestArray>,
                LinearArrayOperation<ArrayType, TestArray, TestArray>,
            >,
        >>::batch(&operation, &context, &[unbatched_zero, batched_zero])
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().into_owned(), batched_type);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero Add inputs");
    }
}
