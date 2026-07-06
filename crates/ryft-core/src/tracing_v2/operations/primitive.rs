use std::fmt::Debug;
use std::ops::BitAnd;

use ryft_macros::{BatchableOperation, DifferentiableOperation, Operation, TransposableOperation};

use crate::operations::BooleanLike;
use crate::operations::arithmetic::{AddOperation, DivOperation, MulOperation, NegOperation, SubOperation};
use crate::operations::compare::CompareOperation;
use crate::operations::constants::{
    ConstantOperation, FillOperation, IotaOperation, OneLikeOperation, OneOperation, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{
    ConditionOperation, MaybeScan, MaybeWhile, ScanOperation, Select, SelectOperation, WhileOperation, WhileParts,
    WhilePredicate,
};
use crate::operations::debugging::PrintOperation;
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, ConcatenateOperation, DynamicSliceOperation, DynamicUpdateSliceOperation,
    GatherOperation, PadOperation, Reshape, ReshapeOperation, ScatterOperation, Slice, SliceOperation,
    TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use crate::operations::tag::{MaybeTag, TagOperation};
use crate::operations::trigonometric::{CosOperation, SinOperation};
use crate::programs::Value;
use crate::tracing_v2::operations::collective::{AxisIndexOperation, CollectiveOperation};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use crate::tracing_v2::operations::dot::MaybeDot;
use crate::tracing_v2::operations::memory::TransferToMemoryOperation;
use crate::tracing_v2::operations::reduce::ReduceOperation;
use crate::tracing_v2::operations::{DotDimensionNumbers, DotOperation, Reduce};
use crate::tracing_v2::rematerialization::RematerializeOperation;
use crate::types::ArrayType;

/// Reusable operation enum for ordinary staged programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates. Most variants are thin tags
/// around one semantic primitive defined elsewhere in [`super`].
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`] and
/// [`Dot`](Self::Dot) a [`DotOperation`].
#[derive(Clone, Debug, Operation, DifferentiableOperation, TransposableOperation, BatchableOperation)]
// TODO(eaplatanios): Verify that we need all of these bounds / that they cannot be simplified.
#[ryft(bounds(
    interpretation(BooleanLike + WhilePredicate + Slice + UpdateSlice + Reshape),
    partial_evaluation(PartialEq + BooleanLike),
    differentiation(PartialEq + BooleanLike),
    batching(
        BooleanLike + BitAnd<Output = V> + Select<Condition = V> + Broadcast + Reduce + Slice + UpdateSlice + Reshape
    ),
))]
pub enum ArrayOperation<V: Value<ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<ArrayType, V>),
    // TODO(eaplatanios): Why is this limited to `f64`?
    Fill(FillOperation<ArrayType, f64>),
    Iota(IotaOperation<ArrayType>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    StopGradient(StopGradientOperation),
    Tag(TagOperation),
    Print(PrintOperation),
    TransferToMemory(TransferToMemoryOperation),
    Dot(DotOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    Broadcast(BroadcastOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(DynamicSliceOperation),
    DynamicUpdateSlice(DynamicUpdateSliceOperation),
    Pad(PadOperation),
    Concatenate(ConcatenateOperation),
    Gather(GatherOperation),
    Scatter(ScatterOperation),
    Reduce(ReduceOperation),
    Compare(CompareOperation),
    Not(NotOperation),
    And(AndOperation),
    Or(OrOperation),
    Xor(XorOperation),
    #[ryft(batching(active))]
    Collective(CollectiveOperation),
    #[ryft(batching(active))]
    AxisIndex(AxisIndexOperation),
    Select(SelectOperation),
    Condition(Box<ConditionOperation<ArrayType, V, Self>>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    Scan(Box<ScanOperation<ArrayType, V, Self>>),
    CustomJvp(Box<CustomJvpOperation<ArrayType, V, Self>>),
    CustomVjp(Box<CustomVjpOperation<ArrayType, V, Self>>),
    CustomVjpTangent(Box<CustomVjpTangentOperation<ArrayType, V, Self>>),
    Rematerialize(Box<RematerializeOperation<ArrayType, V, Self>>),
}

// TODO(eaplatanios): Should this be derived as part of one of our macros?
impl<V> MaybeTag for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn key(&self) -> Option<&str> {
        match self {
            Self::Tag(operation) => Some(operation.key()),
            _ => None,
        }
    }
}

// TODO(eaplatanios): Should this be derived as part of one of our macros?
impl<V> MaybeDot for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        match self {
            Self::Dot(operation) => Some(operation.dimensions()),
            _ => None,
        }
    }
}

// TODO(eaplatanios): Should this be derived as part of one of our macros?
impl<V> MaybeWhile<ArrayType, V, ArrayOperation<V>> for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn as_while(&self) -> Option<WhileParts<'_, ArrayType, V, ArrayOperation<V>>> {
        match self {
            Self::While(operation) => operation.as_while(),
            _ => None,
        }
    }
}

// TODO(eaplatanios): Should this be derived as part of one of our macros?
impl<V> MaybeScan<ArrayType, V, ArrayOperation<V>> for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn scan_body(&self) -> Option<&crate::programs::Program<ArrayType, V, ArrayOperation<V>, Vec<V>, Vec<V>>> {
        match self {
            Self::Scan(operation) => Some(operation.body()),
            _ => None,
        }
    }
}
