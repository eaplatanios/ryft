use std::fmt::Debug;

use ryft_macros::Operation;

use crate::backends::scalars::Scalar;
use crate::operations::compare::CompareOperation;
use crate::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use crate::operations::constants::{
    ConstantOperation, FillOperation, IotaOperation, OneLikeOperation, OneOperation, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{ConditionOperation, ScanOperation, SelectOperation, WhileOperation};
use crate::operations::debugging::PrintOperation;
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::operations::manipulation::{
    BroadcastOperation, ConcatenateOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, GatherOperation,
    PadOperation, ReshapeOperation, ScatterOperation, SliceOperation, TransposeOperation, UpdateSliceOperation,
};
use crate::operations::math::{
    AbsOperation, AddOperation, Atan2Operation, CosOperation, DivOperation, MulOperation, NegOperation, SinOperation,
    SubOperation,
};
use crate::operations::math::{ExpOperation, LogOperation, SqrtOperation};
use crate::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use crate::operations::tag::TagOperation;
use crate::programs::Value;
use crate::tracing_v2::operations::DotOperation;
use crate::tracing_v2::operations::collective::{AxisIndexOperation, CollectiveOperation};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use crate::tracing_v2::operations::memory::TransferToMemoryOperation;
use crate::tracing_v2::operations::reduce::ReduceOperation;
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
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(batching, differentiation, transposition))]
pub enum ArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<V>),
    Fill(FillOperation<ArrayType, Scalar>),
    Iota(IotaOperation<ArrayType>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    Atan2(Atan2Operation),
    Exp(ExpOperation),
    Log(LogOperation),
    Sqrt(SqrtOperation),
    Abs(AbsOperation),
    Complex(ComplexOperation),
    Conjugate(ConjugateOperation),
    Real(RealOperation),
    Imaginary(ImaginaryOperation),
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
    Collective(CollectiveOperation),
    AxisIndex(AxisIndexOperation),
    Select(SelectOperation),
    Condition(ConditionOperation<V>),
    While(WhileOperation),
    Scan(ScanOperation<V>),
    CustomJvp(CustomJvpOperation),
    CustomVjp(CustomVjpOperation),
    CustomVjpTangent(CustomVjpTangentOperation),
    Rematerialize(RematerializeOperation),
}
