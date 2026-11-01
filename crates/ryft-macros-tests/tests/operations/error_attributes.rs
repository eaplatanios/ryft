use ryft::{ArrayType, ZeroOperation};

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), unknown = "value")]
enum UnknownAttributeOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), identity)]
enum RemovedIdentityAttributeOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), dispatch())]
enum EmptyDispatchOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), dispatch(lowering))]
enum UnknownDispatcherOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), dispatch(identity, batching, identity))]
enum DuplicateDispatcherOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), dispatch(batching))]
#[ryft(dispatch(transposition))]
enum DuplicateDispatchAttributeOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), members())]
enum EmptyMembersOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), members(ArrayType, ArrayType))]
enum DuplicateMemberOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), members(ArrayType, replicated(DimensionType)))]
enum UnknownMemberRoleOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), members(structural(ArrayType, DimensionType)))]
enum TooManyRoleMembersOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array), members(ArrayType))]
#[ryft(members(DimensionType))]
enum DuplicateMembersAttributeOperation {
    Zero(ZeroOperation<ArrayType>),
}

fn main() {}
