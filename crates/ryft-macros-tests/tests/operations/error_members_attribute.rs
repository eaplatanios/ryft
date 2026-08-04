struct DataType;
struct ArrayType;
struct DimensionType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
#[ryft(members())]
enum EmptyMembersOperation<V: Value<Type = DataType>> {
    Member(MemberOperation<V>),
}

#[derive(ryft::Operation)]
#[ryft(members(ArrayType, ArrayType))]
enum DuplicateMemberOperation<V: Value<Type = DataType>> {
    Member(MemberOperation<V>),
}

#[derive(ryft::Operation)]
#[ryft(members(ArrayType, replicated(DimensionType)))]
enum UnknownMemberRoleOperation<V: Value<Type = DataType>> {
    Member(MemberOperation<V>),
}

#[derive(ryft::Operation)]
#[ryft(members(structural(ArrayType, DimensionType)))]
enum TooManyRoleMembersOperation<V: Value<Type = DataType>> {
    Member(MemberOperation<V>),
}

#[derive(ryft::Operation)]
#[ryft(members(ArrayType))]
#[ryft(members(DimensionType))]
enum DuplicateMembersAttributeOperation<V: Value<Type = DataType>> {
    Member(MemberOperation<V>),
}

fn main() {}
