struct DataType;
struct ArrayType;
struct DimensionType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(composite_member(ArrayType))]
    UnknownClass(MemberOperation<V>),

    #[ryft(projected(ArrayType, replicated))]
    UnknownProjectedRole(MemberOperation<V>),

    #[ryft(mixed(ArrayType, replicated))]
    UnknownMixedRole(MemberOperation<V>),

    #[ryft(projected(ArrayType, structural, DimensionType))]
    TooManyProjectedArguments(MemberOperation<V>),

    #[ryft(mixed(ArrayType, structural, DimensionType))]
    TooManyMixedArguments(MemberOperation<V>),

    #[ryft(skip_from, skip_from)]
    DuplicateSkipFrom(MemberOperation<V>),
}

fn main() {}
