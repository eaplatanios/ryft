struct DataType;
struct ArrayType;
struct DimensionType;
struct ScalarType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
#[ryft(members(ArrayType, structural(DimensionType)))]
enum UndeclaredMemberOperation<V: Value<Type = DataType>> {
    #[ryft(projected(ScalarType))]
    Projected(MemberOperation<V>),

    #[ryft(mixed(ScalarType))]
    Mixed(MemberOperation<V>),
}

#[derive(ryft::Operation)]
#[ryft(members(ArrayType, DimensionType))]
enum AmbiguousDefaultedMemberOperation<V: Value<Type = DataType>> {
    #[ryft(mixed)]
    Mixed(MemberOperation<V>),

    #[ryft(mixed(structural))]
    MixedStructural(MemberOperation<V>),
}

fn main() {}
