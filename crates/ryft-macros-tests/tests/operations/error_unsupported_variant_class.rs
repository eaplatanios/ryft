struct DataType;
struct ArrayType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(projected(ArrayType, DataType))]
    TooManyTypes(MemberOperation<V>),

    #[ryft(composite_member(ArrayType))]
    UnknownClass(MemberOperation<V>),
}

fn main() {}
