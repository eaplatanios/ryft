struct DataType;
struct ArrayType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    Member(#[ryft(projected(ArrayType))] MemberOperation<V>),
}

fn main() {}
