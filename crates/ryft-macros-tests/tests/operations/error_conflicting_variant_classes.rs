struct DataType;
struct ArrayType;
struct DimensionType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(projected(ArrayType), replicated(DimensionType))]
    Member(MemberOperation<V>),
}

fn main() {}
