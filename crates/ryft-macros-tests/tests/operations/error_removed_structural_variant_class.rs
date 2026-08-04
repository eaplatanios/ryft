struct DataType;
struct DimensionType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(structural(DimensionType))]
    Member(MemberOperation<V>),
}

fn main() {}
