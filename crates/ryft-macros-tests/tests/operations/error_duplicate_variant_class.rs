struct DataType;
struct ArrayType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(projected(ArrayType), projected(ArrayType))]
    Member(MemberOperation<V>),
}

fn main() {}
