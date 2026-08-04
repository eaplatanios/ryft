struct DataType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(replicated)]
    Member(MemberOperation<V>),
}

fn main() {}
