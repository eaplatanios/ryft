struct DataType;

trait Value {
    type Type;
}

struct MemberOperation<V>(V);

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    #[ryft(projected)]
    Member(MemberOperation<V>),

    #[ryft(mixed)]
    MixedMember(MemberOperation<V>),
}

fn main() {}
