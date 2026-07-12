use std::marker::PhantomData;

struct ArrayType;

trait Value {
    type Type;
}

#[derive(ryft::TransposableOperation)]
#[ryft(bounds(unknown(Clone)))]
enum BadOperation<V: Value<Type = ArrayType>> {
    Operation(PhantomData<V>),
}

fn main() {}
