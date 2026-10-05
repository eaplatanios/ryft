use std::marker::PhantomData;

struct DataType;

trait Value {
    type Type;
}

#[derive(ryft::Operation)]
#[ryft(dispatch(identity, batching, identity))]
enum BadOperation<V: Value<Type = DataType>> {
    Operation(PhantomData<V>),
}

fn main() {}
