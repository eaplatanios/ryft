use std::marker::PhantomData;

struct ArrayType;

trait Value<T> {}

#[derive(ryft::TransposableOperation)]
#[ryft(bounds(differentiation(Clone)))]
enum BadOperation<V: Value<ArrayType>> {
    Operation(PhantomData<V>),
}

fn main() {}
