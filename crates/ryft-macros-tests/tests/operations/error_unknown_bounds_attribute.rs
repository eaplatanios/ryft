use std::marker::PhantomData;

struct DataType;

trait Value<T> {}

#[derive(ryft::Operation)]
#[ryft(bounds(lowering(Clone)))]
enum BadOperation<V: Value<DataType>> {
    Operation(PhantomData<V>),
}

fn main() {}
