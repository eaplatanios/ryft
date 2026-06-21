use std::marker::PhantomData;

struct DataType;

trait Value<T> {}

#[derive(ryft::Operation)]
#[ryft(type = "DataType")]
enum BadOperation<V: Value<DataType>> {
    Operation(PhantomData<V>),
}

fn main() {}
