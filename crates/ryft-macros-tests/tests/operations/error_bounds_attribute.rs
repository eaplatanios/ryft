use std::marker::PhantomData;

struct DataType;

trait Value {
    type Type;
}

#[derive(ryft::Operation)]
#[ryft(bounds = "T: Operation<DataType>")]
enum BadOperation<V: Value<Type = DataType>> {
    Operation(PhantomData<V>),
}

fn main() {}
