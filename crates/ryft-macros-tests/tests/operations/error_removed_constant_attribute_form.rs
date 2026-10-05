use std::marker::PhantomData;

struct DataType;

trait Value {
    type Type;
}

struct Constant;

#[derive(ryft::Operation)]
#[ryft(type(DataType), constant = Constant)]
enum BadOperation<V: Value<Type = DataType>> {
    Operation(PhantomData<V>),
}

fn main() {}
