use std::marker::PhantomData;

struct DataType;
struct ArrayType;

trait Value {
    type Type;
}

#[derive(ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>, W: Value<Type = ArrayType>> {
    Add(AddOperation<V, W>),
}

struct AddOperation<V, W>(PhantomData<(V, W)>);

fn main() {}
