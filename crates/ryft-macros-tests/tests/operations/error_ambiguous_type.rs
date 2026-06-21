use std::marker::PhantomData;

struct DataType;
struct ArrayType;

trait Value<T> {}

#[derive(ryft::Operation)]
enum BadOperation<V: Value<DataType>, W: Value<ArrayType>> {
    Add(AddOperation<V, W>),
}

struct AddOperation<V, W>(PhantomData<(V, W)>);

fn main() {}
