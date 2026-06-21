use std::marker::PhantomData;

struct DataType;
struct TypeError;

trait Value<T> {}

trait Operation<T> {
    fn name(&self) -> &'static str;

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;
}

#[derive(ryft::Operation)]
enum BadOperation<V: Value<DataType>> {
    Add,
    Marker(PhantomData<V>),
}

fn main() {}
