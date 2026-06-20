struct DataType;
struct TypeError;

trait Operation<T> {
    fn name(&self) -> &'static str;

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;
}

#[derive(ryft::Operation)]
#[ryft(type = "DataType")]
enum BadOperation {
    Add,
}

fn main() {}
