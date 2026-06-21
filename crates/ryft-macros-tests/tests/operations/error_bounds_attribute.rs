struct DataType;

#[derive(ryft::Operation)]
#[ryft(type = "DataType", bounds = "T: Operation<DataType>")]
enum BadOperation<T> {
    Operation(T),
}

fn main() {}
