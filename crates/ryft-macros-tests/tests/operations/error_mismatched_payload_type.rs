use ryft::{ArrayType, ConstantOperation, DataType, StopGradientOperation, Value, ZeroOperation};

#[derive(Clone, ryft::Operation)]
enum BadOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    StopGradient(StopGradientOperation<ArrayType>),
    Constant(ConstantOperation<V>),
}

fn main() {}
