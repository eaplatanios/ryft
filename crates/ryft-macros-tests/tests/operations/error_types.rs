use ryft::{ArrayType, ConstantOperation, DimensionType, Value, ZeroOperation};

#[derive(ryft::Operation)]
enum MissingTypeOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
enum AmbiguousTypeOperation<A: Value<Type = ArrayType>, D: Value<Type = DimensionType>> {
    Array(ConstantOperation<A>),
    Dimension(ConstantOperation<D>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType))]
enum MissingConstantOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType, DimensionType), constant(Array))]
enum TooManyTypesOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), type(ArrayType), constant(Array))]
enum DuplicateTypeOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type = ArrayType, constant(Array))]
enum RemovedTypeAttributeFormOperation {
    Zero(ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant = Array)]
enum RemovedConstantAttributeFormOperation {
    Zero(ZeroOperation<ArrayType>),
}

fn main() {}
