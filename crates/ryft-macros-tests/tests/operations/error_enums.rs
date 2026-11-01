use ryft::{ArrayOperation, ArrayType, ConstantOperation, Value, ZeroOperation};

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array))]
struct StructOperation(ZeroOperation<ArrayType>);

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array))]
enum BadVariantsOperation {
    Unit,
    Named { operation: ZeroOperation<ArrayType> },
    Pair(ZeroOperation<ArrayType>, ZeroOperation<ArrayType>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayType), constant(Array))]
enum DuplicatePayloadTypeOperation {
    Zero(ZeroOperation<ArrayType>),
    BoxedZero(Box<ZeroOperation<ArrayType>>),
}

#[derive(ryft::Operation)]
enum DuplicateGenericPayloadTypeOperation<V: Value<Type = ArrayType>> {
    Constant(ConstantOperation<V>),
    #[ryft(skip_from)]
    OtherConstant(ConstantOperation<V>),
}

#[derive(ryft::Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>))]
enum DuplicateMemberTypeOperation {
    #[ryft(projected(ArrayType))]
    Array(ArrayOperation<ryft::Array>),
    #[ryft(projected(ArrayType, structural))]
    Zero(ZeroOperation<ArrayType>),
}

fn main() {}
