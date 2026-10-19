use ryft::{
    Array, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, BatchableOperation,
    EagerContext, GuardOperation, OperationProvider, ProgramError,
};

/// Declares the array member as structural even though its batching policy has no replicated projection for it.
#[derive(Clone, Debug, ryft::Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>), members(structural(ArrayType)), dispatch(batching))]
enum BadOperation {
    #[ryft(projected(ArrayType, structural))]
    Array(ArrayOperation<Array>),
}

impl OperationProvider<ArrayIrType, GuardOperation<ArrayIrType>> for BadOperation {
    type Operation = Self;

    fn provide(_request: GuardOperation<ArrayIrType>, _input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation { message: "guards are unsupported by this test family".to_owned() })
    }
}

fn assert_batchable<O: BatchableOperation<EagerContext<ArrayIrValue<Array>, BadOperation>, ArrayIrBatchingPolicy>>() {}

fn main() {
    assert_batchable::<BadOperation>();
}
