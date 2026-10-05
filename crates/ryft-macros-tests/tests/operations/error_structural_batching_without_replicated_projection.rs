use ryft::{Array, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, BatchableOperation, EagerContext};

/// Declares the array member as structural even though its batching policy has no replicated projection for it.
#[derive(Clone, Debug, ryft::Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>), members(structural(ArrayType)), dispatch(batching))]
enum BadOperation {
    #[ryft(projected(ArrayType, structural))]
    Array(ArrayOperation<Array>),
}

fn assert_batchable<O: BatchableOperation<EagerContext<ArrayIrValue<Array>, BadOperation>, ArrayIrBatchingPolicy>>() {}

fn main() {
    assert_batchable::<BadOperation>();
}
