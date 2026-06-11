use crate::macros::check_count;
use crate::operations::InterpretableOperation;
use crate::operations::control_flow::SelectOperation;
use crate::programs::{ProgramError, Value};
use crate::types::ArrayType;

impl<
    V: Value<ArrayType>
        + crate::operations::manipulation::Broadcast<Output = V>
        + crate::operations::manipulation::Transpose,
    C,
> crate::tracing_v2::batching::BatchableOperation<V, C> for SelectOperation
where
    SelectOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        crate::tracing_v2::batching::apply_elementwise_batch(self, inputs)
    }
}
