use ryft_core::{ArrayType, ProgramError, TagOperation};
use ryft_mlir::ValueRef;

use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for TagOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        // `tag` only affects rematerialization policies; by lowering time it is the identity, so
        // forward the operand without emitting any MLIR operation.
        if input_values.len() != 1 {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        }
        Ok(vec![input_values[0]])
    }
}
