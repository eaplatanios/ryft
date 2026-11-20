use ryft_core::{ArrayType, ProgramError, TagOperation};
use ryft_mlir::ValueRef;

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for TagOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_types, output_types);
        // `tag` only affects rematerialization policies, so by lowering time it is the identity and the operand is
        // forwarded without emitting any MLIR operation.
        if input_values.len() != 1 {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
        }
        Ok(vec![input_values[0]])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, Error as CoreError};
    use ryft_mlir::{Block, Context as MlirContext, Location, Value as MlirValue};

    use crate::lowering::types::lower_tensor_type;
    use crate::operations::FlatProgram;

    use super::*;

    #[test]
    fn test_tag_operation_lower() {
        // A tag is the identity by lowering time, so it forwards its operand without emitting anything, and it takes
        // exactly one operand.
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let block = context.block(&[(lower_tensor_type(&scalar_type, &context, location).unwrap(), location)]);
        let input = block.as_ref().argument(0).unwrap().as_ref();
        let mut lowerer = Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram>::new());
        let tag = TagOperation::<ArrayType>::new("residual");
        let types = [scalar_type];
        assert_eq!(tag.lower(&mut lowerer, &[input], &types, &types).unwrap(), vec![input]);
        assert!(block.as_ref().is_empty());
        assert!(matches!(
            tag.lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
            ),
        ));
    }
}
