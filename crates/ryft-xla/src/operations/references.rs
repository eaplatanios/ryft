use ryft_core::{
    ArrayIrType, ArrayReferenceTransform, ArrayType, Operation as CoreOperation, ReferenceAddUpdateOperation,
    ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
    ReferenceSwapOperation, ReferenceWriteOperation,
};
use ryft_mlir::ValueRef;

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError};

impl LowerableOperation for ReferenceNewOperation<ArrayType, ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceFreezeOperation<ArrayType, ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

#[cfg(test)]
mod tests {
    use ryft_mlir::{Block, Context as MlirContext, Location};

    use crate::operations::FlatProgram;

    use super::*;

    /// Lowers `operation` without inputs or outputs and returns the number of values that it produces. Reference
    /// discharge removes every reference operation before lowering, so each one rejects lowering as a defensive
    /// backstop whatever its inputs are.
    fn lower_without_values<O: LowerableOperation<Type = ArrayIrType>>(operation: &O) -> Result<usize, Error> {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let mut lowerer = Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram>::new());
        operation.lower(&mut lowerer, &[], &[], &[]).map(|values| values.len())
    }

    #[test]
    fn test_reference_new_operation_lower() {
        let operation = ReferenceNewOperation::<ArrayType, ArrayIrType>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_new",
        ));
    }

    #[test]
    fn test_reference_read_operation_lower() {
        let operation = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_read",
        ));
    }

    #[test]
    fn test_reference_write_operation_lower() {
        let operation = ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_write",
        ));
    }

    #[test]
    fn test_reference_swap_operation_lower() {
        let operation = ReferenceSwapOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_swap",
        ));
    }

    #[test]
    fn test_reference_add_update_operation_lower() {
        let operation = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct }))
                if construct == "reference_add_update",
        ));
    }

    #[test]
    fn test_reference_atomic_add_update_operation_lower() {
        let operation = ReferenceAtomicAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct }))
                if construct == "reference_atomic_add_update",
        ));
    }

    #[test]
    fn test_reference_freeze_operation_lower() {
        let operation = ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_freeze",
        ));
    }
}
