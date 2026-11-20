use ryft_core::{
    ArrayReferenceTransform, ArrayType, Operation, ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation,
    ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSwapOperation,
    ReferenceWriteOperation,
};
use ryft_mlir::ValueRef;

use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, Type};

impl LowerableOperation for ReferenceNewOperation<ArrayType, Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceReadOperation<ArrayType, Type, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceWriteOperation<ArrayType, Type, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceSwapOperation<ArrayType, Type, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceAddUpdateOperation<ArrayType, Type, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceAtomicAddUpdateOperation<ArrayType, Type, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.name().to_string() }))
    }
}

impl LowerableOperation for ReferenceFreezeOperation<ArrayType, Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
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
    fn lower_without_values<O: LowerableOperation<Type = Type>>(operation: &O) -> Result<usize, Error> {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let mut lowerer = Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram>::new());
        operation.lower(&mut lowerer, &[], &[], &[]).map(|values| values.len())
    }

    #[test]
    fn test_reference_new_operation_lower() {
        let operation = ReferenceNewOperation::<ArrayType, Type>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_new",
        ));
    }

    #[test]
    fn test_reference_read_operation_lower() {
        let operation = ReferenceReadOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_read",
        ));
    }

    #[test]
    fn test_reference_write_operation_lower() {
        let operation = ReferenceWriteOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_write",
        ));
    }

    #[test]
    fn test_reference_swap_operation_lower() {
        let operation = ReferenceSwapOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_swap",
        ));
    }

    #[test]
    fn test_reference_add_update_operation_lower() {
        let operation = ReferenceAddUpdateOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct }))
                if construct == "reference_add_update",
        ));
    }

    #[test]
    fn test_reference_atomic_add_update_operation_lower() {
        let operation = ReferenceAtomicAddUpdateOperation::<ArrayType, Type, ArrayReferenceTransform>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct }))
                if construct == "reference_atomic_add_update",
        ));
    }

    #[test]
    fn test_reference_freeze_operation_lower() {
        let operation = ReferenceFreezeOperation::<ArrayType, Type>::new();
        assert!(matches!(
            lower_without_values(&operation),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "reference_freeze",
        ));
    }
}
