use ryft_core::{
    ArrayIrType, ArrayReferenceTransform, ArrayType, Operation, ReferenceAddUpdateOperation,
    ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
    ReferenceSwapOperation, ReferenceWriteOperation,
};
use ryft_mlir::ValueRef;

use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ReferenceNewOperation<ArrayType, ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}

impl LowerableOperation for ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}

impl LowerableOperation for ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}

impl LowerableOperation for ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}

impl LowerableOperation for ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}

impl LowerableOperation for ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}

impl LowerableOperation for ReferenceFreezeOperation<ArrayType, ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(LoweringError::UnresolvedReference { construct: self.name().to_string() })
    }
}
