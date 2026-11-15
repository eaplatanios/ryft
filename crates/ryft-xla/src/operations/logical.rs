use ryft_core::{AndOperation, ArrayType, NotOperation, OrOperation, XorOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::operations::normalize_binary_elementwise_operands;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for NotOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::not(input_values[0], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for AndOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let result = lowerer.block().append_operation(stable_hlo::and(left, right, lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for OrOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let result = lowerer.block().append_operation(stable_hlo::or(left, right, lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

impl LowerableOperation for XorOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let result = lowerer.block().append_operation(stable_hlo::xor(left, right, lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayOperation, DataType, Placeholder};

    use crate::lowering::tests::{ProgramBuilder, lowered_unary_module};
    use crate::lowering::to_mlir_module_for_plain_program;
    use crate::values::ArrayConstant;

    use super::*;

    #[test]
    fn test_not_operation_lower() {
        assert_eq!(
            lowered_unary_module(ArrayOperation::Not(NotOperation::new()), DataType::Boolean, vec![4]).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xi1>) -> tensor<4xi1> {
                    %0 = stablehlo.not %arg0 : tensor<4xi1>
                    return %0 : tensor<4xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_and_operation_lower() {
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(ArrayType::new_static(DataType::Boolean, [4]));
        let right = builder.add_input(ArrayType::new_static(DataType::Boolean, [4]));
        let output = builder
            .add_instruction(ArrayOperation::And(AndOperation::new()), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
                    %0 = stablehlo.and %arg0, %arg1 : tensor<4xi1>
                    return %0 : tensor<4xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_or_operation_lower() {
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(ArrayType::new_static(DataType::Boolean, [4]));
        let right = builder.add_input(ArrayType::new_static(DataType::Boolean, [4]));
        let output = builder
            .add_instruction(ArrayOperation::Or(OrOperation::new()), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
                    %0 = stablehlo.or %arg0, %arg1 : tensor<4xi1>
                    return %0 : tensor<4xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_xor_operation_lower() {
        let mut builder = ProgramBuilder::new();
        let left = builder.add_input(ArrayType::new_static(DataType::Boolean, [4]));
        let right = builder.add_input(ArrayType::new_static(DataType::Boolean, [4]));
        let output = builder
            .add_instruction(ArrayOperation::Xor(XorOperation::new()), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
                    %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi1>
                    return %0 : tensor<4xi1>
                  }
                }
            "},
        );
    }
}
