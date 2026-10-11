use ryft_core::macros::check_count;
use ryft_core::{
    AbsOperation, AddOperation, ArrayType, DataType, DivOperation, MulOperation, NegOperation, PowOperation,
    RemOperation, RsqrtOperation, SignOperation, SqrtOperation, SubOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue, ValueRef};

use crate::experimental::lowering::types::{
    lower_constant_elements_attribute, lower_signed_one_bit_to_mlir, lower_tensor_type,
};
use crate::operations::{lower_accuracy, normalize_binary_elementwise_operands};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Recovers signed one-bit values before promotion and computes one-bit arithmetic in a byte carrier.
fn normalize_arithmetic_inputs<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
) -> Result<[ValueRef<'b, 'c, 't>; 2], LoweringError> {
    check_count!("input", input_values, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let mut inputs = [input_values[0], input_values[1]];
    for (input, input_type) in inputs.iter_mut().zip(input_types) {
        if input_type.data_type() == DataType::I1 {
            *input = lower_signed_one_bit_to_mlir(*input, &mut lowerer.block, lowerer.context, lowerer.location)?;
        }
    }
    let output_type = match output_types[0].data_type() {
        DataType::I1 => output_types[0].clone().with_data_type(DataType::I8),
        DataType::U1 => output_types[0].clone().with_data_type(DataType::U8),
        _ => output_types[0].clone(),
    };
    normalize_binary_elementwise_operands(
        &inputs,
        &[output_type],
        &mut lowerer.block,
        lowerer.context,
        lowerer.location,
    )
}

/// Restores wrapping one-bit arithmetic by retaining the low bit before converting to the predicate carrier.
fn lower_arithmetic_output<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    output: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if !matches!(output_type.data_type(), DataType::I1 | DataType::U1) {
        return Ok(output);
    }
    // A direct integer-to-predicate conversion tests nonzero, which would incorrectly turn an even sum into one.
    let tensor_type = output.r#type()?.cast::<TensorTypeRef>().unwrap();
    let scalar_type = lowerer.context.tensor_type(tensor_type.element_type()?, &[], None, lowerer.location)?;
    let data_type = if output_type.data_type() == DataType::I1 { DataType::I8 } else { DataType::U8 };
    let one = lower_constant_elements_attribute(data_type, scalar_type, 1, lowerer.context)?;
    let one = lowerer.block.append_operation(stable_hlo::constant(one, lowerer.location)?)?;
    let one = normalize_binary_elementwise_operands(
        &[one.result(0).unwrap().as_ref(), output],
        &[output_type.clone().with_data_type(data_type)],
        &mut lowerer.block,
        lowerer.context,
        lowerer.location,
    )?[0];
    let masked = lowerer.block.append_operation(stable_hlo::and(output, one, lowerer.location)?)?;
    let narrowed = lowerer.block.append_operation(stable_hlo::convert(
        masked.result(0).unwrap().as_ref(),
        lower_tensor_type(output_type, lowerer.context, lowerer.location)?,
        lowerer.location,
    )?)?;
    Ok(narrowed.result(0).unwrap().as_ref())
}

impl LowerableOperation for AddOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_arithmetic_inputs(input_values, input_types, output_types, lowerer)?;
        let result = lowerer.block.append_operation(stable_hlo::add(left, right, lowerer.location)?)?;
        Ok(vec![lower_arithmetic_output(result.result(0).unwrap().as_ref(), &output_types[0], lowerer)?])
    }
}

impl LowerableOperation for SubOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_arithmetic_inputs(input_values, input_types, output_types, lowerer)?;
        let result = lowerer.block.append_operation(stable_hlo::subtract(left, right, lowerer.location)?)?;
        Ok(vec![lower_arithmetic_output(result.result(0).unwrap().as_ref(), &output_types[0], lowerer)?])
    }
}

impl LowerableOperation for MulOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_arithmetic_inputs(input_values, input_types, output_types, lowerer)?;
        let result = lowerer.block.append_operation(stable_hlo::multiply(left, right, lowerer.location)?)?;
        Ok(vec![lower_arithmetic_output(result.result(0).unwrap().as_ref(), &output_types[0], lowerer)?])
    }
}

impl LowerableOperation for DivOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_arithmetic_inputs(input_values, input_types, output_types, lowerer)?;
        let result = lowerer.block.append_operation(stable_hlo::divide(left, right, lowerer.location)?)?;
        Ok(vec![lower_arithmetic_output(result.result(0).unwrap().as_ref(), &output_types[0], lowerer)?])
    }
}

impl LowerableOperation for NegOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if matches!(output_types[0].data_type(), DataType::I1 | DataType::U1) {
            return Ok(vec![input_values[0]]);
        }
        let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }
}

impl LowerableOperation for SqrtOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::sqrt(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.sqrt should return one result").as_ref()])
    }
}

impl LowerableOperation for RsqrtOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::rsqrt(
            input_values[0],
            lower_accuracy(self.accuracy()),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.rsqrt should return one result").as_ref()])
    }
}

impl LowerableOperation for PowOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::power(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.power should return one result").as_ref()])
    }
}

impl LowerableOperation for SignOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if matches!(output_types[0].data_type(), DataType::I1 | DataType::U1) {
            return Ok(vec![input_values[0]]);
        }
        let result = lowerer.block.append_operation(stable_hlo::sign(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.sign should return one result").as_ref()])
    }
}

impl LowerableOperation for RemOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_arithmetic_inputs(input_values, input_types, output_types, lowerer)?;
        let result = lowerer.block.append_operation(stable_hlo::remainder(left, right, lowerer.location)?)?;
        Ok(vec![lower_arithmetic_output(result.result(0).unwrap().as_ref(), &output_types[0], lowerer)?])
    }
}

impl LowerableOperation for AbsOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::abs(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.abs should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::Placeholder;

    use crate::experimental::lowering::tests::XlaProgramBuilder;
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;

    use super::*;

    #[test]
    fn test_normalize_arithmetic_inputs() {
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(ArrayType::scalar(DataType::I1));
        let right = builder.add_input(ArrayType::scalar(DataType::I8));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i8>) -> tensor<i8> {
                    %0 = stablehlo.convert %arg0 : (tensor<i1>) -> tensor<i8>
                    %1 = stablehlo.negate %0 : tensor<i8>
                    %2 = stablehlo.add %1, %arg1 : tensor<i8>
                    return %2 : tensor<i8>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_arithmetic_output() {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I1));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>) -> tensor<i1> {
                    %0 = stablehlo.convert %arg0 : (tensor<i1>) -> tensor<i8>
                    %1 = stablehlo.negate %0 : tensor<i8>
                    %2 = stablehlo.convert %arg0 : (tensor<i1>) -> tensor<i8>
                    %3 = stablehlo.negate %2 : tensor<i8>
                    %4 = stablehlo.add %1, %3 : tensor<i8>
                    %c = stablehlo.constant dense<1> : tensor<i8>
                    %5 = stablehlo.and %4, %c : tensor<i8>
                    %6 = stablehlo.convert %5 : (tensor<i8>) -> tensor<i1>
                    return %6 : tensor<i1>
                  }
                }
            "#},
        );
    }
}
