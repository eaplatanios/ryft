use ryft_core::macros::check_count;
use ryft_core::{
    AbsOperation, AddOperation, ArrayType, DataType, DivOperation, MulOperation, NegOperation, PowOperation,
    RemOperation, RsqrtOperation, SaturatingSubOperation, SignOperation, SqrtOperation, SubOperation,
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

impl LowerableOperation for SaturatingSubOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_arithmetic_inputs(input_values, input_types, output_types, lowerer)?;
        let data_type = output_types[0].data_type();
        if data_type.is_unsigned() {
            // Clamping the subtrahend first avoids unsigned wraparound, including for U64.
            let right = lowerer.block.append_operation(stable_hlo::minimum(left, right, lowerer.location)?)?;
            let difference = lowerer.block.append_operation(stable_hlo::subtract(
                left,
                right.result(0).unwrap().as_ref(),
                lowerer.location,
            )?)?;
            return Ok(vec![lower_arithmetic_output(
                difference.result(0).unwrap().as_ref(),
                &output_types[0],
                lowerer,
            )?]);
        }
        if !data_type.is_signed() {
            return Err(LoweringError::UnsupportedDataType { data_type });
        }

        let carrier_data_type = if data_type == DataType::I1 { DataType::I8 } else { data_type };
        let carrier_type = output_types[0].clone().with_data_type(carrier_data_type);
        let width = data_type.bit_width();
        let minimum = i64::MIN >> (64 - width);
        let maximum = i64::MAX >> (64 - width);
        let scalar_type = lower_tensor_type(&ArrayType::scalar(carrier_data_type), lowerer.context, lowerer.location)?;
        let constants = [minimum, maximum, 0]
            .into_iter()
            .map(|value| {
                let elements =
                    lower_constant_elements_attribute(carrier_data_type, scalar_type, value, lowerer.context)?;
                let constant = lowerer.block.append_operation(stable_hlo::constant(elements, lowerer.location)?)?;
                // Broadcasting against a normalized input retains its runtime extents for bounded dynamic arrays.
                Ok(normalize_binary_elementwise_operands(
                    &[constant.result(0).unwrap().as_ref(), left],
                    &[carrier_type.clone()],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?[0])
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let [minimum, maximum, zero] = constants.as_slice() else { unreachable!() };

        // Clamp the minuend to [MIN + max(right, 0), MAX + min(right, 0)] before subtracting. Each bound addition
        // stays in range, and the final difference cannot overflow. This also avoids assumptions about intermediate
        // wrapping in native sub-byte arithmetic, while I64 needs no wider representation.
        let positive = lowerer.block.append_operation(stable_hlo::maximum(right, *zero, lowerer.location)?)?;
        let negative = lowerer.block.append_operation(stable_hlo::minimum(right, *zero, lowerer.location)?)?;
        let lower = lowerer.block.append_operation(stable_hlo::add(
            *minimum,
            positive.result(0).unwrap().as_ref(),
            lowerer.location,
        )?)?;
        let upper = lowerer.block.append_operation(stable_hlo::add(
            *maximum,
            negative.result(0).unwrap().as_ref(),
            lowerer.location,
        )?)?;
        let clamped = lowerer.block.append_operation(stable_hlo::clamp(
            lower.result(0).unwrap().as_ref(),
            left,
            upper.result(0).unwrap().as_ref(),
            lowerer.location,
        )?)?;
        let difference = lowerer.block.append_operation(stable_hlo::subtract(
            clamped.result(0).unwrap().as_ref(),
            right,
            lowerer.location,
        )?)?;
        Ok(vec![lower_arithmetic_output(difference.result(0).unwrap().as_ref(), &output_types[0], lowerer)?])
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{ConvertElementTypeOperation, Placeholder};
    use ryft_pjrt::{BufferType, Client, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, test_spmd_compilation_options};
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};

    use super::*;

    /// Executes subtraction after converting to `data_types`, using I64 boundary values to keep sub-byte integer
    /// transport out of the oracle. Unsigned U64 values above I64::MAX use the I64 encoding with the same bits.
    fn execute_saturating_sub(
        client: &Client<'_>,
        data_types: [DataType; 2],
        left: &[i64],
        right: &[i64],
        scalar_right: bool,
    ) -> Vec<i64> {
        let mut builder = XlaProgramBuilder::new();
        let left_input = builder.add_input(ArrayType::new_static(DataType::I64, [left.len()]));
        let right_input = builder.add_input(if scalar_right {
            ArrayType::scalar(DataType::I64)
        } else {
            ArrayType::new_static(DataType::I64, [right.len()])
        });
        let left_value = builder
            .add_instruction(ConvertElementTypeOperation::new(data_types[0], false), Vec::new(), vec![left_input], None)
            .unwrap()[0];
        let right_value = builder
            .add_instruction(
                ConvertElementTypeOperation::new(data_types[1], false),
                Vec::new(),
                vec![right_input],
                None,
            )
            .unwrap()[0];
        let difference = builder
            .add_instruction(SaturatingSubOperation::new(), Vec::new(), vec![left_value, right_value], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ConvertElementTypeOperation::new(DataType::I64, false), Vec::new(), vec![difference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(1))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let left = client
            .buffer(values_to_bytes(left).as_slice(), BufferType::I64, &[left.len() as u64], None, device.clone(), None)
            .unwrap();
        let right_dimensions = if scalar_right { vec![] } else { vec![right.len() as u64] };
        let right = client
            .buffer(values_to_bytes(right).as_slice(), BufferType::I64, &right_dimensions, None, device, None)
            .unwrap();
        let inputs = [
            ExecutionInput { buffer: Arc::new(left), donatable: false },
            ExecutionInput { buffer: Arc::new(right), donatable: false },
        ];
        let mut outputs = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
            )
            .unwrap()
            .block_until_ready()
            .unwrap()
            .remove(0);
        let bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes(bytes.as_slice())
    }

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

    #[test]
    fn test_saturating_sub_operation_lower() {
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(ArrayType::new_static(DataType::U8, [2]));
        let right = builder.add_input(ArrayType::scalar(DataType::U8));
        let output =
            builder.add_instruction(SaturatingSubOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
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
                  func.func @main(%arg0: tensor<2xui8>, %arg1: tensor<ui8>) -> tensor<2xui8> {
                    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui8>) -> tensor<2xui8>
                    %1 = stablehlo.minimum %arg0, %0 : tensor<2xui8>
                    %2 = stablehlo.subtract %arg0, %1 : tensor<2xui8>
                    return %2 : tensor<2xui8>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_saturating_sub_operation_lower_signed_limits() {
        let client = execution_client();
        for data_type in
            [DataType::I1, DataType::I2, DataType::I4, DataType::I8, DataType::I16, DataType::I32, DataType::I64]
        {
            let width = data_type.bit_width();
            let minimum = i64::MIN >> (64 - width);
            let maximum = i64::MAX >> (64 - width);
            let left = [minimum, maximum, minimum, maximum, 0, minimum, 0];
            let right = [maximum, minimum, minimum, maximum, minimum, 0, maximum];
            let expected = [minimum, maximum, 0, 0, maximum, minimum, -maximum];
            assert_eq!(execute_saturating_sub(&client, [data_type; 2], &left, &right, false), expected, "{data_type}");
        }
    }

    #[test]
    fn test_saturating_sub_operation_lower_unsigned_limits() {
        let client = execution_client();
        for data_type in
            [DataType::U1, DataType::U2, DataType::U4, DataType::U8, DataType::U16, DataType::U32, DataType::U64]
        {
            let maximum = (u64::MAX >> (64 - data_type.bit_width())) as i64;
            let left = [0, maximum, maximum, 0];
            let right = [maximum, 0, maximum, 0];
            assert_eq!(
                execute_saturating_sub(&client, [data_type; 2], &left, &right, false),
                [0, maximum, 0, 0],
                "{data_type}",
            );
            assert_eq!(
                execute_saturating_sub(&client, [data_type; 2], &[0, maximum], &[1], true),
                [0, (maximum as u64).wrapping_sub(1) as i64],
                "{data_type}",
            );
        }
    }

    #[test]
    fn test_saturating_sub_operation_lower_promoted_data_type() {
        let client = execution_client();
        assert_eq!(
            execute_saturating_sub(&client, [DataType::I8, DataType::U8], &[-128, 127], &[255], true),
            [-383, -128],
        );
        assert_eq!(execute_saturating_sub(&client, [DataType::I1, DataType::I8], &[-1, 0], &[-128], true), [127, 127],);
        assert_eq!(execute_saturating_sub(&client, [DataType::I1, DataType::I2], &[-1, 0], &[-2], true), [1, 1]);
        assert_eq!(execute_saturating_sub(&client, [DataType::I2, DataType::I4], &[-2, 1], &[-8], true), [6, 7]);
    }
}
