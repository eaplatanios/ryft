use ryft_core::{ArrayType, ProgramError, TransposeOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for TransposeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::transpose(
            input_values[0],
            self.permutation().normalize(self.permutation().len()).map_err(ProgramError::from)?.as_slice(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, Dimension, Placeholder, Shape};
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, dynamic_dimension, ragged_dot_cpu_compilation_options,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_lower_transpose_executes_dynamic_shape() {
        let shape = Shape::new(vec![dynamic_dimension("size", Some(5)), Dimension::Static(2)]);
        let mut builder = XlaProgramBuilder::new();
        let integer_input = builder.add_input(ArrayType::new(DataType::I64, shape.clone()));
        let complex_input = builder.add_input(ArrayType::new(DataType::C64, shape));
        let integer_output = builder
            .add_instruction(TransposeOperation::new([-1, -2]), Vec::new(), vec![integer_input], None)
            .unwrap()[0];
        let complex_output = builder
            .add_instruction(TransposeOperation::new([1, 0]), Vec::new(), vec![complex_input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![integer_output, complex_output],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "transpose_dynamic").unwrap();
        // Keep CPU PJRT's physical boundary static while transposing bounded dynamic intermediates. The output
        // extent must follow the permuted axis, and one executable must also handle an empty runtime shape.
        let module = format!(
            "{}{}",
            module
                .strip_suffix("}\n")
                .unwrap()
                .replace("func.func @transpose_dynamic", "func.func private @transpose_dynamic"),
            indoc! {r#"
              func.func @main(%arg0: tensor<4x2xi64>, %arg1: tensor<4x2xcomplex<f32>>, %arg2: tensor<i32>) -> (tensor<2x4xi64>, tensor<2x4xcomplex<f32>>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
                %integers = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<4x2xi64>, tensor<i32>) -> tensor<?x2xi64, #stablehlo.bounds<4, ?>>
                %complex = stablehlo.set_dimension_size %arg1, %arg2, dim = 0 : (tensor<4x2xcomplex<f32>>, tensor<i32>) -> tensor<?x2xcomplex<f32>, #stablehlo.bounds<4, ?>>
                %integer_result, %complex_result = func.call @transpose_dynamic(%integers, %complex) : (tensor<?x2xi64, #stablehlo.bounds<4, ?>>, tensor<?x2xcomplex<f32>, #stablehlo.bounds<4, ?>>) -> (tensor<2x?xi64, #stablehlo.bounds<?, 4>>, tensor<2x?xcomplex<f32>, #stablehlo.bounds<?, 4>>)
                %integer_rows = stablehlo.get_dimension_size %integer_result, dim = 0 : (tensor<2x?xi64, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                %integer_columns = stablehlo.get_dimension_size %integer_result, dim = 1 : (tensor<2x?xi64, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                %complex_rows = stablehlo.get_dimension_size %complex_result, dim = 0 : (tensor<2x?xcomplex<f32>, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                %complex_columns = stablehlo.get_dimension_size %complex_result, dim = 1 : (tensor<2x?xcomplex<f32>, #stablehlo.bounds<?, 4>>) -> tensor<i32>
                %bound = stablehlo.constant dense<4> : tensor<i32>
                %integer_storage = stablehlo.set_dimension_size %integer_result, %bound, dim = 1 : (tensor<2x?xi64, #stablehlo.bounds<?, 4>>, tensor<i32>) -> tensor<2x4xi64>
                %complex_storage = stablehlo.set_dimension_size %complex_result, %bound, dim = 1 : (tensor<2x?xcomplex<f32>, #stablehlo.bounds<?, 4>>, tensor<i32>) -> tensor<2x4xcomplex<f32>>
                return %integer_storage, %complex_storage, %integer_rows, %integer_columns, %complex_rows, %complex_columns : tensor<2x4xi64>, tensor<2x4xcomplex<f32>>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
              }
            }
            "#},
        );
        let client = execution_client();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // Integers beyond exact f64 precision detect accidental numeric conversion. Negative imaginary parts and
        // signed zeros detect unintended conjugation or loss of exact complex storage.
        let integers = [i64::MAX, i64::MIN, 9_007_199_254_740_993, -9_007_199_254_740_993, 5, 6, 7, 8];
        let complex_components =
            [1_f32, -2.0, -0.0, 0.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        for extent in [2_i32, 0] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&integers).as_slice(),
                                BufferType::I64,
                                &[4, 2],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&complex_components).as_slice(),
                                BufferType::C64,
                                &[4, 2],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&[extent]).as_slice(),
                                BufferType::I32,
                                &[],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap(),
                    ),
                    donatable: false,
                },
            ];
            let execution = executable
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
            let outputs = execution
                .outputs
                .iter()
                .map(|output| output.copy_to_host(None).unwrap().r#await().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values_from_bytes::<i32>(outputs[2].as_slice()), vec![2]);
            assert_eq!(values_from_bytes::<i32>(outputs[3].as_slice()), vec![extent]);
            assert_eq!(values_from_bytes::<i32>(outputs[4].as_slice()), vec![2]);
            assert_eq!(values_from_bytes::<i32>(outputs[5].as_slice()), vec![extent]);
            if extent == 2 {
                // Ignore padding beyond the runtime extent in each physical row.
                let integer_values = values_from_bytes::<i64>(outputs[0].as_slice());
                assert_eq!(&integer_values[..2], &[integers[0], integers[2]]);
                assert_eq!(&integer_values[4..6], &[integers[1], integers[3]]);
                let complex_bits = values_from_bytes::<u32>(outputs[1].as_slice());
                assert_eq!(&complex_bits[..4], &[1_f32, -2.0, 3.0, -4.0].map(f32::to_bits));
                assert_eq!(&complex_bits[8..12], &[-0_f32, 0.0, 5.0, -6.0].map(f32::to_bits));
            }
        }
    }
}
