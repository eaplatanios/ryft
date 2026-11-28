use ryft_core::{ArrayType, ProgramError, TransposeOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::{Error, LowerableOperation, Lowerer, ProgramLoweringOperation};

// TODO(eaplatanios): Review this module.

impl LowerableOperation for TransposeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let result = lowerer.block().append_operation(stable_hlo::transpose(
            input_values[0],
            self.permutation().normalize(self.permutation().len()).map_err(ProgramError::from)?.as_slice(),
            lowerer.location(),
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, Dimension, Placeholder, Shape};
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::lowering::tests::{ProgramBuilder, dynamic_dimension, unproject_plain_program};
    use crate::tests::{execution_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{ArrayConstant, ProgramLoweringExtension, ProgramLoweringOptions};

    use super::*;

    #[test]
    fn test_transpose_operation_lower() {
        // Negative axes normalize against the input rank before they reach `stablehlo.transpose`.
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3, 4]));
        let output =
            builder.add_instruction(TransposeOperation::new([-1, 0, 1]), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            unproject_plain_program(program)
                .to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning())
                .unwrap(),
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
                %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
                return %0 : tensor<4x2x3xf32>
              }
            }
        "#},
        );
    }

    #[test]
    fn test_transpose_operation_lower_dynamic_shape() {
        let shape = Shape::new(vec![dynamic_dimension("size", Some(5)), Dimension::Static(2)]);
        let mut builder = ProgramBuilder::new();
        let integer_input = builder.add_input(ArrayType::new(DataType::I64, shape.clone()));
        let complex_input = builder.add_input(ArrayType::new(DataType::C64, shape));
        let integer_output = builder
            .add_instruction(TransposeOperation::new([-1, -2]), Vec::new(), vec![integer_input], None)
            .unwrap()[0];
        let complex_output = builder
            .add_instruction(TransposeOperation::new([1, 0]), Vec::new(), vec![complex_input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![integer_output, complex_output],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = unproject_plain_program(program)
            .to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning())
            .unwrap();
        let client = execution_client();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // Integers beyond exact f64 precision detect accidental numeric conversion. Negative imaginary parts and
        // signed zeros detect unintended conjugation or loss of exact complex storage.
        let integers = [i64::MAX, i64::MIN, 9_007_199_254_740_993, -9_007_199_254_740_993, 5, 6, 7, 8];
        let complex_components =
            [1f32, -2.0, -0.0, 0.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        for extent in [2i32, 0] {
            let extent_buffer = Arc::new(
                client
                    .buffer(values_to_bytes(&[extent]).as_slice(), BufferType::I32, [], None, device.clone(), None)
                    .unwrap(),
            );
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&integers).as_slice(),
                                BufferType::I64,
                                [4, 2],
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
                                [4, 2],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput { buffer: extent_buffer.clone(), donatable: false },
                ExecutionInput { buffer: extent_buffer, donatable: false },
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
                    &[],
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
                .remove(0);
            assert_eq!(execution.outputs.len(), 4);
            for output in &execution.outputs[..2] {
                assert_eq!(output.unpadded_dimensions().unwrap(), &[2, extent as u64]);
            }
            let outputs = execution
                .outputs
                .iter()
                .map(|output| {
                    let layout = ryft_pjrt::Layout::dense_major_to_minor(output.dimensions().unwrap().len());
                    output.copy_to_host(Some(layout)).unwrap().r#await().unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(values_from_bytes::<i64>(&outputs[2]), vec![i64::from(extent)]);
            assert_eq!(values_from_bytes::<i64>(&outputs[3]), vec![i64::from(extent)]);
            if extent == 2 {
                assert_eq!(
                    values_from_bytes::<i64>(&outputs[0][..4 * size_of::<i64>()]),
                    vec![integers[0], integers[2], integers[1], integers[3]],
                );
                assert_eq!(
                    values_from_bytes::<u32>(&outputs[1][..8 * size_of::<u32>()]),
                    [1f32, -2.0, 3.0, -4.0, -0.0, 0.0, 5.0, -6.0].map(f32::to_bits),
                );
            }
        }
    }
}
