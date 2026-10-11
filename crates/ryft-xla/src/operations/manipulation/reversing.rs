use ryft_core::{ArrayType, ProgramError, ReverseOperation, TypeError};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for ReverseOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::reverse(
            input_values[0],
            self.axes()
                .normalize(output_types[0].rank())
                .map_err(|error| ProgramError::from(TypeError::invalid(error.to_string())))?
                .as_slice(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, Placeholder, Shape};

    use crate::experimental::lowering::tests::{
        MixedValue, XlaProgramBuilder, dynamic_dimension, execute_mixed_program, unproject_plain_program,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::XlaArrayConstant;
    use crate::tests::execution_client;

    use super::*;

    #[test]
    fn test_lower_reverse() {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let output = builder.add_instruction(ReverseOperation::new([-1]), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {r#"
            module {
              func.func @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
                %0 = stablehlo.reverse %arg0, dims = [0] : tensor<3xf64>
                return %0 : tensor<3xf64>
              }
            }
        "#},
        );
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &unproject_plain_program(program),
                &[MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3])],
                &[],
            ),
            Ok(vec![MixedValue::Array(vec![3.0, 2.0, 1.0], vec![3])]),
        );
    }

    #[test]
    fn test_lower_reverse_dynamic_extent() {
        let mut builder = XlaProgramBuilder::new();
        let input =
            builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("length", Some(5))])));
        let output = builder.add_instruction(ReverseOperation::new([0]), Vec::new(), vec![input], None).unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0], vec![4])], &[2]),
            Ok(vec![MixedValue::Array(vec![2.0, 1.0], vec![2]), MixedValue::Dimension(2)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0], vec![4])], &[0]),
            Ok(vec![MixedValue::Array(Vec::new(), vec![0]), MixedValue::Dimension(0)]),
        );
    }
}
