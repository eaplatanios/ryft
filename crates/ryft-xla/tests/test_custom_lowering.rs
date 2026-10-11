//! Checks that downstream crates can lower their own operation families through the public lowering API.

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    AddOperation, ArrayIrType, ArrayType, DataType, Operation, Placeholder, ProgramBuilder, RegionInterface, TypeError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};
use ryft_xla::experimental::ops::{XlaConstant, XlaOperation};
use ryft_xla::{
    LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer, lower_operation_variants,
    to_mlir_module_for_program,
};

/// Downstream operation that doubles its input, which `ryft-xla` knows nothing about.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TwiceOperation;

impl Operation for TwiceOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        "twice"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        _region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl LowerableOperation for TwiceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let doubled =
            lowerer
                .block()
                .append_operation(stable_hlo::add(input_values[0], input_values[0], lowerer.location())?)?;
        Ok(vec![doubled.result(0).unwrap().as_ref()])
    }
}

/// Downstream operation family that extends the backend-owned XLA operations with [`TwiceOperation`].
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(crate = "ryft_core", type(ArrayIrType), constant(XlaConstant))]
enum ExtendedOperation {
    Xla(Box<XlaOperation>),
    Twice(TwiceOperation),
}

impl LowerableOperation for ExtendedOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_operation_variants!(
            self, ExtendedOperation, lowerer, input_values, input_types, output_types;
            Xla => boxed,
            Twice,
        )
    }
}

#[test]
fn test_downstream_operation_family_lowers_through_the_public_api() {
    let vector_type = ArrayType::new_static(DataType::F32, [3]);
    let mut builder = ProgramBuilder::<XlaConstant, ExtendedOperation>::new();
    let input = builder.add_input(vector_type.clone().into());
    let doubled = builder
        .add_instruction(ExtendedOperation::Twice(TwiceOperation), Vec::new(), vec![input], None)
        .unwrap()[0];
    let sum = builder
        .add_instruction(
            ExtendedOperation::Xla(Box::new(XlaOperation::from(AddOperation::<ArrayIrType>::new()))),
            Vec::new(),
            vec![doubled, input],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let module =
        to_mlir_module_for_program(&program, &[], &vec![vector_type.clone()], &vec![vector_type], "main", None, None)
            .unwrap();
    assert_eq!(
        module,
        indoc! {"
            module {
              func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.add %arg0, %arg0 : tensor<3xf32>
                %1 = stablehlo.add %0, %arg0 : tensor<3xf32>
                return %1 : tensor<3xf32>
              }
            }
        "},
    );
}
