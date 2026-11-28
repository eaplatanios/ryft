//! Checks that downstream crates can lower their own operation families through the public lowering API.

use std::collections::HashMap;

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    AddOperation, ArrayType, CompositeKey, DataType, LogicalMesh, MeshAxis, MeshAxisType, MulOperation,
    Operation as CoreOperation, Placeholder, ProgramBuilder, ProgramError, RegionInterface, SCAN_OPERATION_NAME,
    Sharding, TypeError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{AttributeRef, Block, Operation as MlirOperation, StringRef, Value as MlirValue, ValueRef};
use ryft_xla::lowering::NamedCompositionKey;
use ryft_xla::{
    Constant, Error, LowerableOperation, Lowerer, Operation, ProgramLoweringExtension, ProgramLoweringOperation,
    ProgramLoweringOptions, Type, lower_operation_variants,
};

/// Downstream operation that doubles its input, which `ryft-xla` knows nothing about.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TwiceOperation;

impl CoreOperation for TwiceOperation {
    type Type = Type;

    fn name(&self) -> &'static str {
        "twice"
    }

    fn infer_output_types(
        &self,
        input_types: &[Type],
        region_interfaces: &[RegionInterface<Type>],
    ) -> Result<Vec<Type>, TypeError> {
        let _ = region_interfaces;
        Ok(input_types.to_vec())
    }
}

impl LowerableOperation for TwiceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let doubled =
            lowerer
                .block()
                .append_operation(stable_hlo::add(input_values[0], input_values[0], lowerer.location())?)?;
        Ok(vec![doubled.result(0).unwrap().as_ref()])
    }
}

/// Downstream operation that squares its input and lowers to a `stablehlo.composite` named `downstream.square`,
/// whose decomposition it requests from the module that it is lowered into.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SquareOperation;

impl CoreOperation for SquareOperation {
    type Type = Type;

    fn name(&self) -> &'static str {
        "square"
    }

    fn infer_output_types(
        &self,
        input_types: &[Type],
        region_interfaces: &[RegionInterface<Type>],
    ) -> Result<Vec<Type>, TypeError> {
        let _ = region_interfaces;
        Ok(input_types.to_vec())
    }
}

impl LowerableOperation for SquareOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = output_types;
        let Type::Array(array_type) = &input_types[0] else {
            return Err(ProgramError::InvalidArgument { message: "`square` expects an array".to_string() }.into());
        };
        let composite = CompositeKey::new("downstream.square", 1);
        let key = NamedCompositionKey::new(composite, vec![array_type.clone()], vec![array_type.clone()]);
        let symbol = lowerer.require_named_composition(key, || {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_types[0].clone());
            let multiply = Operation::from(MulOperation::<ArrayType>::new());
            let output = builder.add_instruction(multiply, Vec::new(), vec![input, input], None)?[0];
            Ok(builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])?)
        })?;
        let result_type = lowerer.lower_tensor_type(array_type)?;
        let composite = lowerer.block().append_operation(stable_hlo::composite(
            "downstream.square",
            1,
            None::<&HashMap<StringRef, AttributeRef>>,
            &[input_values[0]],
            symbol.as_ref(),
            Vec::new(),
            &[result_type],
            lowerer.location(),
        )?)?;
        Ok(vec![composite.result(0).unwrap().as_ref()])
    }
}

/// Downstream identity that shares the diagnostic name of the built-in scan.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct IdentityOperation;

impl CoreOperation for IdentityOperation {
    type Type = Type;

    fn name(&self) -> &'static str {
        SCAN_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[Type],
        region_interfaces: &[RegionInterface<Type>],
    ) -> Result<Vec<Type>, TypeError> {
        let _ = region_interfaces;
        Ok(input_types.to_vec())
    }
}

impl LowerableOperation for IdentityOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_types, output_types);
        Ok(input_values.to_vec())
    }
}

/// Downstream operation family that extends the backend-owned XLA operations with user-defined payloads.
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(crate = "ryft_core", type(Type), constant(Constant))]
enum ExtendedOperation {
    Xla(Box<Operation>),
    Twice(TwiceOperation),
    Square(SquareOperation),
    Identity(IdentityOperation),
}

impl LowerableOperation for ExtendedOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        lower_operation_variants!(
            self, ExtendedOperation, lowerer, input_values, input_types, output_types;
            Xla => boxed,
            Twice,
            Square,
            Identity,
        )
    }
}

#[test]
fn test_downstream_operation_family_lowers_through_the_public_api() {
    let vector_type = ArrayType::new_static(DataType::F32, [3]);
    let mut builder = ProgramBuilder::<Constant, ExtendedOperation>::new();
    let input = builder.add_input(vector_type.clone().into());
    let doubled = builder
        .add_instruction(ExtendedOperation::Twice(TwiceOperation), Vec::new(), vec![input], None)
        .unwrap()[0];
    let sum = builder
        .add_instruction(
            ExtendedOperation::Xla(Box::new(Operation::from(AddOperation::<ArrayType>::new()))),
            Vec::new(),
            vec![doubled, input],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![sum], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let module = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
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

#[test]
fn test_downstream_operation_named_scan_carries_a_pending_reduction_without_regions() {
    // The pending sum on the explicit axis may pass through an identity on a mixed mesh. The operation shares
    // its diagnostic name with a built-in scan, whose region boundary would reject this input.
    let mesh = LogicalMesh::new(vec![
        MeshAxis::new("e", 2, MeshAxisType::Explicit).unwrap(),
        MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
    ])
    .unwrap();
    let sharding = Sharding::replicated(mesh, 0).with_unreduced_axes(["e"]).unwrap();
    let input_type = ArrayType::scalar(DataType::F32).with_sharding(sharding).unwrap();
    let mut builder = ProgramBuilder::<Constant, ExtendedOperation>::new();
    let input = builder.add_input(input_type.into());
    let output = builder
        .add_instruction(ExtendedOperation::Identity(IdentityOperation), Vec::new(), vec![input], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert_eq!(
        program.to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning()).unwrap(),
        indoc! {r#"
            module {
              sdy.mesh @mesh = <["e"=2, "a"=2]>
              func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], unreduced={"e"}>}) -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], unreduced={"e"}>}) {
                %0 = sdy.sharding_constraint %arg0 <@mesh, [], unreduced={"e"}> : tensor<f32>
                return %0 : tensor<f32>
              }
            }
        "#},
    );
}

#[test]
fn test_downstream_operation_requests_a_named_composition_through_the_public_api() {
    // Both calls have the same key, so the module holds one decomposition function, which is emitted ahead of the
    // entry function because the lowering of the calls requested it.
    let vector_type = ArrayType::new_static(DataType::F32, [3]);
    let mut builder = ProgramBuilder::<Constant, ExtendedOperation>::new();
    let input = builder.add_input(vector_type.into());
    let square = || ExtendedOperation::Square(SquareOperation);
    let squared = builder.add_instruction(square(), Vec::new(), vec![input], None).unwrap()[0];
    let output = builder.add_instruction(square(), Vec::new(), vec![squared], None).unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let module = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
    assert_eq!(
        module,
        indoc! {r#"
            module {
              func.func private @downstream.square(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.multiply %arg0, %arg0 : tensor<3xf32>
                return %0 : tensor<3xf32>
              }
              func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.composite "downstream.square" %arg0 {decomposition = @downstream.square, version = 1 : i32} : (tensor<3xf32>) -> tensor<3xf32>
                %1 = stablehlo.composite "downstream.square" %0 {decomposition = @downstream.square, version = 1 : i32} : (tensor<3xf32>) -> tensor<3xf32>
                return %1 : tensor<3xf32>
              }
            }
        "#},
    );
}
