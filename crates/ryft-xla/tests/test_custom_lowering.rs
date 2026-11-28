//! Checks that downstream crates can lower their own operation families through the public lowering API.

use std::collections::HashMap;

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    ADD_OPERATION_NAME, AddOperation, ArrayType, CompositeKey, DataType, LogicalMesh, MUL_OPERATION_NAME, MeshAxis,
    MeshAxisType, MulOperation, Operation as CoreOperation, Placeholder, ProgramBuilder, ProgramError, RegionInterface,
    SCAN_OPERATION_NAME, Sharding, TypeError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{AttributeRef, Block, Operation as MlirOperation, StringRef, Value as MlirValue, ValueRef};
use ryft_xla::lowering::NamedCompositionKey;
use ryft_xla::{
    Constant, Error, InstructionLowerer, LowerableOperation, Lowerer, LoweringRule, Operation,
    ProgramLoweringExtension, ProgramLoweringOperation, ProgramLoweringOptions, Type, lower_operation_variants,
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
        let symbol = lowerer.require_function(key, "downstream.square", || {
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
    #[ryft(family)]
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

/// Lowering rule that lowers an instruction with two inputs to a `stablehlo.subtract`.
struct SubtractRule;

impl LoweringRule for SubtractRule {
    fn lower<'b, 'c: 'b, 't: 'c>(
        &self,
        instruction: &mut dyn InstructionLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let inputs = instruction.input_values();
        let subtract = stable_hlo::subtract(inputs[0], inputs[1], instruction.location())?;
        let subtract = instruction.block().append_operation(subtract)?;
        Ok(vec![subtract.result(0).unwrap().as_ref()])
    }
}

/// Lowering rule that negates what the operation family lowers a [`TwiceOperation`] to, and that leaves the
/// instructions of every other operation as their family lowers them.
struct NegateTwiceRule;

impl LoweringRule for NegateTwiceRule {
    fn lower<'b, 'c: 'b, 't: 'c>(
        &self,
        instruction: &mut dyn InstructionLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let outputs = instruction.lower_default()?;
        if instruction.payload::<TwiceOperation>().is_none() {
            return Ok(outputs);
        }
        let negate = stable_hlo::negate(outputs[0], instruction.location())?;
        let negate = instruction.block().append_operation(negate)?;
        Ok(vec![negate.result(0).unwrap().as_ref()])
    }
}

/// Returns the program `f(x) = twice(x) + x` over the downstream operation family.
fn twice_plus_input_program() -> ryft_core::FlatProgram<ryft_core::EagerContext<Constant, ExtendedOperation>> {
    let vector_type = ArrayType::new_static(DataType::F32, [3]);
    let mut builder = ProgramBuilder::<Constant, ExtendedOperation>::new();
    let input = builder.add_input(vector_type.into());
    let doubled = builder
        .add_instruction(ExtendedOperation::Twice(TwiceOperation), Vec::new(), vec![input], None)
        .unwrap()[0];
    let add = ExtendedOperation::Xla(Box::new(Operation::from(AddOperation::<ArrayType>::new())));
    let sum = builder.add_instruction(add, Vec::new(), vec![doubled, input], None).unwrap()[0];
    builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![sum], vec![Placeholder], vec![Placeholder])
        .unwrap()
}

#[test]
fn test_lowering_rule_replaces_the_lowering_of_an_operation() {
    // The rule lowers the instruction of the operation that it is registered for, which a member family of the
    // downstream family holds here, and no other instruction, including the `stablehlo.add` that `twice` emits.
    let options = ProgramLoweringOptions::new().with_lowering_rule(ADD_OPERATION_NAME, SubtractRule);
    assert!(options.has_lowering_rule(ADD_OPERATION_NAME));
    assert!(!options.has_lowering_rule(MUL_OPERATION_NAME));
    assert_eq!(
        twice_plus_input_program().to_rendered_mlir("main", &options).unwrap(),
        indoc! {"
            module {
              func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.add %arg0, %arg0 : tensor<3xf32>
                %1 = stablehlo.subtract %0, %arg0 : tensor<3xf32>
                return %1 : tensor<3xf32>
              }
            }
        "},
    );
}

#[test]
fn test_lowering_rule_wraps_the_lowering_of_the_operation_family() {
    // The rule emits what the family lowers the instruction to and then adds to it, for the payload that it
    // recognizes. Registered for another operation as well, it leaves the instructions of that operation alone.
    let options = ProgramLoweringOptions::new()
        .with_lowering_rule("twice", NegateTwiceRule)
        .with_lowering_rule(ADD_OPERATION_NAME, NegateTwiceRule);
    assert_eq!(
        twice_plus_input_program().to_rendered_mlir("main", &options).unwrap(),
        indoc! {"
            module {
              func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.add %arg0, %arg0 : tensor<3xf32>
                %1 = stablehlo.negate %0 : tensor<3xf32>
                %2 = stablehlo.add %1, %arg0 : tensor<3xf32>
                return %2 : tensor<3xf32>
              }
            }
        "},
    );
}

#[test]
fn test_lowering_rule_applies_inside_private_functions() {
    // The decomposition of `square` is a program over the operations of the backend, which the module lowers as a
    // private function. The rule lowers its multiplication, although the lowered program is over another family.
    let vector_type = ArrayType::new_static(DataType::F32, [3]);
    let mut builder = ProgramBuilder::<Constant, ExtendedOperation>::new();
    let input = builder.add_input(vector_type.into());
    let square = ExtendedOperation::Square(SquareOperation);
    let output = builder.add_instruction(square, Vec::new(), vec![input], None).unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let options = ProgramLoweringOptions::new().with_lowering_rule(MUL_OPERATION_NAME, SubtractRule);
    assert_eq!(
        program.to_rendered_mlir("main", &options).unwrap(),
        indoc! {r#"
            module {
              func.func private @downstream.square(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.subtract %arg0, %arg0 : tensor<3xf32>
                return %0 : tensor<3xf32>
              }
              func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
                %0 = stablehlo.composite "downstream.square" %arg0 {decomposition = @downstream.square, version = 1 : i32} : (tensor<3xf32>) -> tensor<3xf32>
                return %0 : tensor<3xf32>
              }
            }
        "#},
    );
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
