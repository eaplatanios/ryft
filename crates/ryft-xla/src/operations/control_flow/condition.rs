use std::rc::Rc;

use ryft_core::{ArrayIrType, ArrayType, CONDITION_OPERATION_NAME, ConditionOperation, Operation, ProgramError};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, current_or_new_token, token_threaded_effects,
};
use crate::experimental::ops::FlatXlaProgram;
use crate::operations::control_flow::{lower_control_flow_region, lower_refined_region_input};
use crate::operations::jit_call::JitCallFunctionMap;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

pub(crate) fn lower_condition_to_if<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    branch_regions: &[FlatXlaProgram<O>],
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let [true_branch, false_branch] = branch_regions else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} expected 2 attached regions but got {}", CONDITION_OPERATION_NAME, branch_regions.len(),),
        });
    };
    let branch_input_types = true_branch.input_types();
    let expected_input_count = branch_input_types.len() + 1;
    if input_values.len() != expected_input_count || input_types.len() != expected_input_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "`{}` expected {} lowered inputs but got {} lowered inputs with {} input types",
                CONDITION_OPERATION_NAME,
                expected_input_count,
                input_values.len(),
                input_types.len(),
            ),
        });
    }
    // Branch inputs only need to refine the branch input types, so each one is converted to the lowered type of its
    // branch input first.
    let branch_inputs = input_values[1..]
        .iter()
        .zip(&input_types[1..])
        .zip(branch_input_types.iter())
        .map(|((value, input_type), branch_input_type)| {
            lower_refined_region_input(*value, input_type, branch_input_type, block, context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let branch_inputs = branch_inputs.as_slice();
    // Each ordered class used by either branch is captured and returned independently. Both branches carry the union
    // so their result signatures agree, returning an entry token unchanged when that branch is pure for the class.
    let threaded_effects = true_branch.effects().classes().union(false_branch.effects().classes());
    let mut entry_effect_tokens = *effect_tokens;
    for effect in token_threaded_effects(threaded_effects) {
        let token = current_or_new_token(effect, effect_tokens, block, location)?;
        entry_effect_tokens.set(effect, token);
    }
    // The declared branch outputs may differ (e.g., `f64[extent]` and `f64[3]`) while agreeing after input refinement,
    // so each branch normalizes its results to the condition's output types.
    let true_branch_region = lower_control_flow_region(
        true_branch,
        branch_inputs,
        output_types,
        context,
        location,
        captured_values,
        nested_functions,
        collective_state,
        entry_effect_tokens,
        threaded_effects,
    )?;
    let false_branch_region = lower_control_flow_region(
        false_branch,
        branch_inputs,
        output_types,
        context,
        location,
        captured_values,
        nested_functions,
        collective_state,
        entry_effect_tokens,
        threaded_effects,
    )?;
    let operation = block.append_operation(stable_hlo::r#if(
        input_values[0],
        true_branch_region.into(),
        false_branch_region.into(),
        location,
    )?)?;
    let output_count = true_branch.output_types().len();
    for (token_index, effect) in token_threaded_effects(threaded_effects).enumerate() {
        effect_tokens.set(
            effect,
            operation
                .result(output_count + token_index)
                .expect("a token-threaded stablehlo.if should return one trailing result per active effect class")
                .as_ref(),
        );
    }
    Ok((0..output_count)
        .map(|index| operation.result(index).expect("stablehlo.if should return one result per output").as_ref())
        .collect())
}

impl LowerableOperation for ConditionOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "higher-order self `{}` must be stored directly in the enclosing backend self family",
                self.name(),
            ),
        }
        .into())
    }
}

impl LowerableOperation for ConditionOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_condition_to_if(
            lowerer.regions.as_slice(),
            input_values,
            input_types,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.captured_values.as_slice(),
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, ArrayType, CompareOperation, ComparisonDirection, ConditionOperation, DataType,
        Dimension, DimensionBounds, DimensionSizeOperation, DimensionType, DimensionVariable,
        DynamicBroadcastOperation, NegOperation, OneOperation, Placeholder, Shape, ZeroOperation,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, ragged_dot_cpu_compilation_options,
        unproject_plain_program, xla_identity_branch, xla_neg_branch,
    };
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::tests::execution_client;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_program_lowers_condition_to_stablehlo_if() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(input_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(predicate_type, input_type.clone()),
            &input_type,
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_constant_predicate_condition_to_stablehlo_if() {
        // A condition whose predicate input is fed by a staged constant still lowers to `stablehlo.if`; folding the
        // constant predicate away is the backend's job (StableHLO canonicalization and XLA's conditional
        // simplification), not ryft's.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(input_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let input = builder.add_input(input_type.clone().into());
        let predicate =
            builder.add_instruction(OneOperation::new(predicate_type), Vec::new(), vec![], None).unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_type, &input_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_dimension_extent_through_condition() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let output_type = dynamic_vector_type.clone();

        let branch = |negate: bool| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let branch_extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0]
            } else {
                scalar
            };
            let output = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()),
                    Vec::new(),
                    vec![scalar, branch_extent],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(false).entry_region_ref());
        let false_region = builder.import_region(branch(true).entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone().into());
        let predicate = builder.add_input(predicate_type.clone().into());
        let scalar = builder.add_input(scalar_type.clone().into());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, extent, scalar],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type, predicate_type, scalar_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.get_dimension_size"), "{stablehlo}");
        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.set_dimension_size").count(), 3, "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_condition_branch_print_with_token_result() {
        use ryft_core::PrintOperation;

        // A print inside one condition branch makes the lowered `stablehlo.if` return the branch's final effect
        // token as one extra trailing result: both branches capture the entry token implicitly, the effectful branch
        // returns its print custom call's token, and the pure branch returns the entry token unchanged.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F64);
        let mut true_builder = CompositeXlaProgramBuilder::new();
        let true_input = true_builder.add_input(input_type.clone().into());
        let printed = true_builder
            .add_instruction(PrintOperation::<ArrayType>::new("taken"), Vec::new(), vec![true_input], None)
            .unwrap()[0];
        let negated = true_builder.add_instruction(NegOperation::new(), Vec::new(), vec![printed], None).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![negated], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![predicate_type, input_type.clone()];
        let output_types = vec![input_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = "stablehlo.if"(%arg0) ({
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "taken"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      %3 = stablehlo.negate %arg1 : tensor<f64>
                      stablehlo.return %3, %2 : tensor<f64>, !stablehlo.token
                    }, {
                      stablehlo.return %arg1, %0 : tensor<f64>, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token)
                    return %1#0 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_threads_branch_effect_class_union_independently() {
        use ryft_core::{AssertOperation, PrintOperation};

        // Each branch uses a different ordered class. StableHLO requires identical branch result signatures, so both
        // branches return the union's two tokens while forwarding the untouched entry token for the unused class.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let true_branch = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::<ArrayType>::new(
                        ComparisonDirection::Equal,
                    ))),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("scalar must equal itself")
                        .with_labels(vec!["scalar".to_owned()]),
                    Vec::new(),
                    vec![predicate, input],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(PrintOperation::<ArrayType>::new("false"), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::Boolean), scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(stablehlo.matches("stablehlo.after_all").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.print").count(), 1, "{stablehlo}");
        let if_line = stablehlo.lines().find(|line| line.contains("\"stablehlo.if\"")).unwrap();
        assert!(if_line.contains(":3 ="), "{stablehlo}");
        assert!(stablehlo.contains("-> (tensor<i64>, !stablehlo.token, !stablehlo.token)"), "{stablehlo}",);
    }

    #[test]
    fn test_condition_without_outputs_executes_selected_effects_in_order_on_cpu() {
        use ryft_core::PrintOperation;

        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};

        // An empty data-flow boundary must retain the selected branch's effects and sequence later effects after it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(PrintOperation::<ArrayType>::new(label), Vec::new(), vec![input], None)
                .unwrap();
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder], vec![]).unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_branch = builder.import_region(branch("true").entry_region_ref());
        let false_branch = builder.import_region(branch("false").entry_region_ref());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let predicate = builder.add_input(predicate_type.clone().into());
        let input = builder.add_input(scalar_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, input],
                None,
            )
            .unwrap();
        assert_eq!(outputs.len(), 0);
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("after"), Vec::new(), vec![input], None)
            .unwrap();
        let program =
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![], vec![Placeholder; 2], vec![]).unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![predicate_type, scalar_type],
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f64>) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = "stablehlo.if"(%arg0) ({
                      %3 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "true"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %3 : !stablehlo.token
                    }, {
                      %3 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "false"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      stablehlo.return %3 : !stablehlo.token
                    }) : (tensor<i1>) -> !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%arg1, %1) {api_version = 4 : i32, backend_config = {label = "after"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                    return
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        for (predicate, expected) in [(true, "true: 7.0"), (false, "false: 7.0")] {
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(&[u8::from(predicate)], BufferType::Predicate, &[], None, device.clone(), None)
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(&7f64.to_ne_bytes(), BufferType::F64, &[], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                },
            ];
            let (outputs, lines) = with_captured_prints(|| {
                executable
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
            });
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].outputs.len(), 0);
            assert_eq!(lines, vec![expected, "after: 7.0"]);
        }
    }

    #[test]
    fn test_condition_converts_a_refined_static_branch_input_to_its_bounded_branch_input_type() {
        // A branch input only needs to refine its branch input type, so a static `f64[3]` input feeds a branch input
        // with the bounded dynamic type `f64[rows]`. The output takes the static extent that the inputs establish for
        // `rows`, so the bounded branch result is sliced back to `f64[3]`. The taken branch computes `a + b`.
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let branch_input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let branch = |add: bool| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let first = builder.add_input(branch_input_type.clone().into());
            let second = builder.add_input(branch_input_type.clone().into());
            let output = if add {
                builder
                    .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![first, second], None)
                    .unwrap()[0]
            } else {
                first
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_branch = builder.import_region(branch(true).entry_region_ref());
        let false_branch = builder.import_region(branch(false).entry_region_ref());
        let predicate = builder.add_constant(XlaConstant::Boolean(true));
        let dynamic_input = builder.add_input(branch_input_type.clone().into());
        let static_input = builder.add_input(static_type.into());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, dynamic_input, static_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))]);
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[
                    MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]),
                    MixedValue::Array(vec![10.0, 20.0, 30.0], vec![3]),
                ],
                &[3],
            ),
            Ok(vec![MixedValue::Array(vec![11.0, 22.0, 33.0], vec![3])]),
        );
    }

    #[test]
    fn test_condition_normalizes_retained_branch_signatures_before_lowering_on_cpu() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let client = execution_client();
        let mut builder = CompositeXlaProgramBuilder::new();
        builder.add_input(extent_type.clone().into());
        let input = builder.add_input(dynamic_type.clone().into());
        let true_branch = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        builder.add_input(extent_type.clone().into());
        builder.add_input(dynamic_type.into());
        let output = builder
            .add_instruction(ZeroOperation::new(static_type.clone().into()), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let false_branch = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        for predicate_value in [false, true] {
            // The broad first-class dimension and static array share an extent relationship that structural
            // specialization cannot replace. Direct construction retains both declared branch programs, whose declared
            // outputs (`f64[extent]` and `f64[3]`) agree only after input refinement.
            let mut builder = CompositeXlaProgramBuilder::new();
            let regions = vec![
                builder.import_region(true_branch.entry_region_ref()),
                builder.import_region(false_branch.entry_region_ref()),
            ];
            let predicate = builder.add_constant(XlaConstant::Boolean(predicate_value));
            let extent = builder.add_input(extent_type.clone().into());
            let input = builder.add_input(static_type.clone().into());
            let output = builder
                .add_instruction(ConditionOperation::new(), regions, vec![predicate, extent, input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            assert_eq!(program.output_types(), vec![ArrayIrType::Array(static_type.clone())]);
            let expected = if predicate_value { vec![1.0, 2.0, 3.0] } else { vec![0.0; 3] };
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[MixedValue::Dimension(3), MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3])],
                    &[],
                ),
                Ok(vec![MixedValue::Array(expected, vec![3])]),
            );
        }
    }
}
