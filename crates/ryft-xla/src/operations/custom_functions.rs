use ryft_core::{
    CUSTOM_FUNCTION_OPERATION_NAME, CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME, CustomFunctionOperation,
    CustomFunctionTransposeOperation, Operation, Parameter, ProgramError, RematerializationOptimizationBarrier, Typed,
};
use ryft_mlir::ValueRef;

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl<V: Typed + Parameter, RegionOperation, S> LowerableOperation for CustomFunctionOperation<V, RegionOperation, S>
where
    Self: Operation,
{
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[Self::Type],
        _output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [primal, ..] = lowerer.regions.as_slice() else {
            return Err(LoweringError::UnsupportedOp {
                op: format!("{CUSTOM_FUNCTION_OPERATION_NAME} expected a primal region but got no regions"),
            });
        };
        if lowerer.regions.len() != self.region_slots().len() {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "{} expected {} attached regions but got {}",
                    CUSTOM_FUNCTION_OPERATION_NAME,
                    self.region_slots().len(),
                    lowerer.regions.len(),
                ),
            });
        }
        // Custom function regions are traced through fresh-root contexts whose local capture tables are discarded, so
        // they can never legally reference the enclosing function's captures (the trace boundary rejects bodies that
        // register captures). Lowering the primal region with an empty capture namespace turns any capture-referencing
        // constant that still sneaks in into a loud `MissingCapturedConstant` error instead of silently aliasing
        // whatever value occupies the referenced slot of the enclosing capture prefix. Only the primal region is
        // lowered: attached rule regions are dormant under lowering, and retained rules are callbacks rather than
        // regions. Nested-traced regions (`while`/`scan`/`condition`/`linear_call`) share the enclosing capture scope
        // and keep inheriting it.
        lower_nested_program_inline(
            primal,
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            &[],
            &RematerializationOptimizationBarrier::None,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
        )
    }
}

impl<V: Typed + Parameter, RegionOperation, S> LowerableOperation
    for CustomFunctionTransposeOperation<V, RegionOperation, S>
where
    Self: Operation,
{
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[Self::Type],
        _output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "a `{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` has no forward program and cannot be lowered to \
                 StableHLO; transpose it before lowering",
            ),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{AddOperation, ArrayIrType, ArrayType, MulOperation, Placeholder, ProgramError};

    use crate::experimental::lowering::tests::test_vector_type;
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::{LoweringError, to_mlir_module_for_program};

    #[test]
    fn test_custom_function_lowering_inlines_only_the_primal_program() {
        use ryft_core::{CustomFunctionJvpRule, CustomFunctionOperation, PrintOperation};

        // A retained `custom_function` call lowers only its primal program and threads its effects onto the
        // enclosing ordered-I/O chain, for every rule layout. Nothing from the user-supplied rule programs (marked
        // here by the multiplies on the tangent and cotangent sides) reaches the emitted module.
        let vector_type = test_vector_type(4);
        let primal = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let printed = builder
                .add_instruction(PrintOperation::<ArrayType>::new("primal"), Vec::new(), vec![input], None)
                .unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let tangent = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            let output_tangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![tangent, tangent], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output, output_tangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let forward = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output, input],
                    vec![Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let backward = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let input_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![input_cotangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        for (operation, rules) in [
            (CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, false), vec![&jvp]),
            (
                CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Absent, true),
                vec![&forward, &backward],
            ),
            (
                CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, true),
                vec![&jvp, &forward, &backward],
            ),
        ] {
            let mut builder = CompositeXlaProgramBuilder::new();
            let mut regions = vec![builder.import_region(primal.entry_region_ref())];
            regions.extend(rules.into_iter().map(|rule| builder.import_region(rule.entry_region_ref())));
            let input = builder.add_input(vector_type.clone().into());
            let output = builder
                .add_instruction(XlaOperation::CustomFunction(operation), regions, vec![input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let input_types = vec![vector_type.clone()];
            let output_types = vec![vector_type.clone()];
            let module =
                to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

            assert_eq!(
                module,
                indoc! {r#"
                    module {
                      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                        %0 = stablehlo.after_all  : !stablehlo.token
                        %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "primal"}, has_side_effect = true} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                        %2 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                        return %2 : tensor<4xf32>
                      }
                    }
                "#},
            );
        }
    }

    #[test]
    fn test_custom_rule_lowering_inlines_only_the_primal_program() {
        use ryft_core::{
            CustomFunctionOperation, CustomFunctionTransposeOperation, CustomRuleDefinition, CustomRuleRegistration,
        };

        // A custom rule call lowers only its primal program: its derivative rules are retained callbacks, not regions.
        let vector_type = test_vector_type(4);
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::<XlaConstant, XlaOperation>::new("doubled"));
        let primal = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let input = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::CustomFunction(CustomFunctionOperation::new(rules.reference())),
                vec![primal_region],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![vector_type.clone()];
        let output_types = vec![vector_type.clone()];
        assert_eq!(
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        // An unresolved reverse-mode carrier has no forward program to lower.
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let carrier = CustomFunctionTransposeOperation::new(
            rules.reference(),
            0,
            vec![vector_type.clone().into()],
            vec![vector_type.clone().into()],
        );
        let output = builder
            .add_instruction(XlaOperation::CustomFunctionTranspose(carrier), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None)
                .unwrap_err()
                .to_string(),
            "a `custom_function_transpose` has no forward program and cannot be lowered to StableHLO; transpose it \
             before lowering",
        );
    }

    #[test]
    fn test_custom_function_transpose_lowering_is_rejected() {
        use ryft_core::CustomFunctionTransposeOperation;

        // A custom function carrier with an attached backward rule is reverse-mode-only and must be transposed away
        // before lowering, so lowering it is rejected.
        let vector_type = test_vector_type(4);
        let backward = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let operation = CustomFunctionTransposeOperation::<XlaConstant, XlaOperation>::from_backward_region(
            1,
            vec![ArrayIrType::Array(vector_type.clone())],
            vec![ArrayIrType::Array(vector_type.clone())],
        );
        let mut builder = CompositeXlaProgramBuilder::new();
        let backward_region = builder.import_region(backward.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone().into());
        let residual = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::CustomFunctionTranspose(operation),
                vec![backward_region],
                vec![tangent, residual],
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
        let input_types = vec![vector_type.clone(), vector_type.clone()];
        let output_types = vec![vector_type];

        assert!(matches!(
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None),
            Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "a `custom_function_transpose` has no forward program and cannot be lowered to \
                               StableHLO; transpose it before lowering",
        ));
    }
}
