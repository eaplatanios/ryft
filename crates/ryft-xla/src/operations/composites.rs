use std::hash::Hash;
use std::rc::Rc;

use ryft_core::{
    ArrayIrType, COMPOSITE_OPERATION_NAME, COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeKey, CompositeOperation,
    CompositeTransposeOperation, CustomRuleSource, DifferentiableType, Operation, Parameter, ProgramError,
    RematerializationOptimizationBarrier, Typed,
};
use ryft_mlir::ValueRef;
#[cfg(test)]
use ryft_mlir::{Block as _, Operation as _, Value as _};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::operations::jit_call::{JitCallFunctionMap, lower_shared_function_call};
use crate::{Error, LowerableOperation, LowerableOperationFamily, XlaLowerer};

/// Returns `true` if lowering `operation` at the provided boundary never reads its attached regions, because it is a
/// named composite call (refer to [`LowerableOperation::named_composite`]) that a backend implementation
/// replaces or whose body lowers to a shared function. The
/// lowering loops skip materializing the regions of such instructions, which would otherwise copy the body per call.
pub(crate) fn lowers_without_regions<O: LowerableOperationFamily>(
    operation: &O,
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    target_platform: Option<&str>,
) -> bool {
    let Some(key) = operation.named_composite() else {
        return false;
    };
    NamedCompositeImplementation::select(key, input_types, output_types, target_platform).is_some()
        || nested_functions.is_some_and(|functions| functions.named(key, input_types, output_types).is_some())
}

/// Backend implementation that replaces the calls of a named composite when the XLA lowering targets a
/// platform that supports it. Implementations are selected from an explicit table by the name and version of the
/// definition, together with the boundary of the call and the target platform, before anything is emitted, so that
/// a call either lowers to its implementation or never touches one (i.e., declining emits nothing). A call that no
/// implementation replaces lowers its body.
pub(crate) enum NamedCompositeImplementation {
    /// Test implementation of `ryft.tests.xla_named_square` version 1 on CUDA, which squares its input.
    #[cfg(test)]
    TestSquare,
}

impl NamedCompositeImplementation {
    /// Returns the implementation that replaces the calls of the named composite `key` with the provided input
    /// and output types when lowering for `target_platform`, or [`None`] when the calls lower their body.
    pub(crate) fn select(
        key: &CompositeKey,
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
        target_platform: Option<&str>,
    ) -> Option<Self> {
        match (key.name(), key.version(), target_platform) {
            #[cfg(test)]
            ("ryft.tests.xla_named_square", 1, Some("cuda")) => Some(Self::TestSquare),
            _ => None,
        }
    }

    /// Lowers one call with this implementation, returning one MLIR value per output.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        match *self {
            #[cfg(test)]
            Self::TestSquare => {
                let multiply = _lowerer.block.append_operation(ryft_mlir::dialects::stable_hlo::multiply(
                    _input_values[0],
                    _input_values[0],
                    _lowerer.location,
                )?)?;
                Ok(vec![multiply.result(0).unwrap().as_ref()])
            }
        }
    }
}

impl<
    V: Typed<Type: DifferentiableType + Eq + Hash + Into<ArrayIrType>> + Parameter,
    RegionOperation,
    S: CustomRuleSource<V, RegionOperation>,
> LowerableOperation for CompositeOperation<V, RegionOperation, S>
where
    Self: Operation<Type = V::Type>,
{
    #[inline]
    fn named_composite(&self) -> Option<&CompositeKey> {
        self.named()
    }

    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Self::Type],
        output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // A named composite call lowers to the backend implementation that replaces it, if any, and otherwise to a call of the
        // shared function that implements its body, if it has one, before falling back to inlining its body.
        if let Some(key) = self.named() {
            let input_types = input_types.iter().cloned().map(Into::into).collect::<Vec<ArrayIrType>>();
            let output_types = output_types.iter().cloned().map(Into::into).collect::<Vec<ArrayIrType>>();
            let target_platform = lowerer.collective_state.target_platform();
            if let Some(backend) =
                NamedCompositeImplementation::select(key, &input_types, &output_types, target_platform)
            {
                return backend.lower(lowerer, input_values);
            }
            let function = lowerer
                .nested_functions
                .as_ref()
                .and_then(|functions| functions.named(key, &input_types, &output_types));
            if let Some(function) = function.filter(|_| input_types.len() == input_values.len()) {
                return lower_shared_function_call(
                    &function.symbol,
                    &function.output_types,
                    input_values,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                );
            }

            // Calls whose key and boundary are shared by calls with another body are not indexed, so they find their
            // shared function by their own body instead (refer to `JitCallFunctionMap::named`).
            let function =
                lowerer.nested_functions.as_ref().zip(lowerer.regions.first()).and_then(|(functions, primal)| {
                    functions.get(primal).filter(|function| function.input_types.len() == input_values.len())
                });
            if let Some(function) = function {
                return lower_shared_function_call(
                    &function.symbol,
                    &function.output_types,
                    input_values,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                );
            }
        }

        let [primal, ..] = lowerer.regions.as_slice() else {
            return Err(Error::from(ProgramError::MalformedProgram(format!(
                "{COMPOSITE_OPERATION_NAME} expected a primal region but got no regions"
            ))));
        };
        if lowerer.regions.len() != self.region_slots().len() {
            return Err(Error::from(ProgramError::MalformedProgram(format!(
                "{} expected {} attached regions but got {}",
                COMPOSITE_OPERATION_NAME,
                self.region_slots().len(),
                lowerer.regions.len(),
            ))));
        }
        // Composite call regions are traced through fresh-root contexts whose local capture tables are discarded, so
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

impl<V: Typed + Parameter, RegionOperation, S> LowerableOperation for CompositeTransposeOperation<V, RegionOperation, S>
where
    Self: Operation,
{
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[Self::Type],
        _output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "a `{COMPOSITE_TRANSPOSE_OPERATION_NAME}` has no forward program and cannot be lowered to \
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
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as ArrayIrXlaProgramBuilder};
    use crate::{Error, to_mlir_module_for_program};

    #[test]
    fn test_composite_lowering_inlines_only_the_primal_program() {
        use ryft_core::{CompositeJvpRule, CompositeOperation, PrintOperation};

        // A retained `composite` call lowers only its primal program and threads its effects onto the enclosing
        // ordered-I/O chain, for every rule layout. Nothing from the user-supplied rule programs (marked here by the
        // multiplies on the tangent and cotangent sides) reaches the emitted module.
        let vector_type = test_vector_type(4);
        let primal = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
            (CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false), vec![&jvp]),
            (CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true), vec![&forward, &backward]),
            (CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, true), vec![&jvp, &forward, &backward]),
        ] {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let mut regions = vec![builder.import_region(primal.entry_region_ref())];
            regions.extend(rules.into_iter().map(|rule| builder.import_region(rule.entry_region_ref())));
            let input = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(XlaOperation::Composite(operation), regions, vec![input], None).unwrap()[0];
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
            CompositeOperation, CompositeTransposeOperation, CustomRuleDefinition, CustomRuleRegistration,
        };

        // A custom rule call lowers only its primal program: its derivative rules are retained callbacks, not regions.
        let vector_type = test_vector_type(4);
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::<XlaConstant, XlaOperation>::new("doubled"));
        let primal = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let input = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::Composite(CompositeOperation::new(rules.reference())),
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
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let carrier = CompositeTransposeOperation::new(
            rules.reference(),
            0,
            vec![vector_type.clone().into()],
            vec![vector_type.clone().into()],
        );
        let output = builder
            .add_instruction(XlaOperation::CompositeTranspose(carrier), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None)
                .unwrap_err()
                .to_string(),
            "a `composite_transpose` has no forward program and cannot be lowered to StableHLO; transpose it \
             before lowering",
        );
    }

    #[test]
    fn test_composite_transpose_lowering_is_rejected() {
        use ryft_core::CompositeTransposeOperation;

        // A composite transpose with an attached backward rule is reverse-mode-only and must be transposed away before
        // lowering, so lowering it is rejected.
        let vector_type = test_vector_type(4);
        let backward = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
        let operation = CompositeTransposeOperation::<XlaConstant, XlaOperation>::from_backward_region(
            1,
            vec![ArrayIrType::Array(vector_type.clone())],
            vec![ArrayIrType::Array(vector_type.clone())],
        );
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let backward_region = builder.import_region(backward.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone().into());
        let residual = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::CompositeTranspose(operation),
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
            Err(Error::Core(ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })))
                if message == "a `composite_transpose` has no forward program and cannot be lowered to \
                               StableHLO; transpose it before lowering",
        ));
    }

    /// Returns a program that applies `calls` named composite calls of `rules` to the same input, each attaching the interned
    /// `body`, and returns their outputs.
    fn named_calls_program(
        rules: &ryft_core::CustomRuleRegistration<XlaConstant, XlaOperation>,
        body: &std::sync::Arc<ryft_core::Program<XlaConstant, XlaOperation, Vec<XlaConstant>, Vec<XlaConstant>>>,
        calls: usize,
    ) -> crate::experimental::ops::FlatXlaProgram {
        use ryft_core::CompositeOperation;

        let vector_type = test_vector_type(4);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let outputs = (0..calls)
            .map(|_| {
                let operation = XlaOperation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; calls])
            .unwrap()
    }

    /// Returns a registration of a named composite with the provided key, whose body doubles its input.
    fn named_rules(key: &str) -> ryft_core::CustomRuleRegistration<XlaConstant, XlaOperation> {
        use ryft_core::{CompositeKey, CustomRuleDefinition, NamedComposite};

        let key = CompositeKey::new(key.to_string(), 1);
        let output_types = |types: &[ArrayIrType]| Ok(types.to_vec());
        let named = NamedComposite::new(key, output_types, |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]));
        ryft_core::CustomRuleRegistration::new(CustomRuleDefinition::new("doubled").with_named(named))
    }

    /// Returns the program `x ↦ x + x` over the test vector type, shared as a named composite body.
    fn doubled_body()
    -> std::sync::Arc<ryft_core::Program<XlaConstant, XlaOperation, Vec<XlaConstant>, Vec<XlaConstant>>> {
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        std::sync::Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Lowers `program`, whose outputs all have the test vector type, for `target_platform`.
    fn lower_named_calls(
        program: &crate::experimental::ops::FlatXlaProgram,
        target_platform: Option<&str>,
    ) -> Result<String, Error> {
        let input_types = vec![test_vector_type(4)];
        let output_types = vec![test_vector_type(4); program.output_ids().len()];
        crate::experimental::lowering::lower_mlir_module_for_program(
            program,
            &[],
            &input_types,
            &output_types,
            "main",
            None,
            None,
            target_platform,
        )
        .map(|module| module.stable_hlo)
    }

    #[test]
    fn test_named_composite_lowering_shares_one_function_per_specialization() {
        // Repeated named composite calls with one body specialization lower to calls of one shared private function named
        // after the definition, while a single call inlines its body.
        let rules = named_rules("ryft.tests.xla_named_shared");
        let body = doubled_body();
        assert_eq!(
            lower_named_calls(&named_calls_program(&rules, &body, 2), None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_shared.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_shared.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_shared.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lower_named_calls(&named_calls_program(&rules, &body, 1), None).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_named_composite_lowering_selects_backend_implementations() {
        // A backend implementation replaces the calls on its target, which then need no shared function, and calls
        // lower their body elsewhere.
        let rules = named_rules("ryft.tests.xla_named_square");
        let body = doubled_body();
        let program = named_calls_program(&rules, &body, 2);
        assert_eq!(
            lower_named_calls(&program, Some("cuda")).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %1 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
        assert!(lower_named_calls(&program, None).unwrap().contains("stablehlo.add"));

        // Calls of definitions that no backend implementation recognizes lower their body on every target.
        let rules = named_rules("ryft.tests.xla_named_unrecognized");
        let program = named_calls_program(&rules, &body, 1);
        assert!(lower_named_calls(&program, Some("cuda")).unwrap().contains("stablehlo.add"));
    }

    #[test]
    fn test_named_composite_lowering_separates_conflicting_bodies() {
        use ryft_core::CompositeOperation;

        // Two separately registered definitions share one key and boundary but carry different bodies (e.g., because
        // their named axes differ). Each call must still lower to the shared function of its own body.
        let first = named_rules("ryft.tests.xla_named_conflict");
        let second = named_rules("ryft.tests.xla_named_conflict");
        let doubled = doubled_body();
        let squared = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let input = builder.add_input(test_vector_type(4).into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            std::sync::Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &doubled), (&second, &squared), (&first, &doubled), (&second, &squared)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = XlaOperation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_conflict.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func private @ryft.tests.xla_named_conflict.v1_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_conflict.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_conflict.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = call @ryft.tests.xla_named_conflict.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %3 = call @ryft.tests.xla_named_conflict.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }
}
