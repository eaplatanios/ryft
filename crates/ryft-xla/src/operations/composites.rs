use std::hash::Hash;
use std::rc::Rc;

use ryft_core::{
    ArrayIrType, ArrayType, AttentionConfiguration, AttentionImplementation, AttentionInputs, COMPOSITE_OPERATION_NAME,
    COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeKey, CompositeOperation, CompositeTransposeOperation,
    CustomRuleSource, DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
    DifferentiableType, Operation as CoreOperation, Parameter, Placeholder, ProgramError,
    RematerializationOptimizationBarrier, Typed,
};
use ryft_mlir::ValueRef;
#[cfg(test)]
use ryft_mlir::{Block as _, Operation as _, Value as _};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::operations::attention::{
    lower_fused_attention, lower_fused_attention_backward, select_attention_implementation,
};
use crate::operations::jit_call::{JitCallFunctionMap, lower_shared_function_call};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError};

/// Returns `true` if lowering `operation` at the provided boundary never reads its attached regions, because it is a
/// named composite call (refer to [`LowerableOperation::composite_key`]) that a backend implementation
/// replaces or whose body lowers to a shared function. The
/// lowering loops skip materializing the regions of such instructions, which would otherwise copy the body per call.
pub(crate) fn lowers_without_regions<O: LowerableOperationFamily>(
    operation: &O,
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    target_platform: Option<&str>,
) -> bool {
    let Some(key) = operation.composite_key() else {
        return false;
    };
    NamedCompositeImplementation::select(key, input_types, output_types, target_platform).is_some()
        || nested_functions.is_some_and(|functions| functions.named(key, input_types, output_types).is_some())
}

/// Backend implementation that replaces the calls of a named composite when the XLA lowering targets a
/// platform that supports it. Implementations are selected from an explicit table by the name and version of the
/// definition, together with the boundary of the call and the target platform, before anything is emitted, so that
/// a call either lowers to its implementation or never touches one (i.e., declining emits nothing). A call that no
/// implementation replaces lowers its body. Every implementation must validate the attributes of the key and the
/// boundary that it supports and decline any other call, because its key alone does not fix them.
pub(crate) enum NamedCompositeImplementation {
    /// cuDNN fused attention kernels for a call of the forward attention definition (refer to
    /// [`select_attention_implementation`]).
    Attention {
        /// Configuration of the definition.
        configuration: AttentionConfiguration,

        /// Structure of the inputs of the definition (i.e., which optional inputs are present).
        input_structure: AttentionInputs<Placeholder>,

        /// Whether the definition also returns the log-sum-exp residual of its softmax.
        return_residual: bool,
    },

    /// cuDNN fused attention backward kernels for a call of the backward attention definition (refer to
    /// [`select_attention_implementation`]).
    AttentionBackward {
        /// Configuration of the definition.
        configuration: AttentionConfiguration,

        /// Structure of the inputs of the definition (i.e., which optional inputs are present).
        input_structure: AttentionInputs<Placeholder>,
    },

    /// Implementation of an attention definition that requests the fused kernels for a call that they cannot
    /// implement, which reports why instead of silently lowering the portable body.
    FusedAttentionUnavailable {
        /// Name of the definition.
        name: &'static str,

        /// Reason why the fused kernels are unavailable.
        reason: String,
    },

    /// Test implementation of `ryft.tests.xla_named_square` version 1 on CUDA, which squares its input.
    #[cfg(test)]
    TestSquare,
}

impl NamedCompositeImplementation {
    /// Returns the implementation that replaces the calls of the named composite `key` with the provided input
    /// and output types when lowering for `target_platform`, or [`None`] when the calls lower their body.
    pub(crate) fn select(
        key: &CompositeKey,
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
        target_platform: Option<&str>,
    ) -> Option<Self> {
        match (key.name(), key.version(), target_platform) {
            #[cfg(test)]
            ("ryft.tests.xla_named_square", 1, Some("cuda")) => Some(Self::TestSquare),
            _ => select_attention_implementation(key, input_types, output_types, target_platform),
        }
    }

    /// Returns the error that a call with this implementation reports when it cannot use the implementation and would
    /// otherwise lower its portable body instead (e.g., when eager execution replays the body of a call whose concrete
    /// inputs refine the types at which it was staged), or [`None`] when the body is an acceptable substitute. Calls
    /// that explicitly request a backend implementation (e.g., fused attention) never silently use their body.
    pub(crate) fn body_substitution_error(&self) -> Option<Error> {
        let (name, reason) = match self {
            Self::Attention { configuration, .. } | Self::AttentionBackward { configuration, .. }
                if configuration.implementation() != AttentionImplementation::Fused =>
            {
                return None;
            }
            Self::Attention { .. } => (DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, None),
            Self::AttentionBackward { .. } => (DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, None),
            Self::FusedAttentionUnavailable { name, reason } => (*name, Some(reason.as_str())),
            #[cfg(test)]
            Self::TestSquare => return None,
        };
        let reason = reason.unwrap_or(
            "eager execution replays the body of a call whose inputs refine the types at which it was staged; \
             execute the compiled program instead",
        );
        Some(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: name.to_string(),
            reason: format!("the fused attention implementation is unavailable because {reason}"),
        }))
    }

    /// Lowers one call with this implementation, whose boundary has the provided input and output types, returning
    /// one MLIR value per output.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let array_types = |types: &[ArrayIrType]| {
            types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProgramError::from)
        };
        match self {
            Self::Attention { configuration, input_structure, return_residual } => lower_fused_attention(
                *configuration,
                *input_structure,
                *return_residual,
                input_values,
                array_types(input_types)?.as_slice(),
                array_types(output_types)?.as_slice(),
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::AttentionBackward { configuration, input_structure } => lower_fused_attention_backward(
                *configuration,
                *input_structure,
                input_values,
                array_types(input_types)?.as_slice(),
                array_types(output_types)?.as_slice(),
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::FusedAttentionUnavailable { name, reason } => {
                Err(Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: name.to_string(),
                    reason: format!("the fused attention implementation is unavailable because {reason}"),
                }))
            }
            #[cfg(test)]
            Self::TestSquare => {
                let multiply = lowerer.block.append_operation(ryft_mlir::dialects::stable_hlo::multiply(
                    input_values[0],
                    input_values[0],
                    lowerer.location,
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
    Self: CoreOperation<Type = V::Type>,
{
    #[inline]
    fn composite_key(&self) -> Option<&CompositeKey> {
        self.key()
    }

    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Self::Type],
        output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // A named composite call lowers to the backend implementation that replaces it, if any, and otherwise to a call
        // of the shared function that implements its body, if it has one, before falling back to inlining its body.
        if let Some(key) = self.key() {
            let input_types = input_types.iter().cloned().map(Into::into).collect::<Vec<ArrayIrType>>();
            let output_types = output_types.iter().cloned().map(Into::into).collect::<Vec<ArrayIrType>>();
            let target_platform = lowerer.collective_state.target_platform();
            if let Some(backend) =
                NamedCompositeImplementation::select(key, &input_types, &output_types, target_platform)
            {
                return backend.lower(lowerer, input_values, &input_types, &output_types);
            }
            let function = lowerer
                .nested_functions
                .as_ref()
                .and_then(|functions| functions.named(key, &input_types, &output_types));
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
        // whatever value occupies the referenced slot of the enclosing capture prefix. Runtime arrays that such a body
        // refers to are session constants instead, whose module-scoped values the empty namespace keeps reachable.
        // Only the primal region is lowered: attached rule regions are dormant under lowering, and retained rules are
        // callbacks rather than regions. Nested-traced regions (`while`/`scan`/`condition`/`linear_call`) share the
        // enclosing capture scope and keep inheriting it.
        lower_nested_program_inline(
            primal,
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.collective_state.session_values(lowerer.captured_values.as_slice()),
            &RematerializationOptimizationBarrier::None,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
        )
    }
}

impl<V: Typed + Parameter, RegionOperation, S> LowerableOperation for CompositeTransposeOperation<V, RegionOperation, S>
where
    Self: CoreOperation,
{
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        _lowerer: &mut Lowerer<'b, 'c, 't, O>,
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrType, ArrayType, CompositeJvpRule, CompositeKey, CompositeOperation,
        CompositeTransposeOperation, CustomRuleDefinition, CustomRuleRegistration, DimensionValue, MulOperation,
        NamedComposite, Placeholder, PrintOperation, Program as CoreProgram, ProgramError,
    };

    use crate::experimental::lowering::lower_mlir_module_for_program;
    use crate::experimental::lowering::tests::test_vector_type;
    use crate::experimental::ops::{Constant, FlatProgram, Operation, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::{Error, to_mlir_module_for_program};

    #[test]
    fn test_composite_lowering_inlines_only_the_primal_program() {
        // A retained `composite` call lowers only its primal program and threads its effects onto the enclosing
        // ordered-I/O chain, for every rule layout. Nothing from the user-supplied rule programs (marked here by the
        // multiplies on the tangent and cotangent sides) reaches the emitted module.
        let vector_type = test_vector_type(4);
        let primal = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let printed = builder
                .add_instruction(PrintOperation::<ArrayType>::new("primal"), Vec::new(), vec![input], None)
                .unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let tangent = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            let output_tangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![tangent, tangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![output, output_tangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let forward = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![output, input],
                    vec![Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let backward = {
            let mut builder = ArrayIrProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let input_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
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
            let mut builder = ArrayIrProgramBuilder::new();
            let mut regions = vec![builder.import_region(primal.entry_region_ref())];
            regions.extend(rules.into_iter().map(|rule| builder.import_region(rule.entry_region_ref())));
            let input = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(Operation::Composite(operation), regions, vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        // A custom rule call lowers only its primal program: its derivative rules are retained callbacks, not regions.
        let vector_type = test_vector_type(4);
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::<Constant, Operation>::new("doubled"));
        let primal = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let input = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::Composite(CompositeOperation::new(rules.reference())),
                vec![primal_region],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let carrier = CompositeTransposeOperation::new(
            rules.reference(),
            0,
            vec![vector_type.clone().into()],
            vec![vector_type.clone().into()],
        );
        let output = builder
            .add_instruction(Operation::CompositeTranspose(carrier), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        // A composite transpose with an attached backward rule is reverse-mode-only and must be transposed away before
        // lowering, so lowering it is rejected.
        let vector_type = test_vector_type(4);
        let backward = {
            let mut builder = ArrayIrProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = CompositeTransposeOperation::<Constant, Operation>::from_backward_region(
            1,
            vec![ArrayIrType::Array(vector_type.clone())],
            vec![ArrayIrType::Array(vector_type.clone())],
        );
        let mut builder = ArrayIrProgramBuilder::new();
        let backward_region = builder.import_region(backward.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone().into());
        let residual = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::CompositeTranspose(operation),
                vec![backward_region],
                vec![tangent, residual],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![vector_type.clone(), vector_type.clone()];
        let output_types = vec![vector_type];

        assert!(matches!(
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "a `composite_transpose` has no forward program and cannot be lowered to \
                        StableHLO; transpose it before lowering",
            ),
        ));
    }

    /// Returns a program that applies `calls` named composite calls of `rules` to the same input, each attaching the
    /// interned `body`, and returns their outputs.
    fn named_calls_program(
        rules: &CustomRuleRegistration<Constant, Operation>,
        body: &Arc<CoreProgram<Constant, Operation, Vec<Constant>, Vec<Constant>>>,
        calls: usize,
    ) -> FlatProgram {
        let vector_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let outputs = (0..calls)
            .map(|_| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; calls])
            .unwrap()
    }

    /// Returns a registration of a named composite with the provided key, whose body doubles its input.
    fn named_rules(key: &str) -> CustomRuleRegistration<Constant, Operation> {
        named_rules_with_version(key, 1)
    }

    /// Returns a named composite definition `x ↦ x + x` with the provided key name and version.
    fn named_rules_with_version(key: &str, version: u32) -> CustomRuleRegistration<Constant, Operation> {
        let key = CompositeKey::new(key.to_string(), version);
        let output_types = |types: &[ArrayIrType]| Ok(types.to_vec());
        let named = NamedComposite::new(key, output_types, |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]));
        CustomRuleRegistration::new(CustomRuleDefinition::new("doubled").with_named_composite(named))
    }

    /// Returns the program `x ↦ x + x` over the test vector type, shared as a named composite body.
    fn doubled_body() -> Arc<CoreProgram<Constant, Operation, Vec<Constant>, Vec<Constant>>> {
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Lowers `program`, whose outputs all have the test vector type, for `target_platform`.
    fn lower_named_calls(
        program: &crate::experimental::ops::FlatProgram,
        target_platform: Option<&str>,
    ) -> Result<String, Error> {
        let input_types = vec![test_vector_type(4)];
        let output_types = vec![test_vector_type(4); program.output_ids().len()];
        lower_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None, target_platform)
            .map(|module| module.stable_hlo)
    }

    #[test]
    fn test_named_composite_lowering_shares_one_function_per_specialization() {
        // Repeated named composite calls with one body specialization lower to calls of one shared private function
        // named after the definition, while a single call inlines its body.
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
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_square.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_square.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_square.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );

        // Calls of definitions that no backend implementation recognizes lower their body on every target.
        let rules = named_rules("ryft.tests.xla_named_unrecognized");
        let program = named_calls_program(&rules, &body, 1);
        assert_eq!(
            lower_named_calls(&program, Some("cuda")).unwrap(),
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
    fn test_named_composite_lowering_separates_conflicting_bodies() {
        // Two separately registered definitions share one key and boundary but carry different bodies (e.g., because
        // their named axes differ). Each call must still lower to the shared function of its own body.
        let first = named_rules("ryft.tests.xla_named_conflict");
        let second = named_rules("ryft.tests.xla_named_conflict");
        let doubled = doubled_body();
        let squared = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(test_vector_type(4).into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &doubled), (&second, &squared), (&first, &doubled), (&second, &squared)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
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

    #[test]
    fn test_named_composite_lowering_separates_bodies_calling_different_versions() {
        // Two definitions with one key have bodies that differ only in the version of the named composite that they
        // call. Their renderings include that version, so their bodies are not merged and each keeps its own call. The
        // nested calls of both versions have structurally equal bodies, which share one function named after the first.
        let inner_body = doubled_body();
        let outer_body = |version: u32| {
            let inner = named_rules_with_version("ryft.tests.xla_named_inner", version);
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(test_vector_type(4).into());
            let regions = vec![builder.intern_callee(&inner_body, None).unwrap()];
            let operation = Operation::Composite(CompositeOperation::new(inner.reference()));
            let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
            (inner, Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()))
        };
        let (_first_inner, first_body) = outer_body(1);
        let (_second_inner, second_body) = outer_body(2);
        assert_ne!(first_body.to_string(), second_body.to_string());
        let first = named_rules("ryft.tests.xla_named_outer");
        let second = named_rules("ryft.tests.xla_named_outer");
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &first_body), (&second, &second_body), (&first, &first_body), (&second, &second_body)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_inner.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func private @ryft.tests.xla_named_outer.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @ryft.tests.xla_named_inner.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func private @ryft.tests.xla_named_outer.v1_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @ryft.tests.xla_named_inner.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_outer.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_outer.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = call @ryft.tests.xla_named_outer.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %3 = call @ryft.tests.xla_named_outer.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_named_composite_lowering_selects_backends_per_key_of_a_shared_body() {
        // Calls of two versions share one interned body region, but only the first version has a backend implementation
        // on CUDA. The second version's calls must still share a function of their own instead of following the first
        // version's selection.
        let replaced = named_rules_with_version("ryft.tests.xla_named_square", 1);
        let lowered = named_rules_with_version("ryft.tests.xla_named_square", 2);
        let body = doubled_body();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [&replaced, &lowered, &lowered]
            .into_iter()
            .map(|rules| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(&body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, Some("cuda")).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_square.v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_square.v2(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = call @ryft.tests.xla_named_square.v2(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_named_composite_lowering_separates_bodies_that_cannot_be_shared() {
        // Two definitions share one key and boundary, and the body of the first holds a constant, which structural
        // deduplication cannot key. Its calls must inline their own body instead of calling the second body's function.
        let first = named_rules("ryft.tests.xla_named_unshareable");
        let second = named_rules("ryft.tests.xla_named_unshareable");
        let unshareable = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(test_vector_type(4).into());
            builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let doubled = doubled_body();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &unshareable), (&second, &doubled), (&first, &unshareable), (&second, &doubled)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_unshareable.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_unshareable.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %3 = call @ryft.tests.xla_named_unshareable.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }
}
