use ryft_core::{
    ArrayType, LINEAR_CALL_OPERATION_NAME, LinearCallOperation, Operation as CoreOperation, ProgramError,
    REMATERIALIZE_OPERATION_NAME, RematerializationOptimizationBarrier, RematerializeOperation, StopGradientOperation,
};
use ryft_mlir::ValueRef;

use crate::lowering::regions::lower_nested_program_inline;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, Type};

impl LowerableOperation for StopGradientOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_types, output_types);
        // `stop_gradient` only affects differentiation, so by lowering time it is the identity and every operand is
        // forwarded without emitting any MLIR operation, which matches JAX's lowering.
        Ok(input_values.to_vec())
    }
}

impl LowerableOperation for RematerializeOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(ProgramError::UnsupportedOperation {
            message: "higher-order operation must be stored directly in the enclosing backend operation family"
                .to_string(),
        }
        .into())
    }
}

impl LowerableOperation for LinearCallOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
                self.name(),
            ),
        }
        .into())
    }
}

impl LowerableOperation for RematerializeOperation<Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        let [_] = lowerer.regions() else {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: self.name().to_string(),
                reason: format!(
                    "`{REMATERIALIZE_OPERATION_NAME}` expected 1 attached region but got {}",
                    lowerer.regions().len(),
                ),
            }));
        };
        // Rematerialized bodies are traced through contexts with fresh roots and lower with an empty capture namespace,
        // for the reasons that the `CompositeOperation` lowering explains. A differentiated call recomputes values that
        // the original computation also computes, so a barrier on its selected inputs keeps XLA from merging the two
        // computations or from scheduling the recomputation before those inputs are available.
        let optimization_barrier = match self.differentiated() {
            true => self.optimization_barrier(),
            false => &RematerializationOptimizationBarrier::None,
        };
        let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
        let (effect_tokens, regions, captured_values, nested_functions, collective_state) =
            lowerer.split_effect_tokens();
        lower_nested_program_inline(
            &regions[0],
            input_values,
            &mut block,
            context,
            location,
            collective_state.session_values(captured_values),
            optimization_barrier,
            nested_functions,
            collective_state,
            effect_tokens,
        )
    }
}

impl LowerableOperation for LinearCallOperation<Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        // The forward region is traced as a nested program, so it shares the enclosing capture scope. The transpose
        // region is dormant under lowering.
        if lowerer.regions().len() != 2 {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: self.name().to_string(),
                reason: format!(
                    "`{LINEAR_CALL_OPERATION_NAME}` expected 2 attached regions but got {}",
                    lowerer.regions().len(),
                ),
            }));
        }
        lowerer.lower_region_inline(0, input_values)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, CaptureReference, Context, CosOperation, DataType, DomainTracingContext,
        EverythingSavable, Memory, MulOperation, NothingSavable, Placeholder, PrintOperation, ResidualPolicyReference,
        SaveAndOffloadOnlyTheseNames, SinOperation, StopGradient, TagOperation, TypedProgram, Value as CoreValue,
        differentiate_at, rematerialize,
    };

    use ryft_mlir::{Block, Context as MlirContext, Location, Value as MlirValue};

    use crate::lowering::tests::test_vector_type;
    use crate::lowering::types::lower_tensor_type;
    use crate::operations::{FlatProgram, Operation, Program, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::sessions::tests::{domain_mesh, read_first_shard, replicated_vector_type};
    use crate::tests::{execution_client, values_to_bytes};
    use crate::{Array, CompiledFunction, Constant, Session, Tracer, compile, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_stop_gradient_operation_lower() {
        // A gradient barrier is the identity by lowering time, so it forwards every operand without emitting anything.
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let tensor_type = lower_tensor_type(&scalar_type, &context, location).unwrap();
        let block = context.block(&[(tensor_type, location), (tensor_type, location)]);
        let inputs = [0, 1].map(|index| block.as_ref().argument(index).unwrap().as_ref());
        let mut lowerer = Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram>::new());
        let types = [scalar_type.clone(), scalar_type];
        assert_eq!(
            StopGradientOperation::<ArrayType>::new().lower(&mut lowerer, &inputs, &types, &types).unwrap(),
            inputs.to_vec(),
        );
        assert!(block.as_ref().is_empty());
    }

    #[test]
    fn test_stop_gradient_operation_lower_executes_as_the_identity() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.clone() * x.stop_gradient().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

        let observed = read_first_shard::<f32>(client, &output);
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input * input;
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }

        // Lowering preserves every result of one variadic barrier without materializing an MLIR operation.
        let variadic: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(first, second)| {
                let context = first.domain();
                let mut outputs = context.bind(StopGradientOperation::new(), Vec::new(), &[first, second]).unwrap();
                let second = outputs.pop().unwrap();
                let first = outputs.pop().unwrap();
                (first, second)
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let other_values = [4.0f32, 3.0, 2.0, 1.0];
        let first = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let second =
            Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes::<f32>(&other_values).as_slice())
                .unwrap();
        let (first, second) = variadic.executable().execute(&engine, (first, second)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &first), values);
        assert_eq!(read_first_shard::<f32>(client, &second), other_values);
    }

    #[test]
    fn test_rematerialize_operation_lower_rejects_malformed_instructions() {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let operation = RematerializeOperation::<Type>::new(ResidualPolicyReference::new(NothingSavable));

        // The plain array family cannot carry the body region.
        let mut lowerer = Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram>::new());
        assert!(matches!(
            RematerializeOperation::<ArrayType>::new(ResidualPolicyReference::new(NothingSavable))
                .lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "higher-order operation must be stored directly in the enclosing backend operation \
                        family",
            ),
        ));

        // An array IR call needs exactly its body region.
        assert!(matches!(
            operation.lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "rematerialize" && reason == "`rematerialize` expected 1 attached region but got 0",
        ));
    }

    #[test]
    fn test_rematerialize_operation_lower_inlines_body_effects_on_the_enclosing_token_chain() {
        // Rematerialization is a transform boundary, not an execution boundary. Lowering inlines its body, so prints
        // immediately before, inside, and after the call must form one ordered I/O token chain.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(PrintOperation::<ArrayType>::new("body"), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let input = builder.add_input(scalar_type.clone().into());
        let before = builder
            .add_instruction(PrintOperation::<ArrayType>::new("before"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let rematerialized = builder
            .add_instruction(
                Operation::Rematerialize(RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable))),
                vec![body],
                vec![before],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("after"), Vec::new(), vec![rematerialized], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &scalar_type, &scalar_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "before"}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%arg0, %1) {api_version = 4 : i32, backend_config = {label = "body"}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                    %3 = stablehlo.custom_call @ryft.print(%arg0, %2) {api_version = 4 : i32, backend_config = {label = "after"}, has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                    return %arg0 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rematerialize_operation_lower_rejects_capture_constants_in_its_body() {
        // Rematerialized bodies are traced through contexts with fresh roots and can therefore never legally reference
        // the enclosing function's captures. A capture constant smuggled into such a body must fail loudly instead of
        // silently resolving against the enclosing function's capture prefix, which would alias whichever captured
        // value happens to occupy the referenced slot.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let captured =
                builder.add_constant(Constant::Captured(CaptureReference::new(0, Type::Array(scalar_type.clone()))));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, captured], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let input = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::Rematerialize(RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable))),
                vec![body],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The enclosing function's capture prefix has a matching slot #0, so before rematerialized bodies were scoped
        // to an empty capture namespace this lowering silently forwarded that unrelated captured value.
        let result = to_mlir_module_for_program(
            &program,
            std::slice::from_ref(&scalar_type),
            &scalar_type,
            &scalar_type,
            "main",
            None,
            None,
        );
        assert!(matches!(result, Err(Error::Lowering(LoweringError::MissingCapturedConstant { index: 0 }))));
    }

    #[test]
    fn test_rematerialize_operation_lower_places_an_input_barrier_on_differentiated_calls() {
        // Only a differentiated call whose optimization barrier is enabled wraps its inputs in a barrier, which keeps
        // XLA from merging the recomputation with the original computation.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone().into());
            let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let lower = |operation: RematerializeOperation<Type>| {
            let mut builder = ArrayIrProgramBuilder::new();
            let body = builder.import_region(body.entry_region_ref());
            let input = builder.add_input(scalar_type.clone().into());
            let output =
                builder.add_instruction(Operation::Rematerialize(operation), vec![body], vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            to_mlir_module_for_program(&program, &[], &scalar_type, &scalar_type, "main", None, None).unwrap()
        };
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable));
        assert_eq!(
            lower(operation.clone().with_differentiated(true)),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.optimization_barrier %arg0 : tensor<f64>
                    %1 = stablehlo.sine %0 : tensor<f64>
                    return %1 : tensor<f64>
                  }
                }
            "#},
        );
        let unbarriered = indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                %0 = stablehlo.sine %arg0 : tensor<f64>
                return %0 : tensor<f64>
              }
            }
        "#};
        assert_eq!(lower(operation.clone()), unbarriered);
        assert_eq!(
            lower(
                operation
                    .with_differentiated(true)
                    .with_optimization_barrier(RematerializationOptimizationBarrier::None),
            ),
            unbarriered,
        );
    }

    #[test]
    fn test_rematerialize_operation_lower_gradient_with_an_input_barrier_on_its_recomputation() {
        // `jit(grad(rematerialize(x ↦ sin(x²))))` saves only `x`, so the backward computation recomputes `cos(x²)`
        // inside the differentiated call. Its optimization barrier, which is placed on the inputs of that call unless
        // it is disabled, keeps XLA from reusing the forward computation instead. When the barrier selects no input of
        // the function, it still covers the cotangent that the backward computation receives.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let lower = |optimization_barrier: RematerializationOptimizationBarrier| {
            let function = rematerialize(|x: Tracer| -> Result<Tracer, ProgramError> {
                let context = x.context();
                let square =
                    context.bind(ArrayOperation::Mul(MulOperation::new()), Vec::new(), &[x.clone(), x.clone()])?;
                Ok(context.bind(ArrayOperation::Sin(SinOperation::new()), Vec::new(), &square)?.remove(0))
            })
            .with_optimization_barrier(optimization_barrier);
            let (_, program) = DomainTracingContext::<Session>::trace(
                |x: Tracer| Ok(differentiate_at(x).gradient(|x| function.call(x))?),
                Type::from(scalar_type.clone()),
            )
            .unwrap();
            to_mlir_module_for_program(&program, &[], &scalar_type, &scalar_type, "main", None, None).unwrap()
        };
        assert_eq!(
            lower(RematerializationOptimizationBarrier::All),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %1 = stablehlo.sine %0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %2:2 = stablehlo.optimization_barrier %cst, %arg0 : tensor<f64>, tensor<f64>
                    %3 = stablehlo.multiply %2#1, %2#1 : tensor<f64>
                    %4 = stablehlo.cosine %3 : tensor<f64>
                    %5 = stablehlo.multiply %4, %2#0 : tensor<f64>
                    %6 = stablehlo.multiply %2#1, %5 : tensor<f64>
                    %7 = stablehlo.multiply %2#1, %5 : tensor<f64>
                    %8 = stablehlo.add %6, %7 : tensor<f64>
                    return %8 : tensor<f64>
                  }
                }
            "#},
        );
        assert_eq!(
            lower(RematerializationOptimizationBarrier::Inputs(vec![false])),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %1 = stablehlo.sine %0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %2 = stablehlo.optimization_barrier %cst : tensor<f64>
                    %3 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %4 = stablehlo.cosine %3 : tensor<f64>
                    %5 = stablehlo.multiply %4, %2 : tensor<f64>
                    %6 = stablehlo.multiply %arg0, %5 : tensor<f64>
                    %7 = stablehlo.multiply %arg0, %5 : tensor<f64>
                    %8 = stablehlo.add %6, %7 : tensor<f64>
                    return %8 : tensor<f64>
                  }
                }
            "#},
        );
        assert_eq!(
            lower(RematerializationOptimizationBarrier::None),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %1 = stablehlo.sine %0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %3 = stablehlo.cosine %2 : tensor<f64>
                    %4 = stablehlo.multiply %3, %cst : tensor<f64>
                    %5 = stablehlo.multiply %arg0, %4 : tensor<f64>
                    %6 = stablehlo.multiply %arg0, %4 : tensor<f64>
                    %7 = stablehlo.add %5, %6 : tensor<f64>
                    return %7 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rematerialize_operation_lower_higher_order_derivatives_with_their_input_barriers() {
        // Differentiating the staged gradient of `rematerialize(x ↦ sin(x²))` again, in forward mode (i.e., a
        // forward over reverse Hessian-vector product) or in reverse mode (i.e., the gradient of the gradient), keeps
        // differentiated calls, whose recomputations therefore stay behind optimization barriers.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let function = rematerialize(|x: Tracer| -> Result<Tracer, ProgramError> {
            let context = x.context();
            let square = context.bind(ArrayOperation::Mul(MulOperation::new()), Vec::new(), &[x.clone(), x.clone()])?;
            Ok(context.bind(ArrayOperation::Sin(SinOperation::new()), Vec::new(), &square)?.remove(0))
        });
        let (_, gradient) = DomainTracingContext::<Session>::trace(
            |x: Tracer| Ok(differentiate_at(x).gradient(|x| function.call(x))?),
            Type::from(scalar_type.clone()),
        )
        .unwrap();
        let gradient = gradient.into_flat_program();
        let lower = |program: &Program<Vec<Constant>, Vec<Constant>>| {
            let array_types = |types: Vec<Type>| {
                types.iter().map(|r#type| <&ArrayType>::try_from(r#type).unwrap().clone()).collect::<Vec<_>>()
            };
            let input_types = array_types(program.input_types().into_owned());
            let output_types = array_types(program.output_types().into_owned());
            to_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None).unwrap()
        };
        assert_eq!(
            lower(&gradient.jvp().unwrap()),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %1 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                    %2 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                    %3 = stablehlo.add %1, %2 : tensor<f64>
                    %4 = stablehlo.sine %0 : tensor<f64>
                    %5 = stablehlo.cosine %0 : tensor<f64>
                    %6 = stablehlo.multiply %5, %3 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %7:3 = stablehlo.optimization_barrier %cst, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
                    %8 = stablehlo.multiply %7#1, %7#1 : tensor<f64>
                    %9 = stablehlo.multiply %7#1, %7#2 : tensor<f64>
                    %10 = stablehlo.multiply %7#1, %7#2 : tensor<f64>
                    %11 = stablehlo.add %9, %10 : tensor<f64>
                    %12 = stablehlo.cosine %8 : tensor<f64>
                    %13 = stablehlo.sine %8 : tensor<f64>
                    %14 = stablehlo.multiply %13, %11 : tensor<f64>
                    %15 = stablehlo.negate %14 : tensor<f64>
                    %16 = stablehlo.multiply %12, %7#0 : tensor<f64>
                    %17 = stablehlo.multiply %7#0, %15 : tensor<f64>
                    %18 = stablehlo.multiply %7#1, %16 : tensor<f64>
                    %19 = stablehlo.multiply %16, %7#2 : tensor<f64>
                    %20 = stablehlo.multiply %7#1, %17 : tensor<f64>
                    %21 = stablehlo.add %19, %20 : tensor<f64>
                    %22 = stablehlo.multiply %7#1, %16 : tensor<f64>
                    %23 = stablehlo.multiply %16, %7#2 : tensor<f64>
                    %24 = stablehlo.multiply %7#1, %17 : tensor<f64>
                    %25 = stablehlo.add %23, %24 : tensor<f64>
                    %26 = stablehlo.add %18, %22 : tensor<f64>
                    %27 = stablehlo.add %21, %25 : tensor<f64>
                    return %26, %27 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
        assert_eq!(
            lower(&gradient.linearize().unwrap().pullback().unwrap().into_program()),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> tensor<f64> {
                    %0:3 = stablehlo.optimization_barrier %arg0, %arg1, %arg3 : tensor<f64>, tensor<f64>, tensor<f64>
                    %1 = stablehlo.multiply %0#1, %0#0 : tensor<f64>
                    %2 = stablehlo.multiply %0#1, %0#1 : tensor<f64>
                    %3 = stablehlo.cosine %2 : tensor<f64>
                    %4 = stablehlo.multiply %3, %0#2 : tensor<f64>
                    %5 = stablehlo.multiply %4, %0#0 : tensor<f64>
                    %6 = stablehlo.multiply %0#1, %0#0 : tensor<f64>
                    %7 = stablehlo.add %1, %6 : tensor<f64>
                    %8 = stablehlo.multiply %0#2, %7 : tensor<f64>
                    %9 = stablehlo.negate %8 : tensor<f64>
                    %10 = stablehlo.sine %2 : tensor<f64>
                    %11 = stablehlo.multiply %10, %9 : tensor<f64>
                    %12 = stablehlo.multiply %0#1, %11 : tensor<f64>
                    %13 = stablehlo.add %5, %12 : tensor<f64>
                    %14 = stablehlo.multiply %0#1, %11 : tensor<f64>
                    %15 = stablehlo.add %13, %14 : tensor<f64>
                    %16 = stablehlo.multiply %4, %0#0 : tensor<f64>
                    %17 = stablehlo.add %15, %16 : tensor<f64>
                    return %17 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rematerialize_operation_lower_gradient_with_rounded_bf16_residuals() {
        // Saving everything for `x ↦ sin(x)²` over `bf16` saves `sin(x)`, which the forward computation also squares.
        // It is rounded to `bf16` right after it is computed, so that XLA cannot use a more precise value for the
        // forward computation than the one that the backward computation receives.
        let scalar_type = ArrayType::scalar(DataType::BF16);
        let function = rematerialize(|x: Tracer| -> Result<Tracer, ProgramError> {
            let context = x.context().clone();
            let sine = context.bind(ArrayOperation::Sin(SinOperation::new()), Vec::new(), &[x])?.remove(0);
            Ok(context.bind(ArrayOperation::Mul(MulOperation::new()), Vec::new(), &[sine.clone(), sine])?.remove(0))
        })
        .with_policy(EverythingSavable);
        let (_, gradient) = DomainTracingContext::<Session>::trace(
            |x: Tracer| Ok(differentiate_at(x).gradient(|x| function.call(x))?),
            Type::from(scalar_type.clone()),
        )
        .unwrap();
        assert_eq!(
            to_mlir_module_for_program(&gradient, &[], &scalar_type, &scalar_type, "main", None, None).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<bf16>) -> tensor<bf16> {
                    %0 = stablehlo.sine %arg0 : tensor<bf16>
                    %1 = stablehlo.reduce_precision %0, format = e8m7 : tensor<bf16>
                    %2 = stablehlo.multiply %1, %1 : tensor<bf16>
                    %3 = stablehlo.cosine %arg0 : tensor<bf16>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
                    %4:3 = stablehlo.optimization_barrier %cst, %3, %1 : tensor<bf16>, tensor<bf16>, tensor<bf16>
                    %5 = stablehlo.multiply %4#2, %4#0 : tensor<bf16>
                    %6 = stablehlo.multiply %4#2, %4#0 : tensor<bf16>
                    %7 = stablehlo.add %5, %6 : tensor<bf16>
                    %8 = stablehlo.multiply %4#1, %7 : tensor<bf16>
                    return %8 : tensor<bf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rematerialize_operation_lower_gradient_with_offloaded_residuals() {
        // Offloading the value tagged `sine` of `x ↦ cos(tag(sin(x), "sine"))` to pinned host memory moves it to the
        // host once it is computed and back to the device right before the backward computation uses it.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let function = rematerialize(|x: Tracer| -> Result<Tracer, ProgramError> {
            let context = x.context().clone();
            let sine = context.bind(ArrayOperation::Sin(SinOperation::new()), Vec::new(), &[x])?;
            let sine = context.bind(ArrayOperation::Tag(TagOperation::new("sine")), Vec::new(), &sine)?;
            Ok(context.bind(ArrayOperation::Cos(CosOperation::new()), Vec::new(), &sine)?.remove(0))
        })
        .with_policy(
            SaveAndOffloadOnlyTheseNames::new(Vec::<String>::new(), ["sine"], Memory::Host { pinned: true }).unwrap(),
        );
        let (_, gradient) = DomainTracingContext::<Session>::trace(
            |x: Tracer| Ok(differentiate_at(x).gradient(|x| function.call(x))?),
            Type::from(scalar_type.clone()),
        )
        .unwrap();
        assert_eq!(
            to_mlir_module_for_program(&gradient, &[], &scalar_type, &scalar_type, "main", None, None).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.sine %arg0 : tensor<f32>
                    %1 = stablehlo.cosine %0 : tensor<f32>
                    %2 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<f32>) -> tensor<f32>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %3:3 = stablehlo.optimization_barrier %cst, %arg0, %2 : tensor<f32>, tensor<f32>, tensor<f32>
                    %4 = stablehlo.negate %3#0 : tensor<f32>
                    %5 = stablehlo.custom_call @annotate_device_placement(%3#2) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "device"}} : (tensor<f32>) -> tensor<f32>
                    %6 = stablehlo.sine %5 : tensor<f32>
                    %7 = stablehlo.multiply %6, %4 : tensor<f32>
                    %8 = stablehlo.cosine %3#1 : tensor<f32>
                    %9 = stablehlo.multiply %8, %7 : tensor<f32>
                    return %9 : tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_linear_call_operation_lower_inlines_only_the_forward_program() {
        let vector_type = test_vector_type(4);
        let forward = {
            let mut builder = ArrayIrProgramBuilder::new();
            let tangent = builder.add_input(vector_type.clone().into());
            let residual = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![tangent, residual], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transpose = {
            let mut builder = ArrayIrProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone().into());
        let residual = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::LinearCall(LinearCallOperation::new(1)),
                vec![forward, transpose],
                vec![tangent, residual],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[vector_type.clone(), vector_type.clone()],
            &[vector_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_linear_call_operation_lower_rejects_malformed_instructions() {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();

        // The plain array family cannot carry the forward and transpose regions.
        let mut lowerer = Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram>::new());
        let expected_message =
            "higher-order operation `linear_call` must be stored directly in the enclosing backend operation family";
        assert!(matches!(
            LinearCallOperation::<ArrayType>::new(1).lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == expected_message,
            ),
        ));

        // An array IR call needs exactly its forward and transpose regions.
        assert!(matches!(
            LinearCallOperation::<Type>::new(1).lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "linear_call" && reason == "`linear_call` expected 2 attached regions but got 0",
        ));
    }
}
