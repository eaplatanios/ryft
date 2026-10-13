use std::rc::Rc;

use ryft_core::{
    ArrayIrType, ArrayType, ComparisonDirection, DataType, Operation, ProgramError, ReductionKind,
    RematerializationOptimizationBarrier, WHILE_OPERATION_NAME, WhileOperation,
};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::{
    lower_array_ir_type, lower_explicit_shape, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, physical_bound_type,
};
use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, current_or_new_token, token_threaded_effects,
};
use crate::experimental::ops::FlatXlaProgram;
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::control_flow::{
    lower_control_flow_region, lower_refined_region_input, lower_refined_region_outputs, lower_region_guard,
    lowered_effect_classes,
};
use crate::operations::jit_call::JitCallFunctionMap;
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

pub(crate) fn lower_while_to_while<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    while_op: &WhileOperation<ArrayIrType>,
    loop_regions: &[FlatXlaProgram<O>],
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayIrType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let [condition, body] = loop_regions else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} expected 2 attached regions but got {}", WHILE_OPERATION_NAME, loop_regions.len()),
        });
    };
    let state_types = body.input_types();
    let state_count = state_types.len();
    if input_values.len() != state_count || input_types.len() != state_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "`{}` expected {} lowered inputs but got {} lowered inputs with {} input types",
                WHILE_OPERATION_NAME,
                state_count,
                input_values.len(),
                input_types.len(),
            ),
        });
    }
    // The condition and the body share the loop state's declared types. Guarding the initial entry before the loop
    // checks what eager entry validation checks, because later iterations enter with the body's own typed outputs.
    lower_region_guard(
        WHILE_OPERATION_NAME,
        &state_types,
        input_types,
        input_values,
        effect_tokens,
        block,
        context,
        location,
    )?;
    // Inputs only need to refine the state types, so each one is converted to the lowered type of its state first.
    let input_values = input_values
        .iter()
        .zip(input_types)
        .zip(state_types.iter())
        .map(|((value, input_type), state_type)| {
            lower_refined_region_input(*value, input_type, state_type, block, context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_values = input_values.as_slice();
    // A batched (non-scalar) predicate lowers with the masked semantics owned by this primitive, mirroring JAX's
    // `_while_lowering`. Rather than re-evaluating the condition in both regions, the per-item predicate is threaded
    // through the loop state as one extra carried value (JAX's `(pred, ..., carry)` layout): the condition region
    // only reduces the carried predicate with a Boolean `or` into the scalar continuation decision, and the body
    // region selects per state element between the body's candidate update and the carried (incoming-state)
    // predicate, then recomputes the predicate on the updated state for the next iteration. The condition is
    // therefore evaluated exactly once per iteration (plus once before the loop to seed the initial predicate),
    // never twice. A batched-predicate loop is always pure (the while signature validation rejects effects in a
    // batched-predicate loop, since observable effects cannot be masked for finished items), so predicate threading
    // and token threading never coexist.
    let condition_output_types = condition.output_types();
    let predicate_type = <&ArrayType>::try_from(&condition_output_types[0]).map_err(ProgramError::from)?.clone();
    let batched_predicate = predicate_type.rank() > 0;
    // StableHLO's while-condition region can return only the continuation predicate, not updated effect tokens.
    // Therefore, an effectful scalar condition is evaluated before the loop and after each body execution, with its
    // predicate carried through the loop state. This preserves exactly-once condition effects and lets the body
    // return their updated tokens. Batched predicates already use the same carried-predicate shape for masking.
    let condition_effects = lowered_effect_classes(condition)?;
    let iteration_bound = while_op.iteration_bound();
    // Bounded loops also thread pure predicates so reaching the bound skips the condition computation entirely,
    // including any nested pure loop that would otherwise diverge on the final state.
    let threaded_predicate = batched_predicate || !condition_effects.is_empty() || iteration_bound.is_some();
    let predicate_dimensions = (0..predicate_type.rank()).collect::<Vec<_>>();
    let predicate_offset = if threaded_predicate { 1 } else { 0 };
    // A semantic iteration bound is enforced by threading an internal `i64` iteration counter through the
    // `stablehlo.while` state (element 0, starting at zero and incremented once per body run) and conjoining
    // `counter < bound` into the lowered condition. The counter is internal extra state: the operation's outputs
    // remain exactly the original state elements. Unbounded loops emit no counter machinery at all.
    let counter_offset = if iteration_bound.is_some() { 1 } else { 0 };
    // Carry one trailing token for each ordered class used by either nested program. Each class advances independently
    // through the body; a body that is pure for one active class returns that class's entry token unchanged.
    let threaded_effects = condition_effects.union(lowered_effect_classes(body)?);
    let threaded_effect_count = token_threaded_effects(threaded_effects).count();
    // State layout: `[counter?, states..., predicate?, ordered-effect tokens...]`.
    let predicate_index = counter_offset + state_count;
    let token_start_index = counter_offset + state_count + predicate_offset;
    let mut full_state_types = Vec::with_capacity(counter_offset + state_count + predicate_offset);
    if iteration_bound.is_some() {
        full_state_types.push(ArrayType::scalar(DataType::I64).into());
    }
    full_state_types.extend(state_types.iter().cloned());
    if threaded_predicate {
        full_state_types.push(predicate_type.clone().into());
    }
    let mut lowered_state_types = full_state_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    for _ in 0..threaded_effect_count {
        lowered_state_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    // Seed the loop state. A threaded predicate is evaluated once on the entry state in the enclosing block. For an
    // effectful scalar condition, this also advances its ordered effect chains before those tokens enter the loop.
    let mut state_values = Vec::with_capacity(lowered_state_types.len());
    if iteration_bound.is_some() {
        state_values.push(lower_static_index_constants(&[0], block, context, location)?[0]);
    }
    state_values.extend_from_slice(input_values);
    if threaded_predicate {
        let initial_predicate = lower_nested_program_inline(
            condition,
            input_values,
            block,
            context,
            location,
            captured_values,
            &RematerializationOptimizationBarrier::None,
            nested_functions,
            collective_state,
            effect_tokens,
        )?;
        if initial_predicate.len() != 1 {
            return Err(LoweringError::UnsupportedOp {
                op: format!("{} condition lowered to {} outputs", WHILE_OPERATION_NAME, initial_predicate.len(),),
            });
        }
        state_values.push(initial_predicate[0]);
    }
    for effect in token_threaded_effects(threaded_effects) {
        state_values.push(current_or_new_token(effect, effect_tokens, block, location)?);
    }

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        // A threaded predicate is read from the loop state. Batched predicates are reduced with Boolean `or`, so the
        // loop continues while any mapped item remains active; effectful scalar predicates are already scalar. Pure
        // unbounded scalar predicates retain the smaller representation and are evaluated directly in this region.
        let loop_predicate = if threaded_predicate {
            let carried_predicate = condition_block_ref
                .argument(predicate_index)
                .expect("predicate-threaded while state should include the carried predicate")
                .as_ref();
            if batched_predicate {
                lower_reduce_to_mlir(
                    ReductionKind::Any,
                    predicate_dimensions.as_slice(),
                    carried_predicate,
                    &ArrayType::scalar(DataType::Boolean),
                    &mut condition_block_ref,
                    context,
                    location,
                )?
            } else {
                carried_predicate
            }
        } else {
            let condition_inputs = (counter_offset..counter_offset + state_count)
                .map(|index| {
                    condition_block_ref.argument(index).expect("while condition should have state arguments").as_ref()
                })
                .collect::<Vec<_>>();
            let mut condition_effect_tokens = EffectTokens::default();
            for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
                condition_effect_tokens.set(
                    effect,
                    condition_block_ref
                        .argument(token_start_index + token_offset)
                        .expect("token-threaded while state should include every active effect token")
                        .as_ref(),
                );
            }
            let condition_outputs = lower_nested_program_inline(
                condition,
                condition_inputs.as_slice(),
                &mut condition_block_ref,
                context,
                location,
                captured_values,
                &RematerializationOptimizationBarrier::None,
                nested_functions,
                collective_state,
                &mut condition_effect_tokens,
            )?;
            if condition_outputs.len() != 1 {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{} condition lowered to {} outputs", WHILE_OPERATION_NAME, condition_outputs.len(),),
                });
            }
            condition_outputs[0]
        };
        let predicate = match iteration_bound {
            Some(bound) => {
                let counter =
                    condition_block_ref.argument(0).expect("bounded while state should include the counter").as_ref();
                let bound_constant =
                    lower_static_index_constants(&[bound], &mut condition_block_ref, context, location)?[0];
                let counter_predicate = lower_compare_to_mlir(
                    ComparisonDirection::LessThan,
                    counter,
                    bound_constant,
                    &mut condition_block_ref,
                    location,
                )?;
                let fused = condition_block_ref.append_operation(stable_hlo::and(
                    loop_predicate,
                    counter_predicate,
                    location,
                )?)?;
                fused.result(0).expect("stablehlo.and should return one result").as_ref()
            }
            None => loop_predicate,
        };
        condition_block_ref.append_operation(stable_hlo::r#return(&[predicate], location)?)?;
    }
    condition_region.append_block(condition_block)?;

    let mut body_region = context.region();
    let body_block = context.block(block_arguments.as_slice());
    {
        let mut body_block_ref = body_block.as_ref();
        let body_inputs = (counter_offset..counter_offset + state_count)
            .map(|index| body_block_ref.argument(index).expect("while body should have state arguments").as_ref())
            .collect::<Vec<_>>();
        let mut body_effect_tokens = EffectTokens::default();
        for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
            body_effect_tokens.set(
                effect,
                body_block_ref
                    .argument(token_start_index + token_offset)
                    .expect("token-threaded while state should include every active effect token")
                    .as_ref(),
            );
        }
        let body_outputs = lower_nested_program_inline(
            body,
            body_inputs.as_slice(),
            &mut body_block_ref,
            context,
            location,
            captured_values,
            &RematerializationOptimizationBarrier::None,
            nested_functions,
            collective_state,
            &mut body_effect_tokens,
        )?;
        if body_outputs.len() != state_count {
            return Err(LoweringError::UnsupportedOp {
                op: format!("{} body lowered to {} outputs", WHILE_OPERATION_NAME, body_outputs.len()),
            });
        }
        // For a batched predicate, mask each carry update under the carried (incoming-state) predicate so finished
        // items freeze. A scalar loop uses the body outputs directly.
        let next_state_values = if batched_predicate {
            let carried_predicate = body_block_ref
                .argument(predicate_index)
                .expect("batched-predicate while state should include the carried predicate")
                .as_ref();
            let masked = body_outputs
                .into_iter()
                .zip(body_inputs.iter())
                .zip(state_types.iter())
                .map(|((candidate, carried), state_type)| {
                    // A first-class dimension carry is loop-invariant under a batched predicate (the contract
                    // documented on `WhileType`), so masking it is the identity and the body's candidate result is
                    // threaded on directly. This matches eager interpretation, whose `mask_select` returns equal
                    // dimension carries unchanged.
                    if matches!(state_type, ArrayIrType::Dimension(_)) {
                        return Ok(candidate);
                    }
                    let state_type = <&ArrayType>::try_from(state_type).map_err(ProgramError::from)?;
                    // The predicate broadcasts to each state element's shape along its leading (prefix) axes; a
                    // state element already shaped like the predicate reuses it directly.
                    let element_mask = if state_type.shape() == predicate_type.shape() {
                        carried_predicate
                    } else {
                        let mask_type = ArrayType::new(DataType::Boolean, state_type.shape().clone());
                        if state_type.static_shape().is_some() {
                            let broadcast = body_block_ref.append_operation(stable_hlo::broadcast(
                                carried_predicate,
                                lower_tensor_type(&mask_type, context, location)?,
                                predicate_dimensions.as_slice(),
                                location,
                            )?)?;
                            broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()
                        } else if let Ok(physical_type) = physical_bound_type(&mask_type) {
                            // Broadcast bounded storage, then restore every logical extent from the carry. A
                            // predicate cannot supply the dynamic sizes of trailing state axes, and native bounded
                            // broadcast results admit at most one dynamic axis.
                            let predicate = lower_physical_bound_value(
                                carried_predicate,
                                &predicate_type,
                                0.0,
                                &mut body_block_ref,
                                context,
                                location,
                            )?;
                            let broadcast = body_block_ref.append_operation(stable_hlo::broadcast(
                                predicate,
                                lower_tensor_type(&physical_type, context, location)?,
                                predicate_dimensions.as_slice(),
                                location,
                            )?)?;
                            let sources = (0..state_type.rank()).map(|axis| (*carried, axis)).collect::<Vec<_>>();
                            lower_restore_dynamic_dimensions(
                                broadcast.result(0).unwrap().as_ref(),
                                &mask_type,
                                &sources,
                                &mut body_block_ref,
                                context,
                                location,
                            )?
                        } else {
                            let extents = (0..state_type.rank())
                                .map(|axis| {
                                    lower_runtime_dimension_size_i64(
                                        *carried,
                                        axis,
                                        &mut body_block_ref,
                                        context,
                                        location,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let shape = lower_explicit_shape(&extents, &mut body_block_ref, context, location)?;
                            let broadcast = body_block_ref.append_operation(stable_hlo::dynamic_broadcast(
                                carried_predicate,
                                shape,
                                predicate_dimensions.as_slice(),
                                None,
                                None,
                                location,
                            )?)?;
                            let result = broadcast.result(0).unwrap().as_ref();
                            let expected_type = lower_tensor_type(&mask_type, context, location)?;
                            if result.r#type()? == expected_type.as_ref() {
                                result
                            } else {
                                body_block_ref
                                    .append_operation(tensor::cast(result, expected_type, location)?)?
                                    .result(0)
                                    .unwrap()
                                    .as_ref()
                            }
                        }
                    };
                    let select = body_block_ref.append_operation(stable_hlo::select(
                        element_mask,
                        candidate,
                        *carried,
                        location,
                    )?)?;
                    Ok(select.result(0).expect("stablehlo.select should return one result").as_ref())
                })
                .collect::<Result<Vec<_>, LoweringError>>()?;
            masked
        } else {
            body_outputs
        };
        let next_counter = if iteration_bound.is_some() {
            let counter = body_block_ref.argument(0).expect("bounded while state should include the counter").as_ref();
            let one = lower_static_index_constants(&[1], &mut body_block_ref, context, location)?[0];
            let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, one, location)?)?;
            Some(next_counter.result(0).expect("stablehlo.add should return one result").as_ref())
        } else {
            None
        };
        // Recompute a threaded predicate after the body. At the semantic bound, the condition must not run again:
        // eager interpretation checks the bound before evaluating it. An extra evaluation could cause observable
        // effects or diverge even when the condition is pure.
        // The guarded branch returns the predicate and every active token, forwarding the incoming tokens unchanged
        // when the bound is reached. The carried predicate is immaterial then because the counter ends the loop.
        let next_predicate = if threaded_predicate {
            if let Some(bound) = iteration_bound {
                let bound_constant = lower_static_index_constants(&[bound], &mut body_block_ref, context, location)?[0];
                let evaluate_condition = lower_compare_to_mlir(
                    ComparisonDirection::LessThan,
                    next_counter.unwrap(),
                    bound_constant,
                    &mut body_block_ref,
                    location,
                )?;
                let condition_region = lower_control_flow_region(
                    condition,
                    next_state_values.as_slice(),
                    &condition_output_types,
                    context,
                    location,
                    captured_values,
                    nested_functions,
                    collective_state,
                    body_effect_tokens,
                    threaded_effects,
                )?;
                let mut finished_region = context.region();
                let finished_block = context.block_with_no_arguments();
                {
                    let mut finished_block_ref = finished_block.as_ref();
                    let mut finished_outputs = vec![
                        body_block_ref
                            .argument(predicate_index)
                            .expect("predicate-threaded while state should include the carried predicate")
                            .as_ref(),
                    ];
                    for effect in token_threaded_effects(threaded_effects) {
                        finished_outputs.push(
                            body_effect_tokens
                                .get(effect)
                                .expect("token-threaded while bodies receive every active effect token"),
                        );
                    }
                    finished_block_ref
                        .append_operation(stable_hlo::r#return(finished_outputs.as_slice(), location)?)?;
                }
                finished_region.append_block(finished_block)?;
                let guarded_condition = body_block_ref.append_operation(stable_hlo::r#if(
                    evaluate_condition,
                    condition_region.into(),
                    finished_region.into(),
                    location,
                )?)?;
                for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
                    body_effect_tokens.set(
                        effect,
                        guarded_condition
                            .result(1 + token_offset)
                            .expect("a guarded while condition returns every active effect token")
                            .as_ref(),
                    );
                }
                Some(guarded_condition.result(0).expect("a guarded while condition returns its predicate").as_ref())
            } else {
                let next_predicate = lower_nested_program_inline(
                    condition,
                    next_state_values.as_slice(),
                    &mut body_block_ref,
                    context,
                    location,
                    captured_values,
                    &RematerializationOptimizationBarrier::None,
                    nested_functions,
                    collective_state,
                    &mut body_effect_tokens,
                )?;
                if next_predicate.len() != 1 {
                    return Err(LoweringError::UnsupportedOp {
                        op: format!("{} condition lowered to {} outputs", WHILE_OPERATION_NAME, next_predicate.len(),),
                    });
                }
                Some(next_predicate[0])
            }
        } else {
            None
        };
        let mut next_state = Vec::with_capacity(lowered_state_types.len());
        if let Some(next_counter) = next_counter {
            next_state.push(next_counter);
        }
        next_state.extend(next_state_values);
        if let Some(next_predicate) = next_predicate {
            next_state.push(next_predicate);
        }
        for effect in token_threaded_effects(threaded_effects) {
            next_state.push(
                body_effect_tokens
                    .get(effect)
                    .expect("token-threaded while bodies receive every active effect token"),
            );
        }
        body_block_ref.append_operation(stable_hlo::r#return(next_state.as_slice(), location)?)?;
    }
    body_region.append_block(body_block)?;

    let operation = block.append_operation(stable_hlo::r#while(
        state_values.as_slice(),
        condition_region.into(),
        body_region.into(),
        location,
    )?)?;
    for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
        effect_tokens.set(
            effect,
            operation
                .result(token_start_index + token_offset)
                .expect("a token-threaded stablehlo.while should return one result per active effect class")
                .as_ref(),
        );
    }
    Ok((0..state_types.len())
        .map(|index| {
            operation
                .result(counter_offset + index)
                .expect("stablehlo.while should return one result per state leaf")
                .as_ref()
        })
        .collect())
}

impl LowerableOperation for WhileOperation<ArrayType> {
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

impl LowerableOperation for WhileOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let results = lower_while_to_while(
            self,
            lowerer.regions.as_slice(),
            input_values,
            input_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.captured_values.as_slice(),
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
        )?;
        let region_output_types = lowerer.regions.get(1).map(|body| body.input_types()).unwrap_or_default();
        lower_refined_region_outputs(
            results,
            region_output_types.as_slice(),
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, CompareOperation, Dimension, DimensionBounds, DimensionSizeOperation,
        DimensionType, DimensionVariable, DynamicBroadcastOperation, Placeholder, Shape, SubOperation,
    };

    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, unproject_plain_program, xla_identity_branch,
    };
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::tests::execution_client;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_program_lowers_while_to_stablehlo_while() {
        let state_type = ArrayType::scalar(DataType::Boolean);
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let body_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let state = builder.add_input(state_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
        // An unbounded while emits no iteration-counter machinery.
        assert!(!stablehlo.contains("stablehlo.and"), "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.add"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_threads_effectful_while_condition_once_per_evaluation() {
        use ryft_core::{AssertOperation, PrintOperation};

        // StableHLO condition regions cannot return tokens. The lowering therefore evaluates an effectful scalar
        // condition once before the loop and once after each body execution, carrying its predicate through the loop
        // state so assertions and prints remain on their independent enclosing effect chains.
        let state_type = ArrayType::scalar(DataType::Boolean);
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("condition must be true"),
                    Vec::new(),
                    vec![state],
                    None,
                )
                .unwrap();
            let predicate = builder
                .add_instruction(PrintOperation::<ArrayType>::new("condition"), Vec::new(), vec![state], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let state = builder.add_input(state_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(1).unwrap()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert_eq!(stablehlo.matches("stablehlo.after_all").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.print").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 2, "{stablehlo}");
        // The second condition site is guarded so reaching the semantic bound never executes its effects.
        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        let condition_region = stablehlo.split_once("cond {").unwrap().1.split_once("} do {").unwrap().0;
        assert!(!condition_region.contains("@ryft.print"), "{stablehlo}");
        assert!(!condition_region.contains("@ryft.assert"), "{stablehlo}");
        let while_header = stablehlo.lines().find(|line| line.contains("stablehlo.while(")).unwrap();
        assert!(while_header.contains("tensor<i1>, !stablehlo.token"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_loop_carried_dimension_extent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let extent = builder.add_input(extent_type.clone().into());
            let predicate = builder
                .add_instruction(
                    XlaOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                    Vec::new(),
                    vec![extent, extent],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let extent = builder.add_input(extent_type.into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![extent], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone().into());
        let scalar = builder.add_input(scalar_type.clone().into());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let carried_extent = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![extent],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, carried_extent], None)
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
            &vec![dynamic_vector_type.clone(), scalar_type],
            &vec![dynamic_vector_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.get_dimension_size"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.set_dimension_size"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_bounded_while_with_fused_counter_condition() {
        // A semantic iteration bound threads an internal i64 counter through the `stablehlo.while` state: the
        // condition region conjoins `counter < bound` into the original predicate via `stablehlo.compare` plus
        // `stablehlo.and`, and the body region increments the counter via `stablehlo.add`. The operation's outputs
        // remain the original state elements.
        let state_type = ArrayType::scalar(DataType::Boolean);
        let while_operation = WhileOperation::new().with_iteration_bound(3).unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let body_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let state = builder.add_input(state_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::While(while_operation),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.and"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.add"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<i64>"), "{stablehlo}");
        // Even a pure scalar condition is carried and recomputed only below the bound. The condition region reads
        // its carried predicate, while the body guards the next condition evaluation with a nested branch.
        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        let condition = stablehlo.split_once("cond {").unwrap().1.split_once("} do {").unwrap().0;
        assert_eq!(condition.matches("stablehlo.compare").count(), 1, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_batched_predicate_while_with_masked_state_updates() {
        // A batched (per-item) predicate lowers with the masked semantics owned by the while lowering: the condition
        // region reduces the `tensor<3xi1>` predicate to the scalar loop-continuation decision with a Boolean `or`
        // reduction, and the body region recomputes the per-item predicate on the incoming state and selects per
        // state element between the body's candidate update and the carried state, freezing finished items. The
        // predicate shape equals the state shape here, so no broadcast is needed for the mask.
        use ryft_core::{CompareOperation, OneLikeOperation, ZeroLikeOperation};
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        // Condition region: `or`-reduce the per-item predicate into the scalar continuation decision.
        assert!(stablehlo.contains("stablehlo.reduce"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.or"), "{stablehlo}");
        // Body region: per-item masked state update under the recomputed predicate.
        assert!(stablehlo.contains("stablehlo.select"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3xi1>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_broadcasts_while_masks_over_dynamic_state_axes() {
        for upper_bound in [Some(5), None] {
            let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)]));
            let state_type = ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Static(2),
                    dynamic_dimension("rows", upper_bound),
                    dynamic_dimension("columns", upper_bound),
                ]),
            );
            let condition = {
                let mut builder = CompositeXlaProgramBuilder::new();
                let predicate = builder.add_input(predicate_type.clone().into());
                builder.add_input(state_type.clone().into());
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                        vec![predicate],
                        vec![Placeholder, Placeholder],
                        vec![Placeholder],
                    )
                    .unwrap()
            };
            let body = {
                let mut builder = CompositeXlaProgramBuilder::new();
                let predicate = builder.add_input(predicate_type.clone().into());
                let state = builder.add_input(state_type.clone().into());
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                        vec![predicate, state],
                        vec![Placeholder, Placeholder],
                        vec![Placeholder, Placeholder],
                    )
                    .unwrap()
            };
            let mut builder = CompositeXlaProgramBuilder::new();
            let condition = builder.import_region(condition.entry_region_ref());
            let body = builder.import_region(body.entry_region_ref());
            let predicate = builder.add_input(predicate_type.clone().into());
            let state = builder.add_input(state_type.clone().into());
            let output = builder
                .add_instruction(
                    XlaOperation::While(WhileOperation::new().with_iteration_bound(2).unwrap()),
                    vec![condition, body],
                    vec![predicate, state],
                    None,
                )
                .unwrap()[1];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![predicate_type, state_type.clone()],
                &vec![state_type],
                "main",
                None,
                None,
            )
            .unwrap();

            assert!(module.contains("stablehlo.select"), "{module}");
            if upper_bound.is_some() {
                // Both dynamic mask axes inherit their actual extents from the state, rather than its capacity.
                let body = module.split_once("} do {").unwrap().1;
                assert_eq!(body.matches("stablehlo.set_dimension_size").count(), 2, "{module}");
                assert!(!body.contains("stablehlo.dynamic_broadcast_in_dim"), "{module}");
            } else {
                assert_eq!(module.matches("stablehlo.dynamic_broadcast_in_dim").count(), 1, "{module}");
            }
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_passes_loop_invariant_dimension_through_masked_batched_predicate_while() {
        // A batched-predicate loop may also carry a first-class dimension, which the relaxed `WhileType` contract
        // requires to be loop-invariant. Masking a loop-invariant carry is the identity, so the lowering threads the
        // body's dimension result on directly: only the array carry gets a `stablehlo.select`, while the condition
        // region still `or`-reduces the per-item predicate into the scalar continuation decision.
        use ryft_core::{CompareOperation, OneLikeOperation, ZeroLikeOperation};
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(extent_type.clone().into());
            let state = builder.add_input(state_type.clone().into());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![predicate],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let carried_extent = builder.add_input(extent_type.into());
            let state = builder.add_input(state_type.clone().into());
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![carried_extent, next],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone().into());
        let state = builder.add_input(state_type.clone().into());
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
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![extent, state],
                None,
            )
            .unwrap()[1];
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
            &vec![dynamic_vector_type, state_type.clone()],
            &state_type,
            "main",
            None,
            None,
        )
        .unwrap();

        // The condition region `or`-reduces the carried `tensor<3xi1>` predicate, the array carry takes one
        // `stablehlo.select` against the carried predicate, and the dimension carry (`%iterArg`) is returned
        // unmasked exactly as the body produced it.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<3xf64>, %arg2: tensor<i32>) -> tensor<3xf64> {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
                    %4 = stablehlo.compare GT, %arg1, %3, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
                    %5:3 = stablehlo.while(%iterArg = %2, %iterArg_0 = %arg1, %iterArg_1 = %4) : tensor<i64>, tensor<3xf64>, tensor<3xi1>
                    cond {
                      %c = stablehlo.constant dense<false> : tensor<i1>
                      %6 = stablehlo.reduce(%iterArg_1 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                      %6 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
                      %7 = stablehlo.subtract %iterArg_0, %6 : tensor<3xf64>
                      %8 = stablehlo.select %iterArg_1, %7, %iterArg_0 : tensor<3xi1>, tensor<3xf64>
                      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %9 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
                      %10 = stablehlo.compare GT, %8, %9, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
                      stablehlo.return %iterArg, %8, %10 : tensor<i64>, tensor<3xf64>, tensor<3xi1>
                    }
                    return %5#1 : tensor<3xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_while_converts_a_refined_static_state_input_to_its_bounded_state_type() {
        // A loop input only needs to refine its state type, so a static `f64[3]` initial state feeds a loop state with
        // the bounded dynamic type `f64[rows]`. The static input is padded to the bound of the state type and given its
        // extent as its runtime size before it enters the loop. Because `rows` is loop-invariant, the outputs take its
        // static extent, so the bounded loop results are sliced back to `f64[3]`. One iteration computes `[a, a + b]`.
        // The dynamic input leaves `rows = 3` unproven, so an entry guard checks its runtime size before the loop.
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(state_type.clone().into());
            builder.add_input(state_type.clone().into());
            let predicate = builder.add_constant(XlaConstant::Boolean(true));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let first = builder.add_input(state_type.clone().into());
            let second = builder.add_input(state_type.clone().into());
            let sum = builder
                .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![first, second], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![first, sum],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let dynamic_input = builder.add_input(state_type.clone().into());
        let static_input = builder.add_input(static_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(1).unwrap()),
                vec![condition, body],
                vec![dynamic_input, static_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.output_types(),
            vec![ArrayIrType::Array(static_type.clone()), ArrayIrType::Array(static_type.clone())],
        );

        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![state_type, static_type.clone()],
            &vec![static_type.clone(), static_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf64>, %arg1: tensor<3xf64>, %arg2: tensor<i32>) -> (tensor<3xf64>, tensor<3xf64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<4xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<4>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %3 = stablehlo.compare EQ, %c, %2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %4 = stablehlo.after_all  : !stablehlo.token
                    %5 = stablehlo.custom_call @ryft.assert(%3, %2, %c, %4) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `rows` must have one extent at the entry of `while`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %c_0 = stablehlo.constant dense<3> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %6 = stablehlo.pad %arg1, %cst, low = [0], high = [1], interior = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<4xf64>
                    %7 = stablehlo.set_dimension_size %6, %c_0, dim = 0 : (tensor<4xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<4>>
                    %c_1 = stablehlo.constant dense<0> : tensor<i64>
                    %c_2 = stablehlo.constant dense<true> : tensor<i1>
                    %8:4 = stablehlo.while(%iterArg = %c_1, %iterArg_5 = %0, %iterArg_6 = %7, %iterArg_7 = %c_2) : tensor<i64>, tensor<?xf64, #stablehlo.bounds<4>>, tensor<?xf64, #stablehlo.bounds<4>>, tensor<i1>
                    cond {
                      %c_8 = stablehlo.constant dense<1> : tensor<i64>
                      %13 = stablehlo.compare LT, %iterArg, %c_8, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %14 = stablehlo.and %iterArg_7, %13 : tensor<i1>
                      stablehlo.return %14 : tensor<i1>
                    } do {
                      %13 = stablehlo.add %iterArg_5, %iterArg_6 : tensor<?xf64, #stablehlo.bounds<4>>
                      %c_8 = stablehlo.constant dense<1> : tensor<i64>
                      %14 = stablehlo.add %iterArg, %c_8 : tensor<i64>
                      %c_9 = stablehlo.constant dense<1> : tensor<i64>
                      %15 = stablehlo.compare LT, %14, %c_9, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %16 = "stablehlo.if"(%15) ({
                        %c_10 = stablehlo.constant dense<true> : tensor<i1>
                        stablehlo.return %c_10 : tensor<i1>
                      }, {
                        stablehlo.return %iterArg_7 : tensor<i1>
                      }) : (tensor<i1>) -> tensor<i1>
                      stablehlo.return %14, %iterArg_5, %13, %16 : tensor<i64>, tensor<?xf64, #stablehlo.bounds<4>>, tensor<?xf64, #stablehlo.bounds<4>>, tensor<i1>
                    }
                    %c_3 = stablehlo.constant dense<4> : tensor<i32>
                    %9 = stablehlo.set_dimension_size %8#1, %c_3, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>, tensor<i32>) -> tensor<4xf64>
                    %10 = stablehlo.slice %9 [0:3] : (tensor<4xf64>) -> tensor<3xf64>
                    %c_4 = stablehlo.constant dense<4> : tensor<i32>
                    %11 = stablehlo.set_dimension_size %8#2, %c_4, dim = 0 : (tensor<?xf64, #stablehlo.bounds<4>>, tensor<i32>) -> tensor<4xf64>
                    %12 = stablehlo.slice %11 [0:3] : (tensor<4xf64>) -> tensor<3xf64>
                    return %10, %12 : tensor<3xf64>, tensor<3xf64>
                  }
                }
            "#},
        );
        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[
                    MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]),
                    MixedValue::Array(vec![10.0, 20.0, 30.0], vec![3])
                ],
                &[3],
            ),
            Ok(vec![
                MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                MixedValue::Array(vec![11.0, 22.0, 33.0], vec![3]),
            ]),
        );
        let error = execute_mixed_program(
            &client,
            &program,
            &[MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]), MixedValue::Array(vec![10.0, 20.0, 30.0], vec![3])],
            &[2],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("input dimension `rows` must have one extent at the entry of `while`"), "{error}");
    }

    #[test]
    fn test_while_converts_a_partially_refined_state_input_preserving_its_dynamic_axes() {
        // The second state has the type `f64[rows, cols]` and is fed an `f64[rows, 3]` input, so only `cols` is refined
        // while `rows` stays dynamic. The input conversion keeps the runtime size of `rows` and gives `cols` its static
        // extent, and the outputs take the type `f64[rows, 3]`, so the results are sliced along `cols` while keeping
        // the runtime size of `rows`. One bounded iteration computes `[a, a + b]`.
        let state_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![dynamic_dimension("rows", Some(5)), dynamic_dimension("cols", Some(5))]),
        );
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![state_type.shape().dimensions()[0].clone(), Dimension::Static(3)]),
        );
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(state_type.clone().into());
            builder.add_input(state_type.clone().into());
            let predicate = builder.add_constant(XlaConstant::Boolean(true));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let first = builder.add_input(state_type.clone().into());
            let second = builder.add_input(state_type.clone().into());
            let sum = builder
                .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![first, second], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![first, sum],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let dynamic_input = builder.add_input(state_type.clone().into());
        let refined_input = builder.add_input(input_type.into());
        let outputs = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(1).unwrap()),
                vec![condition, body],
                vec![dynamic_input, refined_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let first = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let second = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[MixedValue::Array(first.to_vec(), vec![4, 4]), MixedValue::Array(second.to_vec(), vec![4, 3])],
                &[2, 3, 2],
            ),
            Ok(vec![
                MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                MixedValue::Array(vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0], vec![2, 3]),
                MixedValue::Dimension(2),
                MixedValue::Dimension(2),
            ]),
        );
    }

    #[test]
    fn test_batched_predicate_while_executes_with_masked_semantics_on_cpu() {
        use std::sync::Arc;

        use ryft_core::{CompareOperation, OneLikeOperation, ZeroLikeOperation};
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::tests::{values_from_bytes, values_to_bytes};

        // End-to-end proof of the threaded-predicate lowering on the real CPU plugin: a per-item countdown
        // `while (x > 0) { x - 1 }` over `f64[3]` carries the batched `bool[3]` predicate through the loop state, so
        // the condition is evaluated once per iteration (seeded before the loop, recomputed in the body) rather than
        // twice. Items [3, 1, 2] terminate after 3, 1, and 2 iterations and all land at 0; a masking bug (or a
        // rejoining finished item) would leave nonzero or negative entries.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![state_type.clone()];
        let output_types = vec![state_type.clone()];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            individually_defined_output_indices: Vec::new(),
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: std::collections::HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_bytes = values_to_bytes::<f64>(&[3.0, 1.0, 2.0]);
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F64, &[3], None, device.clone(), None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bounded_while_condition_effects_stop_at_bound_on_cpu() {
        use std::sync::Arc;

        use ryft_core::{CompareOperation, OneLikeOperation, PrintOperation, ZeroLikeOperation};
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
        use crate::tests::{values_from_bytes, values_to_bytes};

        // A false entry predicate runs the condition once, natural termination runs it once more than the body,
        // and truncation at the bound runs it only before each permitted body execution.
        let state_type = ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let state = builder
                .add_instruction(PrintOperation::<ArrayType>::new("condition"), Vec::new(), vec![state], None)
                .unwrap()[0];
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            let next = builder
                .add_instruction(PrintOperation::<ArrayType>::new("body"), Vec::new(), vec![next], None)
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type.clone().into());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(2).unwrap()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![state_type.clone()];
        let output_types = vec![state_type.clone()];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            individually_defined_output_indices: Vec::new(),
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: std::collections::HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        for (input, expected, expected_lines) in [
            (-1.0, -1.0, vec!["condition: -1.0"]),
            (1.0, 0.0, vec!["condition: 1.0", "body: 0.0", "condition: 0.0"]),
            (5.0, 3.0, vec!["condition: 5.0", "body: 4.0", "condition: 4.0", "body: 3.0"]),
        ] {
            let input_bytes = values_to_bytes::<f64>(&[input]);
            let (output, lines) = with_captured_prints(|| {
                let inputs = ExecutionDeviceInputs {
                    inputs: &[ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(input_bytes.as_slice(), BufferType::F64, &[], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    }],
                    ..Default::default()
                };
                let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
                let mut outputs = execution.block_until_ready().unwrap().remove(0);
                assert_eq!(outputs.outputs.len(), 1);
                let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f64>(output_bytes.as_slice())
            });
            assert_eq!(output, vec![expected]);
            assert_eq!(lines, expected_lines);
        }
    }
}
