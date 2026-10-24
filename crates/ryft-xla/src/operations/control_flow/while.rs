//! Lowering of [`WhileOperation`] to [StableHLO `while`](https://openxla.org/stablehlo/spec#while).
//!
//! A while loop lowers only from the composite [`ArrayIrType`] operation family, which stores its condition and body
//! regions; the homogeneous [`ArrayType`] payload carries no regions and is rejected. The loop state of the lowered
//! `stablehlo.while` is `[counter?, states..., predicate?, tokens...]`: an `i64` iteration counter when the loop has an
//! iteration bound, the loop states, the carried predicate when the loop threads its predicate through its state, and
//! one token per ordered effect class that either region uses. The predicate is threaded for a batched (non-scalar)
//! predicate, which lowers with masked state updates, for an effectful condition, whose effects a `stablehlo.while`
//! condition region cannot thread, and for a bounded loop, whose condition must not run once the bound is reached. A
//! threaded predicate is evaluated once before the loop and once after each body run, while the pure condition of an
//! unbounded loop with a scalar predicate is evaluated in the condition region directly. An entry guard checks the
//! input relationships that the actual input types leave unproven before the loop, the results are normalized to the
//! refined output types of the loop, and results whose types place them over explicit mesh axes are constrained to that
//! placement. Reference state is discharged before lowering.

use ryft_core::{
    ArrayIrType, ArrayType, ComparisonDirection, DataType, EffectClasses, Operation, ProgramError, ReductionKind,
    RematerializationOptimizationBarrier, WhileOperation,
};
use ryft_mlir::dialects::{stable_hlo, tensor};
use ryft_mlir::{Block, Operation as MlirOperation, Region, Type, Value as MlirValue, ValueRef};

use crate::experimental::lowering::EffectTokens;
use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::{
    lower_array_ir_type, lower_explicit_shape, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, physical_bound_type,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::control_flow::{
    append_loop_effect_tokens, lower_control_flow_region, lower_explicit_sharding_constraints,
    lower_refined_region_input, lower_refined_region_outputs, lower_region_guard, lowered_effect_classes,
    set_threaded_effect_tokens, threaded_effect_token_values, validate_unthreaded_effect_tokens,
};
use crate::operations::reductions::lower_reduce_to_mlir;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

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
                "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
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
        // A while loop lowers to one `stablehlo.while` whose state starts with the loop inputs, which the condition and
        // the body receive as their shared state types. The lowering proceeds in order:
        //
        //   1. An entry guard checks the input relationships that the actual input types leave unproven, and each input
        //      is then converted from its actual type to its state type.
        //   2. A threaded predicate is evaluated once on the entry state, and the tokens of the threaded effect classes
        //      enter the loop state after it.
        //   3. The condition region reads (and reduces) a threaded predicate or evaluates a pure condition, and
        //      conjoins the iteration bound. The body region runs the body, masks the state updates of a batched
        //      predicate, and re-evaluates a threaded predicate unless the bound is reached.
        //   4. The trailing token results become the current tokens of the enclosing scope, the state results are
        //      normalized to the refined `output_types` of the loop, and each result whose type places it over an
        //      explicit mesh axis is constrained to that placement.
        //
        // The program that attaches the loop validated its regions, their signatures, and its input count, so a
        // mismatch here is an internal error, reported as the operation's own region contract and input count would
        // report it.
        self.validate_region_count(lowerer.regions.len())?;
        let (condition, body) = (&lowerer.regions[0], &lowerer.regions[1]);
        let (context, location) = (lowerer.context, lowerer.location);
        let state_types = body.input_types();
        let state_count = state_types.len();
        for actual in [input_values.len(), input_types.len()] {
            if actual != state_count {
                return Err(ProgramError::InvalidInputCount { expected: state_count, actual }.into());
            }
        }
        let captured_values = lowerer.captured_values.as_slice();
        let nested_functions = lowerer.nested_functions.as_ref();
        let collective_state = &lowerer.collective_state;
        let block = &mut lowerer.block;
        let effect_tokens = &mut lowerer.effect_tokens;

        // The condition and the body share the loop state's declared types. Guarding the initial entry before the loop
        // checks what eager entry validation checks, because later iterations enter with the body's own typed outputs.
        lower_region_guard(
            self.name(),
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
        // `_while_lowering`. Rather than re-evaluating the condition in both regions, the per-item predicate is
        // threaded through the loop state as one extra carried value (JAX's `(pred, ..., carry)` layout): the condition
        // region only reduces the carried predicate with a Boolean `or` into the scalar continuation decision, and the
        // body region selects per state element between the body's candidate update and the carried (incoming-state)
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
        let iteration_bound = self.iteration_bound();
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
        // Carry one trailing token for each ordered class used by either nested program. Each class advances
        // independently through the body; a body that is pure for one active class returns that class's entry token
        // unchanged.
        let threaded_effects = condition_effects.union(lowered_effect_classes(body)?);
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

        // Seed the loop state. A threaded predicate is evaluated once on the entry state in the enclosing block. For an
        // effectful scalar condition, this also advances its ordered effect chains before those tokens enter the loop.
        let mut state_values = Vec::with_capacity(token_start_index);
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
            state_values.push(initial_predicate[0]);
        }
        append_loop_effect_tokens(
            threaded_effects,
            effect_tokens,
            &mut state_values,
            &mut lowered_state_types,
            block,
            context,
            location,
        )?;
        let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

        let mut condition_region = context.region();
        let condition_block = context.block(block_arguments.as_slice());
        {
            let mut condition_block_ref = condition_block.as_ref();
            // A threaded predicate is read from the loop state. Batched predicates are reduced with Boolean `or`, so
            // the loop continues while any mapped item remains active; effectful scalar predicates are already scalar.
            // Pure unbounded scalar predicates retain the smaller representation and are evaluated directly in this
            // region.
            let loop_predicate = if threaded_predicate {
                let carried_predicate = condition_block_ref.argument(predicate_index).unwrap().as_ref();
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
                    .map(|index| condition_block_ref.argument(index).unwrap().as_ref())
                    .collect::<Vec<_>>();
                let mut condition_effect_tokens = EffectTokens::default();
                set_threaded_effect_tokens(&mut condition_effect_tokens, threaded_effects, |offset| {
                    condition_block_ref.argument(token_start_index + offset).unwrap().as_ref()
                });
                let entry_effect_tokens = condition_effect_tokens;
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
                // The condition region of a `stablehlo.while` returns only the predicate, so it threads no token.
                validate_unthreaded_effect_tokens(entry_effect_tokens, condition_effect_tokens, EffectClasses::NONE)?;
                condition_outputs[0]
            };
            let predicate = match iteration_bound {
                Some(bound) => {
                    let counter = condition_block_ref.argument(0).unwrap().as_ref();
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
                    fused.result(0).unwrap().as_ref()
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
                .map(|index| body_block_ref.argument(index).unwrap().as_ref())
                .collect::<Vec<_>>();
            let mut body_effect_tokens = EffectTokens::default();
            set_threaded_effect_tokens(&mut body_effect_tokens, threaded_effects, |offset| {
                body_block_ref.argument(token_start_index + offset).unwrap().as_ref()
            });
            let entry_effect_tokens = body_effect_tokens;
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
            // For a batched predicate, mask each carry update under the carried (incoming-state) predicate so finished
            // items freeze. A scalar loop uses the body outputs directly.
            let next_state_values = if batched_predicate {
                let carried_predicate = body_block_ref.argument(predicate_index).unwrap().as_ref();
                body_outputs
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
                                broadcast.result(0).unwrap().as_ref()
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
                        Ok(select.result(0).unwrap().as_ref())
                    })
                    .collect::<Result<Vec<_>, LoweringError>>()?
            } else {
                body_outputs
            };
            let next_counter = if iteration_bound.is_some() {
                let counter = body_block_ref.argument(0).unwrap().as_ref();
                let one = lower_static_index_constants(&[1], &mut body_block_ref, context, location)?[0];
                let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, one, location)?)?;
                Some(next_counter.result(0).unwrap().as_ref())
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
                    let bound_constant =
                        lower_static_index_constants(&[bound], &mut body_block_ref, context, location)?[0];
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
                        let mut finished_outputs = vec![body_block_ref.argument(predicate_index).unwrap().as_ref()];
                        finished_outputs.extend(threaded_effect_token_values(body_effect_tokens, threaded_effects));
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
                    set_threaded_effect_tokens(&mut body_effect_tokens, threaded_effects, |offset| {
                        guarded_condition.result(1 + offset).unwrap().as_ref()
                    });
                    Some(guarded_condition.result(0).unwrap().as_ref())
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
                    Some(next_predicate[0])
                }
            } else {
                None
            };
            validate_unthreaded_effect_tokens(entry_effect_tokens, body_effect_tokens, threaded_effects)?;
            let mut next_state = Vec::with_capacity(lowered_state_types.len());
            next_state.extend(next_counter);
            next_state.extend(next_state_values);
            next_state.extend(next_predicate);
            next_state.extend(threaded_effect_token_values(body_effect_tokens, threaded_effects));
            body_block_ref.append_operation(stable_hlo::r#return(next_state.as_slice(), location)?)?;
        }
        body_region.append_block(body_block)?;

        let operation = block.append_operation(stable_hlo::r#while(
            state_values.as_slice(),
            condition_region.into(),
            body_region.into(),
            location,
        )?)?;
        set_threaded_effect_tokens(effect_tokens, threaded_effects, |offset| {
            operation.result(token_start_index + offset).unwrap().as_ref()
        });
        let results =
            (0..state_count).map(|index| operation.result(counter_offset + index).unwrap().as_ref()).collect();
        let results = lower_refined_region_outputs(results, &state_types, output_types, block, context, location)?;
        lower_explicit_sharding_constraints(results, output_types, &collective_state.bound_manual_axes, block, location)
    }
}

#[cfg(test)]
mod tests {
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, AssertOperation, CompareOperation, Dimension, DimensionBounds, DimensionSizeOperation,
        DimensionType, DimensionVariable, DynamicBroadcastOperation, MeshAxisType, NegOperation, OneLikeOperation,
        Placeholder, PrintOperation, Shape, Sharding, ShardingDimension, SubOperation,
    };

    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, unproject_plain_program, xla_identity_branch,
    };
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::operations::control_flow::tests::{
        compare, execute_on_two_devices, lowered_mixed_module, positive, two_device_mesh,
    };
    use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
    use crate::tests::execution_client;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_while_lowering() {
        // An unbounded loop with a pure scalar condition evaluates its condition in the condition region and threads
        // only its states through the loop.
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
        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<i1>) -> tensor<i1> {
                    %0 = stablehlo.while(%iterArg = %arg0) : tensor<i1>
                    cond {
                      stablehlo.return %iterArg : tensor<i1>
                    } do {
                      stablehlo.return %iterArg : tensor<i1>
                    }
                    return %0 : tensor<i1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_while_lowering_bounded() {
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
        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<i1>) -> tensor<i1> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg0) : tensor<i64>, \
                tensor<i1>, tensor<i1>
                    cond {
                      %c_2 = stablehlo.constant dense<3> : tensor<i64>
                      %1 = stablehlo.compare LT, %iterArg, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %2 = stablehlo.and %iterArg_1, %1 : tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %c_2 = stablehlo.constant dense<1> : tensor<i64>
                      %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
                      %c_3 = stablehlo.constant dense<3> : tensor<i64>
                      %2 = stablehlo.compare LT, %1, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %3 = \"stablehlo.if\"(%2) ({
                        stablehlo.return %iterArg_0 : tensor<i1>
                      }, {
                        stablehlo.return %iterArg_1 : tensor<i1>
                      }) : (tensor<i1>) -> tensor<i1>
                      stablehlo.return %1, %iterArg_0, %3 : tensor<i64>, tensor<i1>, tensor<i1>
                    }
                    return %0#1 : tensor<i1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_while_lowering_forwards_dimension_carries() {
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

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> (tensor<?xf32, \
                #stablehlo.bounds<8>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<8xf32>, tensor<i32>) -> \
                tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %3 = stablehlo.while(%iterArg = %2) : tensor<i64>
                    cond {
                      %9 = stablehlo.compare LT, %iterArg, %iterArg, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %9 : tensor<i1>
                    } do {
                      stablehlo.return %iterArg : tensor<i64>
                    }
                    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %5 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32>
                    %6 = stablehlo.set_dimension_size %4, %5, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, \
                #stablehlo.bounds<8>>
                    %7 = stablehlo.get_dimension_size %6, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %8 = stablehlo.convert %7 : (tensor<i32>) -> tensor<i64>
                    return %6, %8 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_while_execution_converts_refined_static_state_inputs() {
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
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![first, second], None)
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
        // The runtime appends its own context to the assertion message.
        assert!(error.contains("input dimension `rows` must have one extent at the entry of `while`"), "{error}");
    }

    #[test]
    fn test_while_execution_converts_partially_refined_state_inputs() {
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
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![first, second], None)
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
    fn test_while_lowering_threads_effectful_conditions() {
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
        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<i1>) -> tensor<i1> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.custom_call @ryft.assert(%arg0, %0) {api_version = 4 : i32, backend_config = {actor \
                = \"assert\", kind = \"generic\", label_count = \"0\", message = \"condition must be true\"}, \
                has_side_effect = true} : (tensor<i1>, !stablehlo.token) -> !stablehlo.token
                    %2 = stablehlo.after_all  : !stablehlo.token
                    %3 = stablehlo.custom_call @ryft.print(%arg0, %2) {api_version = 4 : i32, backend_config = {label \
                = \"condition\"}, has_side_effect = true} : (tensor<i1>, !stablehlo.token) -> !stablehlo.token
                    %4:5 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg0, %iterArg_2 = %1, \
                %iterArg_3 = %3) : tensor<i64>, tensor<i1>, tensor<i1>, !stablehlo.token, !stablehlo.token
                    cond {
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %5 = stablehlo.compare LT, %iterArg, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %6 = stablehlo.and %iterArg_1, %5 : tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %5 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %6 = stablehlo.compare LT, %5, %c_5, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %7:3 = \"stablehlo.if\"(%6) ({
                        %8 = stablehlo.custom_call @ryft.assert(%iterArg_0, %iterArg_2) {api_version = 4 : i32, \
                backend_config = {actor = \"assert\", kind = \"generic\", label_count = \"0\", message = \"condition \
                must be true\"}, has_side_effect = true} : (tensor<i1>, !stablehlo.token) -> !stablehlo.token
                        %9 = stablehlo.custom_call @ryft.print(%iterArg_0, %iterArg_3) {api_version = 4 : i32, \
                backend_config = {label = \"condition\"}, has_side_effect = true} : (tensor<i1>, !stablehlo.token) -> \
                !stablehlo.token
                        stablehlo.return %iterArg_0, %8, %9 : tensor<i1>, !stablehlo.token, !stablehlo.token
                      }, {
                        stablehlo.return %iterArg_1, %iterArg_2, %iterArg_3 : tensor<i1>, !stablehlo.token, \
                !stablehlo.token
                      }) : (tensor<i1>) -> (tensor<i1>, !stablehlo.token, !stablehlo.token)
                      stablehlo.return %5, %iterArg_0, %7#0, %7#1, %7#2 : tensor<i64>, tensor<i1>, tensor<i1>, \
                !stablehlo.token, !stablehlo.token
                    }
                    return %4#1 : tensor<i1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_while_execution_stops_condition_effects_at_the_bound() {
        // A false entry predicate runs the condition once, natural termination runs it once more than the body,
        // and truncation at the bound runs it only before each permitted body execution.
        let state_type = ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let state = builder
                .add_instruction(PrintOperation::<ArrayType>::new("condition"), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = positive(&mut builder, state);
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
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        for (input, expected, expected_lines) in [
            (-1.0, -1.0, vec!["condition: -1.0"]),
            (1.0, 0.0, vec!["condition: 1.0", "body: 0.0", "condition: 0.0"]),
            (5.0, 3.0, vec!["condition: 5.0", "body: 4.0", "condition: 4.0", "body: 3.0"]),
        ] {
            let (outputs, lines) = with_captured_prints(|| {
                execute_mixed_program(&client, &program, &[MixedValue::Array(vec![input], vec![])], &[])
            });
            assert_eq!(outputs, Ok(vec![MixedValue::Array(vec![expected], vec![])]), "input {input}");
            assert_eq!(lines, expected_lines, "input {input}");
        }
    }

    #[test]
    fn test_while_execution_batched_predicate() {
        // A batched (per-item) predicate lowers with masked semantics: the loop carries the `bool[3]` predicate through
        // its state, so the condition is evaluated once before the loop and once after each body run, the condition
        // region `or`-reduces the carried predicate into the scalar continuation decision, and the body selects per
        // item between the body's candidate update and the carried state, freezing finished items. The predicate shape
        // equals the state shape here, so no broadcast is needed for the mask. The countdown `while (x > 0) { x - 1 }`
        // over the items `[3, 1, 2]` terminates them after 3, 1, and 2 iterations, so all of them land at 0, whereas a
        // masking bug (or a rejoining finished item) would leave nonzero or negative items.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone().into());
            let predicate = positive(&mut builder, state);
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
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
                    %1 = stablehlo.compare GT, %arg0, %0, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
                    %2:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %1) : tensor<3xf64>, tensor<3xi1>
                    cond {
                      %c = stablehlo.constant dense<false> : tensor<i1>
                      %3 = stablehlo.reduce(%iterArg_0 init: %c) applies stablehlo.or across dimensions = [0] : \
                (tensor<3xi1>, tensor<i1>) -> tensor<i1>
                      stablehlo.return %3 : tensor<i1>
                    } do {
                      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                      %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
                      %4 = stablehlo.subtract %iterArg, %3 : tensor<3xf64>
                      %5 = stablehlo.select %iterArg_0, %4, %iterArg : tensor<3xi1>, tensor<3xf64>
                      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %6 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
                      %7 = stablehlo.compare GT, %5, %6, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
                      stablehlo.return %5, %7 : tensor<3xf64>, tensor<3xi1>
                    }
                    return %2#0 : tensor<3xf64>
                  }
                }
            "},
        );
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[MixedValue::Array(vec![3.0, 1.0, 2.0], vec![3])],
                &[]
            ),
            Ok(vec![MixedValue::Array(vec![0.0, 0.0, 0.0], vec![3])]),
        );
    }

    #[test]
    fn test_while_lowering_batched_predicate_broadcasts_masks_over_dynamic_axes() {
        // A bounded state axis gives each dynamic mask axis the actual extent of the state with
        // `stablehlo.set_dimension_size`, rather than its capacity, and an unbounded one broadcasts the mask to the
        // runtime shape of the state with `stablehlo.dynamic_broadcast_in_dim`.
        for (upper_bound, expected) in [
            (
                Some(5),
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<2x4x4xf32>, %arg2: tensor<i32>, %arg3: \
                    tensor<i32>) -> (tensor<2x?x?xf32, #stablehlo.bounds<?, 4, 4>>, tensor<i64>, tensor<i64>) {
                        %0 = stablehlo.set_dimension_size %arg1, %arg2, dim = 1 : (tensor<2x4x4xf32>, tensor<i32>) -> \
                    tensor<2x?x4xf32, #stablehlo.bounds<?, 4, ?>>
                        %1 = stablehlo.set_dimension_size %0, %arg3, dim = 2 : (tensor<2x?x4xf32, #stablehlo.bounds<?, \
                    4, ?>>, tensor<i32>) -> tensor<2x?x?xf32, #stablehlo.bounds<?, 4, 4>>
                        %c = stablehlo.constant dense<0> : tensor<i64>
                        %2:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %1, %iterArg_2 = %arg0) \
                    : tensor<i64>, tensor<2xi1>, tensor<2x?x?xf32, #stablehlo.bounds<?, 4, 4>>, tensor<2xi1>
                        cond {
                          %c_3 = stablehlo.constant dense<false> : tensor<i1>
                          %7 = stablehlo.reduce(%iterArg_2 init: %c_3) applies stablehlo.or across dimensions = [0] : \
                    (tensor<2xi1>, tensor<i1>) -> tensor<i1>
                          %c_4 = stablehlo.constant dense<2> : tensor<i64>
                          %8 = stablehlo.compare LT, %iterArg, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                          %9 = stablehlo.and %7, %8 : tensor<i1>
                          stablehlo.return %9 : tensor<i1>
                        } do {
                          %7 = stablehlo.select %iterArg_2, %iterArg_0, %iterArg_0 : tensor<2xi1>, tensor<2xi1>
                          %8 = stablehlo.broadcast_in_dim %iterArg_2, dims = [0] : (tensor<2xi1>) -> tensor<2x4x4xi1>
                          %9 = stablehlo.get_dimension_size %iterArg_1, dim = 1 : (tensor<2x?x?xf32, \
                    #stablehlo.bounds<?, 4, 4>>) -> tensor<i32>
                          %10 = stablehlo.set_dimension_size %8, %9, dim = 1 : (tensor<2x4x4xi1>, tensor<i32>) -> \
                    tensor<2x?x4xi1, #stablehlo.bounds<?, 4, ?>>
                          %11 = stablehlo.get_dimension_size %iterArg_1, dim = 2 : (tensor<2x?x?xf32, \
                    #stablehlo.bounds<?, 4, 4>>) -> tensor<i32>
                          %12 = stablehlo.set_dimension_size %10, %11, dim = 2 : (tensor<2x?x4xi1, \
                    #stablehlo.bounds<?, 4, ?>>, tensor<i32>) -> tensor<2x?x?xi1, #stablehlo.bounds<?, 4, 4>>
                          %13 = stablehlo.select %12, %iterArg_1, %iterArg_1 : tensor<2x?x?xi1, #stablehlo.bounds<?, \
                    4, 4>>, tensor<2x?x?xf32, #stablehlo.bounds<?, 4, 4>>
                          %c_3 = stablehlo.constant dense<1> : tensor<i64>
                          %14 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                          %c_4 = stablehlo.constant dense<2> : tensor<i64>
                          %15 = stablehlo.compare LT, %14, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                          %16 = \"stablehlo.if\"(%15) ({
                            stablehlo.return %7 : tensor<2xi1>
                          }, {
                            stablehlo.return %iterArg_2 : tensor<2xi1>
                          }) : (tensor<i1>) -> tensor<2xi1>
                          stablehlo.return %14, %7, %13, %16 : tensor<i64>, tensor<2xi1>, tensor<2x?x?xf32, \
                    #stablehlo.bounds<?, 4, 4>>, tensor<2xi1>
                        }
                        %3 = stablehlo.get_dimension_size %2#2, dim = 1 : (tensor<2x?x?xf32, #stablehlo.bounds<?, 4, \
                    4>>) -> tensor<i32>
                        %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
                        %5 = stablehlo.get_dimension_size %2#2, dim = 2 : (tensor<2x?x?xf32, #stablehlo.bounds<?, 4, \
                    4>>) -> tensor<i32>
                        %6 = stablehlo.convert %5 : (tensor<i32>) -> tensor<i64>
                        return %2#2, %4, %6 : tensor<2x?x?xf32, #stablehlo.bounds<?, 4, 4>>, tensor<i64>, tensor<i64>
                      }
                    }
                "},
            ),
            (
                None,
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> {
                        %c = stablehlo.constant dense<0> : tensor<i64>
                        %0:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = \
                    %arg0) : tensor<i64>, tensor<2xi1>, tensor<2x?x?xf32>, tensor<2xi1>
                        cond {
                          %c_3 = stablehlo.constant dense<false> : tensor<i1>
                          %1 = stablehlo.reduce(%iterArg_2 init: %c_3) applies stablehlo.or across dimensions = [0] : \
                    (tensor<2xi1>, tensor<i1>) -> tensor<i1>
                          %c_4 = stablehlo.constant dense<2> : tensor<i64>
                          %2 = stablehlo.compare LT, %iterArg, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                          %3 = stablehlo.and %1, %2 : tensor<i1>
                          stablehlo.return %3 : tensor<i1>
                        } do {
                          %1 = stablehlo.select %iterArg_2, %iterArg_0, %iterArg_0 : tensor<2xi1>, tensor<2xi1>
                          %2 = stablehlo.get_dimension_size %iterArg_1, dim = 0 : (tensor<2x?x?xf32>) -> tensor<i32>
                          %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                          %4 = stablehlo.get_dimension_size %iterArg_1, dim = 1 : (tensor<2x?x?xf32>) -> tensor<i32>
                          %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                          %6 = stablehlo.get_dimension_size %iterArg_1, dim = 2 : (tensor<2x?x?xf32>) -> tensor<i32>
                          %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
                          %8 = stablehlo.reshape %3 : (tensor<i64>) -> tensor<1xi64>
                          %9 = stablehlo.reshape %5 : (tensor<i64>) -> tensor<1xi64>
                          %10 = stablehlo.reshape %7 : (tensor<i64>) -> tensor<1xi64>
                          %11 = stablehlo.concatenate %8, %9, %10, dim = 0 : (tensor<1xi64>, tensor<1xi64>, \
                    tensor<1xi64>) -> tensor<3xi64>
                          %12 = stablehlo.dynamic_broadcast_in_dim %iterArg_2, %11, dims = [0] : (tensor<2xi1>, \
                    tensor<3xi64>) -> tensor<?x?x?xi1>
                          %cast = tensor.cast %12 : tensor<?x?x?xi1> to tensor<2x?x?xi1>
                          %13 = stablehlo.select %cast, %iterArg_1, %iterArg_1 : tensor<2x?x?xi1>, tensor<2x?x?xf32>
                          %c_3 = stablehlo.constant dense<1> : tensor<i64>
                          %14 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                          %c_4 = stablehlo.constant dense<2> : tensor<i64>
                          %15 = stablehlo.compare LT, %14, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                          %16 = \"stablehlo.if\"(%15) ({
                            stablehlo.return %1 : tensor<2xi1>
                          }, {
                            stablehlo.return %iterArg_2 : tensor<2xi1>
                          }) : (tensor<i1>) -> tensor<2xi1>
                          stablehlo.return %14, %1, %13, %16 : tensor<i64>, tensor<2xi1>, tensor<2x?x?xf32>, \
                    tensor<2xi1>
                        }
                        return %0#2 : tensor<2x?x?xf32>
                      }
                    }
                "},
            ),
        ] {
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

            assert_eq!(module, expected, "upper bound {upper_bound:?}");
        }
    }

    #[test]
    fn test_while_lowering_batched_predicate_passes_loop_invariant_dimensions() {
        // A batched-predicate loop may also carry a first-class dimension, which the relaxed `WhileType` contract
        // requires to be loop-invariant. Masking a loop-invariant carry is the identity, so the lowering threads the
        // body's dimension result on directly: only the array carry gets a `stablehlo.select`, while the condition
        // region still `or`-reduces the per-item predicate into the scalar continuation decision.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(extent_type.clone().into());
            let state = builder.add_input(state_type.clone().into());
            let predicate = positive(&mut builder, state);
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
    fn test_while_lowering_constrains_explicit_output_shardings() {
        // The body negates an `f32[8]` vector sharded over the explicit mesh axis `x` while it counts its iterations up
        // to a replicated limit. Each result of the loop is constrained to the placement that its type states.
        let (client, mesh) = two_device_mesh(MeshAxisType::Explicit);
        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let vector_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32).with_sharding(scalar_sharding.clone()).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [8]).with_sharding(vector_sharding.clone()).unwrap();
        let state_types = [scalar_type.clone(), scalar_type.clone(), vector_type.clone()];
        let loop_condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let predicate = builder
                .add_instruction(compare(ComparisonDirection::LessThan), Vec::new(), vec![inputs[0], inputs[1]], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let loop_body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let (index, limit, vector) = (inputs[0], inputs[1], inputs[2]);
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![index], None)
                .unwrap()[0];
            let index = builder.add_instruction(AddOperation::new(), Vec::new(), vec![index, one], None).unwrap()[0];
            let vector = builder.add_instruction(NegOperation::new(), Vec::new(), vec![vector], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![index, limit, vector],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(loop_condition.entry_region_ref());
        let body_region = builder.import_region(loop_body.entry_region_ref());
        let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
        let vector = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                inputs,
                None,
            )
            .unwrap()[2];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![vector], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc!(
                "
                lambda %0:{scalar}, %1:{scalar}, %2:{vector} .
                let %3:{scalar}, %4:{scalar}, %5:{vector} = while %0 %1 %2 [
                    condition={{
                        lambda %0:{scalar}, %1:{scalar}, %2:{vector} .
                        let %3:{predicate} = compare [direction=LessThan] %0 %1
                        in (%3)
                    }},
                    body={{
                        lambda %0:{scalar}, %1:{scalar}, %2:{vector} .
                        let %3:{scalar} = one_like %0
                            %4:{scalar} = add %0 %3
                            %5:{vector} = neg %2
                        in (%4, %1, %5)
                    }},
                ]
                in (%5)",
                scalar = "f32[][sharding={mesh<['x'=2:explicit]>, []}]",
                vector = "f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]",
                predicate = "bool[][sharding={mesh<['x'=2:explicit]>, []}]",
            ),
        );
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &state_types.to_vec(),
            &vec![vector_type.clone()],
            "main",
            Some(&[scalar_sharding.clone(), scalar_sharding, vector_sharding.clone()]),
            Some(&[vector_sharding]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}, \
                %arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}, %arg2: tensor<8xf32> \
                {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> (tensor<8xf32> {sdy.sharding = \
                #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0:3 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %arg1, %iterArg_1 = %arg2) : tensor<f32>, \
                tensor<f32>, tensor<8xf32>
                    cond {
                      %4 = stablehlo.compare LT, %iterArg, %iterArg_0, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      stablehlo.return %4 : tensor<i1>
                    } do {
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %4 = sdy.sharding_constraint %cst <@mesh, [], replicated={\"x\"}> : tensor<f32>
                      %5 = stablehlo.add %iterArg, %4 : tensor<f32>
                      %6 = stablehlo.negate %iterArg_1 : tensor<8xf32>
                      stablehlo.return %5, %iterArg_0, %6 : tensor<f32>, tensor<f32>, tensor<8xf32>
                    }
                    %1 = sdy.sharding_constraint %0#0 <@mesh, [], replicated={\"x\"}> : tensor<f32>
                    %2 = sdy.sharding_constraint %0#1 <@mesh, [], replicated={\"x\"}> : tensor<f32>
                    %3 = sdy.sharding_constraint %0#2 <@mesh, [{\"x\"}]> : tensor<8xf32>
                    return %3 : tensor<8xf32>
                  }
                }
            "},
        );
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Three iterations negate the vector three times.
        assert_eq!(
            execute_on_two_devices(
                &client,
                &mesh,
                &module,
                &[(&scalar_type, &[0.0]), (&scalar_type, &[3.0]), (&vector_type, &values)],
            ),
            vec![vec![vec![-1.0, -2.0, -3.0, -4.0]], vec![vec![-5.0, -6.0, -7.0, -8.0]]],
        );
    }
}
