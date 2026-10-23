use std::rc::Rc;

use ryft_core::{
    ArrayIrType, ArrayType, ComparisonDirection, DataType, Dimension, Operation, ProgramError,
    RematerializationOptimizationBarrier, SCAN_OPERATION_NAME, ScanOperation, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Type, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::{
    lower_array_ir_type, lower_constant_output, lower_static_index_constants, lower_tensor_type,
    stable_hlo_dynamic_dimension_bound, static_dimensions,
};
use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, current_or_new_token, token_threaded_effects,
};
use crate::experimental::ops::FlatXlaProgram;
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::control_flow::{
    lower_refined_region_input, lower_refined_region_outputs, lower_region_guard, lowered_effect_classes,
};
use crate::operations::jit_call::JitCallFunctionMap;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Lowers one shape-counted scan loop to a `stablehlo.while` over the state
/// `[counter, carries..., stacks..., ys...]`.
///
/// The `i64` counter starts at zero. Each loop trip runs `unroll` consecutive logical iterations (body copies) and
/// advances the counter by `unroll`, and the loop runs while a whole trip fits (`counter + unroll - 1 < length`), so it
/// performs `length / unroll` trips. The `length % unroll` remaining iterations follow the loop: as straight-line body
/// copies at static iteration indices for a static length, and as a second loop with one body copy per trip for a
/// dynamic length. Logical iteration `i` computes its iteration index (`i`, or `length - 1 - i` when `reverse` is set),
/// reads one slice of every stacked input with `stablehlo.dynamic_slice` (dropping the unit iteration axis with
/// `stablehlo.reshape`), inlines the lowered body program over `[index, carries..., iteration_slices...]`, and writes
/// each per-iteration output into its preallocated stacked zero accumulator with `stablehlo.dynamic_update_slice`.
/// This is the same strategy JAX uses to lower `lax.scan`, which is not an XLA primitive. When `unroll` is at least a
/// static `length`, no `stablehlo.while` is emitted at all: the body copies inline as straight-line operations at
/// static iteration indices. The provided `input_values` omit the body index input: the first `carry_count` values are
/// carries and the remaining values are stacked operands, followed by the runtime length when the length is dynamic.
/// Reference state is discharged before lowering.
pub(crate) fn lower_scan_to_while<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    scan: &ScanOperation<ArrayIrType>,
    body_program: &FlatXlaProgram<O>,
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
    let (carry_count, length, reverse, unroll) = (scan.carry_count(), scan.length(), scan.reverse(), scan.unroll());
    let body_input_types = body_program.input_types();
    let body_output_types = body_program.output_types();
    // The checked scan signature includes an index supplied by this loop rather than an operation operand.
    let operand_count = body_input_types.len() - 1;
    let runtime_length_count = usize::from(length.variable().is_some());
    if input_values.len() != operand_count + runtime_length_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "`{SCAN_OPERATION_NAME}` expected {} lowered inputs but got {}",
                operand_count + runtime_length_count,
                input_values.len(),
            ),
        });
    }
    let (input_values, runtime_length) = if runtime_length_count == 1 {
        (&input_values[..operand_count], Some(input_values[operand_count]))
    } else {
        (input_values, None)
    };
    let static_length = length.value();
    if unroll == 0 {
        return Err(LoweringError::UnsupportedOp {
            op: format!("`{SCAN_OPERATION_NAME}` unroll factor must be at least 1"),
        });
    }
    // A factor beyond the largest possible trip count unrolls nothing more, so it is clamped to that trip count. Only a
    // dynamic length without an upper bound can leave a factor whose iteration offsets the `i64` counter cannot hold.
    let unroll = static_length
        .or_else(|| stable_hlo_dynamic_dimension_bound(length))
        .map_or(unroll, |maximum_length| unroll.min(maximum_length.max(1)));
    if unroll > i64::MAX as usize {
        return Err(LoweringError::UnsupportedOp {
            op: format!("`{SCAN_OPERATION_NAME}` unroll factor {unroll} must be at most {}", i64::MAX),
        });
    }
    let carry_types = &body_input_types[1..1 + carry_count];
    // Initial carries only need to refine the carry types, so each one is converted to the lowered type of its carry
    // first. Stacked inputs are converted only where the loop state stacks them over the declared length (below).
    if input_types.len() < operand_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "`{SCAN_OPERATION_NAME}` expected at least {operand_count} input types but got {}",
                input_types.len(),
            ),
        });
    }
    let entry_values = input_values;
    let mut converted_input_values = input_values.to_vec();
    for (index, carry_type) in carry_types.iter().enumerate() {
        converted_input_values[index] =
            lower_refined_region_input(input_values[index], &input_types[index], carry_type, block, context, location)?;
    }
    let input_values = converted_input_values.as_slice();
    let x_slice_types = body_input_types[1 + carry_count..]
        .iter()
        .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::from)?;
    let y_slice_types = body_output_types[carry_count..]
        .iter()
        .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::from)?;
    let stacked = |slice_type: &ArrayType| -> ArrayType {
        let mut dimensions = vec![length.clone()];
        dimensions.extend(slice_type.shape().dimensions().iter().cloned());
        ArrayType::new(slice_type.data_type(), ryft_core::arrays::Shape::new(dimensions))
    };

    let initialize_accumulator =
        |slice_type: &ArrayType, block: &mut BlockRef<'b, 'c, 't>| -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
            let stacked_type = stacked(slice_type);
            let physical_length = match length {
                Dimension::Static(length) => *length,
                Dimension::Dynamic(_) => {
                    stable_hlo_dynamic_dimension_bound(length).ok_or_else(|| LoweringError::UnsupportedOp {
                        op: format!(
                            "{SCAN_OPERATION_NAME} length {length} needs a finite upper bound for physical accumulator \
                         allocation",
                        ),
                    })?
                }
            };
            let mut physical_dimensions = stacked_type.shape().dimensions().to_vec();
            physical_dimensions[0] = Dimension::Static(physical_length);
            let physical_type = stacked_type.clone().with_shape(Shape::new(physical_dimensions));
            let accumulator =
                lower_constant_output(std::slice::from_ref(&physical_type), 0, block, context, location)?.remove(0);
            let Some(runtime_length) = runtime_length else {
                return Ok(accumulator);
            };
            let i32_scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
            let converted = block.append_operation(stable_hlo::convert(runtime_length, i32_scalar_type, location)?)?;
            let converted = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
            let tensor_type = lower_tensor_type(&stacked_type, context, location)?;
            let refined = block.append_operation(stable_hlo::set_dimension_size(
                accumulator,
                converted,
                tensor_type,
                0,
                location,
            )?)?;
            Ok(refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref())
        };

    // Logical iterations `first_iteration..length` of a static-length scan whose loop (if any) already ran the earlier
    // iterations are emitted as straight-line body copies at static iteration indices, reading and writing the same
    // stacked inputs and zero accumulators that the loop form threads through its state.
    let lower_static_iterations = |first_iteration: usize,
                                   length: usize,
                                   mut carries: Vec<ValueRef<'b, 'c, 't>>,
                                   x_stacks: Vec<ValueRef<'b, 'c, 't>>,
                                   mut y_accumulators: Vec<ValueRef<'b, 'c, 't>>,
                                   block: &mut BlockRef<'b, 'c, 't>,
                                   effect_tokens: &mut EffectTokens<'b, 'c, 't>|
     -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let zero_index = lower_static_index_constants(&[0], block, context, location)?[0];
        for iteration in first_iteration..length {
            let index = if reverse { length - 1 - iteration } else { iteration };
            let index_value = lower_static_index_constants(&[index], block, context, location)?[0];
            (carries, y_accumulators) = lower_scan_iteration(
                body_program,
                x_slice_types.as_slice(),
                y_slice_types.as_slice(),
                index_value,
                zero_index,
                carries,
                x_stacks.as_slice(),
                y_accumulators,
                block,
                context,
                location,
                captured_values,
                nested_functions,
                collective_state,
                effect_tokens,
            )?;
        }
        carries.extend(y_accumulators);
        Ok(carries)
    };

    // The body enters with the carries and with slices of the stacked inputs, whose identities the declared stacked
    // types carry unchanged next to the leading length axis. Guarding the original carries and stacks therefore checks
    // what eager entry validation checks, which happens whatever the trip count is, so the guard precedes the
    // zero-length and the fully unrolled paths too.
    let entry_types = scan.declared_input_types(&body_input_types).map_err(ProgramError::from)?;
    lower_region_guard(
        SCAN_OPERATION_NAME,
        &entry_types,
        &input_types[..operand_count],
        entry_values,
        effect_tokens,
        block,
        context,
        location,
    )?;

    // A zero-length scan runs no iteration and returns its initial carries and empty stacked outputs. This also
    // applies to a dynamic length whose bounds admit only zero: constructing a loop body would otherwise emit
    // unit-length slice reads or writes against zero-sized physical stacks, which are invalid even in a dead body.
    if static_length == Some(0) || stable_hlo_dynamic_dimension_bound(length) == Some(0) {
        let mut outputs = input_values[..carry_count].to_vec();
        for y_slice_type in &y_slice_types {
            outputs.push(initialize_accumulator(y_slice_type, block)?);
        }
        return Ok(outputs);
    }

    // A scan whose unroll factor covers its whole static length needs no loop at all.
    if let Some(length) = static_length.filter(|length| unroll >= *length) {
        let y_accumulators = y_slice_types
            .iter()
            .map(|y_slice_type| initialize_accumulator(y_slice_type, block))
            .collect::<Result<Vec<_>, _>>()?;
        return lower_static_iterations(
            0,
            length,
            input_values[..carry_count].to_vec(),
            input_values[carry_count..].to_vec(),
            y_accumulators,
            block,
            effect_tokens,
        );
    }

    // Assemble the loop state `[counter, carries..., stacks..., ys...]`, preallocating one zero accumulator per
    // stacked output.
    let mut state_types = Vec::with_capacity(1 + body_input_types.len() + y_slice_types.len());
    state_types.push(ArrayType::scalar(DataType::I64).into());
    state_types.extend(carry_types.iter().cloned());
    let mut state_values = Vec::with_capacity(state_types.len() + x_slice_types.len() + y_slice_types.len());
    state_values.push(lower_static_index_constants(&[0], block, context, location)?[0]);
    state_values.extend_from_slice(&input_values[..carry_count]);
    // An exact runtime length (e.g., that of an eager dimension value) refines the stacked inputs of a dynamic-length
    // scan to static leading extents, while the loop state stacks them over the declared length, so each one is
    // converted to its declared stacked type like the carries.
    for (index, x_slice_type) in x_slice_types.iter().enumerate() {
        let index = carry_count + index;
        let stacked_type = ArrayIrType::from(stacked(x_slice_type));
        state_values.push(lower_refined_region_input(
            input_values[index],
            &input_types[index],
            &stacked_type,
            block,
            context,
            location,
        )?);
        state_types.push(stacked_type);
    }
    for y_slice_type in &y_slice_types {
        let stacked_type = stacked(y_slice_type);
        state_values.push(initialize_accumulator(y_slice_type, block)?);
        state_types.push(stacked_type.into());
    }
    let loop_inputs = ScanLoopInputs {
        body_program,
        carry_count,
        x_slice_types: x_slice_types.as_slice(),
        y_slice_types: y_slice_types.as_slice(),
        state_types: state_types.as_slice(),
        length: match static_length {
            Some(length) => ScanLoopLength::Static(length),
            None => ScanLoopLength::Dynamic(runtime_length.unwrap()),
        },
        reverse,
        captured_values,
        nested_functions,
        collective_state,
    };
    let state = lower_scan_loop(&loop_inputs, state_values, unroll, block, context, location, effect_tokens)?;

    // The iterations that do not fill a whole trip of the unrolled loop follow it.
    let x_end = 1 + carry_count + x_slice_types.len();
    let state = match static_length {
        Some(length) if length % unroll != 0 => {
            return lower_static_iterations(
                length - length % unroll,
                length,
                state[1..1 + carry_count].to_vec(),
                state[1 + carry_count..x_end].to_vec(),
                state[x_end..].to_vec(),
                block,
                effect_tokens,
            );
        }
        None if unroll > 1 => lower_scan_loop(&loop_inputs, state, 1, block, context, location, effect_tokens)?,
        _ => state,
    };
    let mut outputs = state[1..1 + carry_count].to_vec();
    outputs.extend_from_slice(&state[x_end..]);
    Ok(outputs)
}

/// Trip count of a scan loop emitted by [`lower_scan_loop`].
#[derive(Copy, Clone)]
enum ScanLoopLength<'b, 'c: 'b, 't: 'c> {
    /// Static trip count.
    Static(usize),

    /// Lowered runtime length value of a dynamic trip count.
    Dynamic(ValueRef<'b, 'c, 't>),
}

/// Loop-invariant inputs of [`lower_scan_loop`] for one scan.
struct ScanLoopInputs<'o, 'b, 'c: 'b, 't: 'c, O> {
    /// Lowered body program over `[index, carries..., x_slices...]`.
    body_program: &'o FlatXlaProgram<O>,

    /// Number of loop-carried state leaves.
    carry_count: usize,

    /// Per-iteration slice types of the stacked inputs.
    x_slice_types: &'o [ArrayType],

    /// Per-iteration slice types of the stacked outputs.
    y_slice_types: &'o [ArrayType],

    /// Types of the loop state `[counter, carries..., stacks..., ys...]`, excluding effect tokens.
    pub(crate) state_types: &'o [ArrayIrType],

    /// Trip count of the loop.
    length: ScanLoopLength<'b, 'c, 't>,

    /// Whether logical iteration `i` reads and writes iteration index `length - 1 - i`.
    pub(crate) reverse: bool,

    /// Lowered values of the enclosing function's captures.
    pub(crate) captured_values: &'o [ValueRef<'b, 'c, 't>],

    /// Nested functions that `jit_call` operations in the body lower against.
    pub(crate) nested_functions: Option<&'o Rc<JitCallFunctionMap<O>>>,

    /// Collective lowering state shared with the enclosing function.
    pub(crate) collective_state: &'o CollectiveLoweringState,
}

/// Emits one `stablehlo.while` over the scan loop state `[counter, carries..., stacks..., ys...]` that starts at
/// `state_values` and runs `unroll` consecutive logical iterations (body copies) per trip while a whole trip fits
/// (`counter + unroll - 1 < length`), advancing the counter by `unroll` per trip. Ordered effect tokens ride at the end
/// of the loop state. The function returns the final loop state without the tokens, which it installs into
/// `effect_tokens` instead.
fn lower_scan_loop<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    inputs: &ScanLoopInputs<'_, 'b, 'c, 't, O>,
    mut state_values: Vec<ValueRef<'b, 'c, 't>>,
    unroll: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let ScanLoopInputs {
        body_program,
        carry_count,
        x_slice_types,
        y_slice_types,
        state_types,
        length,
        reverse,
        captured_values,
        nested_functions,
        collective_state,
    } = *inputs;
    // The loop carries one trailing token per ordered class used by the body. Straight-line body copies update the
    // enclosing scope's chains directly, while pure scans emit no token machinery.
    let threaded_effects = lowered_effect_classes(body_program)?;
    let length_value = |block: &mut BlockRef<'b, 'c, 't>| -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        match length {
            ScanLoopLength::Static(length) => Ok(lower_static_index_constants(&[length], block, context, location)?[0]),
            ScanLoopLength::Dynamic(length) => Ok(length),
        }
    };

    // Effect tokens ride at the end of the loop state, so the counter, carry, stack, and accumulator positions stay
    // unchanged.
    let token_start_index = state_types.len();
    let mut lowered_state_types = state_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    for effect in token_threaded_effects(threaded_effects) {
        lowered_state_types.push(context.stable_hlo_token_type()?.as_ref());
        state_values.push(current_or_new_token(effect, effect_tokens, block, location)?);
    }
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        let counter = condition_block_ref.argument(0).expect("scan while state should include the counter").as_ref();
        let length_value = length_value(&mut condition_block_ref)?;
        // Subtract the last copy's offset from the length instead of adding it to the counter: after the final full
        // trip, `counter + unroll - 1` can overflow even though both the length and counter are valid `i64` extents.
        // This subtraction remains representable when a dynamic length is smaller than the unroll factor, too.
        let counter_limit = if unroll == 1 {
            length_value
        } else {
            let offset = lower_static_index_constants(&[unroll - 1], &mut condition_block_ref, context, location)?[0];
            let subtraction =
                condition_block_ref.append_operation(stable_hlo::subtract(length_value, offset, location)?)?;
            subtraction.result(0).expect("stablehlo.subtract should return one result").as_ref()
        };
        let predicate = lower_compare_to_mlir(
            ComparisonDirection::LessThan,
            counter,
            counter_limit,
            &mut condition_block_ref,
            location,
        )?;
        condition_block_ref.append_operation(stable_hlo::r#return(&[predicate], location)?)?;
    }
    condition_region.append_block(condition_block)?;

    let mut body_region = context.region();
    let body_block = context.block(block_arguments.as_slice());
    {
        let mut body_block_ref = body_block.as_ref();
        let arguments = (0..state_types.len())
            .map(|index| body_block_ref.argument(index).expect("scan while body should have state arguments").as_ref())
            .collect::<Vec<_>>();
        let counter = arguments[0];
        let zero_index = lower_static_index_constants(&[0], &mut body_block_ref, context, location)?[0];
        // When the visit order is reversed, logical iteration `i` reads iteration `length - 1 - i` (a zero-length
        // reversed scan never runs its body, so the saturated limit constant is inert).
        let reverse_limit = if reverse {
            let length_value = length_value(&mut body_block_ref)?;
            let one = lower_static_index_constants(&[1], &mut body_block_ref, context, location)?[0];
            let limit = body_block_ref.append_operation(stable_hlo::subtract(length_value, one, location)?)?;
            Some(limit.result(0).expect("stablehlo.subtract should return one result").as_ref())
        } else {
            None
        };

        // Each loop trip runs `unroll` consecutive logical iterations (`counter + copy` for each body copy).
        let mut carries = arguments[1..1 + carry_count].to_vec();
        let x_stacks = arguments[1 + carry_count..1 + carry_count + x_slice_types.len()].to_vec();
        let mut y_accumulators = arguments[1 + carry_count + x_slice_types.len()..].to_vec();
        let mut body_effect_tokens = EffectTokens::default();
        for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
            body_effect_tokens.set(
                effect,
                body_block_ref
                    .argument(token_start_index + token_offset)
                    .expect("token-threaded scan state should include every active effect token")
                    .as_ref(),
            );
        }
        for copy in 0..unroll {
            let iteration = if copy == 0 {
                counter
            } else {
                let offset = lower_static_index_constants(&[copy], &mut body_block_ref, context, location)?[0];
                let addition = body_block_ref.append_operation(stable_hlo::add(counter, offset, location)?)?;
                addition.result(0).expect("stablehlo.add should return one result").as_ref()
            };
            let index_value = match reverse_limit {
                Some(limit) => {
                    let subtraction =
                        body_block_ref.append_operation(stable_hlo::subtract(limit, iteration, location)?)?;
                    subtraction.result(0).expect("stablehlo.subtract should return one result").as_ref()
                }
                None => iteration,
            };
            (carries, y_accumulators) = lower_scan_iteration(
                body_program,
                x_slice_types,
                y_slice_types,
                index_value,
                zero_index,
                carries,
                x_stacks.as_slice(),
                y_accumulators,
                &mut body_block_ref,
                context,
                location,
                captured_values,
                nested_functions,
                collective_state,
                &mut body_effect_tokens,
            )?;
        }

        // Assemble the next state: advance the counter by the unroll factor, thread the new carries, pass the input
        // stacks through unchanged, and thread the updated stacked accumulators and active effect tokens.
        let step = lower_static_index_constants(&[unroll], &mut body_block_ref, context, location)?[0];
        let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, step, location)?)?;
        let mut next_state = vec![next_counter.result(0).expect("stablehlo.add should return one result").as_ref()];
        next_state.extend(carries);
        next_state.extend(x_stacks);
        next_state.extend(y_accumulators);
        for effect in token_threaded_effects(threaded_effects) {
            next_state.push(
                body_effect_tokens
                    .get(effect)
                    .expect("token-threaded scan bodies receive every active effect token"),
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
                .expect("a token-threaded scan should return one result per active effect class")
                .as_ref(),
        );
    }
    Ok((0..state_types.len())
        .map(|index| operation.result(index).expect("stablehlo.while should return one result per state leaf").as_ref())
        .collect())
}

/// Emits one scan iteration at iteration index `index_value` into `block`: reads slice `index_value` of every stacked
/// input (dropping the unit iteration axis), inlines the body program over `[index, carries..., x_slices...]`, writes
/// each per-iteration output into its stacked accumulator at `index_value`, and returns the new carries and
/// accumulators. This is the per-iteration building block shared by the looped and straight-line scan lowerings in
/// [`lower_scan_to_while`] and [`lower_scan_loop`].
fn lower_scan_iteration<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    body_program: &FlatXlaProgram<O>,
    x_slice_types: &[ArrayType],
    y_slice_types: &[ArrayType],
    index_value: ValueRef<'b, 'c, 't>,
    zero_index: ValueRef<'b, 'c, 't>,
    carries: Vec<ValueRef<'b, 'c, 't>>,
    x_stacks: &[ValueRef<'b, 'c, 't>],
    y_accumulators: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<(Vec<ValueRef<'b, 'c, 't>>, Vec<ValueRef<'b, 'c, 't>>), LoweringError> {
    // Read one slice of every stacked input and drop the unit iteration axis.
    let carry_count = carries.len();
    let mut iteration_inputs = Vec::with_capacity(1 + carry_count + x_slice_types.len());
    iteration_inputs.push(index_value);
    iteration_inputs.extend(carries);
    for (stack_offset, x_slice_type) in x_slice_types.iter().enumerate() {
        let slice_dimensions = static_dimensions(x_slice_type)?;
        let mut sizes = vec![1];
        sizes.extend(slice_dimensions.iter().copied());
        let mut start_values = vec![index_value];
        start_values.extend(std::iter::repeat_n(zero_index, slice_dimensions.len()));
        let iteration = block.append_operation(stable_hlo::dynamic_slice(
            x_stacks[stack_offset],
            start_values.as_slice(),
            sizes.as_slice(),
            location,
        )?)?;
        let squeezed = block.append_operation(stable_hlo::reshape(
            iteration.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref(),
            slice_dimensions.as_slice(),
            location,
        )?)?;
        iteration_inputs.push(squeezed.result(0).expect("stablehlo.reshape should return one result").as_ref());
    }

    let body_outputs = lower_nested_program_inline(
        body_program,
        iteration_inputs.as_slice(),
        block,
        context,
        location,
        captured_values,
        &RematerializationOptimizationBarrier::None,
        nested_functions,
        collective_state,
        effect_tokens,
    )?;
    if body_outputs.len() != carry_count + y_slice_types.len() {
        return Err(LoweringError::UnsupportedOp {
            op: format!("`{SCAN_OPERATION_NAME}` body lowered to {} outputs", body_outputs.len()),
        });
    }

    // Thread the new carries and write each per-iteration output into its stacked accumulator.
    let new_carries = body_outputs[..carry_count].to_vec();
    let mut new_accumulators = Vec::with_capacity(y_slice_types.len());
    for (y_offset, y_slice_type) in y_slice_types.iter().enumerate() {
        let slice_dimensions = static_dimensions(y_slice_type)?;
        let mut expanded_dimensions = vec![1];
        expanded_dimensions.extend(slice_dimensions.iter().copied());
        let expanded = block.append_operation(stable_hlo::reshape(
            body_outputs[carry_count + y_offset],
            expanded_dimensions.as_slice(),
            location,
        )?)?;
        let mut start_values = vec![index_value];
        start_values.extend(std::iter::repeat_n(zero_index, slice_dimensions.len()));
        let updated = block.append_operation(stable_hlo::dynamic_update_slice(
            y_accumulators[y_offset],
            expanded.result(0).expect("stablehlo.reshape should return one result").as_ref(),
            start_values.as_slice(),
            location,
        )?)?;
        new_accumulators
            .push(updated.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref());
    }
    Ok((new_carries, new_accumulators))
}

impl LowerableOperation for ScanOperation<ArrayType> {
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

impl LowerableOperation for ScanOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [body] = lowerer.regions.as_slice() else {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "`{SCAN_OPERATION_NAME}` expected 1 attached region but got {}",
                    lowerer.regions.as_slice().len()
                ),
            });
        };
        // Each iteration reads and writes one slice per stack at a static shape. A body that was not specialized to the
        // scan's input types may still declare dynamic slice types (refined only at the instruction boundary), which
        // this lowering does not adapt per iteration. A scan whose length admits no iteration never slices, so it is
        // lowered without its body.
        let carry_count = self.carry_count();
        let body_input_types = body.input_types();
        let body_output_types = body.output_types();
        let admits_iterations =
            self.length().value() != Some(0) && stable_hlo_dynamic_dimension_bound(self.length()) != Some(0);
        if admits_iterations {
            let slice_types = body_input_types[1 + carry_count..]
                .iter()
                .enumerate()
                .map(|(index, r#type)| ("input", index, r#type))
                .chain(
                    body_output_types[carry_count..]
                        .iter()
                        .enumerate()
                        .map(|(index, r#type)| ("output", index, r#type)),
                );
            for (role, index, slice_type) in slice_types {
                if let ArrayIrType::Array(slice_type) = slice_type
                    && slice_type.shape().dimensions().iter().any(|dimension| dimension.value().is_none())
                {
                    return Err(LoweringError::UnsupportedOp {
                        op: format!(
                            "`{SCAN_OPERATION_NAME}` with the dynamic slice type `{slice_type}` for stacked {role} \
                             {index}; lowering requires static slice types, so specialize the body to the scan's \
                             input types",
                        ),
                    });
                }
            }
        }
        let results = lower_scan_to_while(
            self,
            body,
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
        // The lowering produces the carries at their declared types and the stacked outputs stacked over the declared
        // length, which the instruction's output types may refine (e.g., to the trip count of an exact runtime length).
        let mut region_output_types = body_input_types[1..1 + carry_count].to_vec();
        for output_type in &body_output_types[carry_count..] {
            region_output_types.push(match output_type {
                ArrayIrType::Array(slice_type) => {
                    let mut dimensions = vec![self.length().clone()];
                    dimensions.extend(slice_type.shape().dimensions().iter().cloned());
                    ArrayIrType::Array(slice_type.clone().with_shape(Shape::new(dimensions)))
                }
                output_type => output_type.clone(),
            });
        }
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
        AddOperation, ArrayOperation, CompareOperation, DimensionBounds, DimensionSizeOperation, DimensionType,
        DimensionValue, DimensionVariable, MulOperation, Placeholder, RandomAlgorithm, RngBitGeneratorOperation,
        ScanOperation,
    };

    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, test_vector_type,
    };
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::tests::execution_client;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_program_lowers_scan_to_while() {
        // A primal scan lowers to a `stablehlo.while` over `[counter, carries..., xs..., ys...]`: each iteration
        // reads one slice of the stacked inputs with `stablehlo.dynamic_slice`, inlines the body, and writes the
        // per-iteration outputs into preallocated zero accumulators with `stablehlo.dynamic_update_slice` (the
        // strategy JAX uses for `lax.scan`, which is not an XLA primitive).
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f32.clone().into());
        let x = body_builder.add_input(scalar_f32.clone().into());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 3);

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone().into());
        let stacked_type = test_vector_type(3);
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3xf32>
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0) \
                : tensor<i64>, tensor<f32>, tensor<3xf32>, tensor<3xf32>
                    cond {
                      %c_3 = stablehlo.constant dense<3> : tensor<i64>
                      %2 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %2 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xf32>, tensor<i64>) \
                -> tensor<1xf32>
                      %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
                      %4 = stablehlo.multiply %iterArg_0, %3 : tensor<f32>
                      %5 = stablehlo.reshape %4 : (tensor<f32>) -> tensor<1xf32>
                      %6 = stablehlo.dynamic_update_slice %iterArg_2, %5, %iterArg : (tensor<3xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<3xf32>
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %7 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                      stablehlo.return %7, %4, %iterArg_1, %6 : tensor<i64>, tensor<f32>, tensor<3xf32>, \
                tensor<3xf32>
                    }
                    return %1#1, %1#3 : tensor<f32>, tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_scan_carried_dimension_extent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));

        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let extent = builder.add_input(extent_type.into());
            let value = builder.add_input(dynamic_vector_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![extent, value],
                    vec![Placeholder, Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let values = builder.add_input(dynamic_vector_type.clone().into());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![values],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(2, 3)),
                vec![body_region],
                vec![extent, values],
                None,
            )
            .unwrap()[1];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type.clone()],
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
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<i32>) -> (tensor<?xf32, \
                #stablehlo.bounds<8>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<8xf32>, tensor<i32>) -> \
                tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> \
                tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %3:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %2, %iterArg_1 = %0) : tensor<i64>, \
                tensor<i64>, tensor<?xf32, #stablehlo.bounds<8>>
                    cond {
                      %c_2 = stablehlo.constant dense<3> : tensor<i64>
                      %6 = stablehlo.compare LT, %iterArg, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %c_2 = stablehlo.constant dense<1> : tensor<i64>
                      %6 = stablehlo.add %iterArg, %c_2 : tensor<i64>
                      stablehlo.return %6, %iterArg_0, %iterArg_1 : tensor<i64>, tensor<i64>, tensor<?xf32, \
                #stablehlo.bounds<8>>
                    }
                    %4 = stablehlo.get_dimension_size %3#2, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> \
                tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %3#2, %5 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dynamic_scan_length_as_scalar_ssa() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length_type = DimensionType::from(length.clone());
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));

        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![carry, carry],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.clone().into());
        let runtime_length = builder.add_input(length_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body_region],
                vec![carry, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(scalar_type.clone(), ArrayType::scalar(DataType::I64)),
            &(scalar_type, stacked_type),
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>) -> (tensor<f32>, tensor<?xf32, \
                #stablehlo.bounds<8>>, tensor<i64>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                    %2 = stablehlo.set_dimension_size %0, %1, dim = 0 : (tensor<8xf32>, tensor<i32>) -> \
                tensor<?xf32, #stablehlo.bounds<8>>
                    %3:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %2) : tensor<i64>, \
                tensor<f32>, tensor<?xf32, #stablehlo.bounds<8>>
                    cond {
                      %6 = stablehlo.compare LT, %iterArg, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %6 = stablehlo.reshape %iterArg_0 : (tensor<f32>) -> tensor<1xf32>
                      %7 = stablehlo.dynamic_update_slice %iterArg_1, %6, %iterArg : (tensor<?xf32, \
                #stablehlo.bounds<8>>, tensor<1xf32>, tensor<i64>) -> tensor<?xf32, #stablehlo.bounds<8>>
                      %c_2 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.add %iterArg, %c_2 : tensor<i64>
                      stablehlo.return %8, %iterArg_0, %7 : tensor<i64>, tensor<f32>, tensor<?xf32, \
                #stablehlo.bounds<8>>
                    }
                    %4 = stablehlo.get_dimension_size %3#2, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> \
                tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %3#1, %3#2, %5 : tensor<f32>, tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_elides_dynamic_scan_with_only_zero_length() {
        // This runtime dimension can only be zero. Its output stack has a physical leading bound of zero, so an
        // emitted loop body would contain an invalid unit-length update even though it could never execute.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(1)).unwrap());
        let length_type = DimensionType::from(length.clone());
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![carry, carry],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.clone().into());
        let runtime_length = builder.add_input(length_type.into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length)).with_unroll(3).unwrap()),
                vec![body_region],
                vec![carry, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(scalar_type.clone(), ArrayType::scalar(DataType::I64)),
            &(scalar_type, stacked_type),
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>) -> (tensor<f32>, \
                  tensor<?xf32, #stablehlo.bounds<0>>, tensor<i64>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<0xf32>
                    %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                    %2 = stablehlo.set_dimension_size %0, %1, dim = 0 : (tensor<0xf32>, tensor<i32>) -> \
                    tensor<?xf32, #stablehlo.bounds<0>>
                    %3 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?xf32, #stablehlo.bounds<0>>) -> tensor<i32>
                    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
                    return %arg0, %2, %4 : tensor<f32>, tensor<?xf32, #stablehlo.bounds<0>>, tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_mapped_rng_as_dynamic_scan() {
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::from(batch.clone());
        let state_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let stacked_state_type =
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]));
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(2)]));
        let stacked_bits_type =
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]));

        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let state = builder.add_input(state_type.into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::RngBitGenerator(RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type)),
                    Vec::new(),
                    vec![state],
                    None,
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    outputs,
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let states = builder.add_input(stacked_state_type.clone().into());
        let runtime_batch = builder.add_input(batch_type.into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(0, Dimension::Dynamic(batch))),
                vec![body_region],
                vec![states, runtime_batch],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(stacked_state_type.clone(), ArrayType::scalar(DataType::I64)),
            &(stacked_state_type, stacked_bits_type),
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<8x2xui64>, %arg1: tensor<i64>, %arg2: tensor<i32>) -> \
                (tensor<?x2xui64, #stablehlo.bounds<8, ?>>, tensor<?x2xui32, #stablehlo.bounds<8, ?>>, tensor<i64>, \
                tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<8x2xui64>, tensor<i32>) -> \
                tensor<?x2xui64, #stablehlo.bounds<8, ?>>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %c_0 = stablehlo.constant dense<0> : tensor<ui64>
                    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<8x2xui64>
                    %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.set_dimension_size %1, %2, dim = 0 : (tensor<8x2xui64>, tensor<i32>) -> \
                tensor<?x2xui64, #stablehlo.bounds<8, ?>>
                    %c_1 = stablehlo.constant dense<0> : tensor<ui32>
                    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<8x2xui32>
                    %5 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
                    %6 = stablehlo.set_dimension_size %4, %5, dim = 0 : (tensor<8x2xui32>, tensor<i32>) -> \
                tensor<?x2xui32, #stablehlo.bounds<8, ?>>
                    %7:4 = stablehlo.while(%iterArg = %c, %iterArg_2 = %0, %iterArg_3 = %3, %iterArg_4 = %6) : \
                tensor<i64>, tensor<?x2xui64, #stablehlo.bounds<8, ?>>, tensor<?x2xui64, #stablehlo.bounds<8, ?>>, \
                tensor<?x2xui32, #stablehlo.bounds<8, ?>>
                    cond {
                      %13 = stablehlo.compare LT, %iterArg, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %13 : tensor<i1>
                    } do {
                      %c_5 = stablehlo.constant dense<0> : tensor<i64>
                      %13 = stablehlo.dynamic_slice %iterArg_2, %iterArg, %c_5, sizes = [1, 2] : (tensor<?x2xui64, \
                #stablehlo.bounds<8, ?>>, tensor<i64>, tensor<i64>) -> tensor<1x2xui64>
                      %14 = stablehlo.reshape %13 : (tensor<1x2xui64>) -> tensor<2xui64>
                      %output_state, %output = stablehlo.rng_bit_generator %14, algorithm = THREE_FRY : \
                (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui32>)
                      %15 = stablehlo.reshape %output_state : (tensor<2xui64>) -> tensor<1x2xui64>
                      %16 = stablehlo.dynamic_update_slice %iterArg_3, %15, %iterArg, %c_5 : (tensor<?x2xui64, \
                #stablehlo.bounds<8, ?>>, tensor<1x2xui64>, tensor<i64>, tensor<i64>) -> tensor<?x2xui64, \
                #stablehlo.bounds<8, ?>>
                      %17 = stablehlo.reshape %output : (tensor<2xui32>) -> tensor<1x2xui32>
                      %18 = stablehlo.dynamic_update_slice %iterArg_4, %17, %iterArg, %c_5 : (tensor<?x2xui32, \
                #stablehlo.bounds<8, ?>>, tensor<1x2xui32>, tensor<i64>, tensor<i64>) -> tensor<?x2xui32, \
                #stablehlo.bounds<8, ?>>
                      %c_6 = stablehlo.constant dense<1> : tensor<i64>
                      %19 = stablehlo.add %iterArg, %c_6 : tensor<i64>
                      stablehlo.return %19, %iterArg_2, %16, %18 : tensor<i64>, tensor<?x2xui64, \
                #stablehlo.bounds<8, ?>>, tensor<?x2xui64, #stablehlo.bounds<8, ?>>, tensor<?x2xui32, \
                #stablehlo.bounds<8, ?>>
                    }
                    %8:2 = stablehlo.optimization_barrier %7#2, %7#3 : tensor<?x2xui64, #stablehlo.bounds<8, ?>>, \
                tensor<?x2xui32, #stablehlo.bounds<8, ?>>
                    %9 = stablehlo.get_dimension_size %7#2, dim = 0 : (tensor<?x2xui64, #stablehlo.bounds<8, ?>>) \
                -> tensor<i32>
                    %10 = stablehlo.convert %9 : (tensor<i32>) -> tensor<i64>
                    %11 = stablehlo.get_dimension_size %7#3, dim = 0 : (tensor<?x2xui32, #stablehlo.bounds<8, ?>>) \
                -> tensor<i32>
                    %12 = stablehlo.convert %11 : (tensor<i32>) -> tensor<i64>
                    return %8#0, %8#1, %10, %12 : tensor<?x2xui64, #stablehlo.bounds<8, ?>>, tensor<?x2xui32, \
                #stablehlo.bounds<8, ?>>, tensor<i64>, tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_fully_unrolled_scan_without_while() {
        // A scan whose unroll factor equals its length lowers to straight-line operations: no `stablehlo.while` is
        // emitted at all and the body inlines once per iteration (three `stablehlo.multiply` copies for `length = 3`).
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f32.clone().into());
        let x = body_builder.add_input(scalar_f32.clone().into());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 3).with_unroll(3).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone().into());
        let stacked_type = test_vector_type(3);
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3xf32>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %1 = stablehlo.dynamic_slice %arg1, %c, sizes = [1] : (tensor<3xf32>, tensor<i64>) -> \
                tensor<1xf32>
                    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
                    %3 = stablehlo.multiply %arg0, %2 : tensor<f32>
                    %4 = stablehlo.reshape %3 : (tensor<f32>) -> tensor<1xf32>
                    %5 = stablehlo.dynamic_update_slice %0, %4, %c : (tensor<3xf32>, tensor<1xf32>, tensor<i64>) -> \
                tensor<3xf32>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                    %6 = stablehlo.dynamic_slice %arg1, %c_0, sizes = [1] : (tensor<3xf32>, tensor<i64>) -> \
                tensor<1xf32>
                    %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
                    %8 = stablehlo.multiply %3, %7 : tensor<f32>
                    %9 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
                    %10 = stablehlo.dynamic_update_slice %5, %9, %c_0 : (tensor<3xf32>, tensor<1xf32>, tensor<i64>) \
                -> tensor<3xf32>
                    %c_1 = stablehlo.constant dense<2> : tensor<i64>
                    %11 = stablehlo.dynamic_slice %arg1, %c_1, sizes = [1] : (tensor<3xf32>, tensor<i64>) -> \
                tensor<1xf32>
                    %12 = stablehlo.reshape %11 : (tensor<1xf32>) -> tensor<f32>
                    %13 = stablehlo.multiply %8, %12 : tensor<f32>
                    %14 = stablehlo.reshape %13 : (tensor<f32>) -> tensor<1xf32>
                    %15 = stablehlo.dynamic_update_slice %10, %14, %c_1 : (tensor<3xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<3xf32>
                    return %13, %15 : tensor<f32>, tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_partially_unrolled_scan() {
        // A scan with `unroll = 2` over `length = 4` keeps the `stablehlo.while` skeleton but runs two body copies
        // per loop trip: the body region contains two `stablehlo.multiply` copies (and one iteration read/write pair
        // per copy) while the counter advances by the unroll factor.
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f32.clone().into());
        let x = body_builder.add_input(scalar_f32.clone().into());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 4).with_unroll(2).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone().into());
        let stacked_type = test_vector_type(4);
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<f32>, tensor<4xf32>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0) \
                : tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xf32>
                    cond {
                      %c_3 = stablehlo.constant dense<4> : tensor<i64>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %2 = stablehlo.subtract %c_3, %c_4 : tensor<i64>
                      %3 = stablehlo.compare LT, %iterArg, %2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %3 : tensor<i1>
                    } do {
                      %2 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<4xf32>, tensor<i64>) \
                -> tensor<1xf32>
                      %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
                      %4 = stablehlo.multiply %iterArg_0, %3 : tensor<f32>
                      %5 = stablehlo.reshape %4 : (tensor<f32>) -> tensor<1xf32>
                      %6 = stablehlo.dynamic_update_slice %iterArg_2, %5, %iterArg : (tensor<4xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<4xf32>
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %7 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                      %8 = stablehlo.dynamic_slice %iterArg_1, %7, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> \
                tensor<1xf32>
                      %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
                      %10 = stablehlo.multiply %4, %9 : tensor<f32>
                      %11 = stablehlo.reshape %10 : (tensor<f32>) -> tensor<1xf32>
                      %12 = stablehlo.dynamic_update_slice %6, %11, %7 : (tensor<4xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<4xf32>
                      %c_4 = stablehlo.constant dense<2> : tensor<i64>
                      %13 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %13, %10, %iterArg_1, %12 : tensor<i64>, tensor<f32>, tensor<4xf32>, \
                tensor<4xf32>
                    }
                    return %1#1, %1#3 : tensor<f32>, tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_checks_unrolled_scan_counter_without_overflow() {
        // A length of `i64::MAX` with an unroll factor of three leaves one remaining iteration. At the final loop
        // condition, adding two to its counter (`i64::MAX - 1`) would wrap and incorrectly continue the loop.
        // The body has only a scalar carry, so lowering this boundary case requires no enormous array allocations.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let index = body_builder.add_input(scalar_type.clone().into());
        body_builder.add_input(scalar_type.clone().into());
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![index], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = ScanOperation::new(1, i64::MAX as usize).with_unroll(3).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![initial], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0) : tensor<i64>, tensor<i64>
                    cond {
                      %c_2 = stablehlo.constant dense<9223372036854775807> : tensor<i64>
                      %c_3 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.subtract %c_2, %c_3 : tensor<i64>
                      %2 = stablehlo.compare LT, %iterArg, %1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %c_2 = stablehlo.constant dense<1> : tensor<i64>
                      %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
                      %c_3 = stablehlo.constant dense<2> : tensor<i64>
                      %2 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                      %c_4 = stablehlo.constant dense<3> : tensor<i64>
                      %3 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %3, %2 : tensor<i64>, tensor<i64>
                    }
                    %c_0 = stablehlo.constant dense<9223372036854775806> : tensor<i64>
                    return %c_0 : tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_partially_unrolled_scan_with_remaining_iterations() {
        // A scan with `unroll = 2` over `length = 5` runs two loop trips of two body copies each and then one
        // straight-line body copy for the remaining iteration: one `stablehlo.while` and three `stablehlo.multiply`
        // copies in total.
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f32.clone().into());
        let x = body_builder.add_input(scalar_f32.clone().into());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 5).with_unroll(2).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone().into());
        let stacked_type = test_vector_type(5);
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<5xf32>) -> (tensor<f32>, tensor<5xf32>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5xf32>
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %0) \
                : tensor<i64>, tensor<f32>, tensor<5xf32>, tensor<5xf32>
                    cond {
                      %c_4 = stablehlo.constant dense<5> : tensor<i64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %7 = stablehlo.subtract %c_4, %c_5 : tensor<i64>
                      %8 = stablehlo.compare LT, %iterArg, %7, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %8 : tensor<i1>
                    } do {
                      %7 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<5xf32>, tensor<i64>) \
                -> tensor<1xf32>
                      %8 = stablehlo.reshape %7 : (tensor<1xf32>) -> tensor<f32>
                      %9 = stablehlo.multiply %iterArg_1, %8 : tensor<f32>
                      %10 = stablehlo.reshape %9 : (tensor<f32>) -> tensor<1xf32>
                      %11 = stablehlo.dynamic_update_slice %iterArg_3, %10, %iterArg : (tensor<5xf32>, \
                tensor<1xf32>, tensor<i64>) -> tensor<5xf32>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %12 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      %13 = stablehlo.dynamic_slice %iterArg_2, %12, sizes = [1] : (tensor<5xf32>, tensor<i64>) -> \
                tensor<1xf32>
                      %14 = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
                      %15 = stablehlo.multiply %9, %14 : tensor<f32>
                      %16 = stablehlo.reshape %15 : (tensor<f32>) -> tensor<1xf32>
                      %17 = stablehlo.dynamic_update_slice %11, %16, %12 : (tensor<5xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<5xf32>
                      %c_5 = stablehlo.constant dense<2> : tensor<i64>
                      %18 = stablehlo.add %iterArg, %c_5 : tensor<i64>
                      stablehlo.return %18, %15, %iterArg_2, %17 : tensor<i64>, tensor<f32>, tensor<5xf32>, \
                tensor<5xf32>
                    }
                    %c_0 = stablehlo.constant dense<4> : tensor<i64>
                    %2 = stablehlo.dynamic_slice %1#2, %c_0, sizes = [1] : (tensor<5xf32>, tensor<i64>) -> \
                tensor<1xf32>
                    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
                    %4 = stablehlo.multiply %1#1, %3 : tensor<f32>
                    %5 = stablehlo.reshape %4 : (tensor<f32>) -> tensor<1xf32>
                    %6 = stablehlo.dynamic_update_slice %1#3, %5, %c_0 : (tensor<5xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<5xf32>
                    return %4, %6 : tensor<f32>, tensor<5xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_zero_length_scan_without_while() {
        // A zero-length scan runs no iteration, regardless of its unroll factor: it returns its initial carry and an
        // empty stacked output without emitting a `stablehlo.while` or any body copy.
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f32.clone().into());
        let x = body_builder.add_input(scalar_f32.clone().into());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 0).with_unroll(2).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone().into());
        let stacked_type = test_vector_type(0);
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<0xf32>) -> (tensor<f32>, tensor<0xf32>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<0xf32>
                    return %arg0, %0 : tensor<f32>, tensor<0xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_scan_body_print_with_token_loop_state() {
        use ryft_core::{PrintOperation, ScanOperation as CoreScanOperation};

        // A print inside a scan body makes the lowered `stablehlo.while` carry the effect token as one extra
        // trailing state element: the entry token is created before the loop, both regions receive it as an extra
        // block argument, the body threads it through the print custom call, and the loop's trailing result
        // continues the chain (unused here because the program ends right after the scan).
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f64.clone().into());
        let x = body_builder.add_input(scalar_f64.clone().into());
        let printed = body_builder
            .add_instruction(PrintOperation::<ArrayType>::new("iteration"), Vec::new(), vec![x], None)
            .unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, printed], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![sum],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 3);

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f64.clone().into());
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![scalar_f64.clone(), stacked_type];
        let output_types = vec![scalar_f64];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> tensor<f64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0) : tensor<i64>, tensor<f64>, tensor<3xf64>, !stablehlo.token
                    cond {
                      %c_3 = stablehlo.constant dense<3> : tensor<i64>
                      %2 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %2 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
                      %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
                      %4 = stablehlo.custom_call @ryft.print(%3, %iterArg_2) {api_version = 4 : i32, backend_config = {label = "iteration"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      %5 = stablehlo.add %iterArg_0, %3 : tensor<f64>
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %6 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                      stablehlo.return %6, %5, %iterArg_1, %4 : tensor<i64>, tensor<f64>, tensor<3xf64>, !stablehlo.token
                    }
                    return %1#1 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_threads_both_ordered_effect_classes_through_scan() {
        use ryft_core::{AssertOperation, PrintOperation, ScanOperation as CoreScanOperation};

        // The scan body uses both ordered classes: the scalar check contributes an assertion and the print
        // contributes ordered I/O. The loop state carries two independent trailing tokens in canonical class order.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let value = body_builder.add_input(scalar_type.clone().into());
        let predicate = body_builder
            .add_instruction(
                XlaOperation::Array(ArrayOperation::Compare(CompareOperation::<ArrayType>::new(
                    ComparisonDirection::Equal,
                ))),
                Vec::new(),
                vec![carry, value],
                None,
            )
            .unwrap()[0];
        body_builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("scan value must equal carry")
                    .with_labels(vec!["value".to_owned(), "carry".to_owned()]),
                Vec::new(),
                vec![predicate, value, carry],
                None,
            )
            .unwrap();
        body_builder
            .add_instruction(PrintOperation::<ArrayType>::new("iteration"), Vec::new(), vec![value], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![carry],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 3);

        let stacked_type = ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let stacked = builder.add_input(stacked_type.clone().into());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![initial, stacked], None)
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
            &vec![scalar_type.clone(), stacked_type],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<i64>, %arg1: tensor<3xi64>) -> tensor<i64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:5 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0, \
                %iterArg_3 = %1) : tensor<i64>, tensor<i64>, tensor<3xi64>, !stablehlo.token, !stablehlo.token
                    cond {
                      %c_4 = stablehlo.constant dense<3> : tensor<i64>
                      %3 = stablehlo.compare LT, %iterArg, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %3 : tensor<i1>
                    } do {
                      %3 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xi64>, tensor<i64>) \
                -> tensor<1xi64>
                      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
                      %5 = stablehlo.compare EQ, %iterArg_0, %4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %6 = stablehlo.custom_call @ryft.assert(%5, %4, %iterArg_0, %iterArg_2) {api_version = 4 : \
                i32, backend_config = {actor = \"assert\", kind = \"generic\", label_0 = \"value\", label_1 = \
                \"carry\", label_count = \"2\", message = \"scan value must equal carry\"}, has_side_effect = true} \
                : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                      %7 = stablehlo.custom_call @ryft.print(%4, %iterArg_3) {api_version = 4 : i32, backend_config \
                = {label = \"iteration\"}, has_side_effect = true} : (tensor<i64>, !stablehlo.token) -> \
                !stablehlo.token
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %8, %iterArg_0, %iterArg_1, %6, %7 : tensor<i64>, tensor<i64>, \
                tensor<3xi64>, !stablehlo.token, !stablehlo.token
                    }
                    return %2#1 : tensor<i64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_scan_body_print_with_token_loop_state_executes_on_cpu() {
        use std::sync::Arc;

        use ryft_core::{PrintOperation, ScanOperation as CoreScanOperation};
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
        use crate::tests::{values_from_bytes, values_to_bytes};

        // Executes the token-in-loop-state lowering on the real CPU plugin: the effect token is a
        // `stablehlo.while` carry and each iteration's print custom call consumes and produces it, so this pins that
        // XLA accepts and runs token-carrying loops (not just the flat token chain).
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_f64.clone().into());
        let x = body_builder.add_input(scalar_f64.clone().into());
        let printed = body_builder
            .add_instruction(PrintOperation::<ArrayType>::new("iteration"), Vec::new(), vec![x], None)
            .unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, printed], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![sum],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::new(1, 3);

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f64.clone().into());
        let stacked_inputs = builder.add_input(stacked_type.clone().into());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![scalar_f64.clone(), stacked_type];
        let output_types = vec![scalar_f64];
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
        let init_bytes = values_to_bytes::<f64>(&[1.0]);
        let stacked_bytes = values_to_bytes::<f64>(&[1.0, 2.0, 3.0]);

        let ((), lines) = with_captured_prints(|| {
            let inputs = ExecutionDeviceInputs {
                inputs: &[
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(init_bytes.as_slice(), BufferType::F64, &[], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(stacked_bytes.as_slice(), BufferType::F64, &[3], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                ],
                ..Default::default()
            };
            let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
            let mut outputs = execution.block_until_ready().unwrap().remove(0);
            assert_eq!(outputs.outputs.len(), 1);
            let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![7.0]);
        });

        // One print per scan iteration, in iteration order.
        assert_eq!(
            lines,
            vec!["iteration: 1.0".to_string(), "iteration: 2.0".to_string(), "iteration: 3.0".to_string()],
        );
    }

    #[test]
    fn test_scan_with_a_zero_runtime_length_under_a_nonzero_bound_executes_no_iteration() {
        // A dynamic length whose bound admits several iterations can still be zero at runtime. Every lowered form (the
        // loop, a partially unrolled loop, and a fully unrolled body) must then run no iteration, returning the initial
        // carry and an empty stacked output, and must run exactly `length` iterations otherwise. The body
        // `(i, c) -> (c + c, c)` doubles its carry and stacks the carry it received.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let doubled = builder
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![carry, carry], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![doubled, carry],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        for unroll in [1, 2, 5] {
            let mut builder = CompositeXlaProgramBuilder::new();
            let body_region = builder.import_region(body.entry_region_ref());
            let carry = builder.add_input(scalar_type.clone().into());
            let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::Scan(
                        ScanOperation::new(1, Dimension::Dynamic(length.clone())).with_unroll(unroll).unwrap(),
                    ),
                    vec![body_region],
                    vec![carry, runtime_length],
                    None,
                )
                .unwrap()
                .to_vec();
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap();
            let client = execution_client();
            let run = |runtime_length: i64| {
                execute_mixed_program(
                    &client,
                    &program,
                    &[MixedValue::Array(vec![1.0], Vec::new()), MixedValue::Dimension(runtime_length)],
                    &[],
                )
            };
            assert_eq!(
                run(0),
                Ok(vec![
                    MixedValue::Array(vec![1.0], Vec::new()),
                    MixedValue::Array(Vec::new(), vec![0]),
                    MixedValue::Dimension(0),
                ]),
                "unroll factor {unroll}",
            );
            assert_eq!(
                run(3),
                Ok(vec![
                    MixedValue::Array(vec![8.0], Vec::new()),
                    MixedValue::Array(vec![1.0, 2.0, 4.0], vec![3]),
                    MixedValue::Dimension(3),
                ]),
                "unroll factor {unroll}",
            );
        }
    }

    #[test]
    fn test_scan_converts_a_refined_static_carry_to_its_bounded_carry_type() {
        // A scan's initial carries only need to refine the carry types, so a static `f64[3]` initial carry feeds a
        // carry with the bounded dynamic type `f64[rows]`. The carry outputs take the static extent that the inputs
        // establish for `rows`, so the bounded loop results are sliced back to `f64[3]`. Two iterations of
        // `(a, b) -> (a, a + b)` compute `[a, 2a + b]`. The dynamic carry leaves `rows = 3` unproven, so an entry guard
        // checks its runtime size before the loop.
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let carry_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let first = builder.add_input(carry_type.clone().into());
            let second = builder.add_input(carry_type.clone().into());
            let sum = builder
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![first, second], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![first, sum],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let dynamic_input = builder.add_input(carry_type.clone().into());
        let static_input = builder.add_input(static_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(2, Dimension::Static(2))),
                vec![body],
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
            vec![ArrayIrType::Array(static_type.clone()), ArrayIrType::Array(static_type)],
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
                MixedValue::Array(vec![12.0, 24.0, 36.0], vec![3]),
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
        assert!(error.contains("input dimension `rows` must have one extent at the entry of `scan`"), "{error}");
    }

    #[test]
    fn test_scan_guards_its_entry_on_every_lowering_path() {
        // The entry of the `(a, b) -> (a, a + b)` scan relates a bounded dynamic `f64[rows]` carry to a static `f64[3]`
        // carry, and eager entry validation checks that relationship whatever the trip count is. A zero-length scan, a
        // fully unrolled scan, and a looping scan therefore all reject a runtime `rows` of 2.
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let carry_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let first = builder.add_input(carry_type.clone().into());
            let second = builder.add_input(carry_type.clone().into());
            let sum = builder
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![first, second], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![first, sum],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let client = execution_client();
        ensure_assertion_handler_registered(&client).unwrap();
        for (length, unroll, expected) in
            [(0, 1, vec![10.0, 20.0, 30.0]), (2, 1, vec![12.0, 24.0, 36.0]), (2, 2, vec![12.0, 24.0, 36.0])]
        {
            let mut builder = CompositeXlaProgramBuilder::new();
            let region = builder.import_region(body.entry_region_ref());
            let dynamic_input = builder.add_input(carry_type.clone().into());
            let static_input = builder.add_input(static_type.clone().into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(2, Dimension::Static(length)).with_unroll(unroll).unwrap()),
                    vec![region],
                    vec![dynamic_input, static_input],
                    None,
                )
                .unwrap()
                .to_vec();
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap();
            let inputs = [
                MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]),
                MixedValue::Array(vec![10.0, 20.0, 30.0], vec![3]),
            ];
            assert_eq!(
                execute_mixed_program(&client, &program, &inputs, &[3]),
                Ok(vec![MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]), MixedValue::Array(expected, vec![3])]),
                "length {length}, unroll factor {unroll}",
            );
            let error = execute_mixed_program(&client, &program, &inputs, &[2]).unwrap_err().to_string();
            assert!(
                error.contains("input dimension `rows` must have one extent at the entry of `scan`"),
                "length {length}, unroll factor {unroll}: {error}",
            );
        }
    }

    #[test]
    fn test_scan_entry_keeps_the_memory_placement_of_stacked_inputs() {
        // The entry of a scan over host-pinned slices declares its stacked inputs in host-pinned memory too, exactly as
        // core `scan` validation does, so the entry check accepts host-pinned stacks whatever the trip count is.
        let memory = ryft_core::Memory::Host { pinned: true };
        let slice_type = ArrayType::new_static(DataType::F64, [3]).with_memory(memory);
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let slice = builder.add_input(slice_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        for length in [0, 2] {
            let stacked_type = ArrayType::new_static(DataType::F64, [length, 3]).with_memory(memory);
            let mut builder = CompositeXlaProgramBuilder::new();
            let region = builder.import_region(body.entry_region_ref());
            let stack = builder.add_input(stacked_type.clone().into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(0, Dimension::Static(length))),
                    vec![region],
                    vec![stack],
                    None,
                )
                .unwrap()
                .to_vec();
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap();
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &[stacked_type.clone()],
                &[stacked_type.clone()],
                "main",
                None,
                None,
            );
            assert!(module.is_ok(), "length {length}: {module:?}");
        }
    }

    #[test]
    fn test_scan_entry_admits_stacks_sharded_along_the_scan_axis_over_auto_mesh_axes() {
        // A stack whose scan axis is sharded over an `Auto` mesh axis has replicated slices, which match a replicated
        // body declaration. The entry check uses the declared stacked types of core `scan` validation, which omit
        // shardings, so it accepts such a stack whatever the trip count is.
        use ryft_core::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let slice_type = ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let slice = builder.add_input(slice_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        for length in [0, 4] {
            let stacked_type = ArrayType::new_static(DataType::F64, [length])
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
                .unwrap();
            let mut builder = CompositeXlaProgramBuilder::new();
            let region = builder.import_region(body.entry_region_ref());
            let stack = builder.add_input(stacked_type.clone().into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(0, Dimension::Static(length))),
                    vec![region],
                    vec![stack],
                    None,
                )
                .unwrap()
                .to_vec();
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap();
            let output_type = program.output_types().remove(0);
            let output_type = <&ArrayType>::try_from(&output_type).unwrap().clone();
            let module =
                to_mlir_module_for_program(&program, &[], &[stacked_type.clone()], &[output_type], "main", None, None);
            assert!(module.is_ok(), "length {length}: {module:?}");
        }
    }

    #[test]
    fn test_scan_converts_static_stacked_inputs_of_an_exact_runtime_length() {
        // An exact runtime length refines the declared dynamic length, so a static `f64[4]` stacked input is valid,
        // while the loop state stacks it over the declared bounded length. A running sum computes `(10, [1, 3, 6, 10])`.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let slice = builder.add_input(scalar_type.clone().into());
            let sum = builder
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![carry, slice], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let stack = builder.add_input(ArrayType::new_static(DataType::F64, [4]).into());
        let runtime_length = builder
            .add_constant(XlaConstant::Dimension(DimensionValue::new(DimensionType::from(length.clone()), 4).unwrap()));
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body],
                vec![carry, stack, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            execute_mixed_program(
                &execution_client(),
                &program,
                &[MixedValue::Array(vec![0.0], vec![]), MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0], vec![4])],
                &[],
            ),
            Ok(vec![MixedValue::Array(vec![10.0], vec![]), MixedValue::Array(vec![1.0, 3.0, 6.0, 10.0], vec![4])]),
        );
    }

    #[test]
    fn test_scan_rejects_a_body_with_a_dynamic_slice_type() {
        // A static stacked input refines a body that declares the bounded slice type `f64[rows]`, so the scan
        // instruction type-checks. Its iterations would read slices at a dynamic shape, which this lowering does not
        // adapt, so it is rejected with a precise error instead of an opaque tensor-type failure. A zero-length scan
        // never slices, so it still lowers.
        let slice_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let scan_program = |length: usize| {
            let body = {
                let mut builder = CompositeXlaProgramBuilder::new();
                builder.add_input(ArrayType::scalar(DataType::I64).into());
                let slice = builder.add_input(slice_type.clone().into());
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap()
            };
            let mut builder = CompositeXlaProgramBuilder::new();
            let body = builder.import_region(body.entry_region_ref());
            let stacked_type = ArrayType::new_static(DataType::F64, [length, 3]);
            let stack = builder.add_input(stacked_type.clone().into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(0, Dimension::Static(length))),
                    vec![body],
                    vec![stack],
                    None,
                )
                .unwrap()
                .to_vec();
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert_eq!(program.output_types(), vec![ArrayIrType::Array(stacked_type.clone())]);
            to_mlir_module_for_program(&program, &[], &[stacked_type.clone()], &[stacked_type], "main", None, None)
        };
        assert_eq!(
            scan_program(2),
            Err(LoweringError::UnsupportedOp {
                op: "`scan` with the dynamic slice type `f64[rows]` for stacked input 0; lowering requires static \
                     slice types, so specialize the body to the scan's input types"
                    .to_string(),
            }),
        );
        let empty_module = scan_program(0).unwrap();
        assert_eq!(
            empty_module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<0x3xf64>) -> tensor<0x3xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<0x?xf64, \
                #stablehlo.bounds<?, 4>>
                    %c = stablehlo.constant dense<4> : tensor<i32>
                    %1 = stablehlo.set_dimension_size %0, %c, dim = 1 : (tensor<0x?xf64, #stablehlo.bounds<?, 4>>, \
                tensor<i32>) -> tensor<0x4xf64>
                    %2 = stablehlo.slice %1 [0:0, 0:3] : (tensor<0x4xf64>) -> tensor<0x3xf64>
                    return %2 : tensor<0x3xf64>
                  }
                }
            "},
        );
    }
}
