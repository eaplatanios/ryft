//! Lowering of [`ScanOperation`] to [StableHLO `while`](https://openxla.org/stablehlo/spec#while).
//!
//! A scan lowers only from the [`ArrayIrType`] operation family, which stores its body region; the homogeneous
//! [`ArrayType`] payload carries no region and is rejected. StableHLO has no scan primitive, so a scan lowers to a
//! `stablehlo.while` over the loop state `[counter, carries..., stacks..., ys...]`: an `i64` counter, the
//! carries, the stacked inputs, and one zero accumulator per stacked output, preallocated at the declared stacked
//! output type. Logical iteration `i` reads one slice of every stacked input at its iteration index (`i`, or
//! `length - 1 - i` when the scan is reversed) with `stablehlo.dynamic_slice`, inlines the body program over
//! `[index, carries..., slices...]`, and writes each stacked output slice into its accumulator with
//! `stablehlo.dynamic_update_slice`.
//!
//! Each loop trip runs `unroll` consecutive logical iterations, and the iterations that do not fill a whole trip follow
//! the loop: as straight-line body copies at static iteration indices for a static length, and as a second loop with
//! one body copy per trip for a dynamic length. A scan whose unroll factor covers its static length lowers to
//! straight-line body copies only, and a scan whose length admits no iteration lowers to its initial carries and empty
//! accumulators. On every path, an entry guard first checks the input relationships that the actual input types leave
//! unproven, including those with the runtime length of a dynamic-length scan. Every ordered effect class that the body
//! uses is threaded through the loop state as one trailing token, the results are normalized to the refined output
//! types of the scan, and results whose types place them over explicit mesh axes are constrained to that placement.
//! Reference state is discharged before lowering.

use std::rc::Rc;

use ryft_core::{
    ArrayIrType, ArrayType, ComparisonDirection, DataType, Dimension, EffectClasses, Operation, ProgramError,
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
use crate::experimental::lowering::{CollectiveLoweringState, EffectTokens};
use crate::experimental::ops::FlatXlaProgram;
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::control_flow::{
    append_loop_effect_tokens, lower_explicit_sharding_constraints, lower_refined_region_input,
    lower_refined_region_outputs, lower_region_guard, lowered_effect_classes, set_threaded_effect_tokens,
    threaded_effect_token_values, validate_unthreaded_effect_tokens,
};
use crate::operations::jit_call::JitCallFunctionMap;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Trip count of the loops that lower one scan.
#[derive(Copy, Clone)]
enum ScanLoopLength<'b, 'c: 'b, 't: 'c> {
    /// Static trip count.
    Static(usize),

    /// Lowered runtime length value of a dynamic trip count.
    Dynamic(ValueRef<'b, 'c, 't>),
}

/// Inputs of the lowering of one scan that its accumulators, its straight-line iterations, and its loops share.
struct ScanLoopInputs<'o, 'b, 'c: 'b, 't: 'c, O> {
    /// Lowered body program over `[index, carries..., x_slices...]`.
    body: &'o FlatXlaProgram<O>,

    /// Number of loop-carried state leaves.
    carry_count: usize,

    /// Per-iteration slice types of the stacked inputs.
    x_slice_types: Vec<ArrayType>,

    /// Per-iteration slice types of the stacked outputs.
    y_slice_types: Vec<ArrayType>,

    /// Declared types of the stacked outputs (refer to [`ScanOperation::declared_output_types`]), which keep the
    /// sharding and memory placement of their slices and which their accumulators therefore take.
    y_stack_types: Vec<ArrayType>,

    /// Types of the loop state `[counter, carries..., stacks..., ys...]`, excluding effect tokens.
    state_types: Vec<ArrayIrType>,

    /// Trip count of the scan.
    length: ScanLoopLength<'b, 'c, 't>,

    /// Whether logical iteration `i` reads and writes iteration index `length - 1 - i`.
    reverse: bool,

    /// Ordered effect classes whose tokens the loops carry as trailing state elements (refer to
    /// [`lowered_effect_classes`]). Straight-line body copies advance the tokens of the enclosing scope directly.
    threaded_effects: EffectClasses,

    /// MLIR context that owns the emitted operations.
    context: &'c MlirContext<'t>,

    /// Location of the emitted operations.
    location: LocationRef<'c, 't>,

    /// Lowered values of the enclosing function's captures.
    captured_values: &'o [ValueRef<'b, 'c, 't>],

    /// Nested functions that `jit_call` operations in the body lower against.
    nested_functions: Option<&'o Rc<JitCallFunctionMap<O>>>,

    /// Collective lowering state shared with the enclosing function.
    collective_state: &'o CollectiveLoweringState,
}

impl<'o, 'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily> ScanLoopInputs<'o, 'b, 'c, 't, O> {
    /// Lowers one zero accumulator per stacked output. Each accumulator is allocated at the physical bound of the scan
    /// length and, for a dynamic length, given the runtime length as the runtime extent of its leading axis with
    /// `stablehlo.set_dimension_size`.
    fn lower_accumulators(&self, block: &mut BlockRef<'b, 'c, 't>) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let (context, location) = (self.context, self.location);
        let mut accumulators = Vec::with_capacity(self.y_stack_types.len());
        for stack_type in &self.y_stack_types {
            let mut physical_dimensions = stack_type.shape().dimensions().to_vec();
            let length = &physical_dimensions[0];
            let physical_length = match length {
                Dimension::Static(length) => *length,
                Dimension::Dynamic(_) => {
                    stable_hlo_dynamic_dimension_bound(length).ok_or_else(|| LoweringError::UnsupportedOp {
                        op: format!(
                            "`{SCAN_OPERATION_NAME}` iteration count `{length}` needs a finite upper bound \
                             to allocate its stacked outputs",
                        ),
                    })?
                }
            };
            physical_dimensions[0] = Dimension::Static(physical_length);
            let physical_type = stack_type.clone().with_shape(Shape::new(physical_dimensions));
            let accumulator =
                lower_constant_output(std::slice::from_ref(&physical_type), 0, block, context, location)?.remove(0);
            let ScanLoopLength::Dynamic(runtime_length) = self.length else {
                accumulators.push(accumulator);
                continue;
            };
            let i32_scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
            let converted = block.append_operation(stable_hlo::convert(runtime_length, i32_scalar_type, location)?)?;
            let refined = block.append_operation(stable_hlo::set_dimension_size(
                accumulator,
                converted.result(0).unwrap().as_ref(),
                lower_tensor_type(stack_type, context, location)?,
                0,
                location,
            )?)?;
            accumulators.push(refined.result(0).unwrap().as_ref());
        }
        Ok(accumulators)
    }

    /// Emits logical iterations `first_iteration..length` of a scan of the static `length` as straight-line body copies
    /// at static iteration indices, which read and write the same stacked inputs and accumulators that a loop threads
    /// through its state, and returns the final carries followed by the final accumulators. The body copies advance
    /// the `effect_tokens` of the enclosing scope directly.
    fn lower_static_iterations(
        &self,
        first_iteration: usize,
        length: usize,
        mut carries: Vec<ValueRef<'b, 'c, 't>>,
        x_stacks: &[ValueRef<'b, 'c, 't>],
        mut y_accumulators: Vec<ValueRef<'b, 'c, 't>>,
        block: &mut BlockRef<'b, 'c, 't>,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let zero_index = lower_static_index_constants(&[0], block, self.context, self.location)?[0];
        for iteration in first_iteration..length {
            let index = if self.reverse { length - 1 - iteration } else { iteration };
            let index_value = lower_static_index_constants(&[index], block, self.context, self.location)?[0];
            (carries, y_accumulators) =
                self.lower_iteration(index_value, zero_index, carries, x_stacks, y_accumulators, block, effect_tokens)?;
        }
        carries.extend(y_accumulators);
        Ok(carries)
    }

    /// Emits one `stablehlo.while` over the loop state `[counter, carries..., stacks..., ys...]` that starts at
    /// `state_values` and runs `unroll` consecutive logical iterations (body copies) per trip while a whole trip fits
    /// (`counter + unroll - 1 < length`), advancing the counter by `unroll` per trip. The tokens of the threaded effect
    /// classes ride at the end of the loop state. The function returns the final loop state without the tokens, which
    /// it installs into `effect_tokens` instead.
    fn lower_loop(
        &self,
        mut state_values: Vec<ValueRef<'b, 'c, 't>>,
        unroll: usize,
        block: &mut BlockRef<'b, 'c, 't>,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let (context, location) = (self.context, self.location);
        let state_count = self.state_types.len();
        let mut state_types = self
            .state_types
            .iter()
            .map(|r#type| lower_array_ir_type(r#type, context, location).map(|tensor_type| tensor_type.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        append_loop_effect_tokens(
            self.threaded_effects,
            effect_tokens,
            &mut state_values,
            &mut state_types,
            block,
            context,
            location,
        )?;
        let block_arguments = state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();
        let length_value = |block: &mut BlockRef<'b, 'c, 't>| -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
            match self.length {
                ScanLoopLength::Static(length) => {
                    Ok(lower_static_index_constants(&[length], block, context, location)?[0])
                }
                ScanLoopLength::Dynamic(length) => Ok(length),
            }
        };

        let mut condition_region = context.region();
        let condition_block = context.block(block_arguments.as_slice());
        {
            let mut condition_block_ref = condition_block.as_ref();
            let counter = condition_block_ref.argument(0).unwrap().as_ref();
            let length_value = length_value(&mut condition_block_ref)?;
            // Subtract the last copy's offset from the length instead of adding it to the counter: after the final full
            // trip, `counter + unroll - 1` can overflow even though both the length and counter are valid `i64`
            // extents. This subtraction remains representable when a dynamic length is smaller than the unroll factor,
            // too.
            let counter_limit = if unroll == 1 {
                length_value
            } else {
                let offset =
                    lower_static_index_constants(&[unroll - 1], &mut condition_block_ref, context, location)?[0];
                let subtraction =
                    condition_block_ref.append_operation(stable_hlo::subtract(length_value, offset, location)?)?;
                subtraction.result(0).unwrap().as_ref()
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
            let arguments =
                (0..state_count).map(|index| body_block_ref.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
            let mut body_effect_tokens = EffectTokens::default();
            set_threaded_effect_tokens(&mut body_effect_tokens, self.threaded_effects, |offset| {
                body_block_ref.argument(state_count + offset).unwrap().as_ref()
            });
            let entry_effect_tokens = body_effect_tokens;
            let counter = arguments[0];
            let zero_index = lower_static_index_constants(&[0], &mut body_block_ref, context, location)?[0];
            // When the visit order is reversed, logical iteration `i` reads iteration `length - 1 - i` (a zero-length
            // reversed scan never runs its body, so the saturated limit constant is inert).
            let reverse_limit = if self.reverse {
                let length_value = length_value(&mut body_block_ref)?;
                let one = lower_static_index_constants(&[1], &mut body_block_ref, context, location)?[0];
                let limit = body_block_ref.append_operation(stable_hlo::subtract(length_value, one, location)?)?;
                Some(limit.result(0).unwrap().as_ref())
            } else {
                None
            };

            // Each loop trip runs `unroll` consecutive logical iterations (`counter + copy` for each body copy).
            let x_end = 1 + self.carry_count + self.x_slice_types.len();
            let mut carries = arguments[1..1 + self.carry_count].to_vec();
            let x_stacks = &arguments[1 + self.carry_count..x_end];
            let mut y_accumulators = arguments[x_end..].to_vec();
            for copy in 0..unroll {
                let iteration = if copy == 0 {
                    counter
                } else {
                    let offset = lower_static_index_constants(&[copy], &mut body_block_ref, context, location)?[0];
                    let addition = body_block_ref.append_operation(stable_hlo::add(counter, offset, location)?)?;
                    addition.result(0).unwrap().as_ref()
                };
                let index_value = match reverse_limit {
                    Some(limit) => {
                        let subtraction =
                            body_block_ref.append_operation(stable_hlo::subtract(limit, iteration, location)?)?;
                        subtraction.result(0).unwrap().as_ref()
                    }
                    None => iteration,
                };
                (carries, y_accumulators) = self.lower_iteration(
                    index_value,
                    zero_index,
                    carries,
                    x_stacks,
                    y_accumulators,
                    &mut body_block_ref,
                    &mut body_effect_tokens,
                )?;
            }
            validate_unthreaded_effect_tokens(entry_effect_tokens, body_effect_tokens, self.threaded_effects)?;

            // Assemble the next state: advance the counter by the unroll factor, thread the new carries, pass the input
            // stacks through unchanged, and thread the updated accumulators and effect tokens.
            let step = lower_static_index_constants(&[unroll], &mut body_block_ref, context, location)?[0];
            let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, step, location)?)?;
            let mut next_state = vec![next_counter.result(0).unwrap().as_ref()];
            next_state.extend(carries);
            next_state.extend_from_slice(x_stacks);
            next_state.extend(y_accumulators);
            next_state.extend(threaded_effect_token_values(body_effect_tokens, self.threaded_effects));
            body_block_ref.append_operation(stable_hlo::r#return(next_state.as_slice(), location)?)?;
        }
        body_region.append_block(body_block)?;

        let operation = block.append_operation(stable_hlo::r#while(
            state_values.as_slice(),
            condition_region.into(),
            body_region.into(),
            location,
        )?)?;
        set_threaded_effect_tokens(effect_tokens, self.threaded_effects, |offset| {
            operation.result(state_count + offset).unwrap().as_ref()
        });
        Ok((0..state_count).map(|index| operation.result(index).unwrap().as_ref()).collect())
    }

    /// Emits one scan iteration at iteration index `index_value` into `block`: reads slice `index_value` of every
    /// stacked input (dropping the unit iteration axis), inlines the body program over `[index, carries...,
    /// x_slices...]`, writes each stacked output slice into its accumulator at `index_value`, and returns the new
    /// carries and accumulators.
    fn lower_iteration(
        &self,
        index_value: ValueRef<'b, 'c, 't>,
        zero_index: ValueRef<'b, 'c, 't>,
        carries: Vec<ValueRef<'b, 'c, 't>>,
        x_stacks: &[ValueRef<'b, 'c, 't>],
        y_accumulators: Vec<ValueRef<'b, 'c, 't>>,
        block: &mut BlockRef<'b, 'c, 't>,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    ) -> Result<(Vec<ValueRef<'b, 'c, 't>>, Vec<ValueRef<'b, 'c, 't>>), LoweringError> {
        let location = self.location;
        let mut iteration_inputs = Vec::with_capacity(1 + self.carry_count + self.x_slice_types.len());
        iteration_inputs.push(index_value);
        iteration_inputs.extend(carries);
        for (x_stack, x_slice_type) in x_stacks.iter().zip(&self.x_slice_types) {
            let slice_dimensions = static_dimensions(x_slice_type)?;
            let mut sizes = vec![1];
            sizes.extend(slice_dimensions.iter().copied());
            let mut start_values = vec![index_value];
            start_values.extend(std::iter::repeat_n(zero_index, slice_dimensions.len()));
            let iteration = block.append_operation(stable_hlo::dynamic_slice(
                *x_stack,
                start_values.as_slice(),
                sizes.as_slice(),
                location,
            )?)?;
            let squeezed = block.append_operation(stable_hlo::reshape(
                iteration.result(0).unwrap().as_ref(),
                slice_dimensions.as_slice(),
                location,
            )?)?;
            iteration_inputs.push(squeezed.result(0).unwrap().as_ref());
        }

        let mut body_outputs = lower_nested_program_inline(
            self.body,
            iteration_inputs.as_slice(),
            block,
            self.context,
            location,
            self.captured_values,
            &RematerializationOptimizationBarrier::None,
            self.nested_functions,
            self.collective_state,
            effect_tokens,
        )?;
        let y_outputs = body_outputs.split_off(self.carry_count);
        let mut new_accumulators = Vec::with_capacity(self.y_slice_types.len());
        for ((y_output, y_slice_type), y_accumulator) in
            y_outputs.into_iter().zip(&self.y_slice_types).zip(y_accumulators)
        {
            let slice_dimensions = static_dimensions(y_slice_type)?;
            let mut expanded_dimensions = vec![1];
            expanded_dimensions.extend(slice_dimensions.iter().copied());
            let expanded =
                block.append_operation(stable_hlo::reshape(y_output, expanded_dimensions.as_slice(), location)?)?;
            let mut start_values = vec![index_value];
            start_values.extend(std::iter::repeat_n(zero_index, slice_dimensions.len()));
            let updated = block.append_operation(stable_hlo::dynamic_update_slice(
                y_accumulator,
                expanded.result(0).unwrap().as_ref(),
                start_values.as_slice(),
                location,
            )?)?;
            new_accumulators.push(updated.result(0).unwrap().as_ref());
        }
        Ok((body_outputs, new_accumulators))
    }
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
                "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
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
        // A scan's inputs are its carries and its stacked inputs, followed by the runtime length when its length is
        // dynamic, and its body receives the iteration index, which the loop supplies, before the carries and the
        // slices of the stacked inputs. The lowering proceeds in order:
        //
        //   1. Each initial carry is converted from its actual type to the type of its carry.
        //   2. An entry guard checks the input relationships that the actual input types leave unproven, including
        //      those with the runtime length. Eager entry validation checks them whatever the trip count is, so the
        //      guard precedes every path below.
        //   3. A scan whose length admits no iteration returns its carries and empty accumulators, and a scan whose
        //      unroll factor covers its static length runs straight-line body copies. Any other scan runs a loop whose
        //      trips run `unroll` body copies each, followed by its remaining iterations (refer to `ScanLoopInputs`).
        //   4. The carries and the accumulators are normalized to the refined `output_types` of the scan, and each
        //      result whose type places it over an explicit mesh axis is constrained to that placement.
        //
        // The program that attaches the scan validated its region and input counts, so a mismatch here is an internal
        // error, reported as the operation's own region contract and input count would report it.
        self.validate_region_count(lowerer.regions.len())?;
        let body = &lowerer.regions[0];
        let (context, location) = (lowerer.context, lowerer.location);
        let (carry_count, length) = (self.carry_count(), self.iteration_count());
        let body_input_types = body.input_types();
        let body_output_types = body.output_types();
        let entry_types = self.declared_input_types(&body_input_types).map_err(ProgramError::from)?;
        let expected = entry_types.len();
        for actual in [input_values.len(), input_types.len()] {
            if actual != expected {
                return Err(ProgramError::InvalidInputCount { expected, actual }.into());
            }
        }
        let stack_end = body_input_types.len() - 1;
        let declared_output_types = self.declared_output_types(&body_output_types).map_err(ProgramError::from)?;

        // Each iteration reads and writes one slice per stack at a static shape. A body that was not specialized to the
        // scan's input types may still declare dynamic slice types (refined only at the instruction boundary), which
        // this lowering does not adapt per iteration. A scan whose length admits no iteration never slices, so it is
        // lowered without its body.
        let static_length = length.value();
        let admits_iterations = static_length != Some(0) && stable_hlo_dynamic_dimension_bound(length) != Some(0);
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

        // A factor beyond the largest possible trip count unrolls nothing more, so it is clamped to that trip count.
        // Only a dynamic length without an upper bound can leave a factor whose iteration offsets the `i64` counter
        // cannot hold.
        let unroll = static_length
            .or_else(|| stable_hlo_dynamic_dimension_bound(length))
            .map_or(self.unroll(), |maximum_length| self.unroll().min(maximum_length.max(1)));
        if unroll > i64::MAX as usize {
            return Err(LoweringError::UnsupportedOp {
                op: format!("`{SCAN_OPERATION_NAME}` unroll factor {unroll} must be at most {}", i64::MAX),
            });
        }
        let array_types = |types: &[ArrayIrType]| {
            types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProgramError::from)
        };
        let y_stack_types = &declared_output_types[carry_count..];
        let mut state_types = Vec::with_capacity(1 + stack_end + y_stack_types.len());
        state_types.push(ArrayType::scalar(DataType::I64).into());
        state_types.extend_from_slice(&entry_types[..stack_end]);
        state_types.extend_from_slice(y_stack_types);
        let region_output_types = entry_types[..carry_count].iter().chain(y_stack_types).cloned().collect::<Vec<_>>();
        let inputs = ScanLoopInputs {
            body,
            carry_count,
            x_slice_types: array_types(&body_input_types[1 + carry_count..])?,
            y_slice_types: array_types(&body_output_types[carry_count..])?,
            y_stack_types: array_types(y_stack_types)?,
            state_types,
            length: match length {
                Dimension::Static(length) => ScanLoopLength::Static(*length),
                Dimension::Dynamic(_) => ScanLoopLength::Dynamic(input_values[stack_end]),
            },
            reverse: self.reverse(),
            threaded_effects: lowered_effect_classes(body)?,
            context,
            location,
            captured_values: lowerer.captured_values.as_slice(),
            nested_functions: lowerer.nested_functions.as_ref(),
            collective_state: &lowerer.collective_state,
        };
        let block = &mut lowerer.block;
        let effect_tokens = &mut lowerer.effect_tokens;

        // Initial carries only need to refine the carry types, so each one is converted to the lowered type of its
        // carry first. Stacked inputs are converted only where the loop state stacks them over the declared length
        // (below).
        let mut carries = input_values[..carry_count]
            .iter()
            .zip(input_types)
            .zip(&entry_types)
            .map(|((value, input_type), carry_type)| {
                lower_refined_region_input(*value, input_type, carry_type, block, context, location)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The body enters with the carries and with slices of the stacked inputs, whose identities the declared stacked
        // types carry unchanged next to the leading length axis, which the runtime length shares. Guarding the original
        // inputs therefore checks what eager entry validation checks.
        lower_region_guard(
            SCAN_OPERATION_NAME,
            &entry_types,
            input_types,
            input_values,
            effect_tokens,
            block,
            context,
            location,
        )?;

        let results = if !admits_iterations {
            // A scan whose length admits no iteration returns its initial carries and empty stacked outputs. This also
            // applies to a dynamic length whose bounds admit only zero: constructing a loop body would otherwise emit
            // unit-length slice reads or writes against zero-sized physical stacks, which are invalid even in a dead
            // body.
            carries.extend(inputs.lower_accumulators(block)?);
            carries
        } else if let Some(length) = static_length.filter(|length| unroll >= *length) {
            let y_accumulators = inputs.lower_accumulators(block)?;
            let x_stacks = &input_values[carry_count..stack_end];
            inputs.lower_static_iterations(0, length, carries, x_stacks, y_accumulators, block, effect_tokens)?
        } else {
            let mut state_values = vec![lower_static_index_constants(&[0], block, context, location)?[0]];
            state_values.extend(carries);
            // An exact runtime length (e.g., that of an eager dimension value) refines the stacked inputs of a
            // dynamic-length scan to static leading extents, while the loop state stacks them over the declared length,
            // so each one is converted to its declared stacked type like the carries.
            for index in carry_count..stack_end {
                state_values.push(lower_refined_region_input(
                    input_values[index],
                    &input_types[index],
                    &entry_types[index],
                    block,
                    context,
                    location,
                )?);
            }
            state_values.extend(inputs.lower_accumulators(block)?);
            let mut state = inputs.lower_loop(state_values, unroll, block, effect_tokens)?;

            // The iterations that do not fill a whole trip of the unrolled loop follow it: as a second loop with one
            // body copy per trip for a dynamic length, and as straight-line body copies for a static length.
            if static_length.is_none() && unroll > 1 {
                state = inputs.lower_loop(state, 1, block, effect_tokens)?;
            }
            let mut y_accumulators = state.split_off(1 + stack_end);
            let x_stacks = state.split_off(1 + carry_count);
            let mut carries = state.split_off(1);
            match static_length {
                Some(length) if length % unroll != 0 => {
                    let first_iteration = length - length % unroll;
                    inputs.lower_static_iterations(
                        first_iteration,
                        length,
                        carries,
                        &x_stacks,
                        y_accumulators,
                        block,
                        effect_tokens,
                    )?
                }
                _ => {
                    carries.append(&mut y_accumulators);
                    carries
                }
            }
        };

        // The lowering produces the carries at their declared types and the stacked outputs stacked over the declared
        // length, which the output types of the scan may refine (e.g., to the trip count of an exact runtime length).
        let results =
            lower_refined_region_outputs(results, &region_output_types, output_types, block, context, location)?;
        lower_explicit_sharding_constraints(
            results,
            output_types,
            &lowerer.collective_state.bound_manual_axes,
            block,
            location,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, AddOperation, ArrayReferenceTransform, ArrayReferenceTransformIndex, AssertOperation, BatchAxis,
        BatchedProgram, ConditionOperation, ConvertElementType, Device, DeviceMesh, DimensionBounds,
        DimensionSizeOperation, DimensionType, DimensionValue, DimensionVariable, EffectClass, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Mul, MulOperation, NegOperation, OneLikeOperation, Placeholder, PrintOperation,
        ProgramBatchingOutputAxesPolicy, RandomAlgorithm, ReferenceNewOperation, ReferenceReadOperation, ReferenceType,
        RngBitGeneratorOperation, ScanOptions, ShardMap, ShardMapOperation, Sharding, ShardingDimension,
        TracingContext, TypeError, WhileOperation, scan,
    };
    use ryft_mlir::Location;
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Layout, Program as PjrtProgram};

    use crate::experimental::XlaDomainError;
    use crate::experimental::lowering::tests::{
        MixedValue, dynamic_dimension, execute_mixed_program, ragged_dot_cpu_compilation_options, test_vector_type,
    };
    use crate::experimental::ops::{
        FlatXlaProgram, XlaConstant, XlaOperation, XlaProgramBuilder as ArrayIrXlaProgramBuilder,
    };
    use crate::operations::control_flow::tests::{
        compare, execute_on_two_devices, lowered_mixed_module, positive, two_device_mesh,
    };
    use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{
        CompiledXlaFunction, FromPjrt, XlaArray, XlaOptions, XlaSession, compile, compile_statefully,
        to_mlir_module_for_program,
    };

    use super::*;

    /// Returns the scan body `(i, c, x) -> (c + x)` over `f64` scalars, which prints each item `x` with the label
    /// `item` before adding it to the sum `c`.
    fn printing_sum_body() -> FlatXlaProgram {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let sum = builder.add_input(scalar_type.clone().into());
        let item = builder.add_input(scalar_type.into());
        let item = builder
            .add_instruction(PrintOperation::<ArrayType>::new("item"), Vec::new(), vec![item], None)
            .unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, item], None).unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap()
    }

    /// Returns a scan of the static `length` and `unroll` factor over the `f32` body `(i, c, x) -> (c * x, c * x)`,
    /// which takes its initial carry and its stacked items as inputs.
    fn product_scan_program(length: usize, unroll: usize) -> FlatXlaProgram {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![product, product],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(test_vector_type(length).into());
        let scan = ScanOperation::new(1, length).with_unroll(unroll).unwrap();
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![carry, items], None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    /// Returns a scan of the given `length` over the `f64` body `(i, c, x) -> (2c + x, c)`, which appends each item `x`
    /// to its carry as a binary digit and stacks the carry that it received, so that both of its results depend on the
    /// order in which its iterations visit the items. The scan takes its initial carry and its stacked items as inputs,
    /// followed by its runtime length when its length is dynamic.
    fn binary_digits_scan_program(length: Dimension, reverse: bool, unroll: usize) -> FlatXlaProgram {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let doubled =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, carry], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![doubled, item], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![next, carry],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![length.clone()])).into());
        let mut inputs = vec![carry, items];
        if let Dimension::Dynamic(length) = &length {
            inputs.push(builder.add_input(DimensionType::from(length.clone()).into()));
        }
        let input_count = inputs.len();
        let scan = ScanOperation::new(1, length).with_reverse(reverse).with_unroll(unroll).unwrap();
        let outputs =
            builder.add_instruction(XlaOperation::Scan(scan), vec![body_region], inputs, None).unwrap().to_vec();
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; input_count], vec![Placeholder; 2])
            .unwrap()
    }

    #[test]
    fn test_scan_lowering() {
        // A scan lowers to a `stablehlo.while` over `[counter, carries..., xs..., ys...]`: each iteration reads one
        // slice of the stacked inputs with `stablehlo.dynamic_slice`, inlines the body, and writes the per-iteration
        // outputs into preallocated zero accumulators with `stablehlo.dynamic_update_slice`.
        assert_eq!(
            lowered_mixed_module(&product_scan_program(3, 1)),
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
    fn test_scan_lowering_rejects_malformed_instructions() {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![carry, carry],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        // The plain array family cannot carry a body region, so scans lower only from the array IR family.
        let mut lowerer = XlaLowerer::new(block.as_ref(), &context, location).with_regions(vec![body.clone()]);
        let expected_message =
            "higher-order operation `scan` must be stored directly in the enclosing backend operation family";
        assert!(matches!(
            ScanOperation::<ArrayType>::new(1, 3).lower(&mut lowerer, &[], &[], &[]),
            Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message })) if message == expected_message,
        ));

        // An array IR scan needs exactly one body region.
        let mut lowerer =
            XlaLowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatXlaProgram>::new());
        assert!(matches!(
            ScanOperation::<ArrayIrType>::new(1, 3).lower(&mut lowerer, &[], &[], &[]),
            Err(LoweringError::Tracing(ProgramError::MalformedProgram(message)))
                if message == "operation `scan` declares 1 region slots but 0 regions were attached",
        ));

        // It also needs one input per carry and per stacked input.
        let mut lowerer = XlaLowerer::new(block.as_ref(), &context, location).with_regions(vec![body.clone()]);
        assert!(matches!(
            ScanOperation::<ArrayIrType>::new(1, 3).lower(&mut lowerer, &[], &[], &[]),
            Err(LoweringError::Tracing(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        ));

        // A dynamic length without an upper bound leaves no physical extent at which to allocate the stacked outputs.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, None).unwrap());
        let input_types: Vec<ArrayIrType> =
            vec![scalar_type.clone().into(), DimensionType::from(length.clone()).into()];
        let output_types: Vec<ArrayIrType> = vec![
            scalar_type.clone().into(),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())])).into(),
        ];
        let block = context.block(
            &input_types
                .iter()
                .map(|r#type| (lower_array_ir_type(r#type, &context, location).unwrap(), location))
                .collect::<Vec<_>>(),
        );
        let input_values = (0..2).map(|index| block.as_ref().argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let mut lowerer = XlaLowerer::new(block.as_ref(), &context, location).with_regions(vec![body]);
        assert!(matches!(
            ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length)).lower(
                &mut lowerer,
                &input_values,
                &input_types,
                &output_types,
            ),
            Err(LoweringError::UnsupportedOp { op })
                if op == "`scan` iteration count `length` needs a finite upper bound to allocate its stacked outputs",
        ));
    }

    #[test]
    fn test_scan_lowering_rejects_dynamic_slice_types() {
        // A static stacked input refines a body that declares the bounded slice type `f64[rows]`, so the scan
        // instruction type-checks. Its iterations would read slices at a dynamic shape, which this lowering does not
        // adapt, so it is rejected with a precise error instead of an opaque tensor-type failure. A zero-length scan
        // never slices, so it still lowers.
        let slice_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let scan_program = |length: usize| {
            let body = {
                let mut builder = ArrayIrXlaProgramBuilder::new();
                builder.add_input(ArrayType::scalar(DataType::I64).into());
                let slice = builder.add_input(slice_type.clone().into());
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap()
            };
            let mut builder = ArrayIrXlaProgramBuilder::new();
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

    #[test]
    fn test_scan_lowering_rejects_unthreaded_token_advances() {
        // The body prints its slice, which advances the ordered I/O token. A loop that threads ordered I/O carries the
        // advanced token through its state, whereas a loop that threads no class could not return it, so lowering it
        // would leave later I/O unordered with respect to the prints.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let stack_type = ArrayType::new_static(DataType::F64, [3]);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let slice = builder.add_input(scalar_type.clone().into());
            let slice = builder
                .add_instruction(PrintOperation::<ArrayType>::new("slice"), Vec::new(), vec![slice], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let state_types: Vec<ArrayIrType> =
            vec![ArrayType::scalar(DataType::I64).into(), stack_type.clone().into(), stack_type.clone().into()];
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block_arguments = state_types
            .iter()
            .map(|r#type| (lower_array_ir_type(r#type, &context, location).unwrap(), location))
            .collect::<Vec<_>>();
        let block = context.block(&block_arguments);
        let collective_state = CollectiveLoweringState::new();
        let lower = |threaded_effects| {
            let inputs = ScanLoopInputs {
                body: &body,
                carry_count: 0,
                x_slice_types: vec![scalar_type.clone()],
                y_slice_types: vec![scalar_type.clone()],
                y_stack_types: vec![stack_type.clone()],
                state_types: state_types.clone(),
                length: ScanLoopLength::Static(3),
                reverse: false,
                threaded_effects,
                context: &context,
                location,
                captured_values: &[],
                nested_functions: None,
                collective_state: &collective_state,
            };
            let state_values = (0..3).map(|index| block.as_ref().argument(index).unwrap().as_ref()).collect();
            inputs.lower_loop(state_values, 1, &mut block.as_ref(), &mut EffectTokens::default())
        };
        assert!(lower(EffectClasses::single(EffectClass::OrderedIo)).is_ok());
        let expected_message =
            "control-flow region lowering advanced the `ordered_io` token, which the region does not thread";
        assert!(matches!(
            lower(EffectClasses::NONE),
            Err(LoweringError::Tracing(ProgramError::MalformedProgram(message))) if message == expected_message,
        ));
    }

    #[test]
    fn test_scan_execution_guards_entry() {
        // The body `(a, b) -> (a, a + b)` declares both carries `f64[rows]`, and the scan receives a bounded dynamic
        // `f64[rows]` initial carry and a static `f64[3]` one, which refines its carry type. The carry outputs take the
        // static extent that the inputs establish for `rows`, so the bounded loop results are sliced back to `f64[3]`,
        // and `n` iterations compute `[a, b + n * a]`. Eager entry validation checks that the runtime extent of `rows`
        // is 3 whatever the trip count is, so a zero-length scan, a fully unrolled scan, and a looping scan all guard
        // their entry and reject a runtime `rows` of 2.
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let carry_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("rows", Some(5))]));
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
        for (length, unroll, expected) in
            [(0, 1, vec![10.0, 20.0, 30.0]), (2, 1, vec![12.0, 24.0, 36.0]), (2, 2, vec![12.0, 24.0, 36.0])]
        {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
            assert_eq!(
                program.output_types(),
                vec![ArrayIrType::Array(static_type.clone()), ArrayIrType::Array(static_type.clone())],
                "length {length}, unroll factor {unroll}",
            );
            let inputs = [
                MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4]),
                MixedValue::Array(vec![10.0, 20.0, 30.0], vec![3]),
            ];
            assert_eq!(
                execute_mixed_program(&client, &program, &inputs, &[3]),
                Ok(vec![MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]), MixedValue::Array(expected, vec![3])]),
                "length {length}, unroll factor {unroll}",
            );
            // The runtime appends its own context to the assertion message.
            let error = execute_mixed_program(&client, &program, &inputs, &[2]).unwrap_err().to_string();
            assert!(
                error.contains("input dimension `rows` must have one extent at the entry of `scan`"),
                "length {length}, unroll factor {unroll}: {error}",
            );
        }
    }

    #[test]
    fn test_scan_execution_guards_runtime_length() {
        // The body declares its carry `f64[length]` with the identity of the scan length, so a static `f64[3]` initial
        // carry refines it only when the runtime length is 3. Neither input type proves that relationship, so an entry
        // guard checks it before the loop. The body `(i, c) -> (c + c)` doubles its carry once per iteration.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(6)).unwrap());
        let carry_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(length.clone())]));
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(carry_type.into());
            let doubled =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, carry], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![doubled], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
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
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:dimension<length ∈ [0, 6)> .
                let %2:f64[3] = scan [carry_count=1, iteration_count=length, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[length] .
                        let %2:f64[length] = add %1 %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64> {
                    %c = stablehlo.constant dense<3> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.pad %arg0, %cst, low = [0], high = [2], interior = [0] : (tensor<3xf64>, \
                tensor<f64>) -> tensor<5xf64>
                    %1 = stablehlo.set_dimension_size %0, %c, dim = 0 : (tensor<5xf64>, tensor<i32>) -> tensor<?xf64, \
                #stablehlo.bounds<5>>
                    %c_0 = stablehlo.constant dense<3> : tensor<i64>
                    %2 = stablehlo.compare EQ, %arg1, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %3 = stablehlo.after_all  : !stablehlo.token
                    %4 = stablehlo.custom_call @ryft.assert(%2, %c_0, %arg1, %3) {api_version = 4 : i32, \
                backend_config = {actor = \"guard\", kind = \"generic\", label_0 = \"expected\", label_1 = \"actual\", \
                label_count = \"2\", message = \"input dimension `length` must have one extent at the entry of \
                `scan`\"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> \
                !stablehlo.token
                    %c_1 = stablehlo.constant dense<0> : tensor<i64>
                    %5:2 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %1) : tensor<i64>, tensor<?xf64, \
                #stablehlo.bounds<5>>
                    cond {
                      %8 = stablehlo.compare LT, %iterArg, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %8 : tensor<i1>
                    } do {
                      %8 = stablehlo.add %iterArg_3, %iterArg_3 : tensor<?xf64, #stablehlo.bounds<5>>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %9, %8 : tensor<i64>, tensor<?xf64, #stablehlo.bounds<5>>
                    }
                    %c_2 = stablehlo.constant dense<5> : tensor<i32>
                    %6 = stablehlo.set_dimension_size %5#1, %c_2, dim = 0 : (tensor<?xf64, #stablehlo.bounds<5>>, \
                tensor<i32>) -> tensor<5xf64>
                    %7 = stablehlo.slice %6 [0:3] : (tensor<5xf64>) -> tensor<3xf64>
                    return %7 : tensor<3xf64>
                  }
                }
            "},
        );
        let client = execution_client();
        let carry = MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(
            execute_mixed_program(&client, &program, &[carry.clone(), MixedValue::Dimension(3)], &[]),
            Ok(vec![MixedValue::Array(vec![8.0, 16.0, 24.0], vec![3])]),
        );
        let error = execute_mixed_program(&client, &program, &[carry, MixedValue::Dimension(2)], &[])
            .unwrap_err()
            .to_string();
        // The runtime appends its own context to the assertion message.
        assert!(error.contains("input dimension `length` must have one extent at the entry of `scan`"), "{error}");
    }

    #[test]
    fn test_scan_execution_guards_stacks_beside_a_symbolic_runtime_length() {
        // The scan stacks a static `f64[3]` input over its nominal runtime length `dimension<length>`, so its declared
        // stacked type `f64[length]` relates the stack's leading extent to the runtime length. Neither input type
        // proves that relationship, so an entry guard checks that the runtime length is 3 before the loop, and the
        // stacked output keeps the static extent `f64[3]`. The body `(i, c, x) -> (c + x, c)` computes a running sum
        // and stacks the carry that each iteration receives.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(6)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![sum, carry],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let stack = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body_region],
                vec![carry, stack, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:dimension<length ∈ [0, 6)> .
                let %3:f64[], %4:f64[3] = scan [carry_count=1, iteration_count=length, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = add %1 %2
                        in (%3, %1)
                    },
                ]
                in (%3, %4)"},
        );
        // One `compare EQ` and `@ryft.assert` guard the runtime length before the loop, whose state holds the stack
        // padded to the bound of `length`. The stacked output is sliced back to `f64[3]`.
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<3xf64>, %arg2: tensor<i64>) -> (tensor<f64>, \
                tensor<3xf64>) {
                    %c = stablehlo.constant dense<3> : tensor<i64>
                    %0 = stablehlo.compare EQ, %arg2, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.assert(%0, %c, %arg2, %1) {api_version = 4 : i32, backend_config \
                = {actor = \"guard\", kind = \"generic\", label_0 = \"expected\", label_1 = \"actual\", label_count = \
                \"2\", message = \"input dimension `length` must have one extent at the entry of `scan`\"}, \
                has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %c_1 = stablehlo.constant dense<3> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %3 = stablehlo.pad %arg1, %cst, low = [0], high = [2], interior = [0] : (tensor<3xf64>, \
                tensor<f64>) -> tensor<5xf64>
                    %4 = stablehlo.set_dimension_size %3, %c_1, dim = 0 : (tensor<5xf64>, tensor<i32>) -> \
                tensor<?xf64, #stablehlo.bounds<5>>
                    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %5 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<5xf64>
                    %6 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
                    %7 = stablehlo.set_dimension_size %5, %6, dim = 0 : (tensor<5xf64>, tensor<i32>) -> tensor<?xf64, \
                #stablehlo.bounds<5>>
                    %8:4 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %arg0, %iterArg_5 = %4, %iterArg_6 = %7) : \
                tensor<i64>, tensor<f64>, tensor<?xf64, #stablehlo.bounds<5>>, tensor<?xf64, #stablehlo.bounds<5>>
                    cond {
                      %11 = stablehlo.compare LT, %iterArg, %arg2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %11 : tensor<i1>
                    } do {
                      %11 = stablehlo.dynamic_slice %iterArg_5, %iterArg, sizes = [1] : (tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<i64>) -> tensor<1xf64>
                      %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
                      %13 = stablehlo.add %iterArg_4, %12 : tensor<f64>
                      %14 = stablehlo.reshape %iterArg_4 : (tensor<f64>) -> tensor<1xf64>
                      %15 = stablehlo.dynamic_update_slice %iterArg_6, %14, %iterArg : (tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<1xf64>, tensor<i64>) -> tensor<?xf64, #stablehlo.bounds<5>>
                      %c_7 = stablehlo.constant dense<1> : tensor<i64>
                      %16 = stablehlo.add %iterArg, %c_7 : tensor<i64>
                      stablehlo.return %16, %13, %iterArg_5, %15 : tensor<i64>, tensor<f64>, tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<?xf64, #stablehlo.bounds<5>>
                    }
                    %c_3 = stablehlo.constant dense<5> : tensor<i32>
                    %9 = stablehlo.set_dimension_size %8#3, %c_3, dim = 0 : (tensor<?xf64, #stablehlo.bounds<5>>, \
                tensor<i32>) -> tensor<5xf64>
                    %10 = stablehlo.slice %9 [0:3] : (tensor<5xf64>) -> tensor<3xf64>
                    return %8#1, %10 : tensor<f64>, tensor<3xf64>
                  }
                }
            "},
        );
        let client = execution_client();
        let inputs = |runtime_length| {
            [
                MixedValue::Array(vec![0.0], vec![]),
                MixedValue::Array(vec![1.0, 2.0, 4.0], vec![3]),
                MixedValue::Dimension(runtime_length),
            ]
        };
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs(3), &[]),
            Ok(vec![MixedValue::Array(vec![7.0], vec![]), MixedValue::Array(vec![0.0, 1.0, 3.0], vec![3])]),
        );
        let error = execute_mixed_program(&client, &program, &inputs(2), &[]).unwrap_err().to_string();
        // The runtime appends its own context to the assertion message.
        assert!(error.contains("input dimension `length` must have one extent at the entry of `scan`"), "{error}");
    }

    #[test]
    fn test_scan_execution_prunes_without_dropping_runtime_length_checks() {
        // The body `(i, r, s) -> (r, s + s)` declares its carry `r` as `f64[length]` with the identity of the scan
        // length and receives a static `f64[3]` for it, so the scan checks the runtime length against 3 at its entry.
        // Only the doubled sum `s` is used, and `r` is an unused pass-through carry, but pruning keeps it because it is
        // the only witness of that relationship: the pruned module still guards its entry, accepts a runtime length of
        // 3, and rejects one of 2.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let related = builder
                .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(length.clone())])).into());
            let sum = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, sum], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![related, sum],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let related = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let sum = builder.add_input(scalar_type.into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(2, Dimension::Dynamic(length))),
                vec![body_region],
                vec![related, sum, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let pruned = program.with_outputs(&[1]).unwrap().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:dimension<length ∈ [1, 8)> .
                let %3:f64[3], %4:f64[] = scan [carry_count=2, iteration_count=length, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[length], %2:f64[] .
                        let %3:f64[] = add %2 %2
                        in (%1, %3)
                    },
                ]
                in (%4)"},
        );
        assert_eq!(
            lowered_mixed_module(&pruned),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<f64>, %arg2: tensor<i64>) -> tensor<f64> {
                    %c = stablehlo.constant dense<3> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.pad %arg0, %cst, low = [0], high = [4], interior = [0] : (tensor<3xf64>, \
                tensor<f64>) -> tensor<7xf64>
                    %1 = stablehlo.set_dimension_size %0, %c, dim = 0 : (tensor<7xf64>, tensor<i32>) -> tensor<?xf64, \
                #stablehlo.bounds<7>>
                    %c_0 = stablehlo.constant dense<3> : tensor<i64>
                    %2 = stablehlo.compare EQ, %arg2, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %3 = stablehlo.after_all  : !stablehlo.token
                    %4 = stablehlo.custom_call @ryft.assert(%2, %c_0, %arg2, %3) {api_version = 4 : i32, \
                backend_config = {actor = \"guard\", kind = \"generic\", label_0 = \"expected\", label_1 = \"actual\", \
                label_count = \"2\", message = \"input dimension `length` must have one extent at the entry of \
                `scan`\"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> \
                !stablehlo.token
                    %c_1 = stablehlo.constant dense<0> : tensor<i64>
                    %5:3 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %1, %iterArg_4 = %arg1) : tensor<i64>, \
                tensor<?xf64, #stablehlo.bounds<7>>, tensor<f64>
                    cond {
                      %8 = stablehlo.compare LT, %iterArg, %arg2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %8 : tensor<i1>
                    } do {
                      %8 = stablehlo.add %iterArg_4, %iterArg_4 : tensor<f64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.add %iterArg, %c_5 : tensor<i64>
                      stablehlo.return %9, %iterArg_3, %8 : tensor<i64>, tensor<?xf64, #stablehlo.bounds<7>>, \
                tensor<f64>
                    }
                    %c_2 = stablehlo.constant dense<7> : tensor<i32>
                    %6 = stablehlo.set_dimension_size %5#1, %c_2, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, \
                tensor<i32>) -> tensor<7xf64>
                    %7 = stablehlo.slice %6 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
                    return %5#2 : tensor<f64>
                  }
                }
            "},
        );
        let client = execution_client();
        let inputs = |extent| {
            [
                MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                MixedValue::Array(vec![1.0], vec![]),
                MixedValue::Dimension(extent),
            ]
        };
        assert_eq!(
            execute_mixed_program(&client, &pruned, &inputs(3), &[]),
            Ok(vec![MixedValue::Array(vec![8.0], vec![])])
        );
        let error = execute_mixed_program(&client, &pruned, &inputs(2), &[]).unwrap_err().to_string();
        // The runtime appends its own context to the assertion message.
        assert!(error.contains("input dimension `length` must have one extent at the entry of `scan`"), "{error}");
    }

    #[test]
    fn test_scan_lowering_zero_length() {
        // A zero-length scan runs no iteration, regardless of its unroll factor: it returns its initial carry and an
        // empty stacked output without emitting a `stablehlo.while` or any body copy.
        assert_eq!(
            lowered_mixed_module(&product_scan_program(0, 2)),
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

        // A dynamic length whose bounds admit only zero has an output stack with a physical leading bound of zero, so
        // an emitted loop body would contain an invalid unit-length update even though it could never execute.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(1)).unwrap());
        let length_type = DimensionType::from(length.clone());
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));
        let mut body_builder = ArrayIrXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![carry, carry],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        let mut builder = ArrayIrXlaProgramBuilder::new();
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
    fn test_scan_lowering_unroll() {
        // A scan whose unroll factor covers its length lowers to straight-line body copies, without a
        // `stablehlo.while`.
        assert_eq!(
            lowered_mixed_module(&product_scan_program(3, 3)),
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

        // A scan with `unroll = 2` over `length = 4` runs two body copies per loop trip, and its counter advances by
        // the unroll factor.
        assert_eq!(
            lowered_mixed_module(&product_scan_program(4, 2)),
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

        // Over `length = 5`, the two loop trips are followed by one straight-line body copy for the remaining
        // iteration.
        assert_eq!(
            lowered_mixed_module(&product_scan_program(5, 2)),
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
    fn test_scan_execution_unrolled_static_length() {
        // A static scan of length 5 unrolled by 2 runs two loop trips of two body copies each and then one
        // straight-line body copy, which reads and writes the stacks at the static index of its iteration: index 4 for
        // a forward scan and index 0 for a reversed scan.
        assert_eq!(
            lowered_mixed_module(&binary_digits_scan_program(Dimension::Static(5), true, 2)),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<5xf64>) -> (tensor<f64>, tensor<5xf64>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<5xf64>
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %0) : \
                tensor<i64>, tensor<f64>, tensor<5xf64>, tensor<5xf64>
                    cond {
                      %c_4 = stablehlo.constant dense<5> : tensor<i64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.subtract %c_4, %c_5 : tensor<i64>
                      %9 = stablehlo.compare LT, %iterArg, %8, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %9 : tensor<i1>
                    } do {
                      %c_4 = stablehlo.constant dense<5> : tensor<i64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.subtract %c_4, %c_5 : tensor<i64>
                      %9 = stablehlo.subtract %8, %iterArg : tensor<i64>
                      %10 = stablehlo.dynamic_slice %iterArg_2, %9, sizes = [1] : (tensor<5xf64>, tensor<i64>) -> \
                tensor<1xf64>
                      %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
                      %12 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<f64>
                      %13 = stablehlo.add %12, %11 : tensor<f64>
                      %14 = stablehlo.reshape %iterArg_1 : (tensor<f64>) -> tensor<1xf64>
                      %15 = stablehlo.dynamic_update_slice %iterArg_3, %14, %9 : (tensor<5xf64>, tensor<1xf64>, \
                tensor<i64>) -> tensor<5xf64>
                      %c_6 = stablehlo.constant dense<1> : tensor<i64>
                      %16 = stablehlo.add %iterArg, %c_6 : tensor<i64>
                      %17 = stablehlo.subtract %8, %16 : tensor<i64>
                      %18 = stablehlo.dynamic_slice %iterArg_2, %17, sizes = [1] : (tensor<5xf64>, tensor<i64>) -> \
                tensor<1xf64>
                      %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
                      %20 = stablehlo.add %13, %13 : tensor<f64>
                      %21 = stablehlo.add %20, %19 : tensor<f64>
                      %22 = stablehlo.reshape %13 : (tensor<f64>) -> tensor<1xf64>
                      %23 = stablehlo.dynamic_update_slice %15, %22, %17 : (tensor<5xf64>, tensor<1xf64>, tensor<i64>) \
                -> tensor<5xf64>
                      %c_7 = stablehlo.constant dense<2> : tensor<i64>
                      %24 = stablehlo.add %iterArg, %c_7 : tensor<i64>
                      stablehlo.return %24, %21, %iterArg_2, %23 : tensor<i64>, tensor<f64>, tensor<5xf64>, \
                tensor<5xf64>
                    }
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %2 = stablehlo.dynamic_slice %1#2, %c_0, sizes = [1] : (tensor<5xf64>, tensor<i64>) -> tensor<1xf64>
                    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
                    %4 = stablehlo.add %1#1, %1#1 : tensor<f64>
                    %5 = stablehlo.add %4, %3 : tensor<f64>
                    %6 = stablehlo.reshape %1#1 : (tensor<f64>) -> tensor<1xf64>
                    %7 = stablehlo.dynamic_update_slice %1#3, %6, %c_0 : (tensor<5xf64>, tensor<1xf64>, tensor<i64>) \
                -> tensor<5xf64>
                    return %5, %7 : tensor<f64>, tensor<5xf64>
                  }
                }
            "},
        );
        let client = execution_client();
        let items = MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        for unroll in [1, 2, 5] {
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &binary_digits_scan_program(Dimension::Static(5), false, unroll),
                    &[MixedValue::Array(vec![0.0], vec![]), items.clone()],
                    &[],
                ),
                Ok(vec![
                    MixedValue::Array(vec![57.0], vec![]),
                    MixedValue::Array(vec![0.0, 1.0, 4.0, 11.0, 26.0], vec![5]),
                ]),
                "unroll factor {unroll}",
            );
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &binary_digits_scan_program(Dimension::Static(5), true, unroll),
                    &[MixedValue::Array(vec![0.0], vec![]), items.clone()],
                    &[],
                ),
                Ok(vec![
                    MixedValue::Array(vec![129.0], vec![]),
                    MixedValue::Array(vec![64.0, 31.0, 14.0, 5.0, 0.0], vec![5]),
                ]),
                "reversed, unroll factor {unroll}",
            );
        }
    }

    #[test]
    fn test_scan_lowering_unroll_counter_does_not_overflow() {
        // A length of `i64::MAX` with an unroll factor of three leaves one remaining iteration. At the final loop
        // condition, adding two to its counter (`i64::MAX - 1`) would wrap and incorrectly continue the loop.
        // The body has only a scalar carry, so lowering this boundary case requires no enormous array allocations.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let mut body_builder = ArrayIrXlaProgramBuilder::new();
        let index = body_builder.add_input(scalar_type.clone().into());
        body_builder.add_input(scalar_type.clone().into());
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![index], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = ScanOperation::new(1, i64::MAX as usize).with_unroll(3).unwrap();

        let mut builder = ArrayIrXlaProgramBuilder::new();
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
    fn test_scan_lowering_dynamic_length() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length_type = DimensionType::from(length.clone());
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));

        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
        let mut builder = ArrayIrXlaProgramBuilder::new();
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
    fn test_scan_lowering_dynamic_length_with_an_empty_reference_carry() {
        // The body adds the element of an empty reference carry at its index to its sum carry, which is valid only when
        // no iteration runs. A runtime length whose bounds admit iterations keeps the body, whose discharged access
        // would slice the empty axis, so the reference discharge that compilation performs before lowering rejects the
        // scan with that typed error, and lowering never emits a `stablehlo.dynamic_slice` of the empty axis that MLIR
        // verification would reject.
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F32);
        let empty_type = ArrayType::new_static(DataType::F32, [0]);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
            let sum = builder.add_input(scalar_type.clone().into());
            let root = builder.add_input(ReferenceType::new(empty_type.clone()).into());
            let read = ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
            let element = builder.add_instruction(read, Vec::new(), vec![root, index], None).unwrap()[0];
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, element], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![sum, root],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let sum = builder.add_input(scalar_type.clone().into());
        let elements = builder.add_input(empty_type.clone().into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![elements], None).unwrap()[0];
        let sum = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(2, Dimension::Dynamic(length.clone()))),
                vec![body_region],
                vec![sum, root, runtime_length],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[0], %2:dimension<length ∈ [0, 8)> .
                let %3:ref<f32[0]> = reference_new %1
                    %4:f32[], %5:ref<f32[0]> = scan [carry_count=2, iteration_count=length, reverse=false] %0 %3 %2 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:ref<f32[0]> .
                            let %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %2 %0
                                %4:f32[] = add %1 %3
                            in (%4, %2)
                        },
                    ]
                in (%4)"},
        );
        let client = execution_client();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let domain = XlaSession::new(&client).domain();
        // Staging succeeds, so the error comes from the discharge that precedes lowering.
        let mut staged = false;
        let error = compile_statefully::<_, (ArrayIrType, ArrayIrType, ArrayIrType), ArrayIrType>(
            |(sum, elements, runtime_length)| {
                let context = sum.context().clone();
                let sum = program.interpret_in_context(&context, vec![sum, elements, runtime_length])?.remove(0);
                staged = true;
                Ok(sum)
            },
            (scalar_type.into(), empty_type.into(), DimensionType::from(length).into()),
            &domain,
            XlaOptions::new(mesh),
        )
        .err()
        .unwrap();
        assert!(staged);
        assert!(matches!(
            error,
            XlaDomainError::Tracing(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`dynamic_slice` size 1 is out of bounds for axis 0 with size 0",
        ));
    }

    #[test]
    fn test_scan_execution_dynamic_length() {
        // Every lowered form of a dynamic-length scan (a loop with one body copy per trip, or an unrolled loop followed
        // by a second loop with one body copy per trip) runs exactly the runtime number of iterations, from none up to
        // the largest number that the bound admits. A reversed scan visits the items backwards while it writes each
        // stacked output at its own iteration index.
        let length = Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::new(0, Some(6)).unwrap()));

        // A reversed scan unrolled by 2 runs iterations `length - 1 - counter` and `length - 2 - counter` per trip of
        // its first loop, and the remaining iteration, if any, in its second loop.
        assert_eq!(
            lowered_mixed_module(&binary_digits_scan_program(length.clone(), true, 2)),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<5xf64>, %arg2: tensor<i64>, %arg3: tensor<i32>) -> \
                (tensor<f64>, tensor<?xf64, #stablehlo.bounds<5>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg1, %arg3, dim = 0 : (tensor<5xf64>, tensor<i32>) -> \
                tensor<?xf64, #stablehlo.bounds<5>>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<5xf64>
                    %2 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
                    %3 = stablehlo.set_dimension_size %1, %2, dim = 0 : (tensor<5xf64>, tensor<i32>) -> tensor<?xf64, \
                #stablehlo.bounds<5>>
                    %4:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %0, %iterArg_2 = %3) : \
                tensor<i64>, tensor<f64>, tensor<?xf64, #stablehlo.bounds<5>>, tensor<?xf64, #stablehlo.bounds<5>>
                    cond {
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.subtract %arg2, %c_3 : tensor<i64>
                      %9 = stablehlo.compare LT, %iterArg, %8, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %9 : tensor<i1>
                    } do {
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.subtract %arg2, %c_3 : tensor<i64>
                      %9 = stablehlo.subtract %8, %iterArg : tensor<i64>
                      %10 = stablehlo.dynamic_slice %iterArg_1, %9, sizes = [1] : (tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<i64>) -> tensor<1xf64>
                      %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
                      %12 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<f64>
                      %13 = stablehlo.add %12, %11 : tensor<f64>
                      %14 = stablehlo.reshape %iterArg_0 : (tensor<f64>) -> tensor<1xf64>
                      %15 = stablehlo.dynamic_update_slice %iterArg_2, %14, %9 : (tensor<?xf64, #stablehlo.bounds<5>>, \
                tensor<1xf64>, tensor<i64>) -> tensor<?xf64, #stablehlo.bounds<5>>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %16 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      %17 = stablehlo.subtract %8, %16 : tensor<i64>
                      %18 = stablehlo.dynamic_slice %iterArg_1, %17, sizes = [1] : (tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<i64>) -> tensor<1xf64>
                      %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
                      %20 = stablehlo.add %13, %13 : tensor<f64>
                      %21 = stablehlo.add %20, %19 : tensor<f64>
                      %22 = stablehlo.reshape %13 : (tensor<f64>) -> tensor<1xf64>
                      %23 = stablehlo.dynamic_update_slice %15, %22, %17 : (tensor<?xf64, #stablehlo.bounds<5>>, \
                tensor<1xf64>, tensor<i64>) -> tensor<?xf64, #stablehlo.bounds<5>>
                      %c_5 = stablehlo.constant dense<2> : tensor<i64>
                      %24 = stablehlo.add %iterArg, %c_5 : tensor<i64>
                      stablehlo.return %24, %21, %iterArg_1, %23 : tensor<i64>, tensor<f64>, tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<?xf64, #stablehlo.bounds<5>>
                    }
                    %5:4 = stablehlo.while(%iterArg = %4#0, %iterArg_0 = %4#1, %iterArg_1 = %4#2, %iterArg_2 = %4#3) : \
                tensor<i64>, tensor<f64>, tensor<?xf64, #stablehlo.bounds<5>>, tensor<?xf64, #stablehlo.bounds<5>>
                    cond {
                      %8 = stablehlo.compare LT, %iterArg, %arg2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %8 : tensor<i1>
                    } do {
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.subtract %arg2, %c_3 : tensor<i64>
                      %9 = stablehlo.subtract %8, %iterArg : tensor<i64>
                      %10 = stablehlo.dynamic_slice %iterArg_1, %9, sizes = [1] : (tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<i64>) -> tensor<1xf64>
                      %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
                      %12 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<f64>
                      %13 = stablehlo.add %12, %11 : tensor<f64>
                      %14 = stablehlo.reshape %iterArg_0 : (tensor<f64>) -> tensor<1xf64>
                      %15 = stablehlo.dynamic_update_slice %iterArg_2, %14, %9 : (tensor<?xf64, #stablehlo.bounds<5>>, \
                tensor<1xf64>, tensor<i64>) -> tensor<?xf64, #stablehlo.bounds<5>>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %16 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %16, %13, %iterArg_1, %15 : tensor<i64>, tensor<f64>, tensor<?xf64, \
                #stablehlo.bounds<5>>, tensor<?xf64, #stablehlo.bounds<5>>
                    }
                    %6 = stablehlo.get_dimension_size %5#3, dim = 0 : (tensor<?xf64, #stablehlo.bounds<5>>) -> \
                tensor<i32>
                    %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
                    return %5#1, %5#3, %7 : tensor<f64>, tensor<?xf64, #stablehlo.bounds<5>>, tensor<i64>
                  }
                }
            "},
        );

        let client = execution_client();
        let xs = MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let forward = [
            (0, 0.0, vec![]),
            (1, 1.0, vec![0.0]),
            (4, 26.0, vec![0.0, 1.0, 4.0, 11.0]),
            (5, 57.0, vec![0.0, 1.0, 4.0, 11.0, 26.0]),
        ];
        let reversed = [
            (0, 0.0, vec![]),
            (1, 1.0, vec![0.0]),
            (4, 49.0, vec![24.0, 11.0, 4.0, 0.0]),
            (5, 129.0, vec![64.0, 31.0, 14.0, 5.0, 0.0]),
        ];
        for (reverse, cases) in [(false, forward), (true, reversed)] {
            for unroll in [1, 2, 3, 5] {
                let program = binary_digits_scan_program(length.clone(), reverse, unroll);
                for (runtime_length, carry, ys) in cases.clone() {
                    let inputs =
                        [MixedValue::Array(vec![0.0], vec![]), xs.clone(), MixedValue::Dimension(runtime_length)];
                    assert_eq!(
                        execute_mixed_program(&client, &program, &inputs, &[runtime_length as i32]),
                        Ok(vec![
                            MixedValue::Array(vec![carry], vec![]),
                            MixedValue::Array(ys, vec![runtime_length as u64]),
                            MixedValue::Dimension(runtime_length),
                        ]),
                        "reverse {reverse}, unroll factor {unroll}, runtime length {runtime_length}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_scan_execution_refines_stacks_of_exact_runtime_length() {
        // An exact runtime length refines the declared dynamic length, so a static `f64[4]` stacked input is valid,
        // while the loop state stacks it over the declared bounded length. A running sum computes
        // `(10, [1, 3, 6, 10])`.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
        let mut builder = ArrayIrXlaProgramBuilder::new();
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
    fn test_scan_execution_forwards_dimension_carries() {
        // The scan carries a first-class dimension next to a vector whose extent it is, and forwards both, so the
        // vector keeps its runtime extent at every extent that its bound admits.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));

        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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

        let mut builder = ArrayIrXlaProgramBuilder::new();
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
                  func.func @main(%arg0: tensor<8xf64>, %arg1: tensor<i32>) -> (tensor<?xf64, #stablehlo.bounds<8>>, \
                tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<8xf64>, tensor<i32>) -> \
                tensor<?xf64, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf64, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %3:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %2, %iterArg_1 = %0) : tensor<i64>, \
                tensor<i64>, tensor<?xf64, #stablehlo.bounds<8>>
                    cond {
                      %c_2 = stablehlo.constant dense<3> : tensor<i64>
                      %6 = stablehlo.compare LT, %iterArg, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %c_2 = stablehlo.constant dense<1> : tensor<i64>
                      %6 = stablehlo.add %iterArg, %c_2 : tensor<i64>
                      stablehlo.return %6, %iterArg_0, %iterArg_1 : tensor<i64>, tensor<i64>, tensor<?xf64, \
                #stablehlo.bounds<8>>
                    }
                    %4 = stablehlo.get_dimension_size %3#2, dim = 0 : (tensor<?xf64, #stablehlo.bounds<8>>) -> \
                tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %3#2, %5 : tensor<?xf64, #stablehlo.bounds<8>>, tensor<i64>
                  }
                }
            "},
        );
        let client = execution_client();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for extent in [1, 8] {
            assert_eq!(
                execute_mixed_program(&client, &program, &[MixedValue::Array(values.clone(), vec![8])], &[extent]),
                Ok(vec![
                    MixedValue::Array(values[..extent as usize].to_vec(), vec![extent as u64]),
                    MixedValue::Dimension(extent as i64),
                ]),
                "extent {extent}",
            );
        }
    }

    #[test]
    fn test_scan_execution_mapped_rng() {
        // A random bit generator applied to each item of a batch of generator states is a scan over the states whose
        // length is the runtime batch size. Each item of the batch is generated independently of the others.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::from(batch.clone());
        let state_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let stacked_state_type =
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]));
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(2)]));
        let stacked_bits_type =
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]));

        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
        let mut builder = ArrayIrXlaProgramBuilder::new();
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

        let client = execution_client();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: stablehlo.into_bytes() }, &ragged_dot_cpu_compilation_options())
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        // Runs the scan over the first `batch` generator states, each a pair of `u64` words, of `states`, and returns
        // the generated states and bits after checking their logical dimensions and runtime extents.
        let run = |states: &[u64], batch: usize| {
            let mut storage = states[..2 * batch].to_vec();
            storage.resize(16, 0);
            let input = |bytes: Vec<u8>, buffer_type: BufferType, dimensions: &[u64]| {
                let buffer = client.buffer(bytes.as_slice(), buffer_type, dimensions, None, device.clone(), None);
                ExecutionInput { buffer: Arc::new(buffer.unwrap()), donatable: false }
            };
            let inputs = [
                input(values_to_bytes(&storage), BufferType::U64, &[8, 2]),
                input(values_to_bytes(&[batch as i64]), BufferType::I64, &[]),
                input(values_to_bytes(&[batch as i32]), BufferType::I32, &[]),
            ];
            let outputs = executable
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
                .remove(0)
                .outputs;
            // A populated dynamic result is copied back as its compact logical elements.
            let logical = |index: usize| {
                let output = &outputs[index];
                let bytes = output.copy_to_host(Some(Layout::dense_major_to_minor(2))).unwrap().r#await().unwrap();
                (bytes, output.unpadded_dimensions().unwrap().to_vec())
            };
            let (state_bytes, state_dimensions) = logical(0);
            let (bits_bytes, bits_dimensions) = logical(1);
            let extents = [2, 3].map(|index| {
                let bytes = outputs[index].copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<i64>(bytes.as_slice())[0]
            });
            assert_eq!(state_dimensions, vec![batch as u64, 2]);
            assert_eq!(bits_dimensions, vec![batch as u64, 2]);
            assert_eq!(extents, [batch as i64; 2]);
            (values_from_bytes::<u64>(&state_bytes[..16 * batch]), values_from_bytes::<u32>(&bits_bytes[..8 * batch]))
        };
        let states = (0..16).map(|value| 1000 + value).collect::<Vec<u64>>();
        let (batched_states, batched_bits) = run(&states, 8);
        assert_ne!(batched_states, states);
        for item in 0..8 {
            assert_eq!(
                run(&states[2 * item..], 1),
                (batched_states[2 * item..2 * item + 2].to_vec(), batched_bits[2 * item..2 * item + 2].to_vec()),
                "item {item}",
            );
        }
    }

    #[test]
    fn test_scan_lowering_threads_effect_class_union() {
        // The scan body uses both ordered classes: the scalar check contributes an assertion and the print
        // contributes ordered I/O. The loop state carries two independent trailing tokens in canonical class order.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let mut body_builder = ArrayIrXlaProgramBuilder::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body_builder.add_input(scalar_type.clone().into());
        let value = body_builder.add_input(scalar_type.clone().into());
        let predicate = body_builder
            .add_instruction(compare(ComparisonDirection::Equal), Vec::new(), vec![carry, value], None)
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
        let scan = ScanOperation::new(1, 3);

        let stacked_type = ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
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
    fn test_scan_execution_threads_print_tokens() {
        // A print inside a scan body makes the lowered `stablehlo.while` carry the effect token as one extra trailing
        // state element: the entry token is created before the loop, both regions receive it as an extra block
        // argument, the body threads it through the print custom call, and the loop's trailing result continues the
        // chain (unused here because the program ends right after the scan). Each iteration prints its item, in
        // iteration order.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(printing_sum_body().entry_region_ref());
        let sum = builder.add_input(scalar_type.into());
        let items = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let sum = builder
            .add_instruction(XlaOperation::Scan(ScanOperation::new(1, 3)), vec![body_region], vec![sum, items], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> tensor<f64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0) : \
                tensor<i64>, tensor<f64>, tensor<3xf64>, !stablehlo.token
                    cond {
                      %c_3 = stablehlo.constant dense<3> : tensor<i64>
                      %2 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %2 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> \
                tensor<1xf64>
                      %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
                      %4 = stablehlo.custom_call @ryft.print(%3, %iterArg_2) {api_version = 4 : i32, backend_config = \
                {label = \"item\"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      %5 = stablehlo.add %iterArg_0, %3 : tensor<f64>
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %6 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                      stablehlo.return %6, %5, %iterArg_1, %4 : tensor<i64>, tensor<f64>, tensor<3xf64>, \
                !stablehlo.token
                    }
                    return %1#1 : tensor<f64>
                  }
                }
            "},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let (outputs, lines) = with_captured_prints(|| {
            execute_mixed_program(
                &client,
                &program,
                &[MixedValue::Array(vec![1.0], vec![]), MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3])],
                &[],
            )
        });
        assert_eq!(outputs, Ok(vec![MixedValue::Array(vec![7.0], vec![])]));
        assert_eq!(lines, vec!["item: 1.0", "item: 2.0", "item: 3.0"]);
    }

    #[test]
    fn test_scan_execution_checks_body_assertions() {
        // The body asserts that each item is positive before it adds the item to the sum. The assertion runs once per
        // iteration, so it fails at the first non-positive item within the runtime length and never checks the items
        // beyond it, including every item of a scan whose runtime length is zero.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let sum = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let is_positive = positive(&mut builder, item);
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("item must be positive").with_labels(vec!["item".to_owned()]),
                    Vec::new(),
                    vec![is_positive, item],
                    None,
                )
                .unwrap();
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, item], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let sum = builder.add_input(scalar_type.into());
        let items = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(length.clone())])).into());
        let runtime_length = builder.add_input(DimensionType::from(length.clone()).into());
        let sum = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body_region],
                vec![sum, items, runtime_length],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[length], %2:dimension<length ∈ [0, 5)> .
                let %3:f64[] = scan [carry_count=1, iteration_count=length, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = zero_like %2
                            %4:bool[] = compare [direction=GreaterThan] %2 %3
                            () = assert [message=\"item must be positive\", labels=[\"item\"]] %4 %2
                            %5:f64[] = add %1 %2
                        in (%5)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<4xf64>, %arg2: tensor<i64>, %arg3: tensor<i32>) -> \
                tensor<f64> {
                    %0 = stablehlo.set_dimension_size %arg1, %arg3, dim = 0 : (tensor<4xf64>, tensor<i32>) -> \
                tensor<?xf64, #stablehlo.bounds<4>>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.after_all  : !stablehlo.token
                    %3:5 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %0, %iterArg_2 = %1, \
                %iterArg_3 = %2) : tensor<i64>, tensor<f64>, tensor<?xf64, #stablehlo.bounds<4>>, !stablehlo.token, \
                !stablehlo.token
                    cond {
                      %4 = stablehlo.compare LT, %iterArg, %arg2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %4 : tensor<i1>
                    } do {
                      %4 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<?xf64, \
                #stablehlo.bounds<4>>, tensor<i64>) -> tensor<1xf64>
                      %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %6 = stablehlo.compare GT, %5, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                      %7 = stablehlo.custom_call @ryft.assert(%6, %5, %iterArg_2) {api_version = 4 : i32, \
                backend_config = {actor = \"assert\", kind = \"generic\", label_0 = \"item\", label_count = \"1\", \
                message = \"item must be positive\"}, has_side_effect = true} : (tensor<i1>, tensor<f64>, \
                !stablehlo.token) -> !stablehlo.token
                      %8 = stablehlo.add %iterArg_0, %5 : tensor<f64>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %9, %8, %iterArg_1, %7, %iterArg_3 : tensor<i64>, tensor<f64>, tensor<?xf64, \
                #stablehlo.bounds<4>>, !stablehlo.token, !stablehlo.token
                    }
                    return %3#1 : tensor<f64>
                  }
                }
            "},
        );
        let client = execution_client();
        let run = |items: Vec<f64>, runtime_length: i64| {
            execute_mixed_program(
                &client,
                &program,
                &[
                    MixedValue::Array(vec![0.0], vec![]),
                    MixedValue::Array(items, vec![4]),
                    MixedValue::Dimension(runtime_length),
                ],
                &[runtime_length as i32],
            )
        };
        assert_eq!(run(vec![-1.0, 2.0, 3.0, 4.0], 0), Ok(vec![MixedValue::Array(vec![0.0], vec![])]));
        assert_eq!(run(vec![1.0, 2.0, -3.0, 4.0], 2), Ok(vec![MixedValue::Array(vec![3.0], vec![])]));
        // The runtime appends its own context to the assertion message.
        let error = run(vec![1.0, 2.0, -3.0, 4.0], 3).unwrap_err().to_string();
        assert!(error.contains("assertion failed: item must be positive; item=-3"), "{error}");
    }

    #[test]
    fn test_scan_lowering_keeps_the_memory_placement_of_stacked_inputs() {
        // The entry of a scan over host-pinned slices declares its stacked inputs in host-pinned memory too, exactly as
        // core `scan` validation does, so the entry check accepts host-pinned stacks whatever the trip count is, and
        // the accumulator of the stacked output is placed in host-pinned memory.
        let memory = Memory::Host { pinned: true };
        let slice_type = ArrayType::new_static(DataType::F64, [3]).with_memory(memory);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let slice = builder.add_input(slice_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        for (length, expected) in [
            (
                0,
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<0x3xf64> {mhlo.memory_kind = \"pinned_host\"}) -> (tensor<0x3xf64> \
                    {mhlo.memory_kind = \"pinned_host\"}) {
                        %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                        %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<0x3xf64>
                        %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = \"\", \
                    has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = \"pinned_host\"}} : \
                    (tensor<0x3xf64>) -> tensor<0x3xf64>
                        return %1 : tensor<0x3xf64>
                      }
                    }
                "},
            ),
            (
                2,
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<2x3xf64> {mhlo.memory_kind = \"pinned_host\"}) -> (tensor<2x3xf64> \
                    {mhlo.memory_kind = \"pinned_host\"}) {
                        %c = stablehlo.constant dense<0> : tensor<i64>
                        %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                        %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64>
                        %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = \"\", \
                    has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = \"pinned_host\"}} : \
                    (tensor<2x3xf64>) -> tensor<2x3xf64>
                        %2:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %1) : tensor<i64>, \
                    tensor<2x3xf64>, tensor<2x3xf64>
                        cond {
                          %c_2 = stablehlo.constant dense<2> : tensor<i64>
                          %3 = stablehlo.compare LT, %iterArg, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                          stablehlo.return %3 : tensor<i1>
                        } do {
                          %c_2 = stablehlo.constant dense<0> : tensor<i64>
                          %3 = stablehlo.dynamic_slice %iterArg_0, %iterArg, %c_2, sizes = [1, 3] : (tensor<2x3xf64>, \
                    tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
                          %4 = stablehlo.reshape %3 : (tensor<1x3xf64>) -> tensor<3xf64>
                          %5 = stablehlo.reshape %4 : (tensor<3xf64>) -> tensor<1x3xf64>
                          %6 = stablehlo.dynamic_update_slice %iterArg_1, %5, %iterArg, %c_2 : (tensor<2x3xf64>, \
                    tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<2x3xf64>
                          %c_3 = stablehlo.constant dense<1> : tensor<i64>
                          %7 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                          stablehlo.return %7, %iterArg_0, %6 : tensor<i64>, tensor<2x3xf64>, tensor<2x3xf64>
                        }
                        return %2#2 : tensor<2x3xf64>
                      }
                    }
                "},
            ),
        ] {
            let stacked_type = ArrayType::new_static(DataType::F64, [length, 3]).with_memory(memory);
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
            )
            .unwrap();
            assert_eq!(module, expected, "length {length}");
        }
    }

    #[test]
    fn test_scan_lowering_admits_stacks_sharded_over_auto_mesh_axes() {
        // A stack whose scan axis is sharded over an `Auto` mesh axis has replicated slices, which match a replicated
        // body declaration. The entry check uses the declared stacked types of core `scan` validation, which omit
        // shardings, so it accepts such a stack whatever the trip count is.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let slice_type = ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let slice = builder.add_input(slice_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![slice], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        for (length, expected) in [
            (
                0,
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<0xf64>) -> tensor<0xf64> {
                        %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                        %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<0xf64>
                        return %0 : tensor<0xf64>
                      }
                    }
                "},
            ),
            (
                4,
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<4xf64>) -> tensor<4xf64> {
                        %c = stablehlo.constant dense<0> : tensor<i64>
                        %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                        %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
                        %1:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %0) : tensor<i64>, \
                    tensor<4xf64>, tensor<4xf64>
                        cond {
                          %c_2 = stablehlo.constant dense<4> : tensor<i64>
                          %2 = stablehlo.compare LT, %iterArg, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                          stablehlo.return %2 : tensor<i1>
                        } do {
                          %2 = stablehlo.dynamic_slice %iterArg_0, %iterArg, sizes = [1] : (tensor<4xf64>, \
                    tensor<i64>) -> tensor<1xf64>
                          %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
                          %4 = stablehlo.reshape %3 : (tensor<f64>) -> tensor<1xf64>
                          %5 = stablehlo.dynamic_update_slice %iterArg_1, %4, %iterArg : (tensor<4xf64>, \
                    tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
                          %c_2 = stablehlo.constant dense<1> : tensor<i64>
                          %6 = stablehlo.add %iterArg, %c_2 : tensor<i64>
                          stablehlo.return %6, %iterArg_0, %5 : tensor<i64>, tensor<4xf64>, tensor<4xf64>
                        }
                        return %1#2 : tensor<4xf64>
                      }
                    }
                "},
            ),
        ] {
            let stacked_type = ArrayType::new_static(DataType::F64, [length])
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
                .unwrap();
            let mut builder = ArrayIrXlaProgramBuilder::new();
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
                to_mlir_module_for_program(&program, &[], &[stacked_type.clone()], &[output_type], "main", None, None)
                    .unwrap();
            assert_eq!(module, expected, "length {length}");
        }
    }

    #[test]
    fn test_scan_lowering_constrains_explicit_output_shardings() {
        // The body negates each `f32[8]` row of an `f32[4, 8]` stack whose second axis is sharded over the explicit
        // mesh axis `x`. The accumulator keeps the sharding of its slices, and the result of the scan is constrained to
        // the placement that its type states.
        let (client, mesh) = two_device_mesh(MeshAxisType::Explicit);
        let logical_mesh = mesh.logical_mesh().clone();
        let slice_type = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let stack_sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let stack_type = ArrayType::new_static(DataType::F32, [4, 8]).with_sharding(stack_sharding.clone()).unwrap();
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let row = builder.add_input(slice_type.into());
            let row = builder.add_instruction(NegOperation::new(), Vec::new(), vec![row], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![row], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let rows = builder.add_input(stack_type.clone().into());
        let rows = builder
            .add_instruction(XlaOperation::Scan(ScanOperation::new(0, 4)), vec![body_region], vec![rows], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![rows], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 8][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] .
                let %1:f32[4, 8][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] = scan [carry_count=0, iteration_count=4, \
                reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                        let %2:f32[8][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = neg %1
                        in (%2)
                    },
                ]
                in (%1)"},
        );
        let types = vec![stack_type.clone()];
        let shardings = vec![stack_sharding];
        let module =
            to_mlir_module_for_program(&program, &[], &types, &types, "main", Some(&shardings), Some(&shardings))
                .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"x\"}]>}) -> \
                (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"x\"}]>}) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x8xf32>
                    %1:3 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %0) : tensor<i64>, \
                tensor<4x8xf32>, tensor<4x8xf32>
                    cond {
                      %c_2 = stablehlo.constant dense<4> : tensor<i64>
                      %3 = stablehlo.compare LT, %iterArg, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %3 : tensor<i1>
                    } do {
                      %c_2 = stablehlo.constant dense<0> : tensor<i64>
                      %3 = stablehlo.dynamic_slice %iterArg_0, %iterArg, %c_2, sizes = [1, 8] : (tensor<4x8xf32>, \
                tensor<i64>, tensor<i64>) -> tensor<1x8xf32>
                      %4 = stablehlo.reshape %3 : (tensor<1x8xf32>) -> tensor<8xf32>
                      %5 = stablehlo.negate %4 : tensor<8xf32>
                      %6 = stablehlo.reshape %5 : (tensor<8xf32>) -> tensor<1x8xf32>
                      %7 = stablehlo.dynamic_update_slice %iterArg_1, %6, %iterArg, %c_2 : (tensor<4x8xf32>, \
                tensor<1x8xf32>, tensor<i64>, tensor<i64>) -> tensor<4x8xf32>
                      %c_3 = stablehlo.constant dense<1> : tensor<i64>
                      %8 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                      stablehlo.return %8, %iterArg_0, %7 : tensor<i64>, tensor<4x8xf32>, tensor<4x8xf32>
                    }
                    %2 = sdy.sharding_constraint %1#2 <@mesh, [{}, {\"x\"}]> : tensor<4x8xf32>
                    return %2 : tensor<4x8xf32>
                  }
                }
            "},
        );
        // Each device holds the negated columns `4d..4d + 4` of every row.
        let expected = (0..2)
            .map(|device| {
                vec![
                    (0..4)
                        .flat_map(|row| (0..4).map(move |column| -((8 * row + 4 * device + column) as f32)))
                        .collect(),
                ]
            })
            .collect::<Vec<Vec<Vec<f32>>>>();
        let values = (0..32).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(execute_on_two_devices(&client, &mesh, &module, &[(&stack_type, &values)]), expected);
    }

    #[test]
    fn test_scan_lowering_constrains_results_of_mixed_explicit_and_auto_meshes() {
        // The mesh has the explicit axis `x` of size 1 and the `Auto` axis `y` of size 2. The body passes a replicated
        // carry through and negates each `f32[8]` row of an `f32[4, 8]` stack whose second axis is sharded over `y`, so
        // neither result places anything over the explicit axis. Their mesh has an explicit axis, though, so both
        // results are still constrained to the placements that their types state.
        let (client, _) = two_device_mesh(MeshAxisType::Auto);
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let carry_sharding = Sharding::replicated(logical_mesh.clone(), 0);
        let carry_type = ArrayType::scalar(DataType::F32).with_sharding(carry_sharding.clone()).unwrap();
        let slice_type = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap())
            .unwrap();
        let stack_sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                .unwrap();
        let stack_type = ArrayType::new_static(DataType::F32, [4, 8]).with_sharding(stack_sharding.clone()).unwrap();
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(carry_type.clone().into());
            let row = builder.add_input(slice_type.into());
            let row = builder.add_instruction(NegOperation::new(), Vec::new(), vec![row], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![carry, row],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(carry_type.clone().into());
        let rows = builder.add_input(stack_type.clone().into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(ScanOperation::new(1, 4)), vec![body_region], vec![carry, rows], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc!(
                "
                lambda %0:{carry}, %1:{stack} .
                let %2:{carry}, %3:{stack} = scan [carry_count=1, iteration_count=4, reverse=false] %0 %1 [
                    body={{
                        lambda %0:i64[], %1:{carry}, %2:{slice} .
                        let %3:{slice} = neg %2
                        in (%1, %3)
                    }},
                ]
                in (%2, %3)",
                carry = "f32[][sharding={mesh<['x'=1:explicit, 'y'=2:auto]>, []}]",
                stack = "f32[4, 8][sharding={mesh<['x'=1:explicit, 'y'=2:auto]>, [{}, {'y'}]}]",
                slice = "f32[8][sharding={mesh<['x'=1:explicit, 'y'=2:auto]>, [{'y'}]}]",
            ),
        );
        let types = vec![carry_type.clone(), stack_type.clone()];
        let shardings = vec![carry_sharding, stack_sharding];
        let module =
            to_mlir_module_for_program(&program, &[], &types, &types, "main", Some(&shardings), Some(&shardings))
                .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=1, \"y\"=2]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}, \
                %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"x\"}>}) -> \
                (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}, tensor<4x8xf32> \
                {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"y\"}], replicated={\"x\"}>}) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x8xf32>
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0) : \
                tensor<i64>, tensor<f32>, tensor<4x8xf32>, tensor<4x8xf32>
                    cond {
                      %c_3 = stablehlo.constant dense<4> : tensor<i64>
                      %4 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %4 : tensor<i1>
                    } do {
                      %c_3 = stablehlo.constant dense<0> : tensor<i64>
                      %4 = stablehlo.dynamic_slice %iterArg_1, %iterArg, %c_3, sizes = [1, 8] : (tensor<4x8xf32>, \
                tensor<i64>, tensor<i64>) -> tensor<1x8xf32>
                      %5 = stablehlo.reshape %4 : (tensor<1x8xf32>) -> tensor<8xf32>
                      %6 = stablehlo.negate %5 : tensor<8xf32>
                      %7 = stablehlo.reshape %6 : (tensor<8xf32>) -> tensor<1x8xf32>
                      %8 = stablehlo.dynamic_update_slice %iterArg_2, %7, %iterArg, %c_3 : (tensor<4x8xf32>, \
                tensor<1x8xf32>, tensor<i64>, tensor<i64>) -> tensor<4x8xf32>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %9, %iterArg_0, %iterArg_1, %8 : tensor<i64>, tensor<f32>, tensor<4x8xf32>, \
                tensor<4x8xf32>
                    }
                    %2 = sdy.sharding_constraint %1#1 <@mesh, [], replicated={\"x\"}> : tensor<f32>
                    %3 = sdy.sharding_constraint %1#3 <@mesh, [{}, {\"y\"}], replicated={\"x\"}> : tensor<4x8xf32>
                    return %2, %3 : tensor<f32>, tensor<4x8xf32>
                  }
                }
            "},
        );
        // Each device holds the whole carry and the negated columns `4d..4d + 4` of every row.
        let expected = (0..2)
            .map(|device| {
                vec![
                    vec![5.0],
                    (0..4)
                        .flat_map(|row| (0..4).map(move |column| -((8 * row + 4 * device + column) as f32)))
                        .collect(),
                ]
            })
            .collect::<Vec<Vec<Vec<f32>>>>();
        let values = (0..32).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(
            execute_on_two_devices(&client, &mesh, &module, &[(&carry_type, &[5.0]), (&stack_type, &values)]),
            expected,
        );
    }

    #[test]
    fn test_scan_execution_in_shard_map() {
        // Each of the two devices of a manual `shard_map` sums its own column of the `f32[2, 2]` items, which are
        // sharded along their second axis, into a sum that starts at its own entry of the `f32[2]` initial sums. The
        // body `(i, c, x) -> (c + x, c + x)` also stacks the running sums, so each device allocates and fills a local
        // `f32[2, 1]` accumulator inside the manual computation.
        let (client, mesh) = two_device_mesh(MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let manual_axes = vec!["x".to_string()];
        let sum_sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let items_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let sum_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sum_sharding.clone()).unwrap();
        let items_type = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(items_sharding.clone()).unwrap();
        let local_sum_type = ArrayType::new_static(DataType::F32, [1])
            .with_sharding(sum_sharding.local_sharding(&manual_axes).unwrap())
            .unwrap();
        let local_items_type = ArrayType::new_static(DataType::F32, [2, 1])
            .with_sharding(items_sharding.local_sharding(&manual_axes).unwrap())
            .unwrap();
        let scan_body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let sum = builder.add_input(local_sum_type.clone().into());
            let item = builder.add_input(local_sum_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, item], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let shard_map_body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let scan_body_region = builder.import_region(scan_body.entry_region_ref());
            let sum = builder.add_input(local_sum_type.into());
            let items = builder.add_input(local_items_type.into());
            let outputs = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(1, 2)),
                    vec![scan_body_region],
                    vec![sum, items],
                    None,
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let shardings = vec![sum_sharding, items_sharding];
        let operation = ShardMapOperation::from_program(
            &shard_map_body,
            vec![sum_type.clone().into(), items_type.clone().into()],
            ShardMap::new(logical_mesh, shardings.clone(), shardings.clone(), manual_axes).unwrap(),
        )
        .unwrap();
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let shard_map_body_region = builder.import_program(shard_map_body);
        let sum = builder.add_input(sum_type.clone().into());
        let items = builder.add_input(items_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::ShardMap(Box::new(operation)),
                vec![shard_map_body_region],
                vec![sum, items],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc!(
                "
                lambda %0:{sum}, %1:{items} .
                let %2:{sum}, %3:{items} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sum_sharding}, {items_sharding}],
                    out_shardings=[{sum_sharding}, {items_sharding}],
                    manual_axes=['x'],
                    global_input_types=[{sum}, {items}],
                    global_output_types=[{sum}, {items}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local_sum}, %1:{local_items} .
                        let %2:{local_sum}, %3:{local_items} = scan [carry_count=1, iteration_count=2, reverse=false] %0 %1 [
                            body={{
                                lambda %0:i64[], %1:{local_sum}, %2:{local_sum} .
                                let %3:{local_sum} = add %1 %2
                                in (%3, %3)
                            }},
                        ]
                        in (%2, %3)
                    }},
                ]
                in (%2, %3)",
                sum = "f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]",
                items = "f32[2, 2][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]",
                sum_sharding = "{mesh<['x'=2:manual]>, [{'x'}]}",
                items_sharding = "{mesh<['x'=2:manual]>, [{}, {'x'}]}",
                local_sum = "f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]",
                local_items = "f32[2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]",
            ),
        );
        let types = vec![sum_type.clone(), items_type.clone()];
        let module =
            to_mlir_module_for_program(&program, &[], &types, &types, "main", Some(&shardings), Some(&shardings))
                .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg1: \
                tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"x\"}]>}) -> (tensor<2xf32> {sdy.sharding \
                = #sdy.sharding<@mesh, [{\"x\"}]>}, tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, \
                {\"x\"}]>}) {
                    %0:2 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, [{}, \
                {\"x\"}]>] out_shardings=[<@mesh, [{\"x\"}]>, <@mesh, [{}, {\"x\"}]>] manual_axes={\"x\"} (%arg2: \
                tensor<1xf32>, %arg3: tensor<2x1xf32>) {
                      %c = stablehlo.constant dense<0> : tensor<i64>
                      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
                      %2:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg2, %iterArg_1 = %arg3, %iterArg_2 = %1) : \
                tensor<i64>, tensor<1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
                      cond {
                        %c_3 = stablehlo.constant dense<2> : tensor<i64>
                        %3 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                        stablehlo.return %3 : tensor<i1>
                      } do {
                        %c_3 = stablehlo.constant dense<0> : tensor<i64>
                        %3 = stablehlo.dynamic_slice %iterArg_1, %iterArg, %c_3, sizes = [1, 1] : (tensor<2x1xf32>, \
                tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
                        %4 = stablehlo.reshape %3 : (tensor<1x1xf32>) -> tensor<1xf32>
                        %5 = stablehlo.add %iterArg_0, %4 : tensor<1xf32>
                        %6 = stablehlo.reshape %5 : (tensor<1xf32>) -> tensor<1x1xf32>
                        %7 = stablehlo.dynamic_update_slice %iterArg_2, %6, %iterArg, %c_3 : (tensor<2x1xf32>, \
                tensor<1x1xf32>, tensor<i64>, tensor<i64>) -> tensor<2x1xf32>
                        %c_4 = stablehlo.constant dense<1> : tensor<i64>
                        %8 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                        stablehlo.return %8, %5, %iterArg_1, %7 : tensor<i64>, tensor<1xf32>, tensor<2x1xf32>, \
                tensor<2x1xf32>
                      }
                      sdy.return %2#1, %2#3 : tensor<1xf32>, tensor<2x1xf32>
                    } : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<2xf32>, tensor<2x2xf32>)
                    return %0#0, %0#1 : tensor<2xf32>, tensor<2x2xf32>
                  }
                }
            "},
        );
        assert_eq!(
            execute_on_two_devices(
                &client,
                &mesh,
                &module,
                &[(&sum_type, &[100.0, 200.0]), (&items_type, &[1.0, 10.0, 2.0, 20.0])],
            ),
            vec![vec![vec![103.0], vec![101.0, 103.0]], vec![vec![230.0], vec![210.0, 230.0]]],
        );
    }

    #[test]
    fn test_scan_execution_nested_in_condition() {
        // The `true` branch sums the items with a scan whose body prints each item, and the `false` branch returns the
        // initial sum. The condition must thread the ordered I/O token through the scan, so that the prints appear in
        // item order and only when the `true` branch runs.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let items_type = ArrayType::new_static(DataType::F64, [3]);
        let true_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let body_region = builder.import_region(printing_sum_body().entry_region_ref());
            let sum = builder.add_input(scalar_type.clone().into());
            let items = builder.add_input(items_type.clone().into());
            let sum = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(1, 3)),
                    vec![body_region],
                    vec![sum, items],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let sum = builder.add_input(scalar_type.clone().into());
            builder.add_input(items_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let selector = builder.add_input(scalar_type.clone().into());
        let sum = builder.add_input(scalar_type.into());
        let items = builder.add_input(items_type.into());
        let predicate = positive(&mut builder, selector);
        let sum = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, sum, items],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[3] .
                let %3:f64[] = zero_like %0
                    %4:bool[] = compare [direction=GreaterThan] %0 %3
                    %5:f64[] = condition %4 %1 %2 [
                        true={
                            lambda %0:f64[], %1:f64[3] .
                            let %2:f64[] = scan [carry_count=1, iteration_count=3, reverse=false] %0 %1 [
                                body={
                                    lambda %0:i64[], %1:f64[], %2:f64[] .
                                    let %3:f64[] = print [label=item] %2
                                        %4:f64[] = add %1 %3
                                    in (%4)
                                },
                            ]
                            in (%2)
                        },
                        false={
                            lambda %0:f64[], %1:f64[3] .
                            in (%0)
                        },
                    ]
                in (%5)"},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<3xf64>) -> tensor<f64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.compare GT, %arg0, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:2 = \"stablehlo.if\"(%0) ({
                      %c = stablehlo.constant dense<0> : tensor<i64>
                      %3:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg1, %iterArg_1 = %arg2, %iterArg_2 = %1) : \
                tensor<i64>, tensor<f64>, tensor<3xf64>, !stablehlo.token
                      cond {
                        %c_3 = stablehlo.constant dense<3> : tensor<i64>
                        %4 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                        stablehlo.return %4 : tensor<i1>
                      } do {
                        %4 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xf64>, tensor<i64>) \
                -> tensor<1xf64>
                        %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
                        %6 = stablehlo.custom_call @ryft.print(%5, %iterArg_2) {api_version = 4 : i32, backend_config \
                = {label = \"item\"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                        %7 = stablehlo.add %iterArg_0, %5 : tensor<f64>
                        %c_3 = stablehlo.constant dense<1> : tensor<i64>
                        %8 = stablehlo.add %iterArg, %c_3 : tensor<i64>
                        stablehlo.return %8, %7, %iterArg_1, %6 : tensor<i64>, tensor<f64>, tensor<3xf64>, \
                !stablehlo.token
                      }
                      stablehlo.return %3#1, %3#3 : tensor<f64>, !stablehlo.token
                    }, {
                      stablehlo.return %arg1, %1 : tensor<f64>, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token)
                    return %2#0 : tensor<f64>
                  }
                }
            "},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let run = |selector: f64| {
            with_captured_prints(|| {
                execute_mixed_program(
                    &client,
                    &program,
                    &[
                        MixedValue::Array(vec![selector], vec![]),
                        MixedValue::Array(vec![10.0], vec![]),
                        MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                    ],
                    &[],
                )
            })
        };
        let (outputs, lines) = run(1.0);
        assert_eq!(outputs, Ok(vec![MixedValue::Array(vec![16.0], vec![])]));
        assert_eq!(lines, vec!["item: 1.0", "item: 2.0", "item: 3.0"]);
        let (outputs, lines) = run(-1.0);
        assert_eq!(outputs, Ok(vec![MixedValue::Array(vec![10.0], vec![])]));
        assert_eq!(lines, Vec::<String>::new());
    }

    #[test]
    fn test_scan_execution_nested_in_while() {
        // Each iteration of the loop sums the items into its running sum with a scan whose body prints each item. The
        // loop must thread the ordered I/O token through the scan, so that the prints appear in item order once per
        // loop iteration.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let items_type = ArrayType::new_static(DataType::F64, [2]);
        let state_types = [scalar_type.clone(), scalar_type.clone(), scalar_type.clone(), items_type.clone()];
        let loop_condition = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let predicate = builder
                .add_instruction(compare(ComparisonDirection::LessThan), Vec::new(), vec![inputs[0], inputs[1]], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder; 4], vec![Placeholder])
                .unwrap()
        };
        let loop_body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let scan_body_region = builder.import_region(printing_sum_body().entry_region_ref());
            let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let (index, limit, sum, items) = (inputs[0], inputs[1], inputs[2], inputs[3]);
            let one = builder
                .add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![index], None)
                .unwrap()[0];
            let index = builder.add_instruction(AddOperation::new(), Vec::new(), vec![index, one], None).unwrap()[0];
            let sum = builder
                .add_instruction(
                    XlaOperation::Scan(ScanOperation::new(1, 2)),
                    vec![scan_body_region],
                    vec![sum, items],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![index, limit, sum, items],
                    vec![Placeholder; 4],
                    vec![Placeholder; 4],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let condition_region = builder.import_region(loop_condition.entry_region_ref());
        let body_region = builder.import_region(loop_body.entry_region_ref());
        let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
        let sum = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                inputs,
                None,
            )
            .unwrap()[2];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[2] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[2] = while %0 %1 %2 %3 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[2] .
                        let %4:bool[] = compare [direction=LessThan] %0 %1
                        in (%4)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[2] .
                        let %4:f64[] = one_like %0
                            %5:f64[] = add %0 %4
                            %6:f64[] = scan [carry_count=1, iteration_count=2, reverse=false] %2 %3 [
                                body={
                                    lambda %0:i64[], %1:f64[], %2:f64[] .
                                    let %3:f64[] = print [label=item] %2
                                        %4:f64[] = add %1 %3
                                    in (%4)
                                },
                            ]
                        in (%5, %1, %6, %3)
                    },
                ]
                in (%6)"},
        );
        assert_eq!(
            lowered_mixed_module(&program),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<2xf64>) -> \
                tensor<f64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:5 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %arg1, %iterArg_1 = %arg2, %iterArg_2 = \
                %arg3, %iterArg_3 = %0) : tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xf64>, !stablehlo.token
                    cond {
                      %2 = stablehlo.compare LT, %iterArg, %iterArg_0, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                      %2 = stablehlo.add %iterArg, %cst : tensor<f64>
                      %c = stablehlo.constant dense<0> : tensor<i64>
                      %3:4 = stablehlo.while(%iterArg_4 = %c, %iterArg_5 = %iterArg_1, %iterArg_6 = %iterArg_2, \
                %iterArg_7 = %iterArg_3) : tensor<i64>, tensor<f64>, tensor<2xf64>, !stablehlo.token
                      cond {
                        %c_8 = stablehlo.constant dense<2> : tensor<i64>
                        %4 = stablehlo.compare LT, %iterArg_4, %c_8, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                        stablehlo.return %4 : tensor<i1>
                      } do {
                        %4 = stablehlo.dynamic_slice %iterArg_6, %iterArg_4, sizes = [1] : (tensor<2xf64>, \
                tensor<i64>) -> tensor<1xf64>
                        %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
                        %6 = stablehlo.custom_call @ryft.print(%5, %iterArg_7) {api_version = 4 : i32, backend_config \
                = {label = \"item\"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                        %7 = stablehlo.add %iterArg_5, %5 : tensor<f64>
                        %c_8 = stablehlo.constant dense<1> : tensor<i64>
                        %8 = stablehlo.add %iterArg_4, %c_8 : tensor<i64>
                        stablehlo.return %8, %7, %iterArg_6, %6 : tensor<i64>, tensor<f64>, tensor<2xf64>, \
                !stablehlo.token
                      }
                      stablehlo.return %2, %iterArg_0, %3#1, %iterArg_2, %3#3 : tensor<f64>, tensor<f64>, tensor<f64>, \
                tensor<2xf64>, !stablehlo.token
                    }
                    return %1#2 : tensor<f64>
                  }
                }
            "},
        );
        let client = execution_client();
        ensure_print_handler_registered(&client).unwrap();
        let (outputs, lines) = with_captured_prints(|| {
            execute_mixed_program(
                &client,
                &program,
                &[
                    MixedValue::Array(vec![0.0], vec![]),
                    MixedValue::Array(vec![2.0], vec![]),
                    MixedValue::Array(vec![0.0], vec![]),
                    MixedValue::Array(vec![1.0, 2.0], vec![2]),
                ],
                &[],
            )
        });
        assert_eq!(outputs, Ok(vec![MixedValue::Array(vec![6.0], vec![])]));
        assert_eq!(lines, vec!["item: 1.0", "item: 2.0", "item: 1.0", "item: 2.0"]);
    }

    #[test]
    fn test_scan_execution_batched() {
        // The body `(i, c, x) -> (c + x, c)` sums the items and stacks the running sums that it received.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let sum = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, item], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![next, sum],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let sum = builder.add_input(scalar_type.into());
        let items = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(ScanOperation::new(1, 3)), vec![body_region], vec![sum, items], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let client = execution_client();
        let batch_type = DimensionType::from(Dimension::Static(2));

        // Mapped items make the sum a mapped carry, which the batched scan instantiates at the batch axis, and the
        // running sums stack along a trailing batch axis.
        let (batched, output_axes) = program
            .batched_with_threaded_extent(
                batch_type.clone(),
                ShardingDimension::replicated(),
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(1)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f64[], %2:f64[2, 3] .
                let %3:f64[2] = broadcast [output_axes=[]] %1 %0
                    %4:f64[3, 2] = transpose [permutation=[1, 0]] %2
                    %5:dimension<2>, %6:f64[2], %7:f64[3, 2] = scan [carry_count=2, iteration_count=3, reverse=false] %0 %3 %4 [
                        body={
                            lambda %0:i64[], %1:dimension<2>, %2:f64[2], %3:f64[2] .
                            let %4:f64[2] = add %2 %3
                            in (%1, %4, %2)
                        },
                    ]
                in (%0, %6, %7)"},
        );
        assert_eq!(
            execute_mixed_program(
                &client,
                &batched,
                &[
                    MixedValue::Dimension(2),
                    MixedValue::Array(vec![10.0], vec![]),
                    MixedValue::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                ],
                &[],
            ),
            Ok(vec![
                MixedValue::Dimension(2),
                MixedValue::Array(vec![16.0, 25.0], vec![2]),
                MixedValue::Array(vec![10.0, 10.0, 11.0, 14.0, 13.0, 19.0], vec![3, 2]),
            ]),
        );

        // A mapped initial sum with replicated items maps the carry only, so each iteration broadcasts its replicated
        // item over the batch.
        let (batched, output_axes) = program
            .batched_with_threaded_extent(
                batch_type,
                ShardingDimension::replicated(),
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(1)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f64[2], %2:f64[3] .
                let %3:dimension<2>, %4:f64[2], %5:f64[3, 2] = scan [carry_count=2, iteration_count=3, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:dimension<2>, %2:f64[2], %3:f64[] .
                        let %4:f64[2] = broadcast [output_axes=[]] %3 %1
                            %5:f64[2] = add %2 %4
                        in (%1, %5, %2)
                    },
                ]
                in (%0, %4, %5)"},
        );
        assert_eq!(
            execute_mixed_program(
                &client,
                &batched,
                &[
                    MixedValue::Dimension(2),
                    MixedValue::Array(vec![10.0, 20.0], vec![2]),
                    MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                ],
                &[],
            ),
            Ok(vec![
                MixedValue::Dimension(2),
                MixedValue::Array(vec![16.0, 26.0], vec![2]),
                MixedValue::Array(vec![10.0, 20.0, 11.0, 21.0, 13.0, 23.0], vec![3, 2]),
            ]),
        );
    }

    #[test]
    fn test_scan_execution_differentiated() {
        // The body `(i, c, x) -> (c * x + 1)` makes the final carry of a forward scan from `c` over `[x0, x1, x2]`
        // equal to `((c * x0 + 1) * x1 + 1) * x2 + 1`, whose gradient with respect to the items needs the carry that
        // each iteration received as a residual stack. A reversed scan visits the items backwards. The primal and the
        // pullback of each scan run in one program.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let body = {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0];
            let one =
                builder.add_instruction(OneLikeOperation::<ArrayType>::new(), Vec::new(), vec![item], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, one], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let value_and_gradient = |scan: ScanOperation<ArrayIrType>, items_type: ArrayType| {
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let body_region = builder.import_region(body.entry_region_ref());
            let carry = builder.add_input(scalar_type.clone().into());
            let items = builder.add_input(items_type.clone().into());
            let mut inputs = vec![carry, items];
            if let Dimension::Dynamic(length) = scan.iteration_count() {
                inputs.push(builder.add_input(DimensionType::from(length.clone()).into()));
            }
            let input_count = inputs.len();
            let output = builder.add_instruction(XlaOperation::Scan(scan), vec![body_region], inputs, None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder; input_count],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize_with_respect_to(&[1]).unwrap();
            let pullback = linearization.pullback().unwrap().into_program();
            let mut builder = ArrayIrXlaProgramBuilder::new();
            let inputs = program.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let cotangent = builder.add_input(scalar_type.clone().into());
            let primal_outputs = builder.splice_program(linearization.primal(), &inputs).unwrap();
            let mut pullback_inputs = vec![cotangent];
            pullback_inputs.extend_from_slice(&primal_outputs[1..]);
            let gradient = builder.splice_program(&pullback, &pullback_inputs).unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![primal_outputs[0], gradient[0]],
                    vec![Placeholder; input_count + 1],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let client = execution_client();
        let static_items_type = ArrayType::new_static(DataType::F64, [3]);
        let static_inputs = [
            MixedValue::Array(vec![1.0], vec![]),
            MixedValue::Array(vec![2.0, 3.0, 4.0], vec![3]),
            MixedValue::Array(vec![1.0], vec![]),
        ];

        // Forward: `c1 = 3`, `c2 = 10`, and `c3 = 41`, with gradient `[c0 x1 x2, c1 x2, c2] = [12, 12, 10]`.
        let program = value_and_gradient(ScanOperation::new(1, 3), static_items_type.clone());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[], %4:f64[3] = scan [carry_count=1, iteration_count=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                            %4:f64[] = one_like %2
                            %5:f64[] = add %3 %4
                        in (%5, %1)
                    },
                ]
                    %5:f64[] = zero [type=f64[]]
                    %6:f64[], %7:f64[3] = scan [carry_count=1, iteration_count=3, reverse=true] %2 %1 %4 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                            let %4:f64[] = mul %2 %1
                                %5:f64[] = mul %3 %1
                            in (%4, %5)
                        },
                    ]
                in (%3, %7)"},
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &static_inputs, &[]),
            Ok(vec![MixedValue::Array(vec![41.0], vec![]), MixedValue::Array(vec![12.0, 12.0, 10.0], vec![3])]),
        );

        // Reversed: `c1 = 5`, `c2 = 16`, and `c3 = 33`, with gradient `[c2, c1 x0, c0 x1 x0] = [16, 10, 6]`.
        let program = value_and_gradient(ScanOperation::new(1, 3).with_reverse(true), static_items_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[], %4:f64[3] = scan [carry_count=1, iteration_count=3, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                            %4:f64[] = one_like %2
                            %5:f64[] = add %3 %4
                        in (%5, %1)
                    },
                ]
                    %5:f64[] = zero [type=f64[]]
                    %6:f64[], %7:f64[3] = scan [carry_count=1, iteration_count=3, reverse=false] %2 %1 %4 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                            let %4:f64[] = mul %2 %1
                                %5:f64[] = mul %3 %1
                            in (%4, %5)
                        },
                    ]
                in (%3, %7)"},
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &static_inputs, &[]),
            Ok(vec![MixedValue::Array(vec![33.0], vec![]), MixedValue::Array(vec![16.0, 10.0, 6.0], vec![3])]),
        );

        // A dynamic length runs the same iterations at runtime length 3 and leaves the carry and an empty gradient at
        // runtime length 0.
        let dynamic_items_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(length.clone())]));
        let program = value_and_gradient(ScanOperation::new(1, Dimension::Dynamic(length)), dynamic_items_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[length], %2:dimension<length ∈ [0, 5)>, %3:f64[] .
                let %4:f64[], %5:f64[length] = scan [carry_count=1, iteration_count=length, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %2
                            %4:f64[] = one_like %2
                            %5:f64[] = add %3 %4
                        in (%5, %1)
                    },
                ]
                    %6:f64[] = zero [type=f64[]]
                    %7:f64[], %8:f64[length] = scan [carry_count=1, iteration_count=length, reverse=true] %3 %1 %5 %2 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                            let %4:f64[] = mul %2 %1
                                %5:f64[] = mul %3 %1
                            in (%4, %5)
                        },
                    ]
                in (%4, %8)"},
        );
        let run = |runtime_length: i64| {
            execute_mixed_program(
                &client,
                &program,
                &[
                    MixedValue::Array(vec![1.0], vec![]),
                    MixedValue::Array(vec![2.0, 3.0, 4.0, 0.0], vec![4]),
                    MixedValue::Dimension(runtime_length),
                    MixedValue::Array(vec![1.0], vec![]),
                ],
                &[runtime_length as i32],
            )
        };
        assert_eq!(
            run(3),
            Ok(vec![
                MixedValue::Array(vec![41.0], vec![]),
                MixedValue::Array(vec![12.0, 12.0, 10.0], vec![3]),
                MixedValue::Dimension(3),
            ]),
        );
        assert_eq!(
            run(0),
            Ok(vec![
                MixedValue::Array(vec![1.0], vec![]),
                MixedValue::Array(vec![], vec![0]),
                MixedValue::Dimension(0),
            ]),
        );
    }

    #[test]
    fn test_scan_frontend_execution() {
        // Tuple carries and output slices retain their parameter order through frontend tracing and XLA lowering.
        // Reversed scans stack suffix sums/products at the original input positions, including the unrolled tail.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let inputs_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let client = execution_client();
        for (reverse, sums, products) in
            [(false, vec![1.0, 3.0, 6.0], vec![1.0, 2.0, 6.0]), (true, vec![6.0, 5.0, 3.0], vec![6.0, 6.0, 3.0])]
        {
            let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
                |(carry, inputs)| {
                    scan(
                        |_, (sum, product), value| {
                            let sum = sum.add(&value)?;
                            let product = product.mul(&value)?;
                            Ok(((sum.clone(), product.clone()), (sum, product)))
                        },
                        carry,
                        inputs,
                        ScanOptions::new().with_reverse(reverse).with_unroll(2)?,
                    )
                },
                ((scalar_type.clone(), scalar_type.clone()), inputs_type.clone()),
            )
            .unwrap();
            let program = program.into_flat_program();
            assert_eq!(
                program.to_string(),
                formatdoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[3] .
                    let %3:f64[], %4:f64[], %5:f64[3], %6:f64[3] = scan \
                    [carry_count=2, iteration_count=3, reverse={reverse}, unroll=2] %0 %1 %2 [
                        body={{
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[] .
                            let %4:f64[] = add %1 %3
                                %5:f64[] = mul %2 %3
                            in (%4, %5, %4, %5)
                        }},
                    ]
                    in (%3, %4, %5, %6)"},
            );
            assert_eq!(
                execute_mixed_program(
                    &client,
                    &program,
                    &[
                        MixedValue::Array(vec![0.0], vec![]),
                        MixedValue::Array(vec![1.0], vec![]),
                        MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]),
                    ],
                    &[],
                ),
                Ok(vec![
                    MixedValue::Array(vec![6.0], vec![]),
                    MixedValue::Array(vec![6.0], vec![]),
                    MixedValue::Array(sums, vec![3]),
                    MixedValue::Array(products, vec![3]),
                ]),
            );
        }
    }

    #[test]
    fn test_scan_frontend_execution_dynamic_count() {
        // The count value supplies the frontend's execution domain when there is no carry or stacked input. Its
        // runtime extent controls the output allocation's logical shape, including the empty result.
        let count_type = DimensionType::new("iterations", DimensionBounds::non_negative(Some(5)).unwrap());
        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |count| {
                let ((), outputs) = scan(
                    |index, (), ()| Ok(((), index.convert_element_type(DataType::F64)?)),
                    (),
                    (),
                    ScanOptions::new().with_iteration_count_value(count).with_unroll(2)?,
                )?;
                Ok(outputs)
            },
            ArrayIrType::Dimension(count_type),
        )
        .unwrap();
        let program = program.into_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<iterations ∈ [0, 5)> .
                let %1:f64[iterations] = scan [carry_count=0, iteration_count=iterations, reverse=false, unroll=2] %0 [
                    body={
                        lambda %0:i64[] .
                        let %1:f64[] = convert_element_type [data_type=f64] %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
        let client = execution_client();
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(3)], &[]),
            Ok(vec![MixedValue::Array(vec![0.0, 1.0, 2.0], vec![3]), MixedValue::Dimension(3)]),
        );
        assert_eq!(
            execute_mixed_program(&client, &program, &[MixedValue::Dimension(0)], &[]),
            Ok(vec![MixedValue::Array(vec![], vec![0]), MixedValue::Dimension(0)]),
        );
    }

    #[test]
    fn test_scan_frontend_execution_through_compile() {
        // The public array-only compilation facade projects its tracers into ArrayType; the frontend's nested body
        // and its region must lift back into the backend's array IR operation family before lowering.
        let client = execution_client();
        let device = client.addressable_devices().unwrap().remove(0);
        let device_id = device.id().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let domain = XlaSession::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let inputs_type = ArrayType::new_static(DataType::F64, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(carry, inputs)| {
                scan(
                    |_, carry, item| {
                        let next = carry.add(&item)?;
                        Ok((next.clone(), next))
                    },
                    carry,
                    inputs,
                    ScanOptions::new(),
                )
                .unwrap()
            },
            (scalar_type.clone(), inputs_type.clone()),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let scalar = "f64[][sharding={mesh<['x'=1:auto]>, []}]";
        let stack = "f64[3][sharding={mesh<['x'=1:auto]>, [{}]}]";
        assert_eq!(
            compiled.source_program().program().to_string(),
            formatdoc! {"
                lambda %0:{scalar}, %1:{stack} .
                let %2:{scalar}, %3:{stack} = scan [carry_count=1, iteration_count=3, reverse=false] %0 %1 [
                    body={{
                        lambda %0:i64[], %1:{scalar}, %2:{scalar} .
                        let %3:{scalar} = add %1 %2
                        in (%3, %3)
                    }},
                ]
                in (%2, %3)"},
        );
        let carry = XlaArray::from_host_buffer(&domain, scalar_type, mesh.clone(), values_to_bytes(&[0f64])).unwrap();
        let inputs =
            XlaArray::from_host_buffer(&domain, inputs_type, mesh, values_to_bytes(&[1f64, 2.0, 3.0])).unwrap();
        let (sum, prefixes) = domain.interpret(&compiled.executable_function(), (carry, inputs)).unwrap();
        let outputs = [sum, prefixes]
            .into_iter()
            .map(|output| {
                let bytes = output
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                values_from_bytes::<f64>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![vec![6.0], vec![1.0, 3.0, 6.0]]);
    }
}
