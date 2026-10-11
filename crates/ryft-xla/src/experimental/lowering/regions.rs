use std::rc::Rc;

use ryft_core::{
    AtomId, Instruction, Operation, Parameterized, Program, ProgramError, RegionRef,
    RematerializationOptimizationBarrier, Type as RyftType, Typed, Value,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::types::lower_constant;
use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, LowerableOperationFamily, LoweringError, XlaLowerer, lower_instruction,
};
use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaProgram};
use crate::operations::jit_call::JitCallFunctionMap;

#[cfg(test)]
use crate::experimental::lowering::types::lower_literal_value;
#[cfg(test)]
use crate::experimental::lowering::{LowerableOperation, MlirLowerableValue};
#[cfg(test)]
use crate::experimental::ops::XlaOperation;
#[cfg(test)]
use ryft_core::ArrayType;
#[cfg(test)]
use ryft_mlir::Type;

/// Inlines a nested sub-program into the given block by mapping the provided input
/// MLIR values to the body's input atoms, lowering constants and instructions in topological
/// order, and returning lowered values corresponding to the program's output atoms.
///
/// `effect_tokens` are the per-class tokens of the lowering scope the program inlines into. They flow into each
/// instruction's lowerer and the updated tokens are read back out after the instruction lowers, so same-class effects
/// chain in program order and the caller observes the program's final tokens.
///
/// When `optimization_barrier` selects any input, one
/// [`stablehlo.optimization_barrier`](https://openxla.org/stablehlo/spec#optimization_barrier) wraps the selected
/// input values together with the present effect tokens before the body is replayed, and the body consumes the
/// barrier's results. XLA cannot look through the barrier, so it can neither schedule the body before the selected
/// inputs are produced nor deduplicate it against an identical computation on the same inputs elsewhere (e.g., a
/// rematerialized recomputation of a primal computation). Because the tokens go through the same barrier and the
/// enclosing scope continues from the barrier's token results, the body's ordered effects are also sequenced after the
/// barrier. Unselected inputs and captured constants, which are not inputs of the body, do not go through the barrier.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_nested_program_inline<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    program: &FlatXlaProgram<O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    optimization_barrier: &RematerializationOptimizationBarrier,
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    lower_nested_region_inline(
        program.entry_region_ref(),
        input_values,
        block,
        context,
        location,
        captured_values,
        optimization_barrier,
        nested_functions,
        collective_state,
        effect_tokens,
    )
}

/// Inlines a borrowed nested region into the given block without materializing the region itself. Refer to the
/// documentation of [`lower_nested_program_inline`] for the effect-token and optimization-barrier contracts.
#[allow(clippy::too_many_arguments)]
fn lower_nested_region_inline<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    region: RegionRef<'_, XlaConstant, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    optimization_barrier: &RematerializationOptimizationBarrier,
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let mut input_values = input_values.to_vec();
    let selected_input_indices =
        (0..input_values.len()).filter(|index| optimization_barrier.selects(*index)).collect::<Vec<_>>();
    if !selected_input_indices.is_empty() {
        // The barrier results mirror its inputs: the selected input values first, followed by the present tokens in
        // the order in which they were appended.
        let barrier_inputs = selected_input_indices
            .iter()
            .map(|index| input_values[*index])
            .chain(effect_tokens.ordered_assertion)
            .chain(effect_tokens.ordered_io)
            .collect::<Vec<_>>();
        let barrier = block.append_operation(stable_hlo::optimization_barrier(barrier_inputs.as_slice(), location)?)?;
        let mut results = (0..barrier_inputs.len()).map(|index| barrier.result(index).unwrap().as_ref());
        for index in selected_input_indices {
            input_values[index] = results.next().unwrap();
        }
        if effect_tokens.ordered_assertion.is_some() {
            effect_tokens.ordered_assertion = results.next();
        }
        if effect_tokens.ordered_io.is_some() {
            effect_tokens.ordered_io = results.next();
        }
    }
    replay_region_ref_into_block(
        region,
        input_values,
        block,
        context,
        location,
        |atom_id, value, block, context, location| {
            lower_constant(atom_id, value, captured_values, block, context, location)
        },
        |instruction, inputs, block, context, location| {
            let input_types = instruction
                .inputs()
                .iter()
                .map(|input| region.atoms()[input.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let output_types = instruction
                .outputs()
                .iter()
                .map(|output| region.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let regions = instruction
                .regions()
                .iter()
                .map(|attached| RegionRef::new(region.arena(), *attached).map(RegionRef::to_program))
                .collect::<Result<Vec<_>, ProgramError>>()?;
            // Every StableHLO operation generated from this one Ryft instruction inherits the instruction's
            // provenance through the lowerer's shared location, with the ambient location as its base.
            let location = collective_state.instruction_location(context, instruction.provenance(), location);
            let mut lowerer = XlaLowerer::new(*block, context, location)
                .with_regions(regions)
                .with_nested_functions(nested_functions.cloned())
                .with_captured_values(captured_values)
                .with_effect_tokens(*effect_tokens)
                .with_collective_state(collective_state.clone());
            let outputs =
                instruction
                    .operation()
                    .lower(&mut lowerer, inputs, input_types.as_slice(), output_types.as_slice())?;
            if lowerer.effect_tokens.ordered_assertion.is_some() {
                collective_state.has_assertions.set(true);
            }
            *effect_tokens = lowerer.effect_tokens;
            Ok(outputs)
        },
    )
}

/// Drives [`Program::interpret_with`] to lower a staged program into MLIR ops appended to `block`.
///
/// The two callbacks plug in lowering policies for [`Atom::Constant`](ryft_core::Atom::Constant)s and [`Instruction`]s
/// respectively while the generic interpreter handles use-count tracking and atom bookkeeping. Each callback receives a
/// mutable [`BlockRef`] because [`BlockRef`] is `Copy` and the helper hands each closure its own copy backed by the
/// same MLIR block.
fn replay_program_into_block<
    'b,
    'c: 'b,
    't: 'c,
    T: RyftType,
    O,
    V: Value<Type = T>,
    Input,
    Output,
    LiftConstant,
    ApplyOp,
>(
    program: &Program<V, O, Input, Output>,
    input_values: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    mut lift_constant: LiftConstant,
    mut apply_op: ApplyOp,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    O: Operation<Type = T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    LiftConstant: FnMut(
        AtomId,
        &V,
        &mut BlockRef<'b, 'c, 't>,
        &'c MlirContext<'t>,
        LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>,
    ApplyOp: FnMut(
        &Instruction<O>,
        &[ValueRef<'b, 'c, 't>],
        &mut BlockRef<'b, 'c, 't>,
        &'c MlirContext<'t>,
        LocationRef<'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>,
{
    let mut block_for_constants = *block;
    let mut block_for_ops = *block;
    program.interpret_with(
        input_values,
        |atom_id, value| lift_constant(atom_id, value, &mut block_for_constants, context, location),
        |instruction, inputs| apply_op(instruction, inputs, &mut block_for_ops, context, location),
    )
}

/// Drives [`RegionRef::interpret_with`] to lower a borrowed region into MLIR ops appended to `block`.
fn replay_region_ref_into_block<'b, 'c: 'b, 't: 'c, O, V, LiftConstant, ApplyOp>(
    region: RegionRef<'_, V, O>,
    input_values: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    mut lift_constant: LiftConstant,
    mut apply_op: ApplyOp,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: Value,
    O: Operation<Type = V::Type>,
    LiftConstant: FnMut(
        AtomId,
        &V,
        &mut BlockRef<'b, 'c, 't>,
        &'c MlirContext<'t>,
        LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>,
    ApplyOp: FnMut(
        &Instruction<O>,
        &[ValueRef<'b, 'c, 't>],
        &mut BlockRef<'b, 'c, 't>,
        &'c MlirContext<'t>,
        LocationRef<'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>,
{
    let mut block_for_constants = *block;
    let mut block_for_ops = *block;
    region.interpret_with(
        input_values,
        |atom_id, value| lift_constant(atom_id, value, &mut block_for_constants, context, location),
        |instruction, inputs| apply_op(instruction, inputs, &mut block_for_ops, context, location),
    )
}

/// Lowers one plain traced program to values inside a block.
#[cfg(test)]
pub(crate) fn lower_plain_program_outputs<'b, 'c: 'b, 't: 'c, O, V, Input, Output>(
    program: &Program<V, O, Input, Output>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableOperation<Type = ArrayType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let input_values = (0..program.input_ids().len())
        .map(|index| block.argument(index).expect("body block arguments should exist").as_ref())
        .collect::<Vec<_>>();
    // Function-body-scoped per-class effect chains are created lazily and dropped at the end of the function body.
    let mut effect_tokens = EffectTokens::default();
    replay_program_into_block(
        program,
        input_values,
        block,
        context,
        location,
        |_, value, block, context, location| lower_literal_value(value, block, context, location),
        |instruction, inputs, block, context, location| {
            let input_types = instruction
                .inputs()
                .iter()
                .map(|input| program.atoms()[input.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let output_types = instruction
                .outputs()
                .iter()
                .map(|output| program.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            if !instruction.regions().is_empty() {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "plain-program lowering does not support attached regions for `{}`; use the production \
                         composite XLA lowerer",
                        instruction.operation().name(),
                    ),
                });
            }
            let location = collective_state.instruction_location(context, instruction.provenance(), location);
            let mut lowerer: XlaLowerer<'_, '_, '_, XlaOperation> = XlaLowerer::new(*block, context, location)
                .with_effect_tokens(std::mem::take(&mut effect_tokens))
                .with_collective_state(collective_state.clone());
            let outputs =
                instruction
                    .operation()
                    .lower(&mut lowerer, inputs, input_types.as_slice(), output_types.as_slice())?;
            if lowerer.effect_tokens.ordered_assertion.is_some() {
                collective_state.has_assertions.set(true);
            }
            effect_tokens = lowerer.effect_tokens;
            Ok(outputs)
        },
    )
}

/// Lowers one traced program with explicit array inputs and the enclosing effect chains.
pub(crate) fn lower_program_outputs_with_inputs<
    'b,
    'c: 'b,
    't: 'c,
    ProgramInput,
    ProgramOutput,
    O: LowerableOperationFamily,
>(
    program: &XlaProgram<ProgramInput, ProgramOutput, O>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    if input_values.len() != program.input_ids().len() {
        return Err(ProgramError::InvalidInputCount {
            expected: program.input_ids().len(),
            actual: input_values.len(),
        }
        .into());
    }
    // Mirror table of every lowered atom value. Shard-map operations look up captured global primals by `AtomId`,
    // so we keep a parallel table alongside `Program::interpret_with`'s use-count-tracked one. `ValueRef` is
    // `Copy`, so this mirror is cheap.
    let mut atom_values = vec![None; program.atoms().len()];
    for (atom_id, value) in program.input_ids().iter().copied().zip(input_values.iter().copied()) {
        atom_values[atom_id.index()] = Some(value);
    }
    let atom_values = std::cell::RefCell::new(atom_values);
    replay_program_into_block(
        program,
        input_values.to_vec(),
        block,
        context,
        location,
        |atom_id, value, block, context, location| {
            let lowered = lower_constant(atom_id, value, captured_values, block, context, location)?;
            atom_values.borrow_mut()[atom_id.index()] = Some(lowered);
            Ok(lowered)
        },
        |instruction, inputs, block, context, location| {
            let mut table = atom_values.borrow_mut();
            let lowered_outputs = lower_instruction(
                program,
                instruction,
                table.as_slice(),
                inputs,
                block,
                context,
                location,
                captured_values,
                nested_functions,
                collective_state,
                effect_tokens,
            )?;
            for (output_atom, lowered_output) in
                instruction.outputs().iter().copied().zip(lowered_outputs.iter().copied())
            {
                table[output_atom.index()] = Some(lowered_output);
            }
            Ok(lowered_outputs)
        },
    )
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{AddOperation, ArrayType, DataType, MulOperation, Placeholder};
    use ryft_mlir::dialects::func;
    use ryft_mlir::{Block, Location, Operation as MlirOperation, Region, TypeAndAttributes, Value as MlirValue};

    use crate::experimental::lowering::tests::{XlaProgramBuilder, unproject_plain_program};
    use crate::experimental::lowering::types::lower_tensor_type;
    use crate::experimental::ops::XlaArrayConstant;

    use super::*;

    /// Inlines `program` into the body of a `main` function behind an optimization barrier on all of its inputs,
    /// returns the inlined outputs from that function, and renders the verified module. When `ordered_io` is set, the
    /// enclosing scope already holds an ordered-I/O token when the program is inlined, so the barrier must thread it.
    fn lower_nested_program_inline_behind_input_barrier(
        program: &FlatXlaProgram,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
        ordered_io: bool,
    ) -> String {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_types = |types: &[ArrayType]| {
            types
                .iter()
                .map(|r#type| lower_tensor_type(r#type, &context, location).unwrap())
                .collect::<Vec<_>>()
        };
        let input_tensor_types = tensor_types(input_types);
        let output_tensor_types = tensor_types(output_types);
        let function_block = context.block(
            input_tensor_types.iter().map(|tensor_type| (*tensor_type, location)).collect::<Vec<_>>().as_slice(),
        );
        {
            let mut block = function_block.as_ref();
            let inputs =
                (0..input_types.len()).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
            let mut effect_tokens = EffectTokens::default();
            if ordered_io {
                let token =
                    block.append_operation(stable_hlo::after_all::<ValueRef, _>(&[], location).unwrap()).unwrap();
                effect_tokens.ordered_io = Some(token.result(0).unwrap().as_ref());
            }
            let outputs = lower_nested_program_inline(
                program,
                inputs.as_slice(),
                &mut block,
                &context,
                location.as_ref(),
                &[],
                &RematerializationOptimizationBarrier::All,
                None,
                &CollectiveLoweringState::new(),
                &mut effect_tokens,
            )
            .unwrap();
            block.append_operation(func::r#return(outputs.as_slice(), location).unwrap()).unwrap();
        }
        let mut function_region = context.region();
        function_region.append_block(function_block).unwrap();
        let function = func::func(
            "main",
            func::FuncAttributes {
                arguments: input_tensor_types
                    .iter()
                    .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                    .collect(),
                results: output_tensor_types
                    .iter()
                    .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                    .collect(),
                ..Default::default()
            },
            function_region,
            location,
        )
        .unwrap();
        module.body().unwrap().append_operation(function).unwrap();
        assert!(module.verify().unwrap());
        module.to_string()
    }

    #[test]
    fn test_lower_nested_program_inline_input_optimization_barrier() {
        // A pure body consumes the barrier's results instead of the enclosing scope's values.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(array_type.clone());
        let right = builder.add_input(array_type.clone());
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let program = unproject_plain_program(program);
        assert_eq!(
            lower_nested_program_inline_behind_input_barrier(
                &program,
                &[array_type.clone(), array_type.clone()],
                &[array_type],
                false,
            ),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
                    %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<2xf32>, tensor<2xf32>
                    %1 = stablehlo.multiply %0#0, %0#1 : tensor<2xf32>
                    return %1 : tensor<2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_nested_program_inline_input_optimization_barrier_orders_effects() {
        use ryft_core::PrintOperation;

        // The enclosing ordered-I/O token goes through the same barrier as the inputs, so the body's print consumes
        // the barrier's token result and is sequenced after the barrier.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("body"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        assert_eq!(
            lower_nested_program_inline_behind_input_barrier(&program, &[array_type.clone()], &[array_type], true),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = stablehlo.optimization_barrier %arg0, %0 : tensor<2xf32>, !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%1#0, %1#1) {api_version = 4 : i32, backend_config = {label = "body"}, has_side_effect = true} : (tensor<2xf32>, !stablehlo.token) -> !stablehlo.token
                    %3 = stablehlo.add %1#0, %1#0 : tensor<2xf32>
                    return %3 : tensor<2xf32>
                  }
                }
            "#},
        );
    }
}
