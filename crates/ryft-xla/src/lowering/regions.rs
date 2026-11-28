use std::rc::Rc;

use ryft_core::{Instruction, RegionRef, Typed};
use ryft_mlir::{Block, BlockRef, LocationRef, ValueRef};

use crate::lowering::emissions::lowers_without_regions;
use crate::lowering::states::ProgramLoweringContext;
use crate::lowering::{EffectTokens, LowerableOperationFamily, Lowerer, lower_instruction_operation};
use crate::operations::call::CallFunctionMap;
use crate::{Constant, Error};

// TODO(eaplatanios): Review this module.

/// Lowers one instruction from `region` through its [`LowerableOperation`](crate::LowerableOperation) implementation
/// and returns its output values. The ordered effect tokens of the enclosing scope flow through `effect_tokens`, and
/// an instruction that uses the assertion token marks the module as requiring the runtime assertion handler.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_instruction<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    region: RegionRef<'_, Constant, O>,
    instruction: &Instruction<O>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<CallFunctionMap<O>>>,
    lowering_context: &ProgramLoweringContext,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let context = block.context();
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
    // Every StableHLO operation generated from this one Ryft instruction inherits the instruction's provenance
    // through the lowerer's shared location, with the ambient location as its base.
    let location = lowering_context.instruction_location(context, instruction.provenance(), location);
    let regions = if lowers_without_regions(
        instruction.operation(),
        &input_types,
        &output_types,
        nested_functions,
        lowering_context.target_platform(),
    ) {
        Vec::new()
    } else {
        instruction
            .regions()
            .iter()
            .map(|attached| RegionRef::new(region.arena(), *attached).map(RegionRef::to_program))
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut lowerer = Lowerer::new(block, context, location, lowering_context.clone())
        .with_regions(regions)
        .with_nested_functions(nested_functions.cloned())
        .with_captured_values(captured_values)
        .with_effect_tokens(*effect_tokens);
    let outputs = lower_instruction_operation(
        instruction.operation(),
        &mut lowerer,
        input_values,
        input_types.as_slice(),
        output_types.as_slice(),
    )?;
    if lowerer.effect_tokens.ordered_assertion.is_some() {
        lowering_context.record_assertion();
    }
    *effect_tokens = lowerer.effect_tokens;
    Ok(outputs)
}
