use std::rc::Rc;

use ryft_core::{
    ArrayIrType, ArrayType, CONDITION_OPERATION_NAME, ConditionOperation, CustomCallOperation, DataType, Dimension,
    EffectClass, EffectClasses, GuardOperation, IotaOperation, MeshAxisType, OneOperation, Parameterized, ProgramError,
    RematerializationOptimizationBarrier, RngBitGeneratorOperation, SCAN_OPERATION_NAME, ScanOperation, Shape, Typed,
    WHILE_OPERATION_NAME, WhileOperation, ZeroOperation,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Type, TypeRef,
    Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::{
    lower_physical_storage, lower_tensor_type, lower_unplaced_constant_output, physical_bound_type,
    reshape_dimension_i64,
};
use crate::experimental::lowering::{
    CollectiveLoweringState, EffectTokens, current_or_new_token, token_threaded_effects,
};
use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaProgram};
use crate::operations::assertions::{lower_guard, validates_output_extents};
use crate::operations::jit_call::JitCallFunctionMap;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{LowerableOperationFamily, LoweringError};

/// StableHLO lowering of `condition`.
pub(crate) mod condition;
/// StableHLO lowering of `scan`.
pub(crate) mod scan;
/// StableHLO lowering of `select`.
pub(crate) mod select;
/// StableHLO lowering of `while`.
pub(crate) mod r#while;

/// Lowers the input relationship guard of a region invoked with `values` of the `actual_types`, before any conversion
/// to the region's declared input types. Lowering retypes each region input independently, so a relationship between
/// inputs that their actual types leave unproven (e.g., a `dimension<n>` input that must equal the extent of an exact
/// `f32[3]` input entering `f32[n]`) is checked by a [`GuardOperation`] before the region runs, exactly as eager
/// validation of region inputs would reject it. Static extents are checked as constants and dynamic ones are read from
/// the values, so the guard needs no conversion and can precede every path that invokes the region (including paths
/// that never run the region). Inputs whose actual types prove every relationship need no guard.
///
/// # Parameters
///
///   - `boundary`: Name of the operation whose region is entered.
///   - `declared_types`: Declared input types of the region.
///   - `actual_types`: Types of the values entering the region.
///   - `values`: Lowered values entering the region, of the `actual_types`.
pub(crate) fn lower_region_guard<'b, 'c: 'b, 't: 'c>(
    boundary: &str,
    declared_types: &[ArrayIrType],
    actual_types: &[ArrayIrType],
    values: &[ValueRef<'b, 'c, 't>],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let Some((guard, members)) =
        GuardOperation::for_entry(boundary, declared_types, actual_types).map_err(ProgramError::from)?
    else {
        return Ok(());
    };
    let member_values = members.iter().map(|&index| values[index]).collect::<Vec<_>>();
    let member_types = members.iter().map(|&index| actual_types[index].clone()).collect::<Vec<_>>();
    lower_guard(
        boundary,
        guard.declared_types(),
        &member_values,
        &member_types,
        effect_tokens,
        block,
        context,
        location,
    )
}

/// Returns the [`EffectClasses`] whose tokens lowering `program` threads: its source effect classes, plus
/// [`EffectClass::OrderedAssertion`] when lowering synthesizes an assertion anywhere in it, at any depth. Lowering
/// synthesizes the input relationship guard of a nested `condition`, `while`, or `scan` whose inputs leave
/// relationships of its region inputs unproven (refer to [`lower_region_guard`]), and the validation of the extent
/// inputs that a dynamic array constructor (`zero`, `one`, or `iota`), a random bit generator, or a custom call
/// consumes for output axes that type inference refined to static extents (refer to `validates_output_extents`). A
/// region that contains such an assertion must thread its tokens out (refer to `token_threaded_effects`) even when its
/// source program is pure, so that later assertions and I/O follow it: a region input guard joins its assertion token
/// into the ordered I/O chain, and every assertion advances the ordered-assertion chain.
pub(crate) fn lowered_effect_classes<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &XlaProgram<ProgramInput, ProgramOutput, O>,
) -> Result<EffectClasses, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    let classes = program.effects().classes();
    if classes.contains(EffectClass::OrderedAssertion) {
        return Ok(classes);
    }
    for region in program.regions().iter() {
        for instruction in region.instructions() {
            let input_types = instruction
                .inputs()
                .iter()
                .map(|id| region.atoms()[id.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let first_region_input_types = || -> Result<Vec<ArrayIrType>, LoweringError> {
                let id = instruction.regions().first().copied().ok_or_else(|| {
                    ProgramError::MalformedProgram("control flow without an attached region".to_string())
                })?;
                Ok(program.region(id)?.input_types())
            };
            let operation = instruction.operation();
            let entry = if operation.projected_payload::<ConditionOperation<ArrayIrType>>().is_some() {
                Some((CONDITION_OPERATION_NAME, first_region_input_types()?, input_types.get(1..).unwrap_or_default()))
            } else if operation.projected_payload::<WhileOperation<ArrayIrType>>().is_some() {
                // Both loop regions share the state types, so the condition's input types are the entry types.
                Some((WHILE_OPERATION_NAME, first_region_input_types()?, input_types.as_slice()))
            } else if let Some(scan) = operation.projected_payload::<ScanOperation<ArrayIrType>>() {
                let entry_types =
                    scan.declared_input_types(&first_region_input_types()?).map_err(ProgramError::from)?;
                Some((SCAN_OPERATION_NAME, entry_types, input_types.as_slice()))
            } else {
                None
            };
            if let Some((boundary, declared, actual)) = entry
                && declared.len() == actual.len()
                && GuardOperation::for_entry(boundary, &declared, actual).map_err(ProgramError::from)?.is_some()
            {
                return Ok(classes.union(EffectClasses::single(EffectClass::OrderedAssertion)));
            }
            let output_types = || {
                instruction
                    .outputs()
                    .iter()
                    .map(|id| region.atoms()[id.index()].r#type().into_owned())
                    .collect::<Vec<_>>()
            };
            let extent_signature = if let Some(zero) = operation.projected_payload::<ZeroOperation<ArrayType>>() {
                Some((std::slice::from_ref(zero.r#type()), output_types()))
            } else if let Some(one) = operation.projected_payload::<OneOperation<ArrayType>>() {
                Some((std::slice::from_ref(one.r#type()), output_types()))
            } else if let Some(iota) = operation.projected_payload::<IotaOperation<ArrayType>>() {
                Some((std::slice::from_ref(iota.r#type()), output_types()))
            } else if let Some(generator) = operation.projected_payload::<RngBitGeneratorOperation<ArrayIrType>>() {
                // The generator's first output is its state, and only its bits have the declared output type.
                Some((std::slice::from_ref(generator.output_type()), output_types().into_iter().skip(1).collect()))
            } else {
                operation
                    .projected_payload::<CustomCallOperation>()
                    .map(|custom_call| (custom_call.output_types(), output_types()))
            };
            if let Some((declared, outputs)) = extent_signature
                && validates_output_extents(declared, &input_types, &outputs)?
            {
                return Ok(classes.union(EffectClasses::single(EffectClass::OrderedAssertion)));
            }
        }
    }
    Ok(classes)
}

/// Converts the lowered value of one control-flow input to the lowered type of the region input that it feeds, which
/// the input type refines (refer to the `while`, `scan`, and `condition` type inference rules in `ryft-core`), or more
/// generally to a type whose dynamic axes take either a static extent of the input or the runtime size of a dynamic
/// axis of the input (e.g., the outputs of a region input guard or a `cast`). Regions receive every input at its
/// declared type, and refinements that only add metadata (e.g., shardings or layouts) lower to the same tensor type, so
/// those inputs are returned unchanged. A static extent that refines a bounded dynamic dimension, and a dynamic
/// dimension whose physical bound differs from the declared one, lower to different tensor types. Such an input is
/// exposed as its full physical buffer, padded up (or sliced down) to the physical bound of the declared type along
/// every such axis, and then given the runtime size of every dynamic dimension of the declared type with
/// `stablehlo.set_dimension_size`: the static extent for a refined axis, and the input's own runtime size for an axis
/// that stays dynamic.
pub(crate) fn lower_refined_region_input<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    input_type: &ArrayIrType,
    region_input_type: &ArrayIrType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let (ArrayIrType::Array(input_type), ArrayIrType::Array(declared_type)) = (input_type, region_input_type) else {
        return Ok(value);
    };
    let dimension_pairs = declared_type.shape().dimensions().iter().zip(input_type.shape().dimensions());
    if !dimension_pairs.clone().any(|pair| match pair {
        (Dimension::Dynamic(_), Dimension::Static(_)) => true,
        (Dimension::Dynamic(declared), Dimension::Dynamic(input)) => {
            declared.bounds().upper() != input.bounds().upper()
        }
        _ => false,
    }) {
        return Ok(value);
    }

    // Read the runtime sizes of the axes that stay dynamic before exposing the physical buffer, which drops them.
    let mut runtime_sizes = Vec::with_capacity(declared_type.rank());
    for (axis, dimensions) in dimension_pairs.enumerate() {
        runtime_sizes.push(match dimensions {
            (Dimension::Static(_), _) => None,
            (Dimension::Dynamic(_), Dimension::Static(extent)) => Some(
                lower_unplaced_constant_output(
                    &[ArrayType::scalar(DataType::I32)],
                    reshape_dimension_i64(*extent)?,
                    block,
                    context,
                    location,
                )?
                .remove(0),
            ),
            (Dimension::Dynamic(_), Dimension::Dynamic(_)) => {
                let size = block.append_operation(stable_hlo::get_dimension_size(value, axis, location)?)?;
                Some(size.result(0).unwrap().as_ref())
            }
        });
    }

    let physical_value = lower_physical_storage(value, input_type, block, context, location)?;
    let physical_input_type = physical_bound_type(input_type)?;
    let physical_type = physical_bound_type(declared_type)?;
    let rank = declared_type.rank();
    // Negative padding slices a physical buffer down to a smaller bound, which keeps every element within the runtime
    // size, because that size lies within both bounds.
    let mut high_padding = vec![0; rank];
    for axis in 0..rank {
        let bound = physical_type.shape().dimensions()[axis].value().unwrap();
        let extent = physical_input_type.shape().dimensions()[axis].value().unwrap();
        high_padding[axis] = reshape_dimension_i64(bound)? - reshape_dimension_i64(extent)?;
    }
    let padding_value =
        lower_unplaced_constant_output(&[ArrayType::scalar(declared_type.data_type())], 0, block, context, location)?
            .remove(0);
    let padded = block.append_operation(stable_hlo::pad(
        physical_value,
        padding_value,
        &vec![0; rank],
        &high_padding,
        &vec![0; rank],
        location,
    )?)?;
    let mut value = padded.result(0).unwrap().as_ref();
    let mut dimensions = physical_type.shape().dimensions().to_vec();
    for (axis, size) in runtime_sizes.into_iter().enumerate() {
        let Some(size) = size else { continue };
        dimensions[axis] = declared_type.shape().dimensions()[axis].clone();
        let refined_type = declared_type.clone().with_shape(Shape::new(dimensions.clone()));
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            value,
            size,
            lower_tensor_type(&refined_type, context, location)?,
            axis,
            location,
        )?)?;
        value = refined.result(0).unwrap().as_ref();
    }
    Ok(value)
}

/// Converts the lowered values of the results of one control-flow instruction from the `region_output_types` that its
/// lowering produced them at to the instruction's `output_types`, which refine them (refer to `refine_output_types` in
/// `ryft-core`). This is the inverse of [`lower_refined_region_input`]: refinements only replace dynamic dimensions
/// with static extents, so a result is exposed as its full physical buffer, sliced to the static extent of every
/// refined axis, and then given back the runtime size of every axis that stays dynamic with
/// `stablehlo.set_dimension_size`. Results whose types the instruction does not refine are returned unchanged.
pub(crate) fn lower_refined_region_outputs<'b, 'c: 'b, 't: 'c>(
    values: Vec<ValueRef<'b, 'c, 't>>,
    region_output_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if values.len() != region_output_types.len() || values.len() != output_types.len() {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "control-flow lowering produced {} results for {} region output types and {} output types",
                values.len(),
                region_output_types.len(),
                output_types.len(),
            ),
        });
    }
    let mut converted = Vec::with_capacity(values.len());
    for ((value, region_output_type), output_type) in values.into_iter().zip(region_output_types).zip(output_types) {
        let (ArrayIrType::Array(declared_type), ArrayIrType::Array(output_type)) = (region_output_type, output_type)
        else {
            converted.push(value);
            continue;
        };
        let dimension_pairs = declared_type.shape().dimensions().iter().zip(output_type.shape().dimensions());
        if !dimension_pairs.clone().any(|pair| matches!(pair, (Dimension::Dynamic(_), Dimension::Static(_)))) {
            converted.push(value);
            continue;
        }

        // Read the runtime sizes of the axes that stay dynamic before exposing the physical buffer, which drops them.
        let mut runtime_sizes = Vec::with_capacity(declared_type.rank());
        for (axis, dimensions) in dimension_pairs.enumerate() {
            runtime_sizes.push(match dimensions {
                (Dimension::Dynamic(_), Dimension::Dynamic(_)) => {
                    let size = block.append_operation(stable_hlo::get_dimension_size(value, axis, location)?)?;
                    Some(size.result(0).unwrap().as_ref())
                }
                _ => None,
            });
        }

        let physical_value = lower_physical_storage(value, declared_type, block, context, location)?;
        let physical_type = physical_bound_type(declared_type)?;
        let limits = physical_type
            .shape()
            .dimensions()
            .iter()
            .zip(output_type.shape().dimensions())
            .map(|(physical, output)| output.value().or(physical.value()).unwrap())
            .collect::<Vec<_>>();
        let rank = limits.len();
        let sliced = block.append_operation(stable_hlo::slice(
            physical_value,
            &vec![0; rank],
            &limits,
            &vec![1; rank],
            location,
        )?)?;
        let mut value = sliced.result(0).unwrap().as_ref();
        let mut dimensions = limits.iter().copied().map(Dimension::Static).collect::<Vec<_>>();
        for (axis, size) in runtime_sizes.into_iter().enumerate() {
            let Some(size) = size else { continue };
            dimensions[axis] = output_type.shape().dimensions()[axis].clone();
            let refined_type = output_type.clone().with_shape(Shape::new(dimensions.clone()));
            let refined = block.append_operation(stable_hlo::set_dimension_size(
                value,
                size,
                lower_tensor_type(&refined_type, context, location)?,
                axis,
                location,
            )?)?;
            value = refined.result(0).unwrap().as_ref();
        }
        converted.push(value);
    }
    Ok(converted)
}

/// Lowers one nested control-flow branch program into a fresh single-block region.
///
/// `entry_effect_tokens` are the enclosing scope's active per-class tokens, referenced inside the region through
/// StableHLO's implicit region capture (the same mechanism that feeds `input_values` into `stablehlo.if` branches).
/// Ordinary results are converted to `output_types` before the region returns one trailing token for each class in
/// `threaded_effects`, in canonical effect order. A branch that is pure for one of those classes returns that class's
/// incoming token unchanged. `threaded_effects` must cover every ordered class whose token the lowering of `program`
/// advances (refer to [`lowered_effect_classes`]): the region cannot return the token of any other class, so an
/// advanced token of such a class is reported as a [`ProgramError::MalformedProgram`] instead of being dropped, which
/// would leave later effects of that class unordered with respect to the effects of the region.
pub(crate) fn lower_control_flow_region<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    program: &FlatXlaProgram<O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayIrType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    entry_effect_tokens: EffectTokens<'b, 'c, 't>,
    threaded_effects: EffectClasses,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let mut region = context.region();
    let block = context.block_with_no_arguments();
    {
        let mut block_ref = block.as_ref();
        let mut region_effect_tokens = entry_effect_tokens;
        let mut outputs = lower_nested_program_inline(
            program,
            input_values,
            &mut block_ref,
            context,
            location,
            captured_values,
            &RematerializationOptimizationBarrier::None,
            nested_functions,
            collective_state,
            &mut region_effect_tokens,
        )?;
        validate_unthreaded_effect_tokens(entry_effect_tokens, region_effect_tokens, threaded_effects)?;
        // Normalize ordinary results before appending tokens so both conditional branches return the instruction's
        // common refined shapes even when their declared dynamic and static result types differ.
        outputs = lower_refined_region_outputs(
            outputs,
            &program.output_types(),
            output_types,
            &mut block_ref,
            context,
            location,
        )?;
        outputs.extend(threaded_effect_token_values(region_effect_tokens, threaded_effects));
        block_ref.append_operation(stable_hlo::r#return(outputs.as_slice(), location)?)?;
    }
    region.append_block(block)?;
    Ok(region)
}

/// Validates that lowering a control-flow region whose tokens were `entry_effect_tokens` when it started and are
/// `region_effect_tokens` when it ends advanced no token of an ordered class outside `threaded_effects`. A region
/// returns only the tokens of its threaded classes (refer to `token_threaded_effects`), so the advanced token of any
/// other class would be dropped, leaving later effects of that class unordered with respect to the effects of the
/// region. The threaded classes come from [`lowered_effect_classes`], which must predict every ordered class whose
/// token lowering advances, so a violation is an internal error. A region that returns no token at all (e.g., the
/// condition region of a `stablehlo.while`) validates with [`EffectClasses::NONE`].
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] naming the first advanced token of an unthreaded class.
pub(crate) fn validate_unthreaded_effect_tokens<'b, 'c: 'b, 't: 'c>(
    entry_effect_tokens: EffectTokens<'b, 'c, 't>,
    region_effect_tokens: EffectTokens<'b, 'c, 't>,
    threaded_effects: EffectClasses,
) -> Result<(), ProgramError> {
    for effect in [EffectClass::OrderedAssertion, EffectClass::OrderedIo] {
        if !token_threaded_effects(threaded_effects).any(|threaded| threaded == effect)
            && region_effect_tokens.get(effect) != entry_effect_tokens.get(effect)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "control-flow region lowering advanced the `{effect}` token, which the region does not thread",
            )));
        }
    }
    Ok(())
}

/// Appends the current token of each class in `threaded_effects` to the initial `state_values` of a
/// `stablehlo.while` and one token type per class to its `state_types`, in canonical effect order (refer to
/// `token_threaded_effects`). A class without a current token gets a fresh one. The loop carries these tokens as its
/// trailing state elements, so its regions read them with [`set_threaded_effect_tokens`] and its results become the
/// current tokens of the enclosing scope the same way.
pub(crate) fn append_loop_effect_tokens<'b, 'c: 'b, 't: 'c>(
    threaded_effects: EffectClasses,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    state_values: &mut Vec<ValueRef<'b, 'c, 't>>,
    state_types: &mut Vec<TypeRef<'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    for effect in token_threaded_effects(threaded_effects) {
        state_values.push(current_or_new_token(effect, effect_tokens, block, location)?);
        state_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    Ok(())
}

/// Sets the token of each class in `threaded_effects` to `token(offset)`, where `offset` is the position of the class
/// in canonical effect order (refer to `token_threaded_effects`), and keeps the tokens of all other classes.
/// Control-flow regions and operations carry the tokens of their threaded classes as trailing block arguments or
/// results in that order, so `token` reads them from the block arguments of a region that it enters, or from the
/// results of the operation that it leaves.
pub(crate) fn set_threaded_effect_tokens<'b, 'c: 'b, 't: 'c>(
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    threaded_effects: EffectClasses,
    mut token: impl FnMut(usize) -> ValueRef<'b, 'c, 't>,
) {
    for (offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
        effect_tokens.set(effect, token(offset));
    }
}

/// Returns the current token of each class in `threaded_effects`, in canonical effect order (refer to
/// `token_threaded_effects`), which a control-flow region returns as its trailing results. Every threaded class has a
/// token in a region whose tokens entered through [`set_threaded_effect_tokens`] or that started from tokens returned
/// by [`append_loop_effect_tokens`].
pub(crate) fn threaded_effect_token_values<'b, 'c: 'b, 't: 'c>(
    effect_tokens: EffectTokens<'b, 'c, 't>,
    threaded_effects: EffectClasses,
) -> Vec<ValueRef<'b, 'c, 't>> {
    token_threaded_effects(threaded_effects).map(|effect| effect_tokens.get(effect).unwrap()).collect()
}

/// Constrains each of the lowered `outputs` of a control-flow operation whose array type in `output_types` carries a
/// sharding over a mesh with at least one `Explicit` axis to that sharding with `sdy.sharding_constraint`, and returns
/// the constrained values. This includes results that such a sharding replicates or shards only over the `Auto` axes
/// of that mesh. Nothing inside a region needs to pin the placement of a result over a mesh with explicit axes (e.g., a
/// region may return a replicated constant), so propagation could otherwise assign the result a placement other than
/// the one that its type states. Results without a sharding, results whose mesh has no explicit axis (e.g., only `Auto`
/// or `Manual` axes), references, and first-class dimensions remain unconstrained. Mesh axes in `bound_manual_axes`,
/// which enclosing manual regions bind, are excluded from the constraints (refer to `lower_sharding_constraint`).
pub(crate) fn lower_explicit_sharding_constraints<'b, 'c: 'b, 't: 'c>(
    outputs: Vec<ValueRef<'b, 'c, 't>>,
    output_types: &[ArrayIrType],
    bound_manual_axes: &[String],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let mut constrained = Vec::with_capacity(outputs.len());
    for (output, output_type) in outputs.into_iter().zip(output_types) {
        let explicit_sharding = match output_type {
            ArrayIrType::Array(output_type) => output_type
                .sharding()
                .filter(|sharding| sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit)),
            ArrayIrType::Dimension(_) | ArrayIrType::Reference(_) => None,
        };
        constrained.push(match explicit_sharding {
            Some(sharding) => {
                lower_sharding_constraint(&[output], sharding, bound_manual_axes, block, location)?.remove(0)
            }
            None => output,
        });
    }
    Ok(constrained)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, AtomId, CompareOperation, ComparisonDirection, ConditionOperation, Device,
        DeviceMesh, LogicalMesh, MeshAxis, Placeholder, PrintOperation, ScanOperation, WhileOperation,
        ZeroLikeOperation,
    };
    use ryft_mlir::Location;
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

    use crate::experimental::lowering::tests::{test_spmd_compilation_options, test_vector_type};
    use crate::experimental::ops::{XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::operations::jit_call::JitCallOperation;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, XlaArray, XlaSession, to_mlir_module_for_program};

    use super::*;

    /// Returns the [`XlaOperation`] that compares two arrays in `direction`.
    pub(super) fn compare(direction: ComparisonDirection) -> XlaOperation {
        XlaOperation::Array(ArrayOperation::Compare(CompareOperation::<ArrayType>::new(direction)))
    }

    /// Stages whether each element of `value` is positive into `builder`. The mixed executable ABI of
    /// `execute_mixed_program` has no boolean inputs, so control flow that executes for both predicate values derives
    /// its predicate from a floating-point selector input.
    pub(super) fn positive(builder: &mut CompositeXlaProgramBuilder, value: AtomId) -> AtomId {
        let zero = builder
            .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![value], None)
            .unwrap()[0];
        builder
            .add_instruction(compare(ComparisonDirection::GreaterThan), Vec::new(), vec![value, zero], None)
            .unwrap()[0]
    }

    /// Lowers `program` with the boundary types of the mixed executable ABI of `execute_mixed_program`, in which
    /// first-class dimensions cross the boundary as `i64` scalars.
    pub(super) fn lowered_mixed_module(program: &FlatXlaProgram) -> String {
        let boundary_type = |r#type: &ArrayIrType| match r#type {
            ArrayIrType::Array(r#type) => r#type.clone(),
            ArrayIrType::Dimension(_) => ArrayType::scalar(DataType::I64),
            ArrayIrType::Reference(_) => panic!("mixed execution fixtures have no reference values"),
        };
        let input_types = program.input_types().iter().map(boundary_type).collect::<Vec<_>>();
        let output_types = program.output_types().iter().map(boundary_type).collect::<Vec<_>>();
        to_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None).unwrap()
    }

    /// Returns a two-device CPU client and a device mesh over its devices with the single mesh axis `x` of `axis_type`.
    pub(super) fn two_device_mesh(axis_type: MeshAxisType) -> (Client<'static>, DeviceMesh) {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        (client, mesh)
    }

    /// Compiles the SPMD `module` for the devices of a [`two_device_mesh`], executes it on global `f32` inputs of the
    /// given types and values, and returns the `f32` shard of each output on each device.
    pub(super) fn execute_on_two_devices(
        client: &Client<'static>,
        mesh: &DeviceMesh,
        module: &str,
        inputs: &[(&ArrayType, &[f32])],
    ) -> Vec<Vec<Vec<f32>>> {
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.as_bytes().to_vec() }, &test_spmd_compilation_options(2))
            .unwrap();
        let domain = XlaSession::new(client).domain();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let inputs = inputs
            .iter()
            .map(|(r#type, values)| {
                XlaArray::from_host_buffer(&domain, (*r#type).clone(), mesh.clone(), values_to_bytes(values)).unwrap()
            })
            .collect();
        let arguments = XlaArray::into_execute_arguments(inputs, device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        outputs
            .iter()
            .map(|output| {
                output
                    .outputs
                    .iter()
                    .map(|buffer| {
                        let bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
                        values_from_bytes::<f32>(bytes.as_slice())
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_lower_control_flow_region_rejects_unthreaded_token_advances() {
        // A print advances the ordered I/O token. A region whose threaded classes include ordered I/O returns the
        // advanced token, whereas a region that threads no class could not return it, so lowering it would leave later
        // I/O unordered with respect to the print.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block(&[(lower_tensor_type(&scalar_type, &context, location).unwrap(), location)]);
        let input = block.as_ref().argument(0).unwrap().as_ref();
        let lower = |threaded_effects| {
            lower_control_flow_region(
                &program,
                &[input],
                &[scalar_type.clone().into()],
                &context,
                location,
                &[],
                None,
                &CollectiveLoweringState::new(),
                EffectTokens::default(),
                threaded_effects,
            )
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
    fn test_validate_unthreaded_effect_tokens() {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let token_type = context.stable_hlo_token_type().unwrap();
        let block = context.block(&[(token_type, location), (token_type, location)]);
        let entry_token = block.as_ref().argument(0).unwrap().as_ref();
        let advanced_token = block.as_ref().argument(1).unwrap().as_ref();
        let mut entry_effect_tokens = EffectTokens::default();
        entry_effect_tokens.set(EffectClass::OrderedIo, entry_token);
        let ordered_io = EffectClasses::single(EffectClass::OrderedIo);

        // A region that leaves every token unchanged threads nothing it would have to return.
        let threaded_effects = EffectClasses::NONE;
        assert_eq!(
            validate_unthreaded_effect_tokens(entry_effect_tokens, entry_effect_tokens, threaded_effects),
            Ok(()),
        );

        // A region may advance the token of a class that it threads, and threading assertions threads I/O too.
        let mut region_effect_tokens = entry_effect_tokens;
        region_effect_tokens.set(EffectClass::OrderedIo, advanced_token);
        assert_eq!(validate_unthreaded_effect_tokens(entry_effect_tokens, region_effect_tokens, ordered_io), Ok(()));
        let threaded_effects = EffectClasses::single(EffectClass::OrderedAssertion);
        assert_eq!(
            validate_unthreaded_effect_tokens(entry_effect_tokens, region_effect_tokens, threaded_effects),
            Ok(()),
        );

        // The advanced token of a class that the region does not thread would be dropped, and so would a token that the
        // region creates for a class that had none when it started.
        assert!(matches!(
            validate_unthreaded_effect_tokens(entry_effect_tokens, region_effect_tokens, EffectClasses::NONE),
            Err(ProgramError::MalformedProgram(message))
                if message
                    == "control-flow region lowering advanced the `ordered_io` token, which the region does not thread",
        ));
        let mut region_effect_tokens = entry_effect_tokens;
        region_effect_tokens.set(EffectClass::OrderedAssertion, advanced_token);
        assert!(matches!(
            validate_unthreaded_effect_tokens(entry_effect_tokens, region_effect_tokens, ordered_io),
            Err(ProgramError::MalformedProgram(message))
                if message
                    == "control-flow region lowering advanced the `ordered_assertion` token, which the region does \
                        not thread",
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_differentiated_composite_control_flow() {
        let scalar_boolean = ArrayType::scalar(DataType::Boolean);
        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let vector_f32 = test_vector_type(2);

        let branch = |double: bool| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_f32.clone().into());
            let output = if double {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            } else {
                input
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut scan_body_builder = CompositeXlaProgramBuilder::new();
        scan_body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = scan_body_builder.add_input(scalar_f32.clone().into());
        let item = scan_body_builder.add_input(scalar_f32.clone().into());
        let next_carry =
            scan_body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0];
        let scan_body = scan_body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![next_carry, next_carry],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut while_condition_builder = CompositeXlaProgramBuilder::new();
        let state = while_condition_builder.add_input(scalar_f32.clone().into());
        let while_predicate = while_condition_builder
            .add_instruction(compare(ComparisonDirection::LessThan), Vec::new(), vec![state, state], None)
            .unwrap()[0];
        let while_condition = while_condition_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![while_predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut while_body_builder = CompositeXlaProgramBuilder::new();
        let state = while_body_builder.add_input(scalar_f32.clone().into());
        let next_state = while_body_builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![state, state], None)
            .unwrap()[0];
        let while_body = while_body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next_state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(true).entry_region_ref());
        let false_region = builder.import_region(branch(false).entry_region_ref());
        let scan_region = builder.import_region(scan_body.entry_region_ref());
        let while_condition_region = builder.import_region(while_condition.entry_region_ref());
        let while_body_region = builder.import_region(while_body.entry_region_ref());
        let predicate = builder.add_input(scalar_boolean.clone().into());
        let input = builder.add_input(scalar_f32.clone().into());
        let items = builder.add_input(vector_f32.clone().into());
        let selected = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let scan_outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, 2)),
                vec![scan_region],
                vec![selected, items],
                None,
            )
            .unwrap()
            .to_vec();
        let final_state = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(2).unwrap()),
                vec![while_condition_region, while_body_region],
                vec![scan_outputs[0]],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![final_state, scan_outputs[1]],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Keep the complete mixed control-flow program behind one compiled-call boundary. Linearization must preserve
        // that boundary while deriving its primal and pullback callees, and lowering must still reach the nested
        // condition and loop operations.
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(program.entry_region_ref());
        let predicate = builder.add_input(scalar_boolean.clone().into());
        let input = builder.add_input(scalar_f32.clone().into());
        let items = builder.add_input(vector_f32.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![predicate, input, items],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let primal = linearization.primal();
        let array_types = |types: Vec<ArrayIrType>| {
            types.into_iter().map(|r#type| <&ArrayType>::try_from(&r#type).unwrap().clone()).collect::<Vec<_>>()
        };
        let primal_module = to_mlir_module_for_program(
            primal,
            &[],
            &array_types(primal.input_types()),
            &array_types(primal.output_types()),
            "main",
            None,
            None,
        )
        .unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        let pullback_module = to_mlir_module_for_program(
            &pullback,
            &[],
            &array_types(pullback.input_types()),
            &array_types(pullback.output_types()),
            "main",
            None,
            None,
        )
        .unwrap();
        // The primal returns the predicate of the condition and the predicate of each loop iteration as residuals, and
        // the pullback replays the loop and the condition backwards over them.
        assert_eq!(
            primal.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[2] .
                let %3:f32[], %4:f32[2], %5:bool[], %6:bool[2] = jit_call %0 %1 %2 [
                    callee={
                        lambda %0:bool[], %1:f32[], %2:f32[2] .
                        let %3:f32[] = condition %0 %1 [
                            true={
                                lambda %0:f32[] .
                                let %1:f32[] = add %0 %0
                                in (%1)
                            },
                            false={
                                lambda %0:f32[] .
                                in (%0)
                            },
                        ]
                            %4:f32[], %5:f32[2] = scan [carry_count=1, length=2, reverse=false] %3 %2 [
                                body={
                                    lambda %0:i64[], %1:f32[], %2:f32[] .
                                    let %3:f32[] = add %1 %2
                                    in (%3, %3)
                                },
                            ]
                            %6:i64[] = zero [type=i64[]]
                            %7:bool[2] = zero [type=bool[2]]
                            %8:f32[], %9:i64[], %10:bool[2] = while [iteration_bound=2] %4 %6 %7 [
                                condition={
                                    lambda %0:f32[], %1:i64[], %2:bool[2] .
                                    let %3:bool[] = compare [direction=LessThan] %0 %0
                                    in (%3)
                                },
                                body={
                                    lambda %0:f32[], %1:i64[], %2:bool[2] .
                                    let %3:f32[] = add %0 %0
                                        %4:i64[] = one [type=i64[]]
                                        %5:i64[] = add %1 %4
                                        %6:bool[] = one [type=bool[]]
                                        %7:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %6
                                        %8:bool[2] = dynamic_update_slice %2 %7 %1
                                    in (%3, %5, %8)
                                },
                            ]
                        in (%8, %5, %0, %10)
                    },
                ]
                in (%3, %4, %5, %6)"},
        );
        assert_eq!(
            primal_module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<2xf32>) -> (tensor<f32>, \
                tensor<2xf32>, tensor<i1>, tensor<2xi1>) {
                    %0 = \"stablehlo.if\"(%arg0) ({
                      %6 = stablehlo.add %arg1, %arg1 : tensor<f32>
                      stablehlo.return %6 : tensor<f32>
                    }, {
                      stablehlo.return %arg1 : tensor<f32>
                    }) : (tensor<i1>) -> tensor<f32>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                    %2:4 = stablehlo.while(%iterArg = %c, %iterArg_3 = %0, %iterArg_4 = %arg2, %iterArg_5 = %1) : \
                tensor<i64>, tensor<f32>, tensor<2xf32>, tensor<2xf32>
                    cond {
                      %c_6 = stablehlo.constant dense<2> : tensor<i64>
                      %6 = stablehlo.compare LT, %iterArg, %c_6, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %6 = stablehlo.dynamic_slice %iterArg_4, %iterArg, sizes = [1] : (tensor<2xf32>, tensor<i64>) -> \
                tensor<1xf32>
                      %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
                      %8 = stablehlo.add %iterArg_3, %7 : tensor<f32>
                      %9 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
                      %10 = stablehlo.dynamic_update_slice %iterArg_5, %9, %iterArg : (tensor<2xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<2xf32>
                      %c_6 = stablehlo.constant dense<1> : tensor<i64>
                      %11 = stablehlo.add %iterArg, %c_6 : tensor<i64>
                      stablehlo.return %11, %8, %iterArg_4, %10 : tensor<i64>, tensor<f32>, tensor<2xf32>, tensor<2xf32>
                    }
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %c_1 = stablehlo.constant dense<false> : tensor<i1>
                    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i1>) -> tensor<2xi1>
                    %c_2 = stablehlo.constant dense<0> : tensor<i64>
                    %4 = stablehlo.compare LT, %2#1, %2#1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                    %5:5 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %2#1, %iterArg_4 = %c_0, %iterArg_5 = %3, \
                %iterArg_6 = %4) : tensor<i64>, tensor<f32>, tensor<i64>, tensor<2xi1>, tensor<i1>
                    cond {
                      %c_7 = stablehlo.constant dense<2> : tensor<i64>
                      %6 = stablehlo.compare LT, %iterArg, %c_7, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %7 = stablehlo.and %iterArg_6, %6 : tensor<i1>
                      stablehlo.return %7 : tensor<i1>
                    } do {
                      %6 = stablehlo.add %iterArg_3, %iterArg_3 : tensor<f32>
                      %c_7 = stablehlo.constant dense<1> : tensor<i64>
                      %7 = stablehlo.add %iterArg_4, %c_7 : tensor<i64>
                      %c_8 = stablehlo.constant dense<true> : tensor<i1>
                      %8 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i1>) -> tensor<1xi1>
                      %c_9 = stablehlo.constant dense<0> : tensor<i64>
                      %c_10 = stablehlo.constant dense<2> : tensor<i64>
                      %9 = stablehlo.compare LT, %iterArg_4, %c_9, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %10 = stablehlo.add %iterArg_4, %c_10 : tensor<i64>
                      %11 = stablehlo.select %9, %10, %iterArg_4 : tensor<i1>, tensor<i64>
                      %12 = stablehlo.dynamic_update_slice %iterArg_5, %8, %11 : (tensor<2xi1>, tensor<1xi1>, \
                tensor<i64>) -> tensor<2xi1>
                      %c_11 = stablehlo.constant dense<1> : tensor<i64>
                      %13 = stablehlo.add %iterArg, %c_11 : tensor<i64>
                      %c_12 = stablehlo.constant dense<2> : tensor<i64>
                      %14 = stablehlo.compare LT, %13, %c_12, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      %15 = \"stablehlo.if\"(%14) ({
                        %16 = stablehlo.compare LT, %6, %6, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                        stablehlo.return %16 : tensor<i1>
                      }, {
                        stablehlo.return %iterArg_6 : tensor<i1>
                      }) : (tensor<i1>) -> tensor<i1>
                      stablehlo.return %13, %6, %7, %12, %15 : tensor<i64>, tensor<f32>, tensor<i64>, tensor<2xi1>, \
                tensor<i1>
                    }
                    return %5#1, %2#3, %arg0, %5#3 : tensor<f32>, tensor<2xf32>, tensor<i1>, tensor<2xi1>
                  }
                }
            "},
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[2], %2:bool[], %3:bool[2] .
                let %4:f32[], %5:f32[2] = jit_call %0 %1 %2 %3 [
                    callee={
                        lambda %0:f32[], %1:f32[2], %2:bool[], %3:bool[2] .
                        let %4:f32[] = scan [carry_count=1, length=2, reverse=true] %0 %3 [
                            body={
                                lambda %0:i64[], %1:f32[], %2:bool[] .
                                let %3:f32[] = condition %2 %1 [
                                    true={
                                        lambda %0:f32[] .
                                        let %1:f32[] = add %0 %0
                                        in (%1)
                                    },
                                    false={
                                        lambda %0:f32[] .
                                        in (%0)
                                    },
                                ]
                                in (%3)
                            },
                        ]
                            %5:f32[], %6:f32[2] = scan [carry_count=1, length=2, reverse=true] %4 %1 [
                                body={
                                    lambda %0:i64[], %1:f32[], %2:f32[] .
                                    let %3:f32[] = add %1 %2
                                    in (%3, %3)
                                },
                            ]
                            %7:f32[] = condition %2 %5 [
                                true={
                                    lambda %0:f32[] .
                                    let %1:f32[] = add %0 %0
                                    in (%1)
                                },
                                false={
                                    lambda %0:f32[] .
                                    in (%0)
                                },
                            ]
                        in (%7, %6)
                    },
                ]
                in (%4, %5)"},
        );
        assert_eq!(
            pullback_module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<2xf32>, %arg2: tensor<i1>, %arg3: tensor<2xi1>) -> \
                (tensor<f32>, tensor<2xf32>) {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg3) : tensor<i64>, \
                tensor<f32>, tensor<2xi1>
                    cond {
                      %c_3 = stablehlo.constant dense<2> : tensor<i64>
                      %4 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %4 : tensor<i1>
                    } do {
                      %c_3 = stablehlo.constant dense<2> : tensor<i64>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %4 = stablehlo.subtract %c_3, %c_4 : tensor<i64>
                      %5 = stablehlo.subtract %4, %iterArg : tensor<i64>
                      %6 = stablehlo.dynamic_slice %iterArg_2, %5, sizes = [1] : (tensor<2xi1>, tensor<i64>) -> \
                tensor<1xi1>
                      %7 = stablehlo.reshape %6 : (tensor<1xi1>) -> tensor<i1>
                      %8 = \"stablehlo.if\"(%7) ({
                        %10 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<f32>
                        stablehlo.return %10 : tensor<f32>
                      }, {
                        stablehlo.return %iterArg_1 : tensor<f32>
                      }) : (tensor<i1>) -> tensor<f32>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %9 = stablehlo.add %iterArg, %c_5 : tensor<i64>
                      stablehlo.return %9, %8, %iterArg_2 : tensor<i64>, tensor<f32>, tensor<2xi1>
                    }
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                    %2:4 = stablehlo.while(%iterArg = %c_0, %iterArg_1 = %0#1, %iterArg_2 = %arg1, %iterArg_3 = %1) : \
                tensor<i64>, tensor<f32>, tensor<2xf32>, tensor<2xf32>
                    cond {
                      %c_4 = stablehlo.constant dense<2> : tensor<i64>
                      %4 = stablehlo.compare LT, %iterArg, %c_4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %4 : tensor<i1>
                    } do {
                      %c_4 = stablehlo.constant dense<2> : tensor<i64>
                      %c_5 = stablehlo.constant dense<1> : tensor<i64>
                      %4 = stablehlo.subtract %c_4, %c_5 : tensor<i64>
                      %5 = stablehlo.subtract %4, %iterArg : tensor<i64>
                      %6 = stablehlo.dynamic_slice %iterArg_2, %5, sizes = [1] : (tensor<2xf32>, tensor<i64>) -> \
                tensor<1xf32>
                      %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
                      %8 = stablehlo.add %iterArg_1, %7 : tensor<f32>
                      %9 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
                      %10 = stablehlo.dynamic_update_slice %iterArg_3, %9, %5 : (tensor<2xf32>, tensor<1xf32>, \
                tensor<i64>) -> tensor<2xf32>
                      %c_6 = stablehlo.constant dense<1> : tensor<i64>
                      %11 = stablehlo.add %iterArg, %c_6 : tensor<i64>
                      stablehlo.return %11, %8, %iterArg_2, %10 : tensor<i64>, tensor<f32>, tensor<2xf32>, tensor<2xf32>
                    }
                    %3 = \"stablehlo.if\"(%arg2) ({
                      %4 = stablehlo.add %2#1, %2#1 : tensor<f32>
                      stablehlo.return %4 : tensor<f32>
                    }, {
                      stablehlo.return %2#1 : tensor<f32>
                    }) : (tensor<i1>) -> tensor<f32>
                    return %3, %2#3 : tensor<f32>, tensor<2xf32>
                  }
                }
            "},
        );
        assert_eq!(linearization.residual_count(), 2);
    }
}
