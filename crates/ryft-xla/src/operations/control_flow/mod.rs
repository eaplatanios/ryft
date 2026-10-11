use std::rc::Rc;

use ryft_core::{
    ArrayIrType, ArrayType, DataType, Dimension, EffectClasses, RematerializationOptimizationBarrier, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::{
    lower_physical_storage, lower_tensor_type, lower_unplaced_constant_output, physical_bound_type,
    reshape_dimension_i64,
};
use crate::experimental::lowering::{CollectiveLoweringState, EffectTokens, token_threaded_effects};
use crate::experimental::ops::FlatXlaProgram;
use crate::operations::jit_call::JitCallFunctionMap;
use crate::{LowerableOperationFamily, LoweringError};

/// StableHLO lowering of `condition`.
pub(crate) mod condition;
/// StableHLO lowering of `scan`.
pub(crate) mod scan;
/// StableHLO lowering of `select`.
pub(crate) mod select;
/// StableHLO lowering of `while`.
pub(crate) mod r#while;

/// Converts the lowered value of one control-flow input to the lowered type of the region input that it feeds, which
/// the input type refines (refer to the `while`, `scan`, and `condition` type inference rules in `ryft-core`). Regions
/// receive every input at its declared type, and refinements that only add metadata (e.g., shardings or layouts) lower
/// to the same tensor type, so those inputs are returned unchanged. A static extent that refines a bounded dynamic
/// dimension lowers to a different tensor type. Such an input is exposed as its full physical buffer, padded up to the
/// physical bound of the declared type along every refined axis, and then given the runtime size of every dynamic
/// dimension of the declared type with `stablehlo.set_dimension_size`: the static extent for a refined axis, and the
/// input's own runtime size for an axis that stays dynamic.
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
    if !dimension_pairs.clone().any(|pair| matches!(pair, (Dimension::Dynamic(_), Dimension::Static(_)))) {
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
                Some(size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref())
            }
        });
    }

    let physical_value = lower_physical_storage(value, input_type, block, context, location)?;
    let physical_input_type = physical_bound_type(input_type)?;
    let physical_type = physical_bound_type(declared_type)?;
    let rank = declared_type.rank();
    let mut high_padding = vec![0; rank];
    for axis in 0..rank {
        let bound = physical_type.shape().dimensions()[axis].value().unwrap();
        let extent = physical_input_type.shape().dimensions()[axis].value().unwrap();
        high_padding[axis] = reshape_dimension_i64(bound - extent)?;
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
    let mut value = padded.result(0).expect("stablehlo.pad should return one result").as_ref();
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
        value = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
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
                    Some(size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref())
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
        let mut value = sliced.result(0).expect("stablehlo.slice should return one result").as_ref();
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
            value = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
        }
        converted.push(value);
    }
    Ok(converted)
}

/// Lowers one nested control-flow branch program into a fresh single-block region.
///
/// `entry_effect_tokens` are the enclosing scope's active per-class tokens, referenced inside the region through
/// StableHLO's implicit region capture (the same mechanism that feeds `input_values` into `stablehlo.if` branches).
/// Ordinary results are converted to `output_types` before the region returns one trailing token for each class
/// in `threaded_effects`, in canonical effect order. A branch that is pure for one of those classes returns that
/// class's entry token unchanged.
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
        // Normalize ordinary results before appending tokens so both conditional branches return the instruction's
        // common refined geometry even when their declared dynamic and static result types differ.
        outputs = lower_refined_region_outputs(
            outputs,
            &program.output_types(),
            output_types,
            &mut block_ref,
            context,
            location,
        )?;
        for effect in token_threaded_effects(threaded_effects) {
            outputs.push(
                region_effect_tokens
                    .get(effect)
                    .expect("token-returning control-flow regions receive one entry token per active class"),
            );
        }
        block_ref.append_operation(stable_hlo::r#return(outputs.as_slice(), location)?)?;
    }
    region.append_block(block)?;
    Ok(region)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, CompareOperation, ComparisonDirection, ConditionOperation, Placeholder,
        ScanOperation, WhileOperation,
    };

    use crate::experimental::lowering::tests::test_vector_type;
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::operations::jit_call::JitCallOperation;
    use crate::to_mlir_module_for_program;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_program_lowers_differentiated_composite_control_flow() {
        let scalar_boolean = ArrayType::scalar(DataType::Boolean);
        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let vector_f32 = test_vector_type(2);

        let branch = |double: bool| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let operand = builder.add_input(scalar_f32.clone().into());
            let output = if double {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![operand, operand], None).unwrap()[0]
            } else {
                operand
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
            .add_instruction(
                XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, state],
                None,
            )
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
        let operand = builder.add_input(scalar_f32.clone().into());
        let items = builder.add_input(vector_f32.clone().into());
        let selected = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, operand],
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
        let operand = builder.add_input(scalar_f32.clone().into());
        let items = builder.add_input(vector_f32.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![predicate, operand, items],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let primal = linearization.primal();
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(primal.instructions().len(), 1);
        let primal_inputs = vec![scalar_boolean, scalar_f32.clone(), vector_f32.clone()];
        let primal_outputs = primal
            .output_types()
            .iter()
            .map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => r#type.clone(),
                ArrayIrType::Dimension(_) => panic!("this fixture has only array residuals"),
                ArrayIrType::Reference(_) => panic!("this fixture has no reference residuals"),
            })
            .collect::<Vec<_>>();
        let primal_stablehlo =
            to_mlir_module_for_program(primal, &[], &primal_inputs, &primal_outputs, "main", None, None).unwrap();
        assert!(primal_stablehlo.contains("\"stablehlo.if\""), "{primal_stablehlo}");
        assert!(primal_stablehlo.matches("stablehlo.while").count() >= 2, "{primal_stablehlo}");

        let pullback = linearization.pullback().unwrap();
        let pullback_inputs = pullback
            .input_types()
            .iter()
            .map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => r#type.clone(),
                ArrayIrType::Dimension(_) => panic!("this fixture has only array residuals"),
                ArrayIrType::Reference(_) => panic!("this fixture has no reference residuals"),
            })
            .collect::<Vec<_>>();
        let pullback_outputs = vec![scalar_f32, vector_f32];
        let pullback_stablehlo =
            to_mlir_module_for_program(&pullback, &[], &pullback_inputs, &pullback_outputs, "main", None, None)
                .unwrap();
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback_stablehlo.contains("\"stablehlo.if\""), "{pullback_stablehlo}");
        assert!(pullback_stablehlo.contains("stablehlo.while"), "{pullback_stablehlo}");
    }
}
