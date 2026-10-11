use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, DataType, Dimension, EffectClass, Operation, PadOperation, ProgramError, Shape,
    Type as RyftType, TypeError,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, Location, LocationRef, Operation as MlirOperation, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::EffectTokens;
use crate::experimental::lowering::types::{
    lower_physical_bound_value_with_padding, lower_runtime_dimension_size_i64, lower_tensor_type, physical_bound_type,
};
use crate::operations::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_DETAIL_ATTRIBUTE, ASSERT_KIND_ATTRIBUTE, ASSERT_PAD_KIND,
    lower_assertion_custom_call, lower_deliberately_false_assertion_predicate,
};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

impl LowerableOperation for PadOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_pad_to_mlir(
            self,
            input_values,
            input_types,
            output_types,
            None,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

/// Validates that one pad interior amount can be represented by StableHLO's signed 64-bit attribute storage.
fn validate_pad_interior_padding(value: usize) -> Result<(), LoweringError> {
    i64::try_from(value).map(|_| ()).map_err(|_| LoweringError::PadInteriorPaddingOutOfRange { value })
}

/// Lowers a pad through the shared StableHLO path used by plain, generic-array, and shard-map dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_pad_to_mlir<'b, 'c: 'b, 't: 'c, T: RyftType, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    operation: &PadOperation<T>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    output_extents: Option<&[ValueRef<'b, 'c, 't>]>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 2, ProgramError);
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let rank = input_types[0].rank();
    if let Some(extents) = output_extents {
        check_count!("output", extents, rank, ProgramError);
    }
    if operation.interior_padding().len() != rank {
        return Err(ProgramError::from(TypeError::invalid(
            "`pad` padding configuration must match the input rank".to_string(),
        ))
        .into());
    }
    let output_type = &output_types[0];
    // Keep ordinary static padding in its canonical native form. Interior amounts on zero/one-element axes are
    // unobservable and need not fit a native attribute. Extreme opposing edges require the safe cropped form below.
    let interior_padding = operation
        .interior_padding()
        .iter()
        .copied()
        .enumerate()
        .map(|(axis, interior)| {
            if input_types[0].shape().dimensions()[axis].value().is_some_and(|extent| extent <= 1) {
                0
            } else {
                interior
            }
        })
        .collect::<Vec<_>>();
    let ordinary_static_padding = input_types[0].static_shape().is_some()
        && output_type.static_shape().is_some()
        && input_types[0].shape().dimensions().iter().enumerate().all(|(axis, dimension)| {
            let extent = dimension.value().unwrap() as i128;
            let expanded = extent
                + (extent - 1).max(0) * interior_padding[axis] as i128
                + i128::from(operation.edge_padding_low()[axis].max(0))
                + i128::from(operation.edge_padding_high()[axis].max(0));
            expanded <= i128::from(i64::MAX)
                && operation.edge_padding_low()[axis] != i64::MIN
                && operation.edge_padding_high()[axis] != i64::MIN
        });
    if ordinary_static_padding {
        interior_padding.iter().copied().try_for_each(validate_pad_interior_padding)?;
        let result = block.append_operation(stable_hlo::pad(
            input_values[0],
            input_values[1],
            operation.edge_padding_low(),
            operation.edge_padding_high(),
            &interior_padding,
            location,
        )?)?;
        return Ok(vec![result.result(0).unwrap().as_ref()]);
    }

    let physical_input_type = physical_bound_type(&input_types[0])?;
    let physical_output_type = physical_bound_type(output_type)?;
    let mut starts = Vec::with_capacity(rank);
    let mut limits = Vec::with_capacity(rank);
    let mut low_padding = Vec::with_capacity(rank);
    let mut high_padding = Vec::with_capacity(rank);
    let mut retained_interior = Vec::with_capacity(rank);
    let mut entirely_padding = false;
    for axis in 0..rank {
        let input_extent = physical_input_type.shape().dimensions()[axis].value().unwrap() as i128;
        let output_extent = physical_output_type.shape().dimensions()[axis].value().unwrap() as i128;
        let low = i128::from(operation.edge_padding_low()[axis]);
        let stride = operation.interior_padding()[axis] as i128 + 1;
        // Input position k lands at low + k * stride. Intersect those positions with the output capacity before
        // constructing a native pad, so neither huge interior gaps nor opposing edges create oversized temporaries.
        let first = if low < 0 { (-low + stride - 1) / stride } else { 0 };
        let limit = if output_extent <= low { 0 } else { ((output_extent - 1 - low) / stride + 1).min(input_extent) };
        if first >= limit || output_extent == 0 {
            entirely_padding = true;
            break;
        }
        let retained = limit - first;
        let first_position = low + first * stride;
        let last_position = first_position + (retained - 1) * stride;
        starts.push(first as usize);
        limits.push(limit as usize);
        low_padding.push(first_position as i64);
        high_padding.push((output_extent - last_position - 1) as i64);
        retained_interior.push(if retained <= 1 { 0 } else { operation.interior_padding()[axis] });
    }
    let mut block = block.as_ref();
    let location = location.as_ref();
    let mut result = if entirely_padding {
        block
            .append_operation(stable_hlo::broadcast(
                input_values[1],
                lower_tensor_type(&physical_output_type, context, location)?,
                &[],
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref()
    } else {
        // Inactive input lanes must contain the actual padding scalar, not a converted approximation or zero. This
        // makes every retained output prefix correct even when the runtime input is smaller than its capacity.
        let input = lower_physical_bound_value_with_padding(
            input_values[0],
            &input_types[0],
            0.0,
            Some(input_values[1]),
            &mut block,
            context,
            location,
        )?;
        let input = if starts.iter().any(|start| *start != 0)
            || limits
                .iter()
                .enumerate()
                .any(|(axis, limit)| *limit != physical_input_type.shape().dimensions()[axis].value().unwrap())
        {
            block
                .append_operation(stable_hlo::slice(input, &starts, &limits, &vec![1; rank], location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        } else {
            input
        };
        retained_interior.iter().copied().try_for_each(validate_pad_interior_padding)?;
        block
            .append_operation(stable_hlo::pad(
                input,
                input_values[1],
                &low_padding,
                &high_padding,
                &retained_interior,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref()
    };
    let mut refined_type = physical_output_type;
    for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let extent = match output_extents {
            Some(extents) => extents[axis],
            // Homogeneous padding retains a dynamic axis only when its size is unchanged.
            None => lower_runtime_dimension_size_i64(input_values[0], axis, &mut block, context, location)?,
        };
        let extent = block
            .append_operation(stable_hlo::convert(
                extent,
                lower_tensor_type(&ArrayType::scalar(DataType::I32), context, location)?,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let mut dimensions = refined_type.shape().dimensions().to_vec();
        dimensions[axis] = dimension.clone();
        refined_type = refined_type.with_shape(Shape::new(dimensions));
        result = block
            .append_operation(stable_hlo::set_dimension_size(
                result,
                extent,
                lower_tensor_type(&refined_type, context, location)?,
                axis,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
    }
    Ok(vec![result])
}

/// Checks padding geometry using actual extents and wide host arithmetic before native dimension narrowing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_pad_extent_assertion<'b, 'c: 'b, 't: 'c>(
    operation: &PadOperation<ArrayIrType>,
    axis: usize,
    input: ValueRef<'b, 'c, 't>,
    output: ValueRef<'b, 'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let predicate = lower_deliberately_false_assertion_predicate(output, block, context, location)?;
    let detail = format!(
        "{axis}:{}:{}:{}",
        operation.edge_padding_low()[axis],
        operation.edge_padding_high()[axis],
        operation.interior_padding()[axis],
    );
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute("pad")),
        context.named_attribute(context.identifier(ASSERT_KIND_ATTRIBUTE), context.string_attribute(ASSERT_PAD_KIND)),
        context.named_attribute(context.identifier(ASSERT_DETAIL_ATTRIBUTE), context.string_attribute(detail.as_str())),
    ]);
    lower_assertion_custom_call(predicate, &[input, output], backend_config, effect_tokens, block, context, location)
}

impl LowerableOperation for PadOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let context = lowerer.context;
        let location = lowerer.location;
        let effect_tokens = &mut lowerer.effect_tokens;
        let block = &mut lowerer.block;
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type).map_err(|error| LoweringError::Tracing(error.into()))?;
        let expected_input_count = output_type.rank() + 2;
        if input_values.len() != expected_input_count {
            return Err(
                ProgramError::InvalidInputCount { expected: expected_input_count, actual: input_values.len() }.into()
            );
        }
        if self.effects().classes().contains(EffectClass::OrderedAssertion) {
            for axis in 0..output_type.rank() {
                let input_extent = lower_runtime_dimension_size_i64(input_values[0], axis, block, context, location)?;
                lower_pad_extent_assertion(
                    self,
                    axis,
                    input_extent,
                    input_values[axis + 2],
                    effect_tokens,
                    block,
                    context,
                    location,
                )?;
            }
        }
        let results = lower_pad_to_mlir(
            self,
            &input_values[..2],
            &input_types[..2]
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProgramError::from)?,
            std::slice::from_ref(output_type),
            Some(&input_values[2..]),
            block,
            context,
            location,
        )?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ConstantOperation, DimensionBounds, DimensionOperation, DimensionType, DimensionValue, Placeholder,
        SliceOperation,
    };
    use ryft_mlir::{Operation as MlirOperation, Type, Value as MlirValue};
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, dynamic_dimension, ragged_dot_cpu_compilation_options, test_vector_type,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::lowering::types::lower_static_index_constants;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaProgramBuilder as CompositeXlaProgramBuilder};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{ToPjrt, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_validate_pad_interior_padding() {
        if usize::MAX <= i64::MAX as usize {
            return;
        }

        assert_eq!(
            validate_pad_interior_padding(usize::MAX),
            Err(LoweringError::PadInteriorPaddingOutOfRange { value: usize::MAX }),
        );
    }

    #[test]
    fn test_lower_pad_to_mlir() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let input_type = ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(1)]));
        let padding_type = ArrayType::scalar(DataType::I64);
        let output_type = input_type.clone().with_shape(Shape::new(vec![Dimension::Static(3)]));
        let input_tensor = lower_tensor_type(&input_type, &context, location).unwrap();
        let padding_tensor = lower_tensor_type(&padding_type, &context, location).unwrap();
        let mut block = context.block(&[(input_tensor, location), (padding_tensor, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let padding = block.argument(1).unwrap().as_ref();
        assert_eq!(
            lower_pad_to_mlir(
                &PadOperation::<ArrayType>::new(vec![1], vec![1], vec![usize::MAX]).unwrap(),
                &[input, padding],
                &[input_type.clone(), padding_type.clone()],
                &[output_type.clone()],
                Some(&[]),
                &mut block,
                &context,
                location,
            )
            .unwrap_err(),
            LoweringError::Tracing(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        // Interior padding is unobservable on a one-element axis, so it need not fit the native attribute type.
        let results = lower_pad_to_mlir(
            &PadOperation::<ArrayType>::new(vec![1], vec![1], vec![usize::MAX]).unwrap(),
            &[input, padding],
            &[input_type, padding_type],
            &[output_type.clone()],
            None,
            &mut block,
            &context,
            location,
        )
        .unwrap();
        assert_eq!(results[0].r#type().unwrap(), lower_tensor_type(&output_type, &context, location).unwrap().as_ref());
    }

    #[test]
    fn test_lower_pad_to_mlir_reconciles_requested_dynamic_bound() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("input", Some(5))]));
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let requested_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("output", Some(12))]));
        let input_tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let padding_value_tensor_type = lower_tensor_type(&padding_value_type, &context, location).unwrap();
        let mut block = context.block(&[(input_tensor_type, location), (padding_value_tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let padding_value = block.argument(1).unwrap().as_ref();

        let output_extent =
            lower_static_index_constants(&[6], &mut block.as_ref(), &context, location.as_ref()).unwrap()[0];
        let results = lower_pad_to_mlir(
            &PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
            &[input, padding_value],
            &[input_type, padding_value_type],
            std::slice::from_ref(&requested_output_type),
            Some(&[output_extent]),
            &mut block,
            &context,
            location,
        )
        .unwrap();

        assert_eq!(
            results[0].r#type().unwrap(),
            lower_tensor_type(&requested_output_type, &context, location).unwrap().as_ref(),
        );
        let operation_names = block
            .operations()
            .unwrap()
            .map(|operation| operation.unwrap().name().as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(operation_names.iter().filter(|name| *name == "stablehlo.pad").count(), 1);
        assert!(!operation_names.iter().any(|name| name == "tensor.cast"));
    }

    #[test]
    fn test_lower_pad_to_mlir_extreme_edge_padding() {
        let client = execution_client();
        for (size, low, high, interior, expected) in [
            (1_usize, i64::MIN, i64::MAX, 0_usize, Vec::new()),
            (2, i64::MIN, i64::MAX, 0, vec![i64::MAX]),
            (2, i64::MIN, 0, i64::MAX as usize, vec![i64::MIN]),
        ] {
            let input_type = ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(size)]));
            let padding_type = ArrayType::scalar(DataType::I64);
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type);
            let padding = builder.add_input(padding_type);
            let result = builder
                .add_instruction(
                    PadOperation::<ArrayType>::new(vec![low], vec![high], vec![interior]).unwrap(),
                    Vec::new(),
                    vec![input, padding],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![result],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
                .unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            let inputs = vec![
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                values_to_bytes(&vec![i64::MIN; size]).as_slice(),
                                BufferType::I64,
                                [size as u64],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap(),
                    ),
                    donatable: false,
                },
                ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(values_to_bytes(&[i64::MAX]).as_slice(), BufferType::I64, [], None, device, None)
                            .unwrap(),
                    ),
                    donatable: false,
                },
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
                .unwrap();
            let bytes = outputs[0].outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<i64>(&bytes), expected);
        }
    }

    #[test]
    fn test_lower_pad_to_mlir_execution() {
        let client = execution_client();
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        // Exact bits survive interior insertion, negative-edge cropping, empty inputs, and changes to allocation
        // bounds. Static result refinements still check the actual runtime input extent.
        for data_type in [DataType::I64, DataType::C64] {
            let element = if data_type == DataType::I64 { "i64" } else { "complex<f32>" };
            let input = if data_type == DataType::I64 {
                vec![i64::MAX, i64::MIN, 9_007_199_254_740_993, 4]
            } else {
                values_from_bytes::<i64>(&values_to_bytes(&[1_f32, -2.0, -0.0, 0.0, 3.0, -4.0, 5.0, -6.0]))
            };
            let padding = if data_type == DataType::I64 {
                9_007_199_254_740_993
            } else {
                values_from_bytes::<i64>(&values_to_bytes(&[-0_f32, 7.0]))[0]
            };
            for (size, low, high, interior, expected) in [
                (0_i32, 1_i64, 2_i64, 1_usize, vec![padding; 3]),
                (2, 1, 2, 1, vec![padding, input[0], padding, input[1], padding, padding]),
                (2, -1, 0, 1, vec![padding, input[1]]),
                (2, -3, 0, 1, Vec::new()),
                (2, i64::MIN, i64::MAX, 0, vec![padding]),
                (2, i64::MIN, 0, i64::MAX as usize, vec![input[1]]),
            ] {
                for bound in [None, Some(expected.len() + 1), Some(14)] {
                    let extent = match bound {
                        None => DimensionValue::constant(expected.len()).unwrap(),
                        Some(bound) => DimensionValue::new(
                            DimensionType::new("output", DimensionBounds::non_negative(Some(bound + 1)).unwrap()),
                            expected.len(),
                        )
                        .unwrap(),
                    };
                    let input_type = ArrayType::new(data_type, Shape::new(vec![dynamic_dimension("input", Some(5))]));
                    let padding_type = ArrayType::scalar(data_type);
                    let mut builder = CompositeXlaProgramBuilder::new();
                    let array = builder.add_input(input_type.clone().into());
                    let scalar = builder.add_input(padding_type.clone().into());
                    let extent = builder
                        .add_instruction(
                            DimensionOperation::from(ConstantOperation::new(extent)),
                            Vec::new(),
                            Vec::new(),
                            None,
                        )
                        .unwrap()[0];
                    let operation = PadOperation::<ArrayIrType>::new(vec![low], vec![high], vec![interior]).unwrap();
                    let result =
                        builder.add_instruction(operation, Vec::new(), vec![array, scalar, extent], None).unwrap()[0];
                    let program = builder
                        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                            vec![result],
                            vec![Placeholder; 2],
                            vec![Placeholder],
                        )
                        .unwrap();
                    let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
                    let module = to_mlir_module_for_program(
                        &program,
                        &[],
                        &vec![input_type, padding_type],
                        &vec![output_type.clone()],
                        "pad_dynamic",
                        None,
                        None,
                    )
                    .unwrap();
                    let capacity = physical_bound_type(&output_type).unwrap().dimension(0).value().unwrap();
                    let storage_type = format!("tensor<{capacity}x{element}>");
                    let body = if output_type.static_shape().is_some() {
                        format!(
                            indoc! {r#"
                            %result = func.call @pad_dynamic(%input, %padding, %size)
                                : (tensor<4x{element}>, tensor<{element}>, tensor<i32>) -> {storage_type}
                            return %result : {storage_type}
                        "#},
                            element = element,
                            storage_type = storage_type,
                        )
                    } else {
                        let result_type = format!("tensor<?x{element}, #stablehlo.bounds<{capacity}>>");
                        format!(
                            indoc! {r#"
                            %result, %extent = func.call @pad_dynamic(%input, %padding, %size)
                                : (tensor<4x{element}>, tensor<{element}>, tensor<i32>) -> ({result_type}, tensor<i64>)
                            %bound = stablehlo.constant dense<{capacity}> : tensor<i32>
                            %storage = stablehlo.set_dimension_size %result, %bound, dim = 0
                                : ({result_type}, tensor<i32>) -> {storage_type}
                            return %storage : {storage_type}
                        "#},
                            element = element,
                            result_type = result_type,
                            capacity = capacity,
                            storage_type = storage_type,
                        )
                    };
                    let prefix = module
                        .strip_suffix("}\n")
                        .unwrap()
                        .replace("func.func @pad_dynamic", "func.func private @pad_dynamic");
                    let module = format!(
                        indoc! {r#"
                        {prefix}
                        func.func @main(%input: tensor<4x{element}>, %padding: tensor<{element}>, %size: tensor<i32>) -> {storage_type} {{
                          {body}
                        }}
                        }}
                    "#},
                        prefix = prefix,
                        element = element,
                        storage_type = storage_type,
                        body = body,
                    );
                    let executable = client
                        .compile(
                            &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                            &ragged_dot_cpu_compilation_options(),
                        )
                        .unwrap();
                    let device = executable.addressable_devices().unwrap().remove(0);
                    for actual_size in [size, size + 1, 0] {
                        let inputs = vec![
                            ExecutionInput {
                                buffer: Arc::new(
                                    client
                                        .buffer(
                                            values_to_bytes(&input).as_slice(),
                                            data_type.to_pjrt(),
                                            [4],
                                            None,
                                            device.clone(),
                                            None,
                                        )
                                        .unwrap(),
                                ),
                                donatable: false,
                            },
                            ExecutionInput {
                                buffer: Arc::new(
                                    client
                                        .buffer(
                                            values_to_bytes(&[padding]).as_slice(),
                                            data_type.to_pjrt(),
                                            [],
                                            None,
                                            device.clone(),
                                            None,
                                        )
                                        .unwrap(),
                                ),
                                donatable: false,
                            },
                            ExecutionInput {
                                buffer: Arc::new(
                                    client
                                        .buffer(
                                            values_to_bytes(&[actual_size]).as_slice(),
                                            BufferType::I32,
                                            [],
                                            None,
                                            device.clone(),
                                            None,
                                        )
                                        .unwrap(),
                                ),
                                donatable: false,
                            },
                        ];
                        let execution = executable
                            .execute(
                                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                                Vec::new(),
                                0,
                                None,
                                Some(file!()),
                                None,
                                None,
                            )
                            .and_then(|execution| execution.block_until_ready());
                        if actual_size != size {
                            let error = execution.err().unwrap().to_string();
                            if actual_size == 0 && low < 0 || interior == i64::MAX as usize {
                                assert!(error.contains("outside the native dimension range on axis 0"));
                            } else {
                                assert!(error.contains("`pad` result extent on axis 0 must equal"));
                            }
                        } else {
                            let outputs = execution.unwrap();
                            let bytes = outputs[0].outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                            assert_eq!(&values_from_bytes::<i64>(&bytes)[..expected.len()], expected.as_slice());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_strided_slice_and_pad() {
        let vector_type = test_vector_type(6);
        let pad_input_type = test_vector_type(3);
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let vector = builder.add_input(vector_type);
        let pad_input = builder.add_input(pad_input_type);
        let padding_value = builder.add_input(padding_value_type);
        let strided = builder
            .add_instruction(
                SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let padded = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![pad_input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![strided, padded],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<6xf32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>) -> (tensor<3xf32>, tensor<8xf32>) {
                    %0 = stablehlo.slice %arg0 [1:6:2] : (tensor<6xf32>) -> tensor<3xf32>
                    %1 = stablehlo.pad %arg1, %arg2, low = [1], high = [2], interior = [1] : (tensor<3xf32>, tensor<f32>) -> tensor<8xf32>
                    return %0, %1 : tensor<3xf32>, tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_negative_and_mixed_pad() {
        let input_type = test_vector_type(5);
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let padding_value = builder.add_input(padding_value_type);
        let trimmed = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![-1], vec![-2], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            )
            .unwrap()[0];
        let mixed = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![-1], vec![2], vec![2]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![trimmed, mixed],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<f32>) -> (tensor<2xf32>, tensor<14xf32>) {
                    %0 = stablehlo.pad %arg0, %arg1, low = [-1], high = [-2], interior = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<2xf32>
                    %1 = stablehlo.pad %arg0, %arg1, low = [-1], high = [2], interior = [2] : (tensor<5xf32>, tensor<f32>) -> tensor<14xf32>
                    return %0, %1 : tensor<2xf32>, tensor<14xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_preserves_dynamic_pad_bound() {
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("input", Some(5))]));
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let padding_value = builder.add_input(padding_value_type);
        assert_eq!(
            builder.add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` dynamic axis 0 requires an explicit result-dimension input".to_string(),
            ))),
        );
    }
}
