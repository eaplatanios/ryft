use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, ComparisonDirection, DataType, Dimension, ReductionKind, ScatterMode, ScatterOperation,
    ScatterReductionKind, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::types::{
    lower_explicit_shape, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, lower_unplaced_constant_output,
    physical_bound_type,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::extrema::lower_extremum_to_mlir;
use crate::operations::reductions::build_reduce_body_region;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Builds the scalar combiner region of a `stablehlo.scatter` for the given [`ScatterReductionKind`], modeled on
/// [`build_reduce_body_region`]. The region's block takes the existing operand scalar and the update scalar and
/// returns the combined value: `Overwrite` returns the update directly (no combine op), arithmetic kinds apply the
/// matching elementwise StableHLO operation, and extrema use [`lower_extremum_to_mlir`].
fn build_scatter_combiner_region<'c, 't>(
    kind: ScatterReductionKind,
    element_type: DataType,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(element_type), context, location)?;
    let block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut region = context.region();
    let mut block_ref = region.append_block(block)?;
    let lhs = block_ref.argument(0)?.as_ref();
    let rhs = block_ref.argument(1)?.as_ref();
    let body_value = if element_type == DataType::Zero {
        lhs
    } else if matches!(element_type, DataType::I1 | DataType::U1) {
        // Predicate carriers need bitwise arithmetic modulo two. Signed I1 reverses the extrema order because its
        // set bit denotes -1, whereas U1's set bit denotes +1.
        match kind {
            ScatterReductionKind::Overwrite => rhs,
            ScatterReductionKind::Add => {
                block_ref.append_operation(stable_hlo::xor(lhs, rhs, location)?)?.result(0).unwrap().as_ref()
            }
            ScatterReductionKind::Mul => {
                block_ref.append_operation(stable_hlo::and(lhs, rhs, location)?)?.result(0).unwrap().as_ref()
            }
            ScatterReductionKind::Min | ScatterReductionKind::Max => {
                let use_or = (kind == ScatterReductionKind::Min) == (element_type == DataType::I1);
                if use_or {
                    block_ref.append_operation(stable_hlo::or(lhs, rhs, location)?)?.result(0).unwrap().as_ref()
                } else {
                    block_ref.append_operation(stable_hlo::and(lhs, rhs, location)?)?.result(0).unwrap().as_ref()
                }
            }
        }
    } else {
        match kind {
            ScatterReductionKind::Overwrite => rhs,
            ScatterReductionKind::Add => block_ref
                .append_operation(stable_hlo::add(lhs, rhs, location)?)?
                .result(0)
                .expect("stablehlo.add should return one result")
                .as_ref(),
            ScatterReductionKind::Mul => block_ref
                .append_operation(stable_hlo::multiply(lhs, rhs, location)?)?
                .result(0)
                .expect("stablehlo.multiply should return one result")
                .as_ref(),
            ScatterReductionKind::Min | ScatterReductionKind::Max => lower_extremum_to_mlir(
                kind == ScatterReductionKind::Max,
                element_type,
                lhs,
                rhs,
                &mut block_ref,
                location,
            )?,
        }
    };
    block_ref.append_operation(stable_hlo::r#return(&[body_value], location)?)?;
    Ok(region)
}

/// Broadcasts an index bound or mask while retaining the runtime query dimensions of an indices tensor.
fn lower_index_broadcast<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    shape_source: Option<ValueRef<'b, 'c, 't>>,
    axes: &[usize],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if let Some(exemplar) = shape_source {
        let input_type = ArrayType::new(
            output_type.data_type(),
            Shape::new(axes.iter().map(|axis| output_type.shape().dimensions()[*axis].clone()).collect()),
        );
        let value = lower_physical_bound_value(value, &input_type, 0.0, block, context, location)?;
        let physical_type = physical_bound_type(output_type)?;
        let result = block
            .append_operation(stable_hlo::broadcast(
                value,
                lower_tensor_type(&physical_type, context, location)?,
                axes,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let sources = (0..output_type.rank()).map(|axis| (exemplar, axis)).collect::<Vec<_>>();
        lower_restore_dynamic_dimensions(result, output_type, &sources, block, context, location)
    } else {
        Ok(block
            .append_operation(stable_hlo::broadcast(
                value,
                lower_tensor_type(output_type, context, location)?,
                axes,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref())
    }
}

/// Lowers scatter by clipping explicit starts or dropping complete out-of-bounds windows. Native scatter already
/// drops windows outside static allocation bounds; dynamic input bounds require an additional logical-size check.
pub(crate) fn lower_scatter_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ScatterOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    target_platform: Option<&str>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 3, ProgramError);
    check_count!("input", input_types, 3, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let dimensions = operation.dimensions();
    let window_axes = (0..input_types[0].rank())
        .filter(|axis| {
            !dimensions.inserted_window_dimensions().contains(axis)
                && !dimensions.input_batching_dimensions().contains(axis)
        })
        .collect::<Vec<_>>();
    for (&axis, &update_axis) in window_axes.iter().zip(dimensions.update_window_dimensions()) {
        let update_dimension = input_types[2].dimension(update_axis);
        if matches!(update_dimension, Dimension::Dynamic(_)) && update_dimension != input_types[0].dimension(axis) {
            return Err(LoweringError::UnsupportedOp {
                op: format!(
                    "`scatter` dynamic update window axis {update_axis} must match input axis {axis}; independently \
                     sized dynamic windows are unsupported by XLA",
                ),
            });
        }
    }
    let index_vector_dimension = input_types[1].rank() - 1;
    let index_data_type = if input_types[1].data_type().is_unsigned() { DataType::U64 } else { DataType::I64 };
    let indices_type = input_types[1].clone().with_data_type(index_data_type);
    let mut indices = block
        .append_operation(stable_hlo::convert(
            input_values[1],
            lower_tensor_type(&indices_type, context, location)?,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    if input_types[1].data_type() == DataType::I1 {
        indices = block.append_operation(stable_hlo::negate(indices, location)?)?.result(0).unwrap().as_ref();
    }
    let adjust_indices = operation.mode() == ScatterMode::Clip
        || (operation.mode() == ScatterMode::Drop && input_types[0].static_shape().is_none());
    if adjust_indices && !dimensions.scatter_dimensions_to_operand_dimensions().is_empty() {
        let shape_source = indices_type.static_shape().is_none().then_some(input_values[1]);
        let mut upper_bounds = Vec::new();
        let mut sentinels = Vec::new();
        for &axis in dimensions.scatter_dimensions_to_operand_dimensions() {
            let extent = lower_runtime_dimension_size_i64(input_values[0], axis, block, context, location)?;
            let window = match window_axes.iter().position(|window_axis| *window_axis == axis) {
                Some(position) => lower_runtime_dimension_size_i64(
                    input_values[2],
                    dimensions.update_window_dimensions()[position],
                    block,
                    context,
                    location,
                )?,
                None => lower_static_index_constants(&[1], block, context, location)?[0],
            };
            upper_bounds.push(
                block.append_operation(stable_hlo::subtract(extent, window, location)?)?.result(0).unwrap().as_ref(),
            );
            if operation.mode() == ScatterMode::Drop {
                let Dimension::Static(capacity) = physical_bound_type(&input_types[0])?.shape().dimensions()[axis]
                else {
                    unreachable!()
                };
                sentinels.push(lower_static_index_constants(&[capacity], block, context, location)?[0]);
            }
        }
        let bound_type = ArrayType::new_static(index_data_type, [upper_bounds.len()]);
        let upper_bounds = lower_explicit_shape(&upper_bounds, block, context, location)?;
        let upper_bounds = block
            .append_operation(stable_hlo::convert(
                upper_bounds,
                lower_tensor_type(&bound_type, context, location)?,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let upper_bounds = lower_index_broadcast(
            upper_bounds,
            &indices_type,
            shape_source,
            &[index_vector_dimension],
            block,
            context,
            location,
        )?;
        let zero =
            lower_unplaced_constant_output(&[ArrayType::scalar(index_data_type)], 0, block, context, location)?[0];
        let zero = lower_index_broadcast(zero, &indices_type, shape_source, &[], block, context, location)?;
        if operation.mode() == ScatterMode::Clip {
            indices =
                block.append_operation(stable_hlo::maximum(indices, zero, location)?)?.result(0).unwrap().as_ref();
            indices = block
                .append_operation(stable_hlo::minimum(indices, upper_bounds, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
        } else {
            let lower = lower_compare_to_mlir(ComparisonDirection::GreaterThanOrEqual, indices, zero, block, location)?;
            let upper =
                lower_compare_to_mlir(ComparisonDirection::LessThanOrEqual, indices, upper_bounds, block, location)?;
            let mask = block.append_operation(stable_hlo::and(lower, upper, location)?)?.result(0).unwrap().as_ref();
            let initial =
                lower_unplaced_constant_output(&[ArrayType::scalar(DataType::Boolean)], 1, block, context, location)?
                    [0];
            let body = build_reduce_body_region(ReductionKind::All, DataType::Boolean, context, location)?;
            let mask = block
                .append_operation(stable_hlo::reduce(&[mask], &[initial], &[index_vector_dimension], body, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
            let mask = lower_index_broadcast(
                mask,
                &indices_type.clone().with_data_type(DataType::Boolean),
                shape_source,
                &(0..index_vector_dimension).collect::<Vec<_>>(),
                block,
                context,
                location,
            )?;
            let sentinels = lower_explicit_shape(&sentinels, block, context, location)?;
            let sentinels = block
                .append_operation(stable_hlo::convert(
                    sentinels,
                    lower_tensor_type(&bound_type, context, location)?,
                    location,
                )?)?
                .result(0)
                .unwrap()
                .as_ref();
            let sentinels = lower_index_broadcast(
                sentinels,
                &indices_type,
                shape_source,
                &[index_vector_dimension],
                block,
                context,
                location,
            )?;
            indices = block
                .append_operation(stable_hlo::select(mask, indices, sentinels, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
        }
    }
    let attribute = context.stable_hlo_scatter_dimensions(
        dimensions.update_window_dimensions(),
        dimensions.inserted_window_dimensions(),
        dimensions.input_batching_dimensions(),
        dimensions.scatter_indices_batching_dimensions(),
        dimensions.scatter_dimensions_to_operand_dimensions(),
        index_vector_dimension,
    )?;
    if target_platform == Some("cuda")
        && output_types[0].data_type() == DataType::C128
        && operation.kind() == ScatterReductionKind::Add
    {
        // GPU scatter-add lacks a native complex128 atomic update. Accumulate its independent real components
        // using floating-point atomics, then reconstruct the complex result.
        let mut components = Vec::with_capacity(2);
        for imaginary in [false, true] {
            let input = if imaginary {
                block.append_operation(stable_hlo::imag(input_values[0], location)?)?.result(0).unwrap().as_ref()
            } else {
                block.append_operation(stable_hlo::real(input_values[0], location)?)?.result(0).unwrap().as_ref()
            };
            let updates = if imaginary {
                block.append_operation(stable_hlo::imag(input_values[2], location)?)?.result(0).unwrap().as_ref()
            } else {
                block.append_operation(stable_hlo::real(input_values[2], location)?)?.result(0).unwrap().as_ref()
            };
            let combiner = build_scatter_combiner_region(ScatterReductionKind::Add, DataType::F64, context, location)?;
            components.push(
                block
                    .append_operation(stable_hlo::scatter(
                        &[input],
                        indices,
                        &[updates],
                        attribute,
                        combiner,
                        !adjust_indices && operation.indices_are_sorted(),
                        !adjust_indices && operation.unique_indices(),
                        location,
                    )?)?
                    .result(0)
                    .unwrap()
                    .as_ref(),
            );
        }
        return Ok(vec![
            block
                .append_operation(stable_hlo::complex(components[0], components[1], location)?)?
                .result(0)
                .unwrap()
                .as_ref(),
        ]);
    }
    let combiner = build_scatter_combiner_region(operation.kind(), output_types[0].data_type(), context, location)?;
    // Clipping can merge formerly unique indices and change their lexicographic order; masked sentinels likewise
    // introduce duplicates. Do not propagate either promise through that normalization.
    let result = block.append_operation(stable_hlo::scatter(
        &[input_values[0]],
        indices,
        &[input_values[2]],
        attribute,
        combiner,
        !adjust_indices && operation.indices_are_sorted(),
        !adjust_indices && operation.unique_indices(),
        location,
    )?)?;
    Ok(vec![result.result(0).unwrap().as_ref()])
}

impl LowerableOperation for ScatterOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let values = lower_scatter_to_mlir(
            self,
            input_values,
            input_types,
            output_types,
            lowerer.collective_state.target_platform(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        if self.output_sharding().is_some() {
            lower_sharding_constraint(
                &values,
                output_types[0].sharding().unwrap(),
                &lowerer.collective_state.bound_manual_axes,
                &mut lowerer.block,
                lowerer.location,
            )
        } else {
            Ok(values)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayOperation, ConstantOperation, DimensionBounds, DimensionVariable, Placeholder, ProgramBuilder,
        ScatterDimensionNumbers, i1, u1,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, ragged_dot_cpu_compilation_options, unproject_plain_program,
    };
    use crate::experimental::lowering::{lower_mlir_module_for_program, to_mlir_module_for_plain_program};
    use crate::experimental::ops::XlaArrayConstant;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{ToPjrt, to_mlir_module_for_program};

    use super::*;

    #[test]
    fn test_lower_scatter_index_modes() {
        let client = execution_client();
        for mode in [ScatterMode::Clip, ScatterMode::Drop] {
            for (unsigned, dynamic_input, dynamic_query) in
                [(false, false, false), (true, false, false), (false, true, false), (false, false, true)]
            {
                let input_values = [10_i64, 20, 30, 40];
                let input_type = if dynamic_input {
                    ArrayType::new(
                        DataType::I64,
                        Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "rows",
                            DimensionBounds::new(2, Some(5)).unwrap(),
                        ))]),
                    )
                } else {
                    ArrayType::new_static(DataType::I64, [4])
                };
                let query = if dynamic_query {
                    Dimension::Dynamic(DimensionVariable::new("query", DimensionBounds::new(0, Some(5)).unwrap()))
                } else {
                    Dimension::Static(4)
                };
                let index_data_type = if unsigned { DataType::U64 } else { DataType::I64 };
                let indices_type =
                    ArrayType::new(index_data_type, Shape::new(vec![query.clone(), Dimension::Static(1)]));
                let updates_type = ArrayType::new(DataType::I64, Shape::new(vec![query]));
                let mut builder = XlaProgramBuilder::new();
                let inputs = [input_type.clone(), indices_type.clone(), updates_type.clone()]
                    .into_iter()
                    .map(|r#type| builder.add_input(r#type))
                    .collect::<Vec<_>>();
                let operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                    ScatterReductionKind::Add,
                )
                .with_mode(mode);
                let output = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0];
                let program = unproject_plain_program(
                    builder
                        .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                            vec![output],
                            vec![Placeholder; 3],
                            vec![Placeholder],
                        )
                        .unwrap(),
                );
                let module = to_mlir_module_for_program(
                    &program,
                    &[],
                    &vec![input_type.clone(), indices_type, updates_type],
                    &vec![input_type],
                    "main",
                    None,
                    None,
                )
                .unwrap();
                let executable = client
                    .compile(
                        &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                        &ragged_dot_cpu_compilation_options(),
                    )
                    .unwrap();
                let device = executable.addressable_devices().unwrap().remove(0);
                let mut values = vec![
                    (values_to_bytes(&input_values), BufferType::I64, vec![4]),
                    (
                        if unsigned {
                            values_to_bytes(&[0_u64, u64::MAX, u64::MAX, 1])
                        } else {
                            values_to_bytes(&[-2_i64, -1, 4, 9])
                        },
                        index_data_type.to_pjrt(),
                        vec![4, 1],
                    ),
                    (values_to_bytes(&[1_i64, 2, 3, 4]), BufferType::I64, vec![4]),
                ];
                if dynamic_input {
                    values.push((values_to_bytes(&[2_i32]), BufferType::I32, vec![]));
                }
                if dynamic_query {
                    values.push((values_to_bytes(&[2_i32]), BufferType::I32, vec![]));
                    values.push((values_to_bytes(&[2_i32]), BufferType::I32, vec![]));
                }
                let inputs = values
                    .into_iter()
                    .map(|(bytes, data_type, shape)| ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(bytes.as_slice(), data_type, shape.as_slice(), None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    })
                    .collect::<Vec<_>>();
                let outputs = executable
                    .execute(
                        vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                        vec![],
                        0,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                    .block_until_ready()
                    .unwrap()
                    .remove(0)
                    .outputs;
                let expected = match (mode, unsigned, dynamic_input, dynamic_query) {
                    (ScatterMode::Clip, true, _, _) => vec![11_i64, 24, 30, 45],
                    (ScatterMode::Drop, true, _, _) => vec![11, 24, 30, 40],
                    (ScatterMode::Clip, _, true, _) => vec![13, 27],
                    (ScatterMode::Clip, _, _, true) => vec![13, 20, 30, 40],
                    (ScatterMode::Clip, _, _, _) => vec![13, 20, 30, 47],
                    (_, _, true, _) => vec![10, 20],
                    _ => vec![10, 20, 30, 40],
                };
                let bytes = outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                assert_eq!(
                    values_from_bytes::<i64>(&bytes[..expected.len() * 8]),
                    expected,
                    "mode={mode}, unsigned={unsigned}, dynamic_input={dynamic_input}, dynamic_query={dynamic_query}",
                );
            }
        }
        // A partially out-of-bounds update is dropped as a whole; Clip moves the complete window into range.
        let mut literal_cases = Vec::new();
        for mode in [ScatterMode::Clip, ScatterMode::Drop] {
            literal_cases.push((
                Array::vector(vec![0_i64; 3]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I64, [1, 1]), &[2_i64]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I64, [1, 2]), &[10_i64, 20]).unwrap(),
                ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![1], vec![], vec![0]),
                    ScatterReductionKind::Add,
                )
                .with_mode(mode),
                values_to_bytes(if mode == ScatterMode::Clip { &[0_i64, 10, 20] } else { &[0_i64, 0, 0] }),
            ));
        }
        // One-bit integers use predicate storage but retain arithmetic modulo two and signed/unsigned extrema.
        for signed in [false, true] {
            for kind in [
                ScatterReductionKind::Overwrite,
                ScatterReductionKind::Add,
                ScatterReductionKind::Mul,
                ScatterReductionKind::Min,
                ScatterReductionKind::Max,
            ] {
                let input = if signed {
                    Array::vector(vec![i1::new(0).unwrap(), i1::new(-1).unwrap(), i1::new(-1).unwrap()]).unwrap()
                } else {
                    Array::vector(vec![u1::new(0).unwrap(), u1::new(1).unwrap(), u1::new(1).unwrap()]).unwrap()
                };
                let updates = if signed {
                    Array::vector(vec![i1::new(-1).unwrap(), i1::new(-1).unwrap(), i1::new(0).unwrap()]).unwrap()
                } else {
                    Array::vector(vec![u1::new(1).unwrap(), u1::new(1).unwrap(), u1::new(0).unwrap()]).unwrap()
                };
                let expected = match kind {
                    ScatterReductionKind::Overwrite => vec![1_u8, 1, 0],
                    ScatterReductionKind::Add => vec![1, 0, 1],
                    ScatterReductionKind::Mul => vec![0, 1, 0],
                    ScatterReductionKind::Min | ScatterReductionKind::Max => {
                        if (kind == ScatterReductionKind::Min) == signed { vec![1, 1, 1] } else { vec![0, 1, 0] }
                    }
                };
                literal_cases.push((
                    input,
                    Array::from_elements(ArrayType::new_static(DataType::I64, [3, 1]), &[0_i64, 1, 2]).unwrap(),
                    updates,
                    ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind),
                    expected,
                ));
            }
        }
        for (input, indices, updates, operation, expected) in literal_cases {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = [input, indices, updates]
                .into_iter()
                .map(|value| {
                    builder.add_instruction(ConstantOperation::new(value), Vec::new(), Vec::new(), None).unwrap()[0]
                })
                .collect::<Vec<_>>();
            let output = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0];
            let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![], vec![Placeholder]).unwrap();
            let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
                .unwrap();
            let outputs = executable
                .execute(
                    vec![ExecutionDeviceInputs { inputs: &[], ..Default::default() }],
                    vec![],
                    0,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
                .remove(0)
                .outputs;
            assert_eq!(outputs[0].copy_to_host(None).unwrap().r#await().unwrap(), expected);
        }
    }

    #[test]
    fn test_lower_scatter_complex_add_on_cuda() {
        let client = execution_client();
        let execution_platform = client.platform_name().unwrap();
        for (data_type, platform, scatter_count) in
            [(DataType::C128, Some("cuda"), 2), (DataType::C128, Some("cpu"), 1), (DataType::C64, Some("cuda"), 1)]
        {
            let input_types = vec![
                ArrayType::new_static(data_type, [4]),
                ArrayType::new_static(DataType::I32, [2, 1]),
                ArrayType::new_static(data_type, [2]),
            ];
            let output_types = vec![input_types[0].clone()];
            let mut builder = XlaProgramBuilder::new();
            let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect();
            let output = builder
                .add_instruction(
                    ArrayOperation::Scatter(ScatterOperation::new(
                        ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                        ScatterReductionKind::Add,
                    )),
                    Vec::new(),
                    inputs,
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 3],
                    vec![Placeholder],
                )
                .unwrap();
            let module = lower_mlir_module_for_program(
                &unproject_plain_program(program),
                &[],
                &input_types,
                &output_types,
                "main",
                None,
                None,
                platform,
            )
            .unwrap()
            .stable_hlo;
            assert_eq!(
                module.matches("stablehlo.scatter").count() - module.matches("#stablehlo.scatter").count(),
                scatter_count,
            );
            assert_eq!(module.matches("stablehlo.complex ").count(), usize::from(scatter_count == 2));

            // Execute the selected backend's lowering as well as checking its structure. In particular, the CUDA
            // double-precision complex path must sum real and imaginary updates through its two scatter kernels.
            if !execution_platform.eq_ignore_ascii_case(platform.unwrap()) {
                continue;
            }
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
                .unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            let (input, updates, expected, buffer_type) = if data_type == DataType::C128 {
                (
                    values_to_bytes(&[1_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                    values_to_bytes(&[10_f64, -20.0, 30.0, -40.0]),
                    values_to_bytes(&[1_f64, 2.0, 43.0, -56.0, 5.0, 6.0, 7.0, 8.0]),
                    BufferType::C128,
                )
            } else {
                (
                    values_to_bytes(&[1_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                    values_to_bytes(&[10_f32, -20.0, 30.0, -40.0]),
                    values_to_bytes(&[1_f32, 2.0, 43.0, -56.0, 5.0, 6.0, 7.0, 8.0]),
                    BufferType::C64,
                )
            };
            let inputs = [
                (input, buffer_type, vec![4]),
                (values_to_bytes(&[1_i32, 1]), BufferType::I32, vec![2, 1]),
                (updates, buffer_type, vec![2]),
            ]
            .into_iter()
            .map(|(bytes, data_type, shape)| ExecutionInput {
                buffer: Arc::new(
                    client.buffer(bytes.as_slice(), data_type, &shape, None, device.clone(), None).unwrap(),
                ),
                donatable: false,
            })
            .collect::<Vec<_>>();
            let output = executable
                .execute(
                    vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                    Vec::new(),
                    0,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
                .remove(0)
                .outputs
                .remove(0);
            assert_eq!(output.copy_to_host(None).unwrap().r#await().unwrap(), expected);
        }
    }

    #[test]
    fn test_lower_scatter_dynamic_window() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [3]));
        let indices = builder.add_input(ArrayType::new_static(DataType::I32, [1, 1]));
        let updates = builder.add_input(ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(1),
                Dimension::Dynamic(DimensionVariable::new("window", DimensionBounds::new(0, Some(3)).unwrap())),
            ]),
        ));
        let output = builder
            .add_instruction(
                ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![1], vec![], vec![0]),
                    ScatterReductionKind::Add,
                ),
                Vec::new(),
                vec![input, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::UnsupportedOp {
                op: "`scatter` dynamic update window axis 1 must match input axis 0; independently sized dynamic \
                     windows are unsupported by XLA"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_scatter() {
        use ryft_core::{Dimension, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, Shape};

        // Scatter-add row updates into a [3, 2] operand at the row indices in a [2, 1] index array. Output is [3, 2],
        // and the Add combiner lowers to a `stablehlo.add` inside the scatter region.
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let updates = builder.add_input(updates_type);
        let output = builder
            .add_instruction(ArrayOperation::Scatter(operation), Vec::new(), vec![operand, indices, updates], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.scatter"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.add"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3x2xf32>"), "{stablehlo}");

        // Complex scatter extrema share the explicit lexicographic combiner used by complex reductions.
        let operand_type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Max);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let updates = builder.add_input(updates_type);
        let output = builder
            .add_instruction(ArrayOperation::Scatter(operation), Vec::new(), vec![operand, indices, updates], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.real").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.imag").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.compare").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 2, "{stablehlo}");
    }
}
