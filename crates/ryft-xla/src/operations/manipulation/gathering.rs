use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, ComparisonDirection, DataType, Dimension as CoreDimension, GatherMode, GatherOperation, ReductionKind,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::experimental::lowering::MlirLowerableValue;
use crate::experimental::lowering::types::{
    lower_explicit_shape, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, lower_unplaced_constant_output,
    physical_bound_type,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::reductions::build_reduce_body_region;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

/// Lowers gather with its explicit index-vector axis. Fill mode masks whole out-of-bounds windows, and clip mode
/// clamps against logical input sizes even when the allocation is larger. Comparisons preserve index signedness.
pub(crate) fn lower_gather_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &GatherOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    check_count!("input", input_values, 2, ProgramError);
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    // Native gather expects integer indices; the one-bit MLIR carrier instead denotes a predicate. Widen every
    // index family without changing its signedness and recover I1's numeric value (-1 or 0) explicitly.
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
    let dimensions = operation.dimensions();
    let (input_batching, indices_batching): (Vec<_>, Vec<_>) = dimensions.batching_dimensions().iter().copied().unzip();
    let index_vector_dimension = input_types[1].rank() - 1;
    let attribute = context.stable_hlo_gather_dimensions(
        dimensions.offset_dimensions(),
        dimensions.collapsed_slice_dimensions(),
        &input_batching,
        &indices_batching,
        dimensions.start_index_map(),
        index_vector_dimension,
    )?;
    // Paired axes select matching batch coordinates and do not contribute window coordinates. Use the
    // canonical window entry when the extent proves which form is valid: zero for an empty axis and one for a
    // nonempty axis. A retained symbolic batch may have used zero to admit an empty extent; leaving that entry zero
    // after specialization to a nonempty batch can make XLA replace the entire gather with zeros. If the extent can
    // still be zero or positive, retain the original entry: changing it to one would violate the gather window bound
    // on a runtime-empty input. This does not add support for unspecialized runtime-zero batching.
    let mut slice_sizes = operation.slice_sizes().to_vec();
    for &(axis, _) in dimensions.batching_dimensions() {
        let bounds = input_types[0].dimension(axis).bounds();
        if bounds.upper() == Some(1) {
            slice_sizes[axis] = 0;
        } else if bounds.lower() > 0 {
            slice_sizes[axis] = 1;
        }
    }
    let result = block.append_operation(stable_hlo::gather(
        input_values[0],
        indices,
        attribute,
        &slice_sizes,
        operation.indices_are_sorted(),
        location,
    )?)?;
    let result = result.result(0).unwrap().as_ref();
    if operation.mode() == &GatherMode::PromiseInBounds
        || (operation.mode() == &GatherMode::Clip && input_types[0].static_shape().is_some())
    {
        return Ok(vec![result]);
    }

    // Compute predicates on bounded storage and restore query extents afterwards. This avoids broadcasting scalar
    // constants to dynamic tensor types without supplying their runtime sizes.
    let comparison_type = physical_bound_type(&indices_type)?;
    let physical_output_type = physical_bound_type(&output_types[0])?;
    let indices = lower_physical_bound_value(indices, &indices_type, 0.0, block, context, location)?;

    // Widen before comparing: a legal input extent can exceed an i8/u8 index's range. In particular, converting u64
    // indices to i64 would incorrectly turn its largest values into negative indices or wrap a clipping operation.
    let mut upper_bounds = Vec::with_capacity(dimensions.start_index_map().len());
    for &axis in dimensions.start_index_map() {
        let extent = match input_types[0].shape().dimensions()[axis] {
            CoreDimension::Static(extent) => lower_static_index_constants(&[extent], block, context, location)?[0],
            CoreDimension::Dynamic(_) => {
                lower_runtime_dimension_size_i64(input_values[0], axis, block, context, location)?
            }
        };
        let window = lower_static_index_constants(&[operation.slice_sizes()[axis]], block, context, location)?[0];
        upper_bounds
            .push(block.append_operation(stable_hlo::subtract(extent, window, location)?)?.result(0).unwrap().as_ref());
    }
    let upper_bounds = lower_explicit_shape(&upper_bounds, block, context, location)?;
    let upper_bounds_type = ArrayType::new_static(index_data_type, [dimensions.start_index_map().len()]);
    let upper_bounds = block
        .append_operation(stable_hlo::convert(
            upper_bounds,
            lower_tensor_type(&upper_bounds_type, context, location)?,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let upper_bounds = block
        .append_operation(stable_hlo::broadcast(
            upper_bounds,
            lower_tensor_type(&comparison_type, context, location)?,
            &[index_vector_dimension],
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let zero = lower_unplaced_constant_output(&[comparison_type], 0, block, context, location)?[0];
    if operation.mode() == &GatherMode::Clip {
        // Native gather clamps against allocation capacities, which can exceed a dynamic input's logical size.
        // Clamp explicitly before gathering, then carry query sizes from the original gather's shape computation.
        let clipped =
            block.append_operation(stable_hlo::maximum(indices, zero, location)?)?.result(0).unwrap().as_ref();
        let clipped = block
            .append_operation(stable_hlo::minimum(clipped, upper_bounds, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        // Componentwise clamping can reorder lexicographically sorted index vectors, so discard the sorting hint.
        let clipped = block
            .append_operation(stable_hlo::gather(
                input_values[0],
                clipped,
                attribute,
                operation.slice_sizes(),
                false,
                location,
            )?)?
            .result(0)
            .unwrap()
            .as_ref();
        let clipped = lower_physical_bound_value(clipped, &output_types[0], 0.0, block, context, location)?;
        let sources = (0..output_types[0].rank()).map(|axis| (result, axis)).collect::<Vec<_>>();
        return Ok(vec![lower_restore_dynamic_dimensions(
            clipped,
            &output_types[0],
            &sources,
            block,
            context,
            location,
        )?]);
    }
    let upper_mask =
        lower_compare_to_mlir(ComparisonDirection::LessThanOrEqual, indices, upper_bounds, block, location)?;
    let lower_mask = lower_compare_to_mlir(ComparisonDirection::GreaterThanOrEqual, indices, zero, block, location)?;
    let mask = block
        .append_operation(stable_hlo::and(lower_mask, upper_mask, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let initial =
        lower_unplaced_constant_output(&[ArrayType::scalar(DataType::Boolean)], 1, block, context, location)?[0];
    let reduction = build_reduce_body_region(ReductionKind::All, DataType::Boolean, context, location)?;
    let mask = block
        .append_operation(stable_hlo::reduce(&[mask], &[initial], &[index_vector_dimension], reduction, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let batch_axes = (0..output_types[0].rank())
        .filter(|axis| !dimensions.offset_dimensions().contains(axis))
        .collect::<Vec<_>>();
    let mask_type = physical_output_type.clone().with_data_type(DataType::Boolean);
    let mask = block
        .append_operation(stable_hlo::broadcast(
            mask,
            lower_tensor_type(&mask_type, context, location)?,
            &batch_axes,
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();

    let fill = operation.options().resolved_fill_value(output_types[0].data_type())?;
    let scalar_type = lower_tensor_type(&ArrayType::scalar(output_types[0].data_type()), context, location)?;
    let fill = block
        .append_operation(stable_hlo::constant(fill.to_dense_elements_attribute(scalar_type, context)?, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let fill = block
        .append_operation(stable_hlo::broadcast(
            fill,
            lower_tensor_type(&physical_output_type, context, location)?,
            &[],
            location,
        )?)?
        .result(0)
        .unwrap()
        .as_ref();
    let physical_result = lower_physical_bound_value(result, &output_types[0], 0.0, block, context, location)?;
    let filled = block
        .append_operation(stable_hlo::select(mask, physical_result, fill, location)?)?
        .result(0)
        .unwrap()
        .as_ref();
    let sources = (0..output_types[0].rank()).map(|axis| (result, axis)).collect::<Vec<_>>();
    Ok(vec![lower_restore_dynamic_dimensions(filled, &output_types[0], &sources, block, context, location)?])
}

impl LowerableOperation for GatherOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let values = lower_gather_to_mlir(
            self,
            input_values,
            input_types,
            output_types,
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

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    #[cfg(test)]
    use ryft_core::Complex as ComplexNumber;
    use ryft_core::{
        Array as CoreArray, ArrayBatch, ArrayBatchingPolicy, ArrayIrType, ArrayOperation, ArrayType, BatchAxis,
        BatchableOperation, BatchingContext, DataType, Dimension as CoreDimension, DimensionBounds, DimensionSize,
        DimensionVariable, DynamicArrayExtentBatchingPolicy, EmptyRegionDriver, GatherDimensionNumbers, GatherMode,
        GatherOperation, LogicalMesh, MeshAxis, MeshAxisType, Placeholder, ProgramBuilder as CoreProgramBuilder,
        ProjectedContext, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, Shape, Sharding,
        ShardingDimension, Typed, TypedProgram, ValueProjection, i1,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::domains::tests::{domain_mesh, read_f64s};
    use crate::experimental::lowering::tests::{
        ProgramBuilder, ragged_dot_cpu_compilation_options, unproject_plain_program,
    };
    use crate::experimental::lowering::to_mlir_module_for_plain_program;
    use crate::experimental::ops::ArrayConstant;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{Array, Options, Session, to_mlir_module_for_program};

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_gather() {
        use ryft_core::{
            Dimension as CoreDimension, DimensionBounds, DimensionVariable, GatherDimensionNumbers, GatherOperation,
            Shape,
        };

        // Take whole rows of a [3, 2] matrix at the row indices in a [2, 1] index array: offset axis 1 carries the
        // row (slice sizes [1, 2]); axis 0 is collapsed (start-index driven). Output is [2, 2].
        let operand_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3), CoreDimension::Static(2)]));
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let mut builder = ProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(ArrayOperation::Gather(operation), Vec::new(), vec![operand, indices], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.gather"), "{stablehlo}");
        assert!(stablehlo.contains("offset_dims = [1]"), "{stablehlo}");
        assert!(stablehlo.contains("collapsed_slice_dims = [0]"), "{stablehlo}");
        assert!(stablehlo.contains("start_index_map = [0]"), "{stablehlo}");
        assert!(stablehlo.contains("slice_sizes = array<i64: 1, 2>"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<2x2xf32>"), "{stablehlo}");

        // A dynamic query-batch axis remains the same array identity through gather; it is not reconstructed from a
        // shape operand or lowered through `stablehlo.dynamic_gather`.
        let query = DimensionVariable::new("query", DimensionBounds::new(1, Some(5)).unwrap());
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Dynamic(query), CoreDimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let mut builder = ProgramBuilder::new();
        let operand = builder.add_input(ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Static(3), CoreDimension::Static(2)]),
        ));
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(ArrayOperation::Gather(operation), Vec::new(), vec![operand, indices], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.gather"), "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.dynamic_gather"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<?x2xf32, #stablehlo.bounds<4, ?>>"), "{stablehlo}");
    }

    #[test]
    fn test_lower_gather_fill_windows() {
        let client = execution_client();
        let input =
            CoreArray::from_elements(ArrayType::new_static(DataType::I64, [300]), &(0_i64..300).collect::<Vec<_>>())
                .unwrap();
        let fill = 9_007_199_254_740_993_i64;
        let windows = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![3])
            .with_mode(GatherMode::Fill { value: None });
        let cases = [
            (
                CoreArray::scalar(7_i64).unwrap(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I32, [2, 0]), &[] as &[i32]).unwrap(),
                GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![], vec![]), vec![])
                    .with_mode(GatherMode::Fill { value: None }),
                values_to_bytes(&[7_i64, 7]),
            ),
            (
                CoreArray::from_elements(ArrayType::new_static(DataType::I64, [2, 3]), &[10_i64, 20, 30, 40, 50, 60])
                    .unwrap(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[0_i32, 2, 1, 3]).unwrap(),
                GatherOperation::new(
                    GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![(0, 0)]),
                    vec![1, 1],
                )
                .with_mode(GatherMode::Fill { value: None }),
                values_to_bytes(&[10_i64, 30, 50, i64::MIN]),
            ),
            (
                input.clone(),
                CoreArray::from_elements(
                    ArrayType::new_static(DataType::I1, [2, 1]),
                    &[i1::new(-1).unwrap(), i1::new(0).unwrap()],
                )
                .unwrap(),
                windows.clone(),
                values_to_bytes(&[i64::MIN, i64::MIN, i64::MIN, 0, 1, 2]),
            ),
            (
                input.clone(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I8, [3, 1]), &[-1_i8, 0, 127]).unwrap(),
                windows
                    .clone()
                    .with_mode(GatherMode::Fill { value: Some(Box::new(CoreArray::scalar(fill).unwrap())) }),
                values_to_bytes(&[fill, fill, fill, 0, 1, 2, 127, 128, 129]),
            ),
            (
                input.clone(),
                CoreArray::from_elements(ArrayType::new_static(DataType::U64, [3, 1]), &[0_u64, 299, u64::MAX])
                    .unwrap(),
                windows.clone(),
                values_to_bytes(&[0_i64, 1, 2, i64::MIN, i64::MIN, i64::MIN, i64::MIN, i64::MIN, i64::MIN]),
            ),
            (
                input,
                CoreArray::from_elements(ArrayType::new_static(DataType::I64, [2, 1]), &[297_i64, 298]).unwrap(),
                windows,
                values_to_bytes(&[297_i64, 298, 299, i64::MIN, i64::MIN, i64::MIN]),
            ),
            (
                CoreArray::from_elements(
                    ArrayType::new_static(DataType::C64, [1]),
                    &[ComplexNumber::new(2.0_f32, -3.0)],
                )
                .unwrap(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[0_i32, 1]).unwrap(),
                GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]).with_mode(
                    GatherMode::Fill {
                        value: Some(Box::new(CoreArray::scalar(ComplexNumber::new(-0.0_f32, 7.0)).unwrap())),
                    },
                ),
                values_to_bytes(&[2.0_f32, -3.0, -0.0, 7.0]),
            ),
        ];
        for (input, indices, operation, expected) in cases {
            let mut builder = CoreProgramBuilder::<CoreArray, ArrayOperation<CoreArray>>::new();
            let input_atom = builder.add_input(input.r#type().into_owned());
            let indices_atom = builder.add_input(indices.r#type().into_owned());
            let output =
                builder.add_instruction(operation, Vec::new(), vec![input_atom, indices_atom], None).unwrap()[0];
            let program = builder
                .build::<Vec<CoreArray>, Vec<CoreArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
                .unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            let inputs = [&input, &indices]
                .into_iter()
                .map(|value| ExecutionInput {
                    buffer: Arc::new(
                        client
                            .buffer(
                                value.storage_bytes(),
                                Array::physical_buffer_type(value.r#type().data_type()),
                                value
                                    .r#type()
                                    .static_shape()
                                    .unwrap()
                                    .as_slice()
                                    .iter()
                                    .map(|extent| *extent as u64)
                                    .collect::<Vec<_>>(),
                                None,
                                device.clone(),
                                None,
                            )
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
                    &[],
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
                .remove(0)
                .outputs;
            assert_eq!(outputs[0].copy_to_host(None).unwrap().r#await().unwrap(), expected);
        }
        // Bounds checks use logical input extents rather than padded allocation capacity.
        for mode in [GatherMode::Clip, GatherMode::Fill { value: None }] {
            let input_type = ArrayType::new(
                DataType::I64,
                Shape::new(vec![CoreDimension::Dynamic(DimensionVariable::new(
                    "rows",
                    DimensionBounds::new(1, Some(5)).unwrap(),
                ))]),
            );
            let query =
                CoreDimension::Dynamic(DimensionVariable::new("query", DimensionBounds::new(0, Some(3)).unwrap()));
            let indices_type = ArrayType::new(DataType::I64, Shape::new(vec![query.clone(), CoreDimension::Static(1)]));
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let indices = builder.add_input(indices_type.clone());
            let output = builder
                .add_instruction(
                    GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
                        .with_mode(mode.clone()),
                    Vec::new(),
                    vec![input, indices],
                    None,
                )
                .unwrap()[0];
            let program = unproject_plain_program(
                builder
                    .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                        vec![output],
                        vec![Placeholder; 2],
                        vec![Placeholder],
                    )
                    .unwrap(),
            );
            let module = to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type, indices_type],
                &vec![ArrayType::new(DataType::I64, Shape::new(vec![query]))],
                "main",
                None,
                None,
            )
            .unwrap();
            let executable = client
                .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &ragged_dot_cpu_compilation_options())
                .unwrap();
            let device = executable.addressable_devices().unwrap().remove(0);
            for query_size in [2_i32, 1, 0] {
                let inputs = [
                    (values_to_bytes(&[10_i64, 20, 30, 40]), BufferType::I64, vec![4]),
                    (values_to_bytes(&[1_i64, 3]), BufferType::I64, vec![2, 1]),
                    (values_to_bytes(&[2_i32]), BufferType::I32, vec![]),
                    (values_to_bytes(&[query_size]), BufferType::I32, vec![]),
                ]
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
                        &[],
                    )
                    .unwrap()
                    .block_until_ready()
                    .unwrap()
                    .remove(0)
                    .outputs;
                let expected = if mode == GatherMode::Clip { vec![20_i64, 20] } else { vec![20, i64::MIN] };
                let bytes = outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                let actual = values_from_bytes::<i64>(&bytes[..query_size as usize * 8]);
                assert_eq!(actual, &expected[..query_size as usize]);
                assert_eq!(
                    values_from_bytes::<i64>(&outputs[1].copy_to_host(None).unwrap().r#await().unwrap()),
                    vec![i64::from(query_size)],
                );
            }
        }
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_clip_mode_gather_to_bare_op() {
        use ryft_core::{Dimension as CoreDimension, GatherDimensionNumbers, GatherMode, GatherOperation, Shape};

        // `Clip` is StableHLO `gather`'s default out-of-bounds behavior, so a `Clip`-mode gather lowers to the bare
        // `stablehlo.gather` (no extra clamp ops) just like the in-bounds default rather than erroring.
        let operand_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3), CoreDimension::Static(2)]));
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2])
            .with_mode(GatherMode::Clip);
        let mut builder = ProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(ArrayOperation::Gather(operation), Vec::new(), vec![operand, indices], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.gather"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<2x2xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_lower_indexed_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let mut builder = CoreProgramBuilder::<CoreArray, ArrayOperation<CoreArray>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [4]));
        let indices = builder.add_input(ArrayType::new_static(DataType::I32, [2, 1]));
        let updates = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let gathered = builder
            .add_instruction(
                GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
                    .with_output_sharding(sharding.clone()),
                Vec::new(),
                vec![input, indices],
                None,
            )
            .unwrap()[0];
        let scattered = builder
            .add_instruction(
                ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                    ScatterReductionKind::Add,
                )
                .with_output_sharding(sharding),
                Vec::new(),
                vec![input, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CoreArray>, Vec<CoreArray>>(
                vec![gathered, scattered],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(module.matches("sdy.sharding_constraint").count(), 2);
    }

    #[test]
    fn test_compiled_jointly_mapped_gather_over_possibly_empty_extent() {
        // Jointly mapping the input and the indices of a gather over a dynamic extent whose lower bound is zero
        // stages one gather with paired batching axes and a zero batching window. This runs the transformed program
        // on the CPU PJRT client used by the other tests (the `cutile` feature only enables compilation of the kernel
        // module) at empty and nonempty mapped extents, through both the primal and its pullback.
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(0, Some(6)).unwrap()));
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), CoreDimension::Static(4)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![extent, CoreDimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherMode::Clip);
        // The public `batch` entry runs under the static extent policy, which rejects a dynamic mapped axis before any
        // operation rule runs, so the joint rule is driven through the dynamic extent policy exactly as the ryft-core
        // reproduction does, with the mapped extent read from the staged input.
        let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let extent = inputs[0].dimension_size(0).unwrap();
                let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                    ProjectedContext::new(extent.context().clone()),
                    extent,
                );
                let batches = [
                    ArrayBatch::new(inputs[0].clone(), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(inputs[1].clone(), BatchAxis::new(0)).unwrap(),
                ];
                let outputs = operation.batch(&context, &EmptyRegionDriver, &batches).unwrap();
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                vec![outputs.into_iter().next().unwrap().into_value()]
            },
            vec![input_type, indices_type],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let program = staged.source_program().program().clone();
        let rendered = program.to_string();
        assert_eq!(
            rendered,
            indoc! {"
                lambda %0:f64[n, 4], %1:i32[n, 1] .
                let %2:f64[n] = gather [
                    dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                    slice_sizes=[0, 1],
                    mode=clip,
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert!(rendered.contains("slice_sizes=[0, 1]"));
        assert!(rendered.contains("batching=[(0, 0)]"));
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        let all_indices = [1_i32, 3, 0, 2, 3];
        for size in [0_usize, 4, 5] {
            let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
            let indices = &all_indices[..size];
            let expected =
                indices.iter().enumerate().map(|(row, &index)| values[row * 4 + index as usize]).collect::<Vec<_>>();
            let mut gradient = vec![0_f64; size * 4];
            for (row, &index) in indices.iter().enumerate() {
                gradient[row * 4 + index as usize] = 1.;
            }
            let input_type = ArrayType::new_static(DataType::F64, [size, 4]);
            let output_type = ArrayType::new_static(DataType::F64, [size]);
            let input_types =
                vec![input_type.clone(), ArrayType::new_static(DataType::I32, [size, 1]), output_type.clone()];
            let specialized = program
                .clone()
                .specialize(&input_types[..2].iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(specialized.output_types(), vec![ArrayIrType::from(output_type.clone())], "n={size}");
            let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
                |inputs| {
                    let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                    let cotangent = inputs.pop().unwrap();
                    let context = inputs[0].context().clone();
                    let mut outputs = linearization
                        .primal()
                        .interpret_in_context(&context, inputs)
                        .unwrap_or_else(|error| panic!("n={size} primal: {error}\n{}", linearization.primal()));
                    let mut pullback_inputs = vec![cotangent];
                    pullback_inputs.extend(outputs.split_off(1));
                    outputs.extend(
                        pullback
                            .interpret_in_context(&context, pullback_inputs)
                            .unwrap_or_else(|error| panic!("n={size} pullback: {error}\n{pullback}")),
                    );
                    outputs
                        .into_iter()
                        .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                        .collect()
                },
                input_types.clone(),
                &domain,
                mesh.clone(),
            )
            .unwrap_or_else(|error| panic!("n={size} native trace: {error}"));
            let bytes = [values_to_bytes(&values), values_to_bytes(indices), values_to_bytes(&vec![1_f64; size])];
            let inputs = input_types
                .into_iter()
                .zip(bytes)
                .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
                .collect();
            let outputs = domain.interpret(&compiled.executable_function(), inputs).unwrap();
            assert_eq!(outputs.len(), 2);
            for output in &outputs {
                assert_eq!(output.data_type(), DataType::F64);
                assert_eq!(output.mesh(), &mesh);
            }
            assert_eq!(outputs[0].shape().as_slice(), output_type.static_shape().unwrap().as_slice());
            assert_eq!(read_f64s(&client, &outputs[0]), expected, "n={size}");
            assert_eq!(outputs[1].shape().as_slice(), &[size, 4]);
            assert_eq!(read_f64s(&client, &outputs[1]), gradient, "pullback, n={size}");
        }
    }
}
