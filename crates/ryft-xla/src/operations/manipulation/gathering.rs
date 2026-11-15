use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, ComparisonDirection, DataType, Dimension as CoreDimension, GatherMode, GatherOperation, ReductionKind,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::lowering::MlirLowerableValue;
use crate::lowering::types::{
    lower_explicit_shape, lower_physical_bound_value, lower_restore_dynamic_dimensions,
    lower_runtime_dimension_size_i64, lower_static_index_constants, lower_tensor_type, lower_unplaced_constant_output,
    physical_bound_type,
};
use crate::operations::comparisons::lower_compare_to_mlir;
use crate::operations::extrema::FloatExtremumForm;
use crate::operations::reductions::build_reduce_body_region;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

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
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        if self.output_sharding().is_some() {
            lower_sharding_constraint(
                &values,
                output_types[0].sharding().unwrap(),
                lowerer.collective_state().bound_manual_axes(),
                &mut lowerer.block(),
                lowerer.location(),
            )
        } else {
            Ok(values)
        }
    }
}

/// Lowers a [`GatherOperation`] with its explicit index vector axis. [`GatherMode::Fill`] masks whole out-of-bounds
/// windows, and [`GatherMode::Clip`] clamps against logical input sizes even when the allocation is larger. Comparisons
/// preserve index signedness.
fn lower_gather_to_mlir<'b, 'c: 'b, 't: 'c>(
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
    // Native gather expects integer indices, whereas the one-bit MLIR carrier denotes a predicate. Widen every index
    // family without changing its signedness and explicitly recover the numeric value of I1 (i.e., -1 or 0).
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
    // canonical window entry when the extent proves which form is valid, which is zero for an empty axis and one for a
    // nonempty axis. A retained symbolic batch may have used zero to admit an empty extent, and leaving that entry
    // zero after specialization to a nonempty batch can make XLA replace the entire gather with zeros. If the extent
    // can still be zero or positive, retain the original entry, because changing it to one would violate the gather
    // window bound on an input that is empty at runtime. This does not add support for unspecialized batching over
    // extents that are zero at runtime.
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

    // Widen before comparing, because a legal input extent can exceed the range of an i8 or u8 index. In particular,
    // converting u64 indices to i64 would incorrectly turn its largest values into negative indices or wrap a clipping
    // operation. The bounds themselves are never negative, because type inference requires every slice size to fit the
    // input extent, or the guaranteed minimum extent of a dynamic input axis.
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
            .append_operation(stable_hlo::gather(input_values[0], clipped, attribute, &slice_sizes, false, location)?)?
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
    let reduction = build_reduce_body_region(
        ReductionKind::All,
        DataType::Boolean,
        FloatExtremumForm::TotalOrder,
        context,
        location,
    )?;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayBatch, ArrayBatchingPolicy, ArrayIrType, ArrayOperation, ArrayType, BatchAxis,
        BatchableOperation, BatchingContext, Complex as ComplexNumber, DataType, Diagonal, Dimension as CoreDimension,
        DimensionBounds, DimensionSize, DimensionVariable, DynamicArrayExtentBatchingPolicy, EmptyRegionDriver, Gather,
        GatherDimensionNumbers, GatherMode, GatherOperation, GatherOptions, LogicalMesh, MeshAxis, MeshAxisType,
        Placeholder, ProgramBuilder as CoreProgramBuilder, ProjectedContext, Reduce, ReductionKind,
        ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, Shape, Sharding, ShardingDimension, Typed,
        TypedProgram, ValueProjection, differentiate_at, i1,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::domains::tests::{domain_mesh, read_f64s};
    use crate::lowering::tests::{ProgramBuilder, ragged_dot_cpu_compilation_options, unproject_plain_program};
    use crate::lowering::to_mlir_module_for_plain_program;
    use crate::tests::{execution_client, replicated_array, values_from_bytes, values_to_bytes};
    use crate::values::ArrayConstant;
    use crate::{Array, Options, Session, to_mlir_module_for_program};

    #[test]
    fn test_gather_operation_lower() {
        // Take whole rows of a [3, 2] matrix at the row indices in a [2, 1] index array. Offset axis 1 carries the
        // row with slice sizes [1, 2], and axis 0 is collapsed because the start indices select it. Output is [2, 2].
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

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.convert %arg1 : (tensor<2x1xi32>) -> tensor<2x1xi64>
                    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<3x2xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );

        // A dynamic query batch axis remains the same array identity through gather, so it is not reconstructed from
        // a shape operand or lowered through `stablehlo.dynamic_gather`.
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
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<?x1xi32, #stablehlo.bounds<4, ?>>) -> tensor<?x2xf32, #stablehlo.bounds<4, ?>> {
                    %0 = stablehlo.convert %arg1 : (tensor<?x1xi32, #stablehlo.bounds<4, ?>>) -> tensor<?x1xi64, #stablehlo.bounds<4, ?>>
                    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<3x2xf32>, tensor<?x1xi64, #stablehlo.bounds<4, ?>>) -> tensor<?x2xf32, #stablehlo.bounds<4, ?>>
                    return %1 : tensor<?x2xf32, #stablehlo.bounds<4, ?>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_gather_operation_lower_clip_mode() {
        // `Clip` is the default out-of-bounds behavior of StableHLO `gather`, so a gather in `Clip` mode lowers to the
        // bare `stablehlo.gather` without extra clamp operations, just like the in-bounds default, rather than
        // failing.
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

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.convert %arg1 : (tensor<2x1xi32>) -> tensor<2x1xi64>
                    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2>}> : (tensor<3x2xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_gather_operation_lower_fill_windows() {
        let client = execution_client();
        let input =
            CoreArray::from_elements(ArrayType::new_static(DataType::I64, [300]), &(0i64..300).collect::<Vec<_>>())
                .unwrap();
        let fill = 9_007_199_254_740_993i64;
        let windows = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![3])
            .with_mode(GatherMode::Fill { value: None });
        let cases = [
            (
                CoreArray::scalar(7i64).unwrap(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I32, [2, 0]), &[] as &[i32]).unwrap(),
                GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![], vec![]), vec![])
                    .with_mode(GatherMode::Fill { value: None }),
                values_to_bytes(&[7i64, 7]),
            ),
            (
                CoreArray::from_elements(ArrayType::new_static(DataType::I64, [2, 3]), &[10i64, 20, 30, 40, 50, 60])
                    .unwrap(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[0i32, 2, 1, 3]).unwrap(),
                GatherOperation::new(
                    GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![(0, 0)]),
                    vec![1, 1],
                )
                .with_mode(GatherMode::Fill { value: None }),
                values_to_bytes(&[10i64, 30, 50, i64::MIN]),
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
                CoreArray::from_elements(ArrayType::new_static(DataType::I8, [3, 1]), &[-1i8, 0, 127]).unwrap(),
                windows
                    .clone()
                    .with_mode(GatherMode::Fill { value: Some(Box::new(CoreArray::scalar(fill).unwrap())) }),
                values_to_bytes(&[fill, fill, fill, 0, 1, 2, 127, 128, 129]),
            ),
            (
                input.clone(),
                CoreArray::from_elements(ArrayType::new_static(DataType::U64, [3, 1]), &[0u64, 299, u64::MAX]).unwrap(),
                windows.clone(),
                values_to_bytes(&[0i64, 1, 2, i64::MIN, i64::MIN, i64::MIN, i64::MIN, i64::MIN, i64::MIN]),
            ),
            (
                input,
                CoreArray::from_elements(ArrayType::new_static(DataType::I64, [2, 1]), &[297i64, 298]).unwrap(),
                windows,
                values_to_bytes(&[297i64, 298, 299, i64::MIN, i64::MIN, i64::MIN]),
            ),
            (
                CoreArray::from_elements(
                    ArrayType::new_static(DataType::C64, [1]),
                    &[ComplexNumber::new(2.0f32, -3.0)],
                )
                .unwrap(),
                CoreArray::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[0i32, 1]).unwrap(),
                GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]).with_mode(
                    GatherMode::Fill {
                        value: Some(Box::new(CoreArray::scalar(ComplexNumber::new(-0.0f32, 7.0)).unwrap())),
                    },
                ),
                values_to_bytes(&[2.0f32, -3.0, -0.0, 7.0]),
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
            for query_size in [2i32, 1, 0] {
                let inputs = [
                    (values_to_bytes(&[10i64, 20, 30, 40]), BufferType::I64, vec![4]),
                    (values_to_bytes(&[1i64, 3]), BufferType::I64, vec![2, 1]),
                    (values_to_bytes(&[2i32]), BufferType::I32, vec![]),
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
                let expected = if mode == GatherMode::Clip { vec![20i64, 20] } else { vec![20, i64::MIN] };
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
    fn test_gather_operation_lower_clip_batching_windows() {
        // The paired batching axis has a zero window, which lowering replaces with a window of one because the input
        // has at least one row. Clipping against the logical extents of a dynamic input gathers a second time, and
        // that gather must use the same windows.
        let rows = CoreDimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap()));
        let input_type = ArrayType::new(DataType::I64, Shape::new(vec![rows.clone(), CoreDimension::Static(4)]));
        let indices_type = ArrayType::new(DataType::I64, Shape::new(vec![rows.clone(), CoreDimension::Static(1)]));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let indices = builder.add_input(indices_type.clone());
        let output = builder
            .add_instruction(
                GatherOperation::new(
                    GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![(0, 0)]),
                    vec![0, 1],
                )
                .with_mode(GatherMode::Clip),
                Vec::new(),
                vec![input, indices],
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
        );
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type, indices_type],
            &vec![ArrayType::new(DataType::I64, Shape::new(vec![rows]))],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4x4xi64>, %arg1: tensor<4x1xi64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<?xi64, #stablehlo.bounds<4>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<4x4xi64>, tensor<i32>) -> tensor<?x4xi64, #stablehlo.bounds<4, ?>>
                    %1 = stablehlo.set_dimension_size %arg1, %arg3, dim = 0 : (tensor<4x1xi64>, tensor<i32>) -> tensor<?x1xi64, #stablehlo.bounds<4, ?>>
                    %2 = stablehlo.convert %1 : tensor<?x1xi64, #stablehlo.bounds<4, ?>>
                    %3 = "stablehlo.gather"(%0, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<?x4xi64, #stablehlo.bounds<4, ?>>, tensor<?x1xi64, #stablehlo.bounds<4, ?>>) -> tensor<?xi64, #stablehlo.bounds<4>>
                    %4 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?x1xi64, #stablehlo.bounds<4, ?>>) -> tensor<i32>
                    %c = stablehlo.constant dense<4> : tensor<i32>
                    %5 = stablehlo.set_dimension_size %2, %c, dim = 0 : (tensor<?x1xi64, #stablehlo.bounds<4, ?>>, tensor<i32>) -> tensor<4x1xi64>
                    %6 = stablehlo.iota dim = 0 : tensor<4x1xi32>
                    %7 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<4x1xi32>
                    %8 = stablehlo.compare LT, %6, %7, SIGNED : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x1xi1>
                    %c_0 = stablehlo.constant dense<0> : tensor<i64>
                    %9 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<4x1xi64>
                    %10 = stablehlo.select %8, %5, %9 : tensor<4x1xi1>, tensor<4x1xi64>
                    %c_1 = stablehlo.constant dense<4> : tensor<i64>
                    %c_2 = stablehlo.constant dense<1> : tensor<i64>
                    %11 = stablehlo.subtract %c_1, %c_2 : tensor<i64>
                    %12 = stablehlo.reshape %11 : (tensor<i64>) -> tensor<1xi64>
                    %13 = stablehlo.convert %12 : tensor<1xi64>
                    %14 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<1xi64>) -> tensor<4x1xi64>
                    %c_3 = stablehlo.constant dense<0> : tensor<i64>
                    %15 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4x1xi64>
                    %16 = stablehlo.maximum %10, %15 : tensor<4x1xi64>
                    %17 = stablehlo.minimum %16, %14 : tensor<4x1xi64>
                    %18 = "stablehlo.gather"(%0, %17) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<?x4xi64, #stablehlo.bounds<4, ?>>, tensor<4x1xi64>) -> tensor<4xi64>
                    %19 = stablehlo.get_dimension_size %18, dim = 0 : (tensor<4xi64>) -> tensor<i32>
                    %c_4 = stablehlo.constant dense<4> : tensor<i32>
                    %20 = stablehlo.set_dimension_size %18, %c_4, dim = 0 : (tensor<4xi64>, tensor<i32>) -> tensor<4xi64>
                    %21 = stablehlo.iota dim = 0 : tensor<4xi32>
                    %22 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<4xi32>
                    %23 = stablehlo.compare LT, %21, %22, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
                    %c_5 = stablehlo.constant dense<0> : tensor<i64>
                    %24 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<4xi64>
                    %25 = stablehlo.select %23, %20, %24 : tensor<4xi1>, tensor<4xi64>
                    %26 = stablehlo.get_dimension_size %3, dim = 0 : (tensor<?xi64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %27 = stablehlo.set_dimension_size %25, %26, dim = 0 : (tensor<4xi64>, tensor<i32>) -> tensor<?xi64, #stablehlo.bounds<4>>
                    %28 = stablehlo.get_dimension_size %27, dim = 0 : (tensor<?xi64, #stablehlo.bounds<4>>) -> tensor<i32>
                    %29 = stablehlo.convert %28 : (tensor<i32>) -> tensor<i64>
                    return %27, %29 : tensor<?xi64, #stablehlo.bounds<4>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_gather_operation_lower_fill_differentiation() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let input = replicated_array(&domain, &mesh, DataType::F32, &[3], &values_to_bytes(&[10f32, 20.0, 30.0]));
        let indices = replicated_array(&domain, &mesh, DataType::I32, &[4, 1], &values_to_bytes(&[-1i32, 1, 1, 3]));
        let options = GatherOptions::default()
            .with_mode(GatherMode::Fill { value: Some(Box::new(CoreArray::scalar(7f32).unwrap())) });

        // Out-of-bounds fill contributes to the primal, but never to an input cotangent. Repeated valid indices
        // accumulate their cotangents through the inverse scatter on the execution device.
        let (value, gradient) = differentiate_at(input)
            .with_captures(indices)
            .value_and_gradient(|input, indices| {
                input
                    .gather(&indices, &GatherDimensionNumbers::new(vec![], vec![0], vec![0]), &[1], &options)
                    .unwrap()
                    .reduce(&[0], ReductionKind::Sum)
                    .unwrap()
            })
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(&value.copy_to_host().unwrap()), vec![54.0]);
        assert_eq!(values_from_bytes::<f32>(&gradient.copy_to_host().unwrap()), vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_lower_gather_to_mlir_diagonal_matches_reference_backend() {
        // A diagonal is a slice of its two axes followed by one gather over iota indices, which lowers and executes
        // like the reference backend, including for offsets, axes with different extents, and empty diagonals.
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);
        let values = (0..18).map(|value| value as f32).collect::<Vec<_>>();
        let reference =
            CoreArray::from_elements(ArrayType::new_static(DataType::F32, [3, 2, 3]), values.as_slice()).unwrap();
        let input = replicated_array(&domain, &mesh, DataType::F32, &[3, 2, 3], &values_to_bytes(&values));
        for (offset, axis1, axis2) in [(0, 0, 2), (0, 2, 0), (1, 0, 2), (-1, 0, 2), (1, 1, 2), (-2, 2, 1), (3, 0, 2)] {
            let diagonal = input.diagonal(offset, axis1, axis2).unwrap();
            let reference_diagonal = reference.diagonal(offset, axis1, axis2).unwrap();
            assert_eq!(
                diagonal.r#type().shape(),
                reference_diagonal.r#type().shape(),
                "diagonal({offset}, {axis1}, {axis2})",
            );
            assert_eq!(
                diagonal.copy_to_host().unwrap(),
                reference_diagonal.logical_bytes(),
                "diagonal({offset}, {axis1}, {axis2})",
            );
        }
        assert_eq!(
            values_from_bytes::<f32>(&input.diagonal(1, 0, 2).unwrap().copy_to_host().unwrap()),
            vec![1.0, 4.0, 8.0, 11.0],
        );
    }

    #[test]
    fn test_gather_operation_lower_output_sharding() {
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
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<4xf32>) {
                    %0 = stablehlo.convert %arg1 : (tensor<2x1xi32>) -> tensor<2x1xi64>
                    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf32>, tensor<2x1xi64>) -> tensor<2xf32>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{"x"}]> : tensor<2xf32>
                    %3 = stablehlo.convert %arg1 : (tensor<2x1xi32>) -> tensor<2x1xi64>
                    %4 = "stablehlo.scatter"(%arg0, %3, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
                    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
                      %6 = stablehlo.add %arg3, %arg4 : tensor<f32>
                      stablehlo.return %6 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<2x1xi64>, tensor<2xf32>) -> tensor<4xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %2, %5 : tensor<2xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_gather_operation_lower_jointly_mapped_over_possibly_empty_extent() {
        // Jointly mapping the input and the indices of a gather over a dynamic extent whose lower bound is zero
        // stages one gather with paired batching axes and a zero batching window. This runs the transformed program
        // on the test execution client at empty and nonempty mapped extents, through both the primal and its
        // pullback.
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(0, Some(6)).unwrap()));
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), CoreDimension::Static(4)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![extent, CoreDimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherMode::Clip);
        // The public `batch` entry runs under the static extent policy, which rejects a dynamic mapped axis before any
        // operation rule runs, so the joint rule is driven through the dynamic extent policy directly, with the mapped
        // extent read from the staged input.
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
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
        assert_eq!(
            program.to_string(),
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
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        let all_indices = [1i32, 3, 0, 2, 3];
        for size in [0usize, 4, 5] {
            let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
            let indices = &all_indices[..size];
            let expected =
                indices.iter().enumerate().map(|(row, &index)| values[row * 4 + index as usize]).collect::<Vec<_>>();
            let mut gradient = vec![0f64; size * 4];
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
            let compiled = crate::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
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
            .unwrap_or_else(|error| panic!("n={size} compilation: {error}"));
            let bytes = [values_to_bytes(&values), values_to_bytes(indices), values_to_bytes(&vec![1f64; size])];
            let inputs = input_types
                .into_iter()
                .zip(bytes)
                .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
                .collect();
            let outputs = compiled.executable().execute(&domain, inputs).unwrap();
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
