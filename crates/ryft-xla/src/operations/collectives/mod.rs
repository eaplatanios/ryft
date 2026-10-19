use ryft_core::{ArrayType, DataType, Dimension, ProgramError, SHARD_MAP_OPERATION_NAME};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::LoweringError;
use crate::experimental::lowering::CollectiveLoweringState;
use crate::experimental::lowering::types::lower_tensor_type;

/// StableHLO lowering of `axis_index`.
pub(crate) mod axis_index;
/// StableHLO lowering of `parallel_all_gather`.
pub(crate) mod parallel_all_gather;
/// StableHLO lowering of `parallel_all_to_all`.
pub(crate) mod parallel_all_to_all;
/// StableHLO lowering of `parallel_permute`.
pub(crate) mod parallel_permute;
/// StableHLO lowering of `parallel_ragged_all_to_all`.
pub(crate) mod parallel_ragged_all_to_all;
/// StableHLO lowering of `parallel_reduce`.
pub(crate) mod parallel_reduce;
/// StableHLO lowering of `parallel_sum_scatter`.
pub(crate) mod parallel_sum_scatter;
/// StableHLO lowering of `parallel_vary`.
pub(crate) mod parallel_vary;

/// Resolves the replica groups of one named mesh axis inside the innermost enclosing `shard_map` manual region,
/// returning the groups together with the axis size. Devices are linearized row-major over the mesh axes, so the
/// devices along the named axis with all other coordinates fixed form one replica group: the group base ids
/// enumerate the other axes' coordinate combinations, and the group members step by the named axis's row-major
/// stride. Errors outside manual regions and for axes the region does not bind as manual mesh axes.
pub(crate) fn mesh_axis_replica_groups(
    collective_state: &CollectiveLoweringState,
    axis_name: &str,
) -> Result<(Vec<Vec<usize>>, usize), LoweringError> {
    let Some(shard_map) = collective_state.manual_shard_map() else {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "collective over axis `{axis_name}` can only be lowered inside a \
                     {SHARD_MAP_OPERATION_NAME} manual region",
            ),
        }
        .into());
    };
    let axis_region = collective_state.manual_axis_region(axis_name);
    let mesh = axis_region.unwrap_or(shard_map).mesh();
    if axis_region.is_none() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "collective over axis `{axis_name}` cannot lower inside this shard_map manual region because the \
                region does not bind that axis as a manual mesh axis",
            ),
        }
        .into());
    }
    let axis_index = mesh.axis_index(axis_name).unwrap();
    let axis_size = mesh.axis_size(axis_name).unwrap();
    let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
    let stride: usize = axis_sizes[axis_index + 1..].iter().product();
    let device_count: usize = axis_sizes.iter().product();
    let mut replica_groups: Vec<Vec<usize>> = Vec::with_capacity(device_count / axis_size);
    for device in 0..device_count {
        // A device is a group base exactly when its coordinate along the named axis is zero.
        if (device / stride) % axis_size == 0 {
            replica_groups.push((0..axis_size).map(|position| device + position * stride).collect());
        }
    }
    Ok((replica_groups, axis_size))
}

/// Resolves the physical replica groups for one collective, expanding each logical subgroup within every fixed
/// coordinate of the enclosing mesh's other axes.
pub(crate) fn collective_replica_groups(
    collective_state: &CollectiveLoweringState,
    axis_name: &str,
    axis_size: usize,
    axis_index_groups: Option<&[Vec<usize>]>,
) -> Result<(Vec<Vec<usize>>, usize), LoweringError> {
    let (mesh_groups, mesh_axis_size) = mesh_axis_replica_groups(collective_state, axis_name)?;
    if axis_size != mesh_axis_size {
        return Err(ProgramError::MalformedProgram(format!(
            "collective over axis `{axis_name}` records size {axis_size}, but the enclosing mesh axis has size \
             {mesh_axis_size}",
        ))
        .into());
    }
    let Some(axis_index_groups) = axis_index_groups else {
        return Ok((mesh_groups, mesh_axis_size));
    };
    let group_size = axis_index_groups.first().map(Vec::len).unwrap_or(0);
    let mut replica_groups = Vec::with_capacity(mesh_groups.len() * axis_index_groups.len());
    for mesh_group in mesh_groups {
        for axis_index_group in axis_index_groups {
            replica_groups.push(
                axis_index_group
                    .iter()
                    .map(|&index| {
                        mesh_group.get(index).copied().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "collective over axis `{axis_name}` has group member {index} outside axis size \
                                 {mesh_axis_size}",
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    Ok((replica_groups, group_size))
}

/// Removes one statically singleton tensor axis while preserving every remaining dynamic dimension.
pub(crate) fn collapse_singleton_axis<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let output_type = lower_tensor_type(output_array_type, context, location)?;
    let reshape = block.append_operation(stable_hlo::reshape_with_output_type(value, output_type, location)?)?;
    Ok(reshape.result(0).expect("stablehlo.reshape should return one result").as_ref())
}

/// Applies one axis-ordered explicit dimension operand per result axis to one native collective result.
pub(crate) fn refine_collective_result_dimensions<'b, 'c: 'b, 't: 'c>(
    mut result: ValueRef<'b, 'c, 't>,
    extents: &[ValueRef<'b, 'c, 't>],
    output_type: &ArrayType,
    block: &mut ryft_mlir::BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: ryft_mlir::LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if output_type.rank() != extents.len() {
        return Err(ProgramError::InvalidInputCount { expected: output_type.rank(), actual: extents.len() }.into());
    }
    let i32_scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    for (axis, &extent) in extents.iter().enumerate() {
        if !matches!(output_type.shape().dimensions()[axis], Dimension::Dynamic(_)) {
            continue;
        }
        let extent = block.append_operation(stable_hlo::convert(extent, i32_scalar_type, location)?)?;
        let extent = extent.result(0).expect("stablehlo.convert should return one result").as_ref();
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            result,
            extent,
            output_tensor_type,
            axis,
            location,
        )?)?;
        result = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayIrBatch, ArrayOperation, BatchAxis, BatchingContext, BatchingTracer, DimensionBounds,
        DimensionConstant, DimensionType, DimensionVariable, OneLike, ParallelAllToAllOperation,
        ParallelPermuteOperation, ParallelReduce, Placeholder, RaggedAxis, Reduce, ReductionKind, Shape, ShardMap,
        Sharding, StagingContext, TracingContext,
    };
    use ryft_mlir::{Block, Location, Value as MlirValue};

    use crate::experimental::lowering::tests::{MixedValue, execute_mixed_program, test_manual_mesh, test_vector_type};
    use crate::experimental::lowering::types::lower_array_ir_type;
    use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation};
    use crate::tests::execution_client;
    use crate::{LowerableOperation, MemberLowerableOperation, XlaLowerer};

    use super::*;

    #[test]
    fn test_execute_parallel_reduce_ragged_coordinate_padding() {
        let row = DimensionVariable::new("row", DimensionBounds::new(0, Some(3)).unwrap());
        let column = DimensionVariable::new("column", DimensionBounds::new(0, Some(4)).unwrap());
        let trace = TracingContext::<XlaConstant, XlaOperation>::new();
        let value = trace.input(ArrayType::new_static(DataType::F64, [2, 2, 3]).into());
        let rows = trace.input(ArrayType::new_static(DataType::I32, [2]).into());
        let columns = trace.input(ArrayType::new_static(DataType::I32, [2, 2]).into());
        let context = BatchingContext::new(trace.clone(), trace.dimension_constant(2).unwrap())
            .with_axis_name("items".to_string());
        let input = BatchingTracer::new(
            context,
            ArrayIrBatch::new(value, BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![
                    RaggedAxis::new(1, rows, row, vec![0]),
                    RaggedAxis::new(2, columns, column, vec![0, 1]),
                ])
                .unwrap(),
        );
        let output = input
            .parallel_reduce(ReductionKind::Sum, "items")
            .unwrap()
            .one_like()
            .unwrap()
            .reduce(&[0, 1], ReductionKind::Sum)
            .unwrap();
        let program: FlatXlaProgram = trace
            .builder()
            .borrow()
            .clone()
            .build(vec![output.into_batch().into_value().atom_id().unwrap()], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 2, 3], %1:i32[2], %2:i32[2, 2] .
                let %3:dimension<2> = constant [value=2]
                    %4:dimension<2> = constant [value=2]
                    %5:dimension<2> = constant [value=2]
                    %6:dimension<3> = constant [value=3]
                    %7:i32[2] = iota [type=i32[2], dimension=0]
                    %8:i32[2, 2, 3] = broadcast [output_axes=[1]] %7 %4 %5 %6
                    %9:i32[2, 2, 3] = broadcast [output_axes=[0]] %1 %4 %5 %6
                    %10:bool[2, 2, 3] = compare [direction=LessThan] %8 %9
                    %11:i32[3] = iota [type=i32[3], dimension=0]
                    %12:i32[2, 2, 3] = broadcast [output_axes=[2]] %11 %4 %5 %6
                    %13:i32[2, 2, 3] = broadcast [output_axes=[0, 1]] %2 %4 %5 %6
                    %14:bool[2, 2, 3] = compare [direction=LessThan] %12 %13
                    %15:bool[2, 2, 3] = and %10 %14
                    %16:f64[2, 2, 3] = zero_like %0
                    %17:f64[2, 2, 3] = select %15 %0 %16
                    %18:f64[2, 3] = reduce [kind=sum, axes=[0]] %17
                    %19:i32[] = reduce [kind=max, axes=[0]] %1
                    %20:dimension<2> = constant [value=2]
                    %21:dimension<2> = constant [value=2]
                    %22:i32[2] = iota [type=i32[2], dimension=0]
                    %23:i32[2, 2] = broadcast [output_axes=[1]] %22 %20 %21
                    %24:i32[2, 2] = broadcast [output_axes=[0]] %1 %20 %21
                    %25:bool[2, 2] = compare [direction=LessThan] %23 %24
                    %26:i32[2, 2] = zero_like %2
                    %27:i32[2, 2] = select %25 %2 %26
                    %28:i32[2] = reduce [kind=max, axes=[0]] %27
                    %29:i32[] = constant [value=0]
                    %30:i32[2] = max %28 %29
                    %31:f64[2, 3] = one_like %18
                    %32:dimension<2> = constant [value=2]
                    %33:dimension<3> = constant [value=3]
                    %34:i32[2] = iota [type=i32[2], dimension=0]
                    %35:i32[2, 3] = broadcast [output_axes=[0]] %34 %32 %33
                    %36:i32[2, 3] = broadcast [output_axes=[]] %19 %32 %33
                    %37:bool[2, 3] = compare [direction=LessThan] %35 %36
                    %38:i32[3] = iota [type=i32[3], dimension=0]
                    %39:i32[2, 3] = broadcast [output_axes=[1]] %38 %32 %33
                    %40:i32[2, 3] = broadcast [output_axes=[0]] %30 %32 %33
                    %41:bool[2, 3] = compare [direction=LessThan] %39 %40
                    %42:bool[2, 3] = and %37 %41
                    %43:f64[2, 3] = zero_like %31
                    %44:f64[2, 3] = select %42 %31 %43
                    %45:f64[] = reduce [kind=sum, axes=[0, 1]] %44
                in (%45)"},
        );

        // Only live coordinate entries can define the result's extents. Count its live elements after replacing
        // its data with ones, so a zero-valued padded result cannot hide an incorrectly enlarged extent.
        let client = execution_client();
        let mut inputs = [
            MixedValue::Array(vec![1.0; 12], vec![2, 2, 3]),
            MixedValue::Integers(vec![1, 2], vec![2]),
            MixedValue::Integers(vec![1, 0, 1, 1], vec![2, 2]),
        ];
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[]),
            Ok(vec![MixedValue::Array(vec![2.0], vec![])]),
        );
        inputs[2] = MixedValue::Integers(vec![1, 3, 1, 1], vec![2, 2]);
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[]),
            Ok(vec![MixedValue::Array(vec![2.0], vec![])]),
        );
        inputs[1] = MixedValue::Integers(vec![0, 0], vec![2]);
        assert_eq!(
            execute_mixed_program(&client, &program, &inputs, &[]),
            Ok(vec![MixedValue::Array(vec![0.0], vec![])]),
        );
    }

    #[test]
    fn test_manual_collective_lowering_revalidates_raw_bindings() {
        use ryft_core::CollectiveOptions;

        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(
            ShardMap::new(mesh.clone(), vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap(),
        );
        let exchange = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        let permute = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let incompatible_mesh = test_manual_mesh("x", 3);
        for operation in [
            ArrayOperation::<Array>::ParallelAllToAll(exchange.clone()),
            ArrayOperation::ParallelPermute(permute.clone()),
            ArrayOperation::ParallelAllToAll(exchange.with_mesh(incompatible_mesh.clone())),
            ArrayOperation::ParallelPermute(permute.with_mesh(incompatible_mesh)),
        ] {
            for sharding in [&replicated, &varying, &unreduced] {
                let input_type = test_vector_type(2).with_sharding(sharding.clone()).unwrap();
                // Record the output of the mesh-free payload, reproducing raw binding without the capability's
                // checked mesh metadata. Varying values remain a supported raw-bind path.
                let output_types = vec![input_type.clone()];
                let context = MlirContext::new();
                let location = context.unknown_location();
                let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
                let block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap().as_ref();
                let mut lowerer: XlaLowerer<'_, '_, '_, XlaOperation> =
                    XlaLowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state.clone());
                let result = operation.lower(&mut lowerer, &[input], &[input_type], &output_types);
                let matching_mesh = match &operation {
                    ArrayOperation::ParallelAllToAll(operation) => operation.mesh().is_none(),
                    ArrayOperation::ParallelPermute(operation) => operation.mesh().is_none(),
                    _ => unreachable!(),
                };
                assert_eq!(result.is_ok(), matching_mesh && sharding == &varying, "{operation}: {result:?}");
                if result.is_err() {
                    // Reject the contract before allocating a channel or emitting a communication operation.
                    assert_eq!(state.channel_ids.get(), 1);
                } else {
                    state.channel_ids.set(1);
                }
            }
        }
    }

    #[test]
    fn test_array_ir_manual_collective_lowering_revalidates_dynamic_raw_bindings() {
        use ryft_core::{CollectiveOptions, MemberOperation};

        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(
            ShardMap::new(mesh, vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap(),
        );
        let extent = DimensionType::new("items", DimensionBounds::new(0, Some(9)).unwrap());
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        for sharding in [&replicated, &varying, &unreduced] {
            let input_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.dimension().clone()]))
                .with_sharding(sharding.clone())
                .unwrap();
            let input_types = vec![input_type.into(), extent.clone().into()];
            let output_types = operation.infer_parent_output_types(&input_types, &[]).unwrap();
            let context = MlirContext::new();
            let location = context.unknown_location();
            let mlir_types = input_types
                .iter()
                .map(|r#type| (lower_array_ir_type(r#type, &context, location).unwrap(), location))
                .collect::<Vec<_>>();
            let block = context.block(&mlir_types);
            let inputs =
                (0..input_types.len()).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
            let mut lowerer: XlaLowerer<'_, '_, '_, XlaOperation> =
                XlaLowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state.clone());
            let result = operation.lower_in_parent(&mut lowerer, &inputs, &input_types, &output_types);
            assert_eq!(result.is_ok(), sharding == &varying, "{sharding}: {result:?}");
            if result.is_err() {
                assert_eq!(state.channel_ids.get(), 1);
            } else {
                state.channel_ids.set(1);
            }
        }
    }

    #[test]
    fn test_collective_mesh_validation_uses_the_nearest_manual_binder() {
        let outer_mesh = test_manual_mesh("x", 2);
        let inner_mesh = test_manual_mesh("x", 3);
        let outer = ShardMap::new(outer_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let inner = ShardMap::new(inner_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(outer).enter_manual_region(inner);
        assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 3, None).unwrap(), &inner_mesh);
        assert!(state.checked_collective_mesh("parallel_permute", "x", 3, Some(&outer_mesh)).is_err());
        assert!(matches!(
            state.checked_collective_mesh("parallel_permute", "x", 2, Some(&outer_mesh)),
            Err(LoweringError::Tracing(ProgramError::MalformedProgram(message)))
                if message == "collective over axis `x` records size 2, but the enclosing mesh axis has size 3",
        ));
    }
}
