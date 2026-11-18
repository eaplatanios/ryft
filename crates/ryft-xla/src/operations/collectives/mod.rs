use ryft_core::{ArrayType, CollectiveAxes, Dimension as CoreDimension, ProgramError, SHARD_MAP_OPERATION_NAME};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::Error;
use crate::lowering::CollectiveLoweringState;
use crate::lowering::types::lower_tensor_type;

/// StableHLO lowering of `axis_index`.
pub(crate) mod axis_index;
/// StableHLO lowering of `parallel_all_gather`.
pub(crate) mod parallel_all_gather;
/// StableHLO lowering of `parallel_all_to_all`.
pub(crate) mod parallel_all_to_all;
/// StableHLO lowering of `parallel_broadcast`.
pub(crate) mod parallel_broadcast;
/// StableHLO lowering of `parallel_cast`.
pub(crate) mod parallel_cast;
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

/// Resolves the replica groups of the named mesh axes `axis_name` inside the enclosing `shard_map` manual regions,
/// returning the groups together with the combined axis size. Devices are linearized row-major over the mesh axes, so
/// the devices whose coordinates differ only along the named axes form one replica group. The group bases enumerate
/// the coordinate combinations of the other axes, and the group members are ordered by their positions along the
/// combined axis, which linearizes the named axes in the given order with the first name major (refer to
/// [`CollectiveAxes`]). Returns an error outside manual regions, for axes that no enclosing region binds as manual mesh
/// axes, and for axes whose binding regions have meshes with different topologies. The groups depend only on the names
/// and sizes of the mesh axes in order, so regions whose meshes differ only in axis types (e.g., a nested region inside
/// `auto_axes` or `explicit_axes`, which retype a free axis) bind axes of one collective.
fn mesh_axis_replica_groups<A: Into<CollectiveAxes>>(
    collective_state: &CollectiveLoweringState,
    axis_name: A,
) -> Result<(Vec<Vec<usize>>, usize), Error> {
    let axis_names = axis_name.into();
    if axis_names.names().is_empty() {
        return Err(ProgramError::MalformedProgram("collective requires at least one axis name".to_string()).into());
    }
    if collective_state.manual_shard_map().is_none() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "collective over axis `{axis_names}` can only be lowered inside a `{SHARD_MAP_OPERATION_NAME}` manual \
                 region",
            ),
        }
        .into());
    }
    let mut mesh = None;
    for axis_name in axis_names.names() {
        let Some(axis_region) = collective_state.manual_axis_region(axis_name) else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "collective over axis `{axis_name}` cannot lower inside this `{SHARD_MAP_OPERATION_NAME}` manual \
                     region because the region does not bind that axis as a manual mesh axis",
                ),
            }
            .into());
        };
        match mesh {
            None => mesh = Some(axis_region.mesh()),
            Some(mesh) if mesh.has_same_topology(axis_region.mesh()) => {}
            Some(_) => {
                return Err(ProgramError::MalformedProgram(format!(
                    "collective over axes `{axis_names}` spans manual regions whose meshes have different topologies",
                ))
                .into());
            }
        }
    }
    let mesh = mesh.unwrap();
    let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
    let strides = (0..axis_sizes.len()).map(|axis| axis_sizes[axis + 1..].iter().product()).collect::<Vec<usize>>();
    let named_axes = axis_names.names().iter().map(|name| mesh.axis_index(name).unwrap()).collect::<Vec<_>>();
    let axis_size: usize = named_axes.iter().map(|axis| axis_sizes[*axis]).product();
    let device_count: usize = axis_sizes.iter().product();
    let mut replica_groups: Vec<Vec<usize>> = Vec::with_capacity(device_count / axis_size);
    for device in 0..device_count {
        // A device is a group base exactly when its coordinates along the named axes are zero.
        if named_axes.iter().all(|axis| (device / strides[*axis]) % axis_sizes[*axis] == 0) {
            replica_groups.push(
                (0..axis_size)
                    .map(|mut position| {
                        let mut member = device;
                        for axis in named_axes.iter().rev() {
                            member += (position % axis_sizes[*axis]) * strides[*axis];
                            position /= axis_sizes[*axis];
                        }
                        member
                    })
                    .collect(),
            );
        }
    }
    Ok((replica_groups, axis_size))
}

/// Resolves the physical replica groups for one collective, expanding each logical subgroup within every fixed
/// coordinate of the enclosing mesh's other axes.
pub(crate) fn collective_replica_groups<A: Into<CollectiveAxes>>(
    collective_state: &CollectiveLoweringState,
    axis_name: A,
    axis_size: usize,
    axis_index_groups: Option<&[Vec<usize>]>,
) -> Result<(Vec<Vec<usize>>, usize), Error> {
    let axis_name = axis_name.into();
    let (mesh_groups, mesh_axis_size) = mesh_axis_replica_groups(collective_state, &axis_name)?;
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
fn collapse_singleton_axis<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let output_type = lower_tensor_type(output_array_type, context, location)?;
    let reshape = block.append_operation(stable_hlo::reshape_with_output_type(value, output_type, location)?)?;
    Ok(reshape.result(0).unwrap().as_ref())
}

/// Applies one axis-ordered explicit dimension operand per result axis to one native collective result.
fn refine_collective_result_dimensions<'b, 'c: 'b, 't: 'c>(
    mut result: ValueRef<'b, 'c, 't>,
    extents: &[ValueRef<'b, 'c, 't>],
    output_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    if output_type.rank() != extents.len() {
        return Err(ProgramError::InvalidInputCount { expected: output_type.rank(), actual: extents.len() }.into());
    }
    let i32_scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    for (axis, &extent) in extents.iter().enumerate() {
        if !matches!(output_type.shape().dimensions()[axis], CoreDimension::Dynamic(_)) {
            continue;
        }
        let extent = block.append_operation(stable_hlo::convert(extent, i32_scalar_type, location)?)?;
        let extent = extent.result(0).unwrap().as_ref();
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            result,
            extent,
            output_tensor_type,
            axis,
            location,
        )?)?;
        result = refined.result(0).unwrap().as_ref();
    }
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, Array as CoreArray, ArrayOperation, AxisIndex, CollectiveOptions, ConvertElementType, DataType, Device,
        DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Operation as CoreOperation, ParallelAllGather,
        ParallelAllGatherOutputVariance, ParallelAllToAll, ParallelAllToAllOperation, ParallelCastOperation,
        ParallelCastTarget, ParallelPermute, ParallelPermuteOperation, ParallelReduce, ParallelSumScatter,
        ParallelVaryOperation, ReductionKind, Shape, ShardMapBoundary, ShardMapOptions, Sharding, ShardingDimension,
        TypeError, Value as CoreValue, auto_axes, shard_map,
    };
    use ryft_mlir::{Block, Location, Value as MlirValue};
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::tests::{
        shard_map_module, static_sharded_array_type, test_manual_mesh, test_spmd_compilation_options, test_vector_type,
    };
    use crate::operations::Operation;
    use crate::operations::sharding::shard_map::tests::execute_inputs_on_cpu_devices;
    use crate::sessions::tests::session_with_client;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayTracer, ExecutionArguments, FromPjrt, LowerableOperation, Lowerer, TracedProgram, trace,
        trace_with_context,
    };

    use super::*;

    /// Lowers a local state cast and verifies that it remains the identity on the input SSA value without allocating
    /// a collective channel or emitting a new instruction.
    fn lower_state_cast(
        operation: &ArrayOperation<CoreArray>,
        state: CollectiveLoweringState,
        input_type: ArrayType,
        output_types: &[ArrayType],
    ) -> Result<(), Error> {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(&input_type, &context, location)?;
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut lowerer: Lowerer<'_, '_, '_, Operation> =
            Lowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state.clone());
        let result = operation.lower(&mut lowerer, &[input], &[input_type], output_types);
        assert_eq!(state.next_channel_id(), 1);
        assert!(block.operations()?.collect::<Result<Vec<_>, _>>()?.is_empty());
        let outputs = result?;
        assert_eq!(outputs, vec![input]);
        Ok(())
    }

    #[test]
    fn test_mesh_axis_replica_groups() {
        // Devices are linearized row-major, so device `3 * x + y` has coordinates `(x, y)`.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(mesh, Vec::new(), Vec::new(), Vec::new()).unwrap());
        assert_eq!(mesh_axis_replica_groups(&state, "x").unwrap(), (vec![vec![0, 3], vec![1, 4], vec![2, 5]], 2));
        assert_eq!(mesh_axis_replica_groups(&state, "y").unwrap(), (vec![vec![0, 1, 2], vec![3, 4, 5]], 3));

        // The first name of a combined axis is major.
        assert_eq!(mesh_axis_replica_groups(&state, ["x", "y"]).unwrap(), (vec![vec![0, 1, 2, 3, 4, 5]], 6));
        assert_eq!(mesh_axis_replica_groups(&state, ["y", "x"]).unwrap(), (vec![vec![0, 3, 1, 4, 2, 5]], 6));
    }

    #[test]
    fn test_mesh_axis_replica_groups_execute_multi_axis_collectives_on_cpu() {
        // Over several manual axes, the participants are positioned along the combined axis, which linearizes the axes
        // in the given order with the first axis major. Every device holds its tile of `0..n`, sharded along
        // `("x", "y")` on a 2x2 mesh, so the expected values are those of JAX's collectives with tuple axis names
        // (`.tasks/shard_map_jax_parity_gaps/m3_oracles.py`).
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let run = |n: usize, body: &dyn Fn(ArrayTracer) -> Result<ArrayTracer, ProgramError>| {
            let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
                |_, x: ArrayTracer| {
                    Ok(shard_map(
                        |x: ArrayTracer| body(x),
                        x,
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                        ShardMapOptions::new(),
                    )?)
                },
                ArrayType::new_static(DataType::F32, [n]).with_sharding(sharded.clone()).unwrap(),
            )
            .unwrap();
            let tile = n / 4;
            let values = (0..n).map(|value| value as f32).collect::<Vec<_>>();
            let tiles: [&[f32]; 4] = std::array::from_fn(|device| &values[device * tile..(device + 1) * tile]);
            execute_inputs_on_cpu_devices(traced.to_mlir_module("main").unwrap(), &[(&[tile as u64], tiles)]).concat()
        };
        for (axes, scatter, all_to_all, gather) in [
            (
                ["x", "y"],
                vec![24.0, 28.0, 32.0, 36.0],
                vec![0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0],
                [0.0, 1.0, 2.0, 3.0],
            ),
            (
                ["y", "x"],
                vec![24.0, 32.0, 28.0, 36.0],
                vec![0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0],
                [0.0, 2.0, 1.0, 3.0],
            ),
        ] {
            assert_eq!(run(16, &|x| x.parallel_sum_scatter_tiled(axes, 0)), scatter);
            assert_eq!(run(16, &|x| x.parallel_all_to_all_tiled(axes, 0, 0)), all_to_all);
            assert_eq!(run(4, &|x| x.parallel_all_gather_tiled(axes, 0)), gather.repeat(4));
        }
        let pairs = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        assert_eq!(run(4, &|x| x.parallel_permute(["x", "y"], pairs.clone())), vec![3.0, 0.0, 1.0, 2.0]);
        assert_eq!(run(4, &|x| x.parallel_permute(["y", "x"], pairs.clone())), vec![3.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn test_mesh_axis_replica_groups_rejects_unresolvable_axes() {
        let outer_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let inner_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(outer_mesh, Vec::new(), Vec::new(), Vec::new()).unwrap())
            .enter_manual_region(ShardMapBoundary::new(inner_mesh, Vec::new(), Vec::new(), Vec::new()).unwrap());
        assert!(matches!(
            mesh_axis_replica_groups(&state, Vec::<String>::new()),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if message == "collective requires at least one axis name",
            ),
        ));
        assert!(matches!(
            mesh_axis_replica_groups(&state, ["x", "y"]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if message
                        == "collective over axes `(x, y)` spans manual regions whose meshes have different topologies",
            ),
        ));
        assert!(matches!(
            mesh_axis_replica_groups(&state, "z"),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if message
                        == "collective over axis `z` cannot lower inside this `shard_map` manual region because the \
                            region does not bind that axis as a manual mesh axis",
            ),
        ));
        assert!(matches!(
            mesh_axis_replica_groups(&CollectiveLoweringState::new(), "x"),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if message == "collective over axis `x` can only be lowered inside a `shard_map` manual region",
            ),
        ));
    }

    #[test]
    fn test_collective_replica_groups() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(mesh, Vec::new(), Vec::new(), Vec::new()).unwrap());
        assert_eq!(
            collective_replica_groups(&state, "y", 4, None).unwrap(),
            (vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]], 4),
        );

        // Logical participant groups expand within every fixed coordinate of the other axes, in order.
        assert_eq!(
            collective_replica_groups(&state, "y", 4, Some(&[vec![0, 2], vec![3, 1]])).unwrap(),
            (vec![vec![0, 2], vec![3, 1], vec![4, 6], vec![7, 5]], 2),
        );
        assert!(matches!(
            collective_replica_groups(&state, "y", 2, None),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if message == "collective over axis `y` records size 2, but the enclosing mesh axis has size 4",
            ),
        ));
        assert!(matches!(
            collective_replica_groups(&state, "y", 4, Some(&[vec![0, 4]])),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if message == "collective over axis `y` has group member 4 outside axis size 4",
            ),
        ));
    }

    #[test]
    fn test_collective_replica_groups_execute_grouped_shape_changing_collectives_on_cpu() {
        let client = cpu_client(4);
        let session = session_with_client(client);
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, (ArrayType, ArrayType, ArrayType)> = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            let options =
                                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
                            Ok((
                                local_x.parallel_all_gather_with_options(
                                    "x",
                                    0,
                                    options.clone(),
                                    ParallelAllGatherOutputVariance::Varying,
                                )?,
                                local_x.clone().parallel_sum_scatter_with_options("x", 0, options.clone())?,
                                local_x.parallel_all_to_all_with_options("x", 0, 0, options)?,
                            ))
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        (sharding.clone(), sharding.clone(), sharding.clone()),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(16)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(module.matches("replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>").count(), 3, "{module}");

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = (0..4).map(|offset| (device_index * 4 + offset) as f32).collect::<Vec<_>>();
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[16], sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            ExecutionArguments::from_arrays(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(
                execute_arguments.as_execution_device_inputs(),
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap()
            .block_until_ready()
            .unwrap();

        let expected_gather = [
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0],
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let expected_scatter = [vec![8.0, 10.0], vec![20.0, 22.0], vec![12.0, 14.0], vec![16.0, 18.0]];
        let expected_parallel_all_to_all = [
            vec![0.0, 1.0, 8.0, 9.0],
            vec![14.0, 15.0, 6.0, 7.0],
            vec![2.0, 3.0, 10.0, 11.0],
            vec![12.0, 13.0, 4.0, 5.0],
        ];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 3);
            let actual = output
                .outputs
                .into_iter()
                .map(|buffer| {
                    let bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                vec![
                    expected_gather[device_index].clone(),
                    expected_scatter[device_index].clone(),
                    expected_parallel_all_to_all[device_index].clone(),
                ],
            );
        }
    }

    #[test]
    fn test_manual_state_cast_lowering_revalidates_bindings() {
        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let checked = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        for (operation, input_sharding) in [
            (
                ArrayOperation::<CoreArray>::ParallelVary(ParallelVaryOperation::new("x".to_string())),
                replicated.clone(),
            ),
            (
                ArrayOperation::ParallelCast(ParallelCastOperation::new("x".to_string(), ParallelCastTarget::Reduced)),
                replicated.clone(),
            ),
            (
                ArrayOperation::ParallelCast(ParallelCastOperation::new("x".to_string(), ParallelCastTarget::Varying)),
                replicated.clone().with_reduced_axes(["x"]).unwrap(),
            ),
            (
                ArrayOperation::ParallelCast(ParallelCastOperation::new(
                    "x".to_string(),
                    ParallelCastTarget::Unreduced,
                )),
                replicated.clone().with_varying_manual_axes(["x"]).unwrap(),
            ),
        ] {
            let input_type = test_vector_type(2).with_sharding(input_sharding).unwrap();
            let output_types = operation.infer_output_types(std::slice::from_ref(&input_type), &[]).unwrap();
            let state = CollectiveLoweringState::new().enter_manual_region(checked.clone());
            lower_state_cast(&operation, state, input_type.clone(), &output_types).unwrap();

            // A raw cast is tracked even though its payload carries no mesh. An unchecked frontend would omit it.
            let state =
                CollectiveLoweringState::new().enter_manual_region(checked.clone().with_check_manual_variation(false));
            assert!(matches!(
                lower_state_cast(&operation, state, input_type.clone(), &output_types),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                        if *message == format!(
                            "`{}` cannot use tracked mesh metadata inside an unchecked manual binding",
                            operation.name(),
                        ),
                ),
            ));

            let state = CollectiveLoweringState::new().enter_manual_region(checked.clone());
            assert!(matches!(
                lower_state_cast(&operation, state, input_type.clone(), std::slice::from_ref(&input_type)),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                        if *message == format!(
                            "`{}` output types do not match the enclosing manual mesh contract",
                            operation.name(),
                        ),
                ),
            ));

            // The input's mesh participates in the binding contract even when the axis has the same size.
            let foreign_mesh = LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap();
            let foreign = ShardMapBoundary::new(foreign_mesh, Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
            let state = CollectiveLoweringState::new().enter_manual_region(foreign);
            assert!(matches!(
                lower_state_cast(&operation, state, input_type.clone(), &output_types),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                        if *message == format!(
                            "`{}` operation mesh does not match the enclosing manual axis `x` mesh",
                            operation.name(),
                        ),
                ),
            ));

            let state = CollectiveLoweringState::new();
            assert!(matches!(
                lower_state_cast(&operation, state, input_type, &output_types),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                        if *message
                            == "collective over axis `x` can only be lowered inside a `shard_map` manual region",
                ),
            ));
        }
    }

    #[test]
    fn test_manual_state_cast_lowering_uses_owning_binding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let outer = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
        let inner = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["y".to_string()]).unwrap();
        let input_type = test_vector_type(2).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        for operation in [
            ArrayOperation::<CoreArray>::ParallelVary(ParallelVaryOperation::new("x".to_string())),
            ArrayOperation::ParallelCast(ParallelCastOperation::new("x".to_string(), ParallelCastTarget::Reduced)),
        ] {
            let output_types = operation.infer_output_types(std::slice::from_ref(&input_type), &[]).unwrap();
            let state = CollectiveLoweringState::new()
                .enter_manual_region(outer.clone())
                .enter_manual_region(inner.clone().with_check_manual_variation(false));
            lower_state_cast(&operation, state, input_type.clone(), &output_types).unwrap();

            let state = CollectiveLoweringState::new()
                .enter_manual_region(outer.clone().with_check_manual_variation(false))
                .enter_manual_region(inner.clone());
            assert!(matches!(
                lower_state_cast(&operation, state, input_type.clone(), &output_types),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                        if *message == format!(
                            "`{}` cannot use tracked mesh metadata inside an unchecked manual binding",
                            operation.name(),
                        ),
                ),
            ));

            let state = CollectiveLoweringState::new().enter_manual_region(inner.clone());
            assert!(matches!(
                lower_state_cast(&operation, state, input_type.clone(), &output_types),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                        if *message
                            == "collective over axis `x` cannot lower inside this `shard_map` manual region \
                                because the region does not bind that axis as a manual mesh axis",
                ),
            ));
        }
    }

    #[test]
    fn test_manual_collective_lowering_revalidates_raw_bindings() {
        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let boundary =
            ShardMapBoundary::new(mesh.clone(), vec![replicated.clone()], vec![replicated.clone()], Vec::new())
                .unwrap();
        let exchange = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        let permute = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let incompatible_mesh = test_manual_mesh("x", 3);
        for operation in [
            ArrayOperation::<CoreArray>::ParallelAllToAll(exchange.clone()),
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
                let state = CollectiveLoweringState::new().enter_manual_region(boundary.clone());
                let mut lowerer: Lowerer<'_, '_, '_, Operation> =
                    Lowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state.clone());
                let result = operation.lower(&mut lowerer, &[input], &[input_type], &output_types);
                let matching_mesh = match &operation {
                    ArrayOperation::ParallelAllToAll(operation) => operation.mesh().is_none(),
                    ArrayOperation::ParallelPermute(operation) => operation.mesh().is_none(),
                    _ => unreachable!(),
                };
                let name = operation.name();
                let output = match &operation {
                    ArrayOperation::ParallelAllToAll(_) => "exchanged",
                    _ => "permuted",
                };
                let expected = if !matching_mesh {
                    Some(ProgramError::MalformedProgram(format!(
                        "`{name}` operation mesh does not match the enclosing manual axis `x` mesh",
                    )))
                } else if sharding == &unreduced {
                    Some(ProgramError::Type(TypeError::invalid(format!("`{name}` does not support unreduced inputs"))))
                } else if sharding == &replicated {
                    Some(ProgramError::Type(TypeError::invalid(format!(
                        "`{name}` input must vary over manual axis `x`; pass an invariant value through \
                         `parallel_vary` first so that the {output} output is typed as varying",
                    ))))
                } else {
                    None
                };
                match (result, expected) {
                    (Ok(outputs), None) => assert_eq!(outputs.len(), 1),
                    (Err(Error::Core(error)), Some(expected)) => {
                        assert_eq!(error.as_ref(), &ryft_core::Error::Program(expected));
                        // Reject the contract before allocating a channel or emitting a communication operation, so
                        // the next channel of the state is still its first one.
                        assert_eq!(state.next_channel_id(), 1);
                    }
                    (result, expected) => panic!("{operation}: expected {expected:?} but got {result:?}"),
                }
            }
        }
    }

    #[test]
    fn test_collective_lowering_accepts_free_axes_retyped_by_auto_axes() {
        // Inside a `shard_map` over `x`, `auto_axes` retypes only the free axis `y`, so the collectives over `x` that
        // it stages record the retyped mesh while their binder records the mesh with `y` explicit. Both meshes make
        // the same axes manual, so the collectives lower with the replica groups and coordinates of the binder.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2]);
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        let local_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                .unwrap();

        // `parallel_reduce` over `x`.
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| {
                Ok(auto_axes(
                    |input: ArrayTracer| input.parallel_reduce(ReductionKind::Sum, "x"),
                    input,
                    Some(&["y"]),
                    local_sharding.clone(),
                )?)
            },
            input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            local_sharding.clone(),
            ShardMapOptions::new(),
        );
        assert_eq!(
            module.unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}], replicated={"x"}>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {"y", ?}]>] out_shardings=[<@mesh, [{}, {"y", ?}], replicated={"x"}>] manual_axes={"x"} (%arg1: tensor<2x2xf32>) {
                      %1 = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]> : tensor<2x2xf32>
                      %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
                      %3 = sdy.sharding_constraint %2 <@mesh, [{}, {"y"}]> : tensor<2x2xf32>
                      sdy.return %3 : tensor<2x2xf32>
                    } : (tensor<4x2xf32>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "#},
        );

        // `axis_index` along `x`.
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| {
                Ok(auto_axes(
                    |input: ArrayTracer| {
                        let coordinate = input.domain().axis_index("x")?.convert_element_type(DataType::F32)?;
                        input.add(&coordinate)
                    },
                    input,
                    Some(&["y"]),
                    local_sharding.clone(),
                )?)
            },
            input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding.clone(),
            ShardMapOptions::new(),
        );
        assert_eq!(
            module.unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {"y", ?}]>] out_shardings=[<@mesh, [{"x"}, {"y", ?}]>] manual_axes={"x"} (%arg1: tensor<2x2xf32>) {
                      %1 = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]> : tensor<2x2xf32>
                      %2 = stablehlo.partition_id : tensor<ui32>
                      %3 = stablehlo.convert %2 : (tensor<ui32>) -> tensor<ui64>
                      %c = stablehlo.constant dense<2> : tensor<ui64>
                      %4 = stablehlo.divide %3, %c : tensor<ui64>
                      %c_0 = stablehlo.constant dense<2> : tensor<ui64>
                      %5 = stablehlo.remainder %4, %c_0 : tensor<ui64>
                      %6 = stablehlo.convert %5 : (tensor<ui64>) -> tensor<f32>
                      %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
                      %8 = stablehlo.add %1, %7 : tensor<2x2xf32>
                      %9 = sdy.sharding_constraint %8 <@mesh, [{}, {"y"}]> : tensor<2x2xf32>
                      sdy.return %9 : tensor<2x2xf32>
                    } : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    return %0 : tensor<4x2xf32>
                  }
                }
            "#},
        );

        // `parallel_all_gather` over `x`.
        let gathered_local_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let gathered_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| {
                Ok(auto_axes(
                    |input: ArrayTracer| input.parallel_all_gather("x", 0),
                    input,
                    Some(&["y"]),
                    gathered_local_sharding,
                )?)
            },
            input_type,
            mesh,
            input_sharding,
            gathered_sharding,
            ShardMapOptions::new(),
        );
        assert_eq!(
            module.unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {"y", ?}]>] out_shardings=[<@mesh, [{"x"}, {}, {"y", ?}]>] manual_axes={"x"} (%arg1: tensor<2x2xf32>) {
                      %1 = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]> : tensor<2x2xf32>
                      %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
                      %3 = "stablehlo.all_gather"(%2) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> : (tensor<1x2x2xf32>) -> tensor<2x2x2xf32>
                      %4 = sdy.sharding_constraint %3 <@mesh, [{}, {}, {"y"}]> : tensor<2x2x2xf32>
                      sdy.return %4 : tensor<2x2x2xf32>
                    } : (tensor<4x2xf32>) -> tensor<4x2x2xf32>
                    return %0 : tensor<4x2x2xf32>
                  }
                }
            "#},
        );
    }
}
