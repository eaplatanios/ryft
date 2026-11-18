use ryft_core::macros::check_count;
use ryft_core::{
    AXIS_INDEX_OPERATION_NAME, ArrayType, AxisIndexOperation, DataType, ProgramError, SHARD_MAP_OPERATION_NAME,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::lowering::CollectiveLoweringState;
use crate::lowering::types::{lower_tensor_type, lower_u64_scalar_constant};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

/// Lowers one [`AxisIndexOperation`] inside a `sdy.manual_computation` region to the executing device's coordinate
/// along the named device mesh axis, as a scalar `u64`.
///
/// Devices are linearized row-major over the mesh axes, which is the same linearization that
/// [`mesh_axis_replica_groups`](crate::operations::collectives::mesh_axis_replica_groups) uses for the replica groups
/// of collectives such as
/// [`lower_collective_to_all_reduce`](crate::operations::collectives::parallel_reduce::lower_collective_to_all_reduce).
/// The coordinate along the named axis is therefore `(partition_id / stride) % size`, where `stride` is the product of
/// the mesh axis sizes after the named axis and `size` is the named axis's size. `partition_id` yields the executing
/// device's linear id, which is converted to `u64`, and the coordinate arithmetic is emitted as scalar StableHLO
/// `divide` and `remainder` operations. The division is skipped for a unit stride, and the remainder is skipped for an
/// axis that spans every device. A declared mesh must have the topology of the binding region's
/// mesh and make the same axes manual (i.e., satisfy
/// [`LogicalMesh::has_same_manual_axes`](ryft_core::LogicalMesh::has_same_manual_axes)), so that `auto_axes` and
/// `explicit_axes` may retype the region's free axes, and the coordinate arithmetic uses the region's mesh.
pub(super) fn lower_axis_index_to_coordinate<'b, 'c: 'b, 't: 'c>(
    operation: &AxisIndexOperation,
    collective_state: &CollectiveLoweringState,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let axis_name = operation.axis_name();
    if collective_state.manual_shard_map().is_none() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{AXIS_INDEX_OPERATION_NAME}` for axis `{axis_name}` can only be lowered inside a \
                 `{SHARD_MAP_OPERATION_NAME}` manual region",
            ),
        }
        .into());
    }
    let Some(axis_region) = collective_state.manual_axis_region(axis_name) else {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{AXIS_INDEX_OPERATION_NAME}` for axis `{axis_name}` cannot lower inside this \
                 `{SHARD_MAP_OPERATION_NAME}` manual region because the region does not bind that axis as a manual \
                 mesh axis",
            ),
        }
        .into());
    };
    let mesh = axis_region.mesh();
    if operation.mesh().is_some_and(|declared| !declared.has_same_manual_axes(mesh)) {
        return Err(ProgramError::MalformedProgram(
            "`axis_index` mesh does not match its active manual binding".to_string(),
        )
        .into());
    }
    let axis_index = mesh.axis_index(axis_name).unwrap();
    let axis_size = mesh.axis_size(axis_name).unwrap();
    let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
    let stride: usize = axis_sizes[axis_index + 1..].iter().product();

    let scalar_u64_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let partition_id = block.append_operation(stable_hlo::partition_id(location)?)?;
    let partition_id = partition_id.result(0).unwrap().as_ref();
    let partition_id = block.append_operation(stable_hlo::convert(partition_id, scalar_u64_type, location)?)?;
    let partition_id = partition_id.result(0).unwrap().as_ref();
    // `partition_id / stride` selects the coordinate's digit group, and `% size` isolates this axis's coordinate. A
    // unit stride and a full-size axis make the respective step an identity, so they are skipped.
    let strided = if stride == 1 {
        partition_id
    } else {
        let stride_constant = lower_u64_scalar_constant(stride as u64, block, context, location)?;
        let divided = block.append_operation(stable_hlo::divide(partition_id, stride_constant, location)?)?;
        divided.result(0).unwrap().as_ref()
    };
    if axis_size == mesh.device_count() {
        return Ok(strided);
    }
    let size_constant = lower_u64_scalar_constant(axis_size as u64, block, context, location)?;
    let coordinate = block.append_operation(stable_hlo::remainder(strided, size_constant, location)?)?;
    Ok(coordinate.result(0).unwrap().as_ref())
}

impl LowerableOperation for AxisIndexOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state().clone();
        if let Some(boundary) = collective_state.manual_axis_region(self.axis_name()) {
            if boundary.checks_manual_variation() && self.mesh().is_none() {
                return Err(ProgramError::MalformedProgram(
                    "`axis_index` requires mesh metadata inside a manual region".to_string(),
                )
                .into());
            }
            if !boundary.checks_manual_variation() && self.mesh().is_some() {
                return Err(ProgramError::MalformedProgram(
                    "`axis_index` cannot use tracked mesh metadata inside an unchecked manual binding".to_string(),
                )
                .into());
            }
        }
        let result = lower_axis_index_to_coordinate(
            self,
            &collective_state,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AxisIndex, Broadcast, Context, Device, DeviceMesh, Dimension as CoreDimension, LogicalMesh, MeshAxis,
        MeshAxisType, Shape, ShardMapOptions, Sharding, ShardingDimension, Typed, Value as CoreValue, shard_map,
    };
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::tests::{
        shard_map_module, static_sharded_array_type, test_manual_mesh, test_spmd_compilation_options, test_vector_type,
    };
    use crate::sessions::tests::session_with_client;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, ArrayTracer, ExecutionArguments, FromPjrt, TracedProgram, trace};

    use super::*;

    #[test]
    fn test_axis_index_operation_lower() {
        let client = cpu_client(4);
        let session = session_with_client(client);
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::U64, Shape::new(vec![CoreDimension::Static(4)]));

        // `axis_index("x")` gives each device its own coordinate along the manual mesh axis `"x"`, added to the local
        // shard. A mesh with a single axis gives that axis unit stride and the full size, so the coordinate is just
        // `partition_id`.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            // `axis_index` is a scalar per device. Broadcast it to the shard shape so that the
                            // operands of the add have congruent shapes, because StableHLO has no implicit
                            // broadcasting.
                            let local_type = local_x.r#type().into_owned();
                            let index = local_x.domain().axis_index("x")?;
                            let index_sharding = index
                                .r#type()
                                .sharding()
                                .unwrap()
                                .with_broadcasted_dimensions(local_type.rank(), &[])
                                .unwrap();
                            let index_type = local_type.with_sharding(index_sharding).unwrap();
                            let index = index.broadcast(index_type, &[])?;
                            Ok(local_x + index)
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<4xui64>) -> (tensor<4xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<1xui64>) {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui64>) -> tensor<1xui64>
                      %4 = stablehlo.add %arg1, %3 : tensor<1xui64>
                      sdy.return %4 : tensor<1xui64>
                    } : (tensor<4xui64>) -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes::<u64>(&[10u64]).as_slice(),
                        BufferType::U64,
                        [1u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::U64, &[4], sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
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

        // Each device d holds shard [10] and adds its coordinate d, so device d outputs [10 + d].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [u64; 1] = values_from_bytes::<u64>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [10 + device_index as u64]);
        }
    }

    #[test]
    fn test_axis_index_operation_lower_major_mesh_axis_uses_divide_and_remainder() {
        // On a 2x2 mesh, `axis_index("x")` addresses the major axis, which has stride 2 and size 2 in row-major order,
        // so the device coordinate is `(partition_id / 2) % 2`. This exercises both the divide and the remainder, which
        // a mesh with a single axis skips.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::U64, Shape::new(vec![CoreDimension::Static(4)]));

        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            let local_type = local_x.r#type().into_owned();
                            let index = local_x.domain().axis_index("x")?;
                            let index_sharding = index
                                .r#type()
                                .sharding()
                                .unwrap()
                                .with_broadcasted_dimensions(local_type.rank(), &[])
                                .unwrap();
                            let index_type = local_type.with_sharding(index_sharding).unwrap();
                            index.broadcast(index_type, &[])
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            global_input_type,
        )
        .unwrap();

        // The body reads only the shape of its input, so pruning drops the input from the manual computation.
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4xui64>) -> (tensor<4xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
                    %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} () {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %c = stablehlo.constant dense<2> : tensor<ui64>
                      %3 = stablehlo.divide %2, %c : tensor<ui64>
                      %c_0 = stablehlo.constant dense<2> : tensor<ui64>
                      %4 = stablehlo.remainder %3, %c_0 : tensor<ui64>
                      %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui64>) -> tensor<1xui64>
                      sdy.return %5 : tensor<1xui64>
                    } : () -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_axis_index_operation_lower_rejects_missing_checked_mesh_metadata() {
        let mesh = test_manual_mesh("x", 2);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, input| Ok(input.domain().bind(AxisIndexOperation::new("x".to_string()), Vec::new(), &[])?.remove(0)),
            test_vector_type(4),
            mesh.clone(),
            Sharding::replicated(mesh.clone(), 1),
            Sharding::replicated(mesh, 0),
            ShardMapOptions::new(),
        );
        assert!(matches!(
            module,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if *message == "`axis_index` requires mesh metadata inside a manual region",
            ),
        ));
    }
}
