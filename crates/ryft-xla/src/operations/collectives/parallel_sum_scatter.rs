use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, CollectiveMode, Dimension as CoreDimension, MemberOperation, Operation as CoreOperation,
    ParallelSumScatterOperation, ProgramError, ReductionKind,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::lowering::CollectiveLoweringState;
use crate::lowering::types::lower_tensor_type;
use crate::operations::collectives::{
    collapse_singleton_axis, collective_replica_groups, refine_collective_result_dimensions,
};
use crate::operations::extrema::FloatExtremumForm;
use crate::operations::reductions::build_reduce_body_region;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, MemberLowerableOperation, Type};

impl LowerableOperation for ParallelSumScatterOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state().clone();
        collective_state.validate_manual_collective(
            self.name(),
            self.axes(),
            self.axis_size(),
            self.mesh(),
            output_types,
            |mesh| {
                let operation = match mesh {
                    Some(mesh) => self.clone().with_mesh(mesh.clone()),
                    None => self.clone(),
                };
                operation.infer_output_types(input_types, &[])
            },
        )?;
        lower_parallel_sum_scatter_to_mlir(
            self,
            &collective_state,
            input_values[0],
            &output_types[0],
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )
    }
}

impl MemberLowerableOperation<Type> for ParallelSumScatterOperation {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let context = lowerer.context();
        let location = lowerer.location();
        let collective_state = lowerer.collective_state();
        let block = &mut lowerer.block();
        collective_state.validate_manual_collective(
            self.name(),
            self.axes(),
            self.axis_size(),
            self.mesh(),
            output_types,
            |mesh| {
                let operation = match mesh {
                    Some(mesh) => self.clone().with_mesh(mesh.clone()),
                    None => self.clone(),
                };
                operation.infer_parent_output_types(input_types, &[])
            },
        )?;
        let Some(input) = input_values.first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type)?;
        let result =
            lower_parallel_sum_scatter_to_mlir(self, collective_state, *input, output_type, block, context, location)?
                .remove(0);
        refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
    }
}

/// Lowers one traced `parallel_sum_scatter` to a channeled `stablehlo.reduce_scatter` with a sum reduction over the
/// named mesh axis's replica groups.
fn lower_parallel_sum_scatter_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelSumScatterOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let (replica_groups, _) = collective_replica_groups(
        collective_state,
        operation.axes(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let computation = build_reduce_body_region(
        ReductionKind::Sum,
        output_array_type.data_type(),
        FloatExtremumForm::TotalOrder,
        context,
        location,
    )?;
    let native_output_type = match operation.options().mode() {
        CollectiveMode::Tiled => output_array_type.clone(),
        CollectiveMode::Untiled => output_array_type
            .with_inserted_dimension(operation.scatter_axis(), CoreDimension::Static(1))
            .map_err(ProgramError::from)?,
    };
    let output_type = lower_tensor_type(&native_output_type, context, location)?;
    let result = block.append_operation(stable_hlo::reduce_scatter(
        input_value,
        operation.scatter_axis(),
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        computation,
        output_type,
        location,
    )?)?;
    let result = result.result(0).unwrap().as_ref();
    if operation.options().mode() == CollectiveMode::Tiled {
        return Ok(vec![result]);
    }
    Ok(vec![collapse_singleton_axis(result, output_array_type, block, context, location)?])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        CollectiveOptions, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ParallelSumScatter,
        Shape, ShardMapOptions, Sharding, ShardingDimension, shard_map,
    };
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::tests::{static_sharded_array_type, test_spmd_compilation_options};
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, ArrayTracer, ExecutionArguments, FromPjrt, Session, TracedProgram, trace};

    use super::*;

    #[test]
    fn test_parallel_sum_scatter_operation_lower() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        // `parallel_sum_scatter` over the manual mesh axis `"x"` sums the two local `f32[4]` shards elementwise and
        // scatters the sum, leaving each device with its own `f32[2]` chunk, so the sharded global output is `f32[4]`.
        // The staged collective lowers to a channeled `stablehlo.reduce_scatter` with a sum reduction.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled())
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
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %c = stablehlo.constant dense<4> : tensor<i64>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.divide %c, %c_0 : tensor<i64>
                      %2 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<4xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let scale = if device_index == 0 { 1.0f32 } else { 10.0f32 };
                let shard_values = [scale, scale * 2.0, scale * 3.0, scale * 4.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
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
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
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

        // Shards are [1, 2, 3, 4] and [10, 20, 30, 40], so the elementwise sum is [11, 22, 33, 44]. Device 0
        // receives the first chunk [11, 22] and device 1 the second chunk [33, 44].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[11.0f32, 22.0], [33.0, 44.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_parallel_sum_scatter_operation_lower_untiled_removes_rank() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_sum_scatter_with_options(
                                "x",
                                1,
                                CollectiveOptions::default().with_axis_index_groups(vec![vec![1, 0]]),
                            )
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new_static(DataType::F32, [6, 2]),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        // Group order chooses the recipient of each column, so device 1 receives column 0, and device 0 column 1.
        // Untiled scatter removes dimension 1 after the native reduce scatter leaves it with extent 1.
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<6x2xf32>) -> (tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<3x2xf32>) {
                      %1 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<3x2xf32>) -> tensor<3x1xf32>
                      %2 = stablehlo.reshape %1 : (tensor<3x1xf32>) -> tensor<3xf32>
                      sdy.return %2 : tensor<3xf32>
                    } : (tensor<6x2xf32>) -> tensor<6xf32>
                    return %0 : tensor<6xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let scale = if device_index == 0 { 1.0f32 } else { 10.0f32 };
                let values = [scale, scale * 2.0, scale * 3.0, scale * 4.0, scale * 5.0, scale * 6.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [3u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[6, 2], input_sharding),
            device_mesh,
            None,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
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
        assert_eq!(outputs.len(), 2);
        let expected = [vec![22.0f32, 44.0, 66.0], vec![11.0f32, 33.0, 55.0]];
        for (output, expected) in outputs.into_iter().zip(expected) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), expected);
        }
    }
}
