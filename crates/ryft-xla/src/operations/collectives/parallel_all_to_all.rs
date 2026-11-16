use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CollectiveMode, Dimension as CoreDimension, MemberOperation, Operation as CoreOperation,
    ParallelAllToAllOperation, ProgramError,
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
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, MemberLowerableOperation};

impl LowerableOperation for ParallelAllToAllOperation {
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
        lower_parallel_all_to_all_to_mlir(
            self,
            &collective_state,
            input_values[0],
            &input_types[0],
            &output_types[0],
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )
    }
}

impl MemberLowerableOperation<ArrayIrType> for ParallelAllToAllOperation {
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
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
        let Some(input_type) = input_types.first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type)?;
        let result = lower_parallel_all_to_all_to_mlir(
            self,
            collective_state,
            *input,
            input_type,
            output_type,
            block,
            context,
            location,
        )?
        .remove(0);
        refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
    }
}

/// Lowers one traced `parallel_all_to_all` to a channeled `stablehlo.all_to_all` over the named mesh axis's replica
/// groups.
fn lower_parallel_all_to_all_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelAllToAllOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    input_array_type: &ArrayType,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let (replica_groups, axis_size) = collective_replica_groups(
        collective_state,
        operation.axes(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let (input_value, split_axis, concatenation_axis) = match operation.options().mode() {
        CollectiveMode::Tiled => (input_value, operation.split_axis(), operation.concatenation_axis()),
        CollectiveMode::Untiled if operation.split_axis() == operation.concatenation_axis() => {
            (input_value, operation.split_axis(), operation.concatenation_axis())
        }
        CollectiveMode::Untiled => {
            let mut split_axis = operation.split_axis();
            let mut concatenation_axis = operation.concatenation_axis();
            if split_axis < concatenation_axis {
                concatenation_axis += 1;
            } else {
                split_axis += 1;
            }
            let expanded_type = input_array_type
                .with_inserted_dimension(concatenation_axis, CoreDimension::Static(1))
                .map_err(ProgramError::from)?;
            let expanded_type = lower_tensor_type(&expanded_type, context, location)?;
            let broadcast_dimensions =
                (0..input_array_type.rank() + 1).filter(|&axis| axis != concatenation_axis).collect::<Vec<_>>();
            let expanded = block.append_operation(stable_hlo::broadcast(
                input_value,
                expanded_type,
                broadcast_dimensions.as_slice(),
                location,
            )?)?;
            (expanded.result(0).unwrap().as_ref(), split_axis, concatenation_axis)
        }
    };
    let result = block.append_operation(stable_hlo::all_to_all(
        &[input_value],
        split_axis,
        axis_size,
        concatenation_axis,
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        location,
    )?)?;
    let result = result.result(0).unwrap().as_ref();
    if operation.options().mode() == CollectiveMode::Tiled || split_axis == concatenation_axis {
        return Ok(vec![result]);
    }
    Ok(vec![collapse_singleton_axis(result, output_array_type, block, context, location)?])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        CollectiveOptions, DataType, Device, DeviceMesh, DimensionBounds, DimensionType, LogicalMesh, MeshAxis,
        MeshAxisType, ParallelAllToAll, Shape, ShardMapBoundary, ShardMapOptions, Sharding, ShardingDimension,
        TypeError, shard_map,
    };
    use ryft_mlir::Location;
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::tests::{static_sharded_array_type, test_manual_mesh, test_spmd_compilation_options};
    use crate::lowering::types::lower_array_ir_type;
    use crate::operations::Operation;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, ArrayTracer, ExecutionArguments, FromPjrt, Session, TracedProgram, trace};

    use super::*;

    #[test]
    fn test_parallel_all_to_all_operation_lower() {
        let client = cpu_client(2);
        let session = Session::with_client(&client);
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

        // `parallel_all_to_all` over the manual mesh axis `"x"` with both the split and the concatenation axes at 0
        // keeps the local `f32[4]` shape. Each device splits its shard into two chunks, keeps its own chunk, and
        // receives the matching chunk of its peer. The staged collective lowers to a channeled `stablehlo.all_to_all`.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_all_to_all_with_options("x", 0, 0, CollectiveOptions::tiled())
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
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %1 = "stablehlo.all_to_all"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> {use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
                      sdy.return %1 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let base = device_index as f32 * 4.0;
                let shard_values = [base + 1.0, base + 2.0, base + 3.0, base + 4.0];
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

        // Shards are [1, 2, 3, 4] and [5, 6, 7, 8], and the exchange leaves device 0 with the two first halves
        // [1, 2, 5, 6] and device 1 with the two second halves [3, 4, 7, 8].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[1.0f32, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 4] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_parallel_all_to_all_operation_lower_untiled_exchanges_rank() {
        let client = cpu_client(2);
        let session = Session::with_client(&client);
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let traced: TracedProgram<ArrayType, (ArrayType, ArrayType)> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            // Removing split axis 0 puts the gathered participant axis at output axis 1.
                            // The inverse exchanges axis 1 back into output axis 0.
                            let exchanged = local_x.parallel_all_to_all("x", 0, 1)?;
                            let recovered = exchanged.parallel_all_to_all("x", 1, 0)?;
                            Ok((exchanged, recovered))
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        (output_sharding.clone(), input_sharding.clone()),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(6)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<2x6xf32>) -> (tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<2x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
                    %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}]>] out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<2x3xf32>) {
                      %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
                      %2 = "stablehlo.all_to_all"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> {use_global_device_ids} : (tensor<2x3x1xf32>) -> tensor<1x3x2xf32>
                      %3 = stablehlo.reshape %2 : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
                      %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
                      %5 = "stablehlo.all_to_all"(%4) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> {use_global_device_ids} : (tensor<1x3x2xf32>) -> tensor<2x3x1xf32>
                      %6 = stablehlo.reshape %5 : (tensor<2x3x1xf32>) -> tensor<2x3xf32>
                      sdy.return %3, %6 : tensor<3x2xf32>, tensor<2x3xf32>
                    } : (tensor<2x6xf32>) -> (tensor<6x2xf32>, tensor<2x6xf32>)
                    return %0#0, %0#1 : tensor<6x2xf32>, tensor<2x6xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let offset = device_index as f32 * 9.0;
                let values = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0, offset + 5.0, offset + 6.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64, 3u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[2, 6], input_sharding),
            device_mesh,
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
        let expected_exchange = [vec![1.0f32, 10.0, 2.0, 11.0, 3.0, 12.0], vec![4.0f32, 13.0, 5.0, 14.0, 6.0, 15.0]];
        let expected_recovery = [vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0]];
        assert_eq!(outputs.len(), 2);
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 2);
            assert_eq!(output.outputs[0].dimensions().unwrap(), vec![3, 2]);
            assert_eq!(output.outputs[1].dimensions().unwrap(), vec![2, 3]);
            let exchanged_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let recovered_bytes = output.outputs[1].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(exchanged_bytes.as_slice()), expected_exchange[device_index]);
            assert_eq!(values_from_bytes::<f32>(recovered_bytes.as_slice()), expected_recovery[device_index]);
        }
    }

    #[test]
    fn test_parallel_all_to_all_operation_lower_in_parent_revalidates_dynamic_raw_bindings() {
        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(
            ShardMapBoundary::new(mesh, vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap(),
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
            let mut lowerer: Lowerer<'_, '_, '_, Operation> =
                Lowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state.clone());
            let result = operation.lower_in_parent(&mut lowerer, &inputs, &input_types, &output_types);
            let expected = if sharding == &unreduced {
                Some(ProgramError::Type(TypeError::invalid("`parallel_all_to_all` does not support unreduced inputs")))
            } else if sharding == &replicated {
                Some(ProgramError::Type(TypeError::invalid(
                    "`parallel_all_to_all` input must vary over manual axis `x`; pass an invariant value through \
                     `parallel_vary` first so that the exchanged output is typed as varying",
                )))
            } else {
                None
            };
            match (result, expected) {
                (Ok(outputs), None) => {
                    assert_eq!(outputs.len(), 1);
                    state.channel_ids().set(1);
                }
                (Err(Error::Core(error)), Some(expected)) => {
                    assert_eq!(error.as_ref(), &ryft_core::Error::Program(expected));
                    assert_eq!(state.channel_ids().get(), 1);
                }
                (result, expected) => panic!("{sharding}: expected {expected:?} but got {result:?}"),
            }
        }
    }
}
