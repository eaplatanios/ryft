use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayType, CollectiveMode, Dimension as CoreDimension, MemberOperation, Operation as CoreOperation,
    ParallelAllGatherOperation, ProgramError, Shape,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::lowering::CollectiveLoweringState;
use crate::lowering::types::lower_tensor_type;
use crate::operations::collectives::{collective_replica_groups, refine_collective_result_dimensions};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, MemberLowerableOperation};

impl LowerableOperation for ParallelAllGatherOperation {
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
        lower_parallel_all_gather_to_mlir(
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

impl MemberLowerableOperation<ArrayIrType> for ParallelAllGatherOperation {
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
        let [output_type] = output_types else {
            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: output_types.len() }.into());
        };
        let output_type = <&ArrayType>::try_from(output_type)?;
        let result =
            lower_parallel_all_gather_to_mlir(self, collective_state, *input, output_type, block, context, location)?
                .remove(0);
        refine_collective_result_dimensions(result, &input_values[1..], output_type, block, context, location)
    }
}

/// Lowers one traced `parallel_all_gather` to a channeled `stablehlo.all_gather` over the named mesh axis's replica
/// groups.
fn lower_parallel_all_gather_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelAllGatherOperation,
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
    let input_value =
        insert_untiled_all_gather_axis(operation, input_value, output_array_type, block, context, location)?;
    let output_type = lower_tensor_type(output_array_type, context, location)?;
    let result = block.append_operation(stable_hlo::all_gather(
        &[input_value],
        operation.concatenation_axis(),
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        &[output_type],
        location,
    )?)?;
    Ok(vec![result.result(0).unwrap().as_ref()])
}

/// Returns `input_value` with the unit participant axis that an untiled all-gather inserts at its concatenation axis,
/// so that the all-gather of the result along that axis is tiled, or `input_value` itself for a tiled all-gather.
fn insert_untiled_all_gather_axis<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelAllGatherOperation,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if operation.options().mode() == CollectiveMode::Tiled {
        return Ok(input_value);
    }
    let mut dimensions = output_array_type.shape().dimensions().to_vec();
    dimensions[operation.concatenation_axis()] = CoreDimension::Static(1);
    let input_type =
        lower_tensor_type(&output_array_type.clone().with_shape(Shape::new(dimensions)), context, location)?;
    let broadcast_dimensions = (0..output_array_type.rank())
        .filter(|&axis| axis != operation.concatenation_axis())
        .collect::<Vec<_>>();
    let broadcast = block.append_operation(stable_hlo::broadcast(
        input_value,
        input_type,
        broadcast_dimensions.as_slice(),
        location,
    )?)?;
    Ok(broadcast.result(0).unwrap().as_ref())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        CollectiveOptions, DataType, Device, DeviceMesh, Differentiate, LogicalMesh, MeshAxis, MeshAxisType, Mul,
        ParallelAllGather, ParallelAllGatherOutputVariance, Reduce, ReductionKind, ShardMapOptions, Sharding,
        ShardingDimension, Value as CoreValue, ValueProjection, shard_map,
    };
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::tests::{static_sharded_array_type, test_spmd_compilation_options};
    use crate::sessions::tests::session_with_client;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, ArrayTracer, ExecutionArguments, FromPjrt, TracedProgram, trace};

    use super::*;

    #[test]
    fn test_parallel_all_gather_operation_lower() {
        let client = cpu_client(2);
        let session = session_with_client(&client);
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
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));

        // `parallel_all_gather` over the manual mesh axis `"x"` extends the local shard from `f32[2]` to the full
        // `f32[4]` concatenation on every device. Its output still varies along the manual axis for the purposes of
        // manual variation tracking, which mirrors the VMA tracking of JAX and rejects a replicated output sharding
        // with `OutputVaryingAlongUntiledManualAxis`. The output therefore stays sharded over `"x"`, which gives the
        // global `f32[8]` concatenation of the gathers of all devices. The staged collective lowers to a channeled
        // `stablehlo.all_gather` over the two devices along `"x"`.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_all_gather_with_options(
                                "x",
                                0,
                                CollectiveOptions::tiled(),
                                ParallelAllGatherOutputVariance::Varying,
                            )
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
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %c = stablehlo.constant dense<2> : tensor<i64>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.multiply %c, %c_0 : tensor<i64>
                      %2 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<2xf32>) -> tensor<4xf32>
                      sdy.return %2 : tensor<4xf32>
                    } : (tensor<4xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4], sharding),
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

        // Shards are [1, 2] and [3, 4], so every device receives the full concatenation [1, 2, 3, 4].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 4] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_parallel_all_gather_operation_lower_untiled_inserts_rank() {
        let client = cpu_client(2);
        let session = session_with_client(&client);
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_all_gather_with_options(
                                "x",
                                0,
                                CollectiveOptions::default(),
                                ParallelAllGatherOutputVariance::Varying,
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
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
                      %2 = "stablehlo.all_gather"(%1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<1x2xf32>) -> tensor<2x2xf32>
                      sdy.return %2 : tensor<2x2xf32>
                    } : (tensor<4xf32>) -> tensor<2x4xf32>
                    return %0 : tensor<2x4xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4], input_sharding),
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
        for output in outputs {
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_parallel_all_gather_operation_lower_grouped_preserves_group_order_across_mesh_coordinates() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_all_gather_with_options(
                                "x",
                                0,
                                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                                ParallelAllGatherOutputVariance::Varying,
                            )
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
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(
            module.contains("replica_groups = dense<[[0, 4], [6, 2], [1, 5], [7, 3]]> : tensor<4x2xi64>"),
            "{module}",
        );
        assert!(module.contains("(tensor<2xf32>) -> tensor<4xf32>"), "{module}");
    }

    #[test]
    fn test_parallel_all_gather_operation_lower_executes_invariant_gradient_on_cpu() {
        let client = cpu_client(4);
        let session = session_with_client(&client);
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The gradient of `sum(g * g)`, where `g` is the invariant all-gather of `x`, is `2 * x`. Linearizing the
        // all-gather keeps its gathered extent as a first-class dimension residual, which crosses the boundary between
        // the primal and the tangent `shard_map`s.
        let traced: TracedProgram<ArrayType, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |x: ArrayTracer| {
                    let (value, gradient) = x
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(x.into_value())
                        .value_and_gradient(|x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                            Ok(shard_map(
                                |local_x: ArrayTracer| {
                                    let gathered = local_x.parallel_all_gather_with_options(
                                        "x",
                                        0,
                                        CollectiveOptions::tiled(),
                                        ParallelAllGatherOutputVariance::Invariant,
                                    )?;
                                    gathered.mul(&gathered)?.reduce(&[0], ReductionKind::Sum)
                                },
                                x,
                                mesh.clone(),
                                sharded.clone(),
                                replicated.clone(),
                                ShardMapOptions::new(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                    ]
                }
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharded)),
                Some(&[replicated, sharded.clone()]),
            )
            .unwrap();
        // Recovering the dimension residual in the tangent body checks its bounds with a runtime assertion.
        crate::operations::assertions::ensure_assertion_handler_registered(&client).unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = [1.0f32, 2.0, 3.0, 4.0];
        let buffers = devices
            .iter()
            .zip(inputs)
            .map(|(device, input)| {
                client
                    .buffer(values_to_bytes(&[input]).as_slice(), BufferType::F32, [1], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(DataType::F32, &[4], sharded),
            device_mesh,
            buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = ExecutionArguments::from_arrays(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for (output, input) in outputs.into_iter().zip(inputs) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap())[0])
                .collect::<Vec<_>>();
            assert_eq!(values, vec![30.0, 2.0 * input]);
        }
    }
}
