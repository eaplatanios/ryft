use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Operation as CoreOperation, ParallelPermuteOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, BlockRef, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::lowering::CollectiveLoweringState;
use crate::operations::collectives::collective_replica_groups;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ParallelPermuteOperation {
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
        lower_parallel_permute_to_mlir(
            self,
            &collective_state,
            input_values[0],
            &mut lowerer.block(),
            lowerer.location(),
        )
    }
}

/// Lowers one traced `parallel_permute` to a channeled `stablehlo.collective_permute`, expanding the axis-local
/// `(source, target)` pairs to global device pairs across the named mesh axis's replica groups.
fn lower_parallel_permute_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelPermuteOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let (replica_groups, _) =
        collective_replica_groups(collective_state, operation.axes(), operation.axis_size(), None)?;
    let mut source_target_pairs = Vec::with_capacity(replica_groups.len() * operation.source_target_pairs().len());
    for group in &replica_groups {
        for (source, target) in operation.source_target_pairs() {
            source_target_pairs.push((group[*source], group[*target]));
        }
    }
    let result = block.append_operation(stable_hlo::collective_permute(
        input_value,
        source_target_pairs.as_slice(),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        location,
    )?)?;
    Ok(vec![result.result(0).unwrap().as_ref()])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Context, DataType, Device, DeviceMesh, Dimension as CoreDimension, LogicalMesh, MeshAxis, MeshAxisType,
        ParallelPermute, ProgramError, Shape, ShardMapOptions, Sharding, ShardingDimension, Value as CoreValue,
        shard_map,
    };
    use ryft_pjrt::{BufferType, Program};

    use crate::execution::ExecutionArguments;
    use crate::lowering::tests::{static_sharded_array_type, test_spmd_compilation_options};
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::tracing::{ArrayTracer, TracedProgram, trace};
    use crate::{Array, FromPjrt, Session};

    use super::*;

    #[test]
    fn test_parallel_permute_operation_lower() {
        let client = cpu_client(2);
        let domain = Session::new(&client).domain();
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

        // `parallel_permute` over the manual mesh axis `"x"` with the rotation pairs [(0, 1), (1, 0)] swaps the two
        // local shards without changing their shapes. The staged collective lowers to a channeled
        // `stablehlo.collective_permute` with the axis-local pairs expanded to global device pairs.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| local_x.parallel_permute("x", vec![(0, 1), (1, 0)]),
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
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.collective_permute"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
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
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharding.clone()),
            device_mesh.clone(),
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

        // Shards are [1, 2] and [3, 4], and the rotation swaps them, so device 0 receives [3, 4] and device 1 [1, 2].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[3.0f32, 4.0], [1.0, 2.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }

        // Partial and empty permutations give every untargeted participant a zero-filled shard.
        for (pairs, expected_values_by_device) in
            [(vec![(0, 1)], [[0.0f32, 0.0], [1.0, 2.0]]), (Vec::new(), [[0.0f32, 0.0], [0.0, 0.0]])]
        {
            let mesh = device_mesh.logical_mesh().clone();
            let sharding = sharding.clone();
            let traced: TracedProgram<ArrayType, ArrayType> = trace(
                move |input: ArrayTracer| {
                    shard_map(
                        |local_input: ArrayTracer| local_input.parallel_permute("x", pairs.clone()),
                        input,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                },
                ArrayType::new_static(DataType::F32, [4]),
            )
            .unwrap();
            let program = Program::Mlir { bytecode: traced.to_mlir_module("main").unwrap().into_bytes() };
            let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();
            assert_eq!(
                executable
                    .addressable_devices()
                    .unwrap()
                    .iter()
                    .map(|device| device.id().unwrap())
                    .collect::<Vec<_>>(),
                execution_device_ids,
            );
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
            assert_eq!(outputs.len(), expected_values_by_device.len());
            for (output, expected) in outputs.into_iter().zip(expected_values_by_device) {
                assert_eq!(output.outputs.len(), 1);
                let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), expected);
            }
        }
    }

    #[test]
    fn test_parallel_permute_operation_lower_rejects_axis_size_mismatch() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // Both operations are valid in isolation, but neither records the size of the enclosing manual axis.
        // The larger size also exercises the pair that previously indexed beyond its replica group.
        for (axis_size, pairs) in [(1, vec![(0, 0)]), (3, vec![(2, 0)])] {
            let mesh = mesh.clone();
            let sharding = sharding.clone();
            let traced: TracedProgram<ArrayType, ArrayType> = trace(
                move |input: ArrayTracer| {
                    shard_map(
                        |local_input: ArrayTracer| {
                            Ok(local_input
                                .domain()
                                .bind(
                                    ParallelPermuteOperation::new("x".to_string(), axis_size, pairs.clone()),
                                    Vec::new(),
                                    &[local_input],
                                )?
                                .remove(0))
                        },
                        input,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                },
                ArrayType::new_static(DataType::F32, [4]),
            )
            .unwrap();
            assert!(matches!(
                traced.to_mlir_module("main"),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                        if *message == format!(
                            "collective over axis `x` records size {axis_size}, \
                             but the enclosing mesh axis has size 2",
                        ),
                ),
            ));
        }
    }
}
