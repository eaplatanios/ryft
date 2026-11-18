use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, DomainTracingContext, EmptyRegionDriver, InterpretableOperation, NamedAxis, Operation as CoreOperation,
    PARALLEL_BROADCAST_OPERATION_NAME, ParallelBroadcastOperation, ParallelBroadcastSource, ProgramError,
    RematerializationOptimizationBarrier, Value as CoreValue, ValueProjection,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{Block, BlockRef, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::lowering::CollectiveLoweringState;
use crate::lowering::regions::lower_nested_program_inline;
use crate::operations::collectives::collective_replica_groups;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, Session, Tracer, Type};

impl LowerableOperation for ParallelBroadcastOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let input_count = if self.source() == ParallelBroadcastSource::Dynamic(()) { 2 } else { 1 };
        check_count!("input", input_values, input_count, ProgramError);
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
        if self.source() == ParallelBroadcastSource::Dynamic(()) {
            // Trace the operation's canonical checked interpretation in the actual manual-axis environment. The
            // decomposition retains exact copies and assertion tokens on targets without native root support.
            let axes = collective_state
                .bound_manual_axes()
                .iter()
                .map(|name| {
                    let boundary = collective_state.manual_axis_region(name).unwrap();
                    let mesh = boundary.mesh().clone();
                    let axis = mesh.axis_index(name).unwrap();
                    (
                        name.clone(),
                        NamedAxis::Mesh {
                            size: mesh.axes()[axis].size(),
                            axis,
                            mesh,
                            checks_manual_variation: boundary.checks_manual_variation(),
                        },
                    )
                })
                .collect::<Vec<_>>();
            let (_, program) = DomainTracingContext::<Session>::trace_with_named_axes(
                |inputs: Vec<Tracer>| {
                    let inputs = inputs
                        .into_iter()
                        .map(ValueProjection::<ArrayType>::into_projected)
                        .collect::<Result<Vec<_>, _>>()?;
                    let outputs = self.interpret(&inputs[0].domain(), &EmptyRegionDriver, &inputs)?;
                    Ok(outputs.into_iter().map(|value| value.into_value()).collect::<Vec<_>>())
                },
                input_types.iter().cloned().map(Type::from).collect::<Vec<_>>(),
                axes,
            )?;
            let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
            return lower_nested_program_inline(
                &program.to_flat_program(),
                input_values,
                &mut block,
                context,
                location,
                &[],
                &RematerializationOptimizationBarrier::None,
                None,
                &collective_state,
                lowerer.effect_tokens_mut(),
            );
        }
        lower_static_parallel_broadcast_to_mlir(
            self,
            &collective_state,
            input_values[0],
            &mut lowerer.block(),
            lowerer.location(),
        )
    }
}

/// Lowers one traced `parallel_broadcast` to a channeled `stablehlo.collective_broadcast`. The operation broadcasts
/// from the first member of every replica group, so each group of the named mesh axes is reordered to start with its
/// member at the source position, followed by the remaining members in axis order, as JAX's GPU lowering of
/// `pbroadcast` does. The channel handle and global device IDs in the groups make XLA interpret the devices as
/// partitions of one replica, which is how `shard_map` regions execute.
///
/// A broadcast over a single participant is the identity and emits nothing. Otherwise, the XLA CPU backend cannot
/// compile `collective-broadcast` because its thunk emitter rejects the opcode, so a lowering whose target platform is
/// known to be the CPU fails here with a precise error, while a lowering without target information emits the
/// operation and leaves the rejection to XLA's compiler. JAX registers its lowering for GPUs only.
fn lower_static_parallel_broadcast_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelBroadcastOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    if operation.axis_size() == 1 {
        return Ok(vec![input_value]);
    }
    if collective_state.target_platform().is_some_and(|platform| platform.eq_ignore_ascii_case("cpu")) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` is not supported by the XLA CPU backend"),
        }
        .into());
    }
    let (replica_groups, _) =
        collective_replica_groups(collective_state, operation.axes(), operation.axis_size(), None)?;
    let ParallelBroadcastSource::Static(source) = operation.source() else {
        unreachable!();
    };
    let replica_groups = replica_groups
        .iter()
        .map(|group| {
            std::iter::once(group[source])
                .chain(group[..source].iter().copied())
                .chain(group[source + 1..].iter().copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let replica_groups = replica_groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let result = block.append_operation(stable_hlo::collective_broadcast(
        &[input_value],
        None,
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
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
        ConvertElementType, DataType, LogicalMesh, MeshAxis, MeshAxisType, ParallelBroadcast, ShardMapOptions,
        Sharding, ShardingDimension, shard_map,
    };
    use ryft_pjrt::Program as PjrtProgram;

    use crate::lowering::lower_mlir_module_for_program;
    use crate::lowering::tests::{test_manual_mesh, test_spmd_compilation_options};
    use crate::operations::sharding::shard_map::tests::{execute_inputs_on_cpu_devices, test_logical_mesh_2x2};
    use crate::tests::cpu_client;
    use crate::{ArrayTracer, TracedProgram, to_mlir_module_for_program, trace_with_context};

    use super::*;

    #[test]
    fn test_parallel_broadcast_operation_lower() {
        // Every device receives the shard of the device at the source position. The replica group is reordered so that
        // the source comes first, which yields the module that JAX's GPU lowering of `pbroadcast` emits
        // (`.tasks/shard_map_jax_parity_gaps/m8/pbroadcast.mlir`).
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| input.parallel_broadcast("x", 1),
                    input,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![Type::Array(global_type.clone())],
        )
        .unwrap();
        let types = [global_type];
        let shardings = [sharded];
        let module =
            to_mlir_module_for_program(&program, &[], &types, &types, "main", Some(&shardings), Some(&shardings))
                .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\"}]>] \
                        out_shardings=[<@mesh, [{\"x\"}]>] manual_axes={\"x\"} (%arg1: tensor<2xf32>) {
                      %1 = \"stablehlo.collective_broadcast\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // The XLA CPU backend cannot compile `collective-broadcast`, so compiling the module fails in XLA, and lowering
        // for a CPU target fails earlier with a precise error.
        let client = cpu_client(2);
        assert!(matches!(
            client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2)),
            Err(ryft_pjrt::Error::Unimplemented { message, .. })
                if message.contains("HLO opcode `collective-broadcast` is not supported by XLA:CPU"),
        ));
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &types,
                &types,
                "main",
                Some(&shardings),
                Some(&shardings),
                Some("cpu"),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` is \
                         not supported by the XLA CPU backend",
                    ),
            ),
        ));
    }

    #[test]
    fn test_parallel_broadcast_operation_lower_dynamic_source_executes_on_cpu() {
        let mesh = test_manual_mesh("x", 2);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let input_types = [
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
            ArrayType::scalar(DataType::F32).with_sharding(replicated.clone()).unwrap(),
        ];
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let value = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let source = ValueProjection::<ArrayType>::into_projected(inputs[1].clone())?;
                let output = shard_map(
                    |(value, source): (ArrayTracer, ArrayTracer)| {
                        value.parallel_broadcast("x", &source.convert_element_type(DataType::U64)?)
                    },
                    (value, source),
                    mesh.clone(),
                    (sharded.clone(), replicated.clone()),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            input_types.iter().cloned().map(Type::from).collect::<Vec<_>>(),
        )
        .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &[input_types[0].clone()],
            "main",
            Some(&[sharded.clone(), replicated]),
            Some(&[sharded]),
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}, %arg1: \
                      tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={\"x\"}>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2:3 = sdy.manual_computation(%arg0, %arg1, %0, %1) in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, \
                        [], replicated={\"x\"}>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{\"x\"}]>, \
                        <@mesh, []>, <@mesh, []>] manual_axes={\"x\"} (%arg2: tensor<2xf32>, %arg3: tensor<f32>, \
                        %arg4: !stablehlo.token, %arg5: !stablehlo.token) {
                      %3 = stablehlo.convert %arg3 : (tensor<f32>) -> tensor<ui64>
                      %4 = \"stablehlo.all_reduce\"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, \
                          type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg6: tensor<ui64>, %arg7: tensor<ui64>):
                        %19 = stablehlo.minimum %arg6, %arg7 : tensor<ui64>
                        stablehlo.return %19 : tensor<ui64>
                      }) : (tensor<ui64>) -> tensor<ui64>
                      %5 = \"stablehlo.all_reduce\"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, \
                          type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg6: tensor<ui64>, %arg7: tensor<ui64>):
                        %19 = stablehlo.maximum %arg6, %arg7 : tensor<ui64>
                        stablehlo.return %19 : tensor<ui64>
                      }) : (tensor<ui64>) -> tensor<ui64>
                      %6 = stablehlo.compare EQ, %4, %5, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
                      %7 = stablehlo.custom_call @ryft.assert(%6, %4, %5, %arg4) {api_version = 4 : i32, \
                          backend_config = {actor = \"assert\", kind = \"generic\", label_0 = \"minimum\", label_1 = \
                          \"maximum\", label_count = \"2\", message = \"`parallel_broadcast` runtime sources must \
                          agree across participants\"}, has_side_effect = true} : (tensor<i1>, tensor<ui64>, \
                          tensor<ui64>, !stablehlo.token) -> !stablehlo.token
                      %c = stablehlo.constant dense<2> : tensor<ui64>
                      %8 = stablehlo.compare LT, %5, %c, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
                      %9 = stablehlo.custom_call @ryft.assert(%8, %3, %7) {api_version = 4 : i32, \
                          backend_config = {actor = \"assert\", kind = \"generic\", label_0 = \"source\", label_count \
                          = \"1\", message = \"`parallel_broadcast` runtime source must be smaller than axis size \
                          2\"}, has_side_effect = true} : (tensor<i1>, tensor<ui64>, !stablehlo.token) -> \
                          !stablehlo.token
                      %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
                      %11 = \"stablehlo.all_gather\"(%10) <{all_gather_dim = 0 : i64, channel_handle = \
                          #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1]]> : \
                          tensor<1x2xi64>, use_global_device_ids}> : (tensor<1x2xf32>) -> tensor<2x2xf32>
                      %12 = stablehlo.slice %11 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
                      %13 = stablehlo.reshape %12 : (tensor<1x2xf32>) -> tensor<2xf32>
                      %14 = stablehlo.slice %11 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
                      %15 = stablehlo.reshape %14 : (tensor<1x2xf32>) -> tensor<2xf32>
                      %c_0 = stablehlo.constant dense<1> : tensor<ui64>
                      %16 = stablehlo.compare EQ, %3, %c_0, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
                      %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<2xi1>
                      %18 = stablehlo.select %17, %15, %13 : tensor<2xi1>, tensor<2xf32>
                      sdy.return %18, %9, %arg5 : tensor<2xf32>, !stablehlo.token, !stablehlo.token
                    } : (tensor<4xf32>, tensor<f32>, !stablehlo.token, !stablehlo.token) -> (tensor<4xf32>, \
                        !stablehlo.token, !stablehlo.token)
                    return %2#0 : tensor<4xf32>
                  }
                }
            "
            },
        );
        assert_eq!(
            execute_inputs_on_cpu_devices(
                module.clone(),
                &[(&[2], [&[1f32, 2.0], &[3.0, 4.0]]), (&[], [&[0f32], &[0f32]])],
            ),
            vec![vec![1f32, 2.0], vec![1.0, 2.0]],
        );
        assert_eq!(
            execute_inputs_on_cpu_devices(
                module.clone(),
                &[(&[2], [&[1f32, 2.0], &[3.0, 4.0]]), (&[], [&[1f32], &[1f32]])],
            ),
            vec![vec![3f32, 4.0], vec![3.0, 4.0]],
        );
        let nan = f32::from_bits(0x7fc01234);
        let output =
            execute_inputs_on_cpu_devices(module, &[(&[2], [&[1f32, 2.0], &[-0.0, nan]]), (&[], [&[1f32], &[1f32]])]);
        assert_eq!(
            output.into_iter().flatten().map(f32::to_bits).collect::<Vec<_>>(),
            vec![(-0.0f32).to_bits(), nan.to_bits(), (-0.0f32).to_bits(), nan.to_bits()],
        );
    }

    #[test]
    fn test_parallel_broadcast_operation_lower_degenerate_executes_on_cpu() {
        // A broadcast over the single device of axis `y` is the identity, so it lowers to nothing and also executes on
        // the CPU backend, which cannot compile `collective-broadcast`.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let input = ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?;
                let output = shard_map(
                    |input: ArrayTracer| input.parallel_broadcast("y", 0),
                    input,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?;
                Ok(vec![output.into_value()])
            },
            vec![Type::Array(global_type.clone())],
        )
        .unwrap();
        let types = [global_type];
        let shardings = [sharded];
        let module = lower_mlir_module_for_program(
            &program,
            &[],
            &types,
            &types,
            "main",
            Some(&shardings),
            Some(&shardings),
            Some("cpu"),
        )
        .unwrap()
        .into_stable_hlo();
        assert_eq!(
            execute_inputs_on_cpu_devices(module, &[(&[2], [&[1.0, 2.0], &[3.0, 4.0]])]),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        );
    }

    #[test]
    fn test_parallel_broadcast_operation_lower_multi_axis() {
        // On a 2x2 mesh, device `(x, y)` has ID `2x + y`. Source 2 along `("x", "y")` is device `(1, 0)`, whose ID is
        // 2, and along `("y", "x")` it is device `(0, 1)`, whose ID is 1, so each single replica group starts with that
        // device. Along `y` alone, every row of the mesh forms its own group, which starts with its device at `y = 1`.
        let mesh = test_logical_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace_with_context(
            |_, x: ArrayTracer| {
                Ok(shard_map(
                    |x: ArrayTracer| {
                        let x = x.parallel_broadcast(["x", "y"], 2)?;
                        let x = x.parallel_broadcast(["y", "x"], 2)?;
                        x.parallel_broadcast("y", 1)
                    },
                    x,
                    mesh.clone(),
                    sharded.clone(),
                    sharded.clone(),
                    ShardMapOptions::new(),
                )?)
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"y\"}]>}) -> \
                      (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", \"y\"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{\"x\", \"y\"}]>] \
                        out_shardings=[<@mesh, [{\"x\", \"y\"}]>] manual_axes={\"x\", \"y\"} (%arg1: tensor<1xf32>) {
                      %1 = \"stablehlo.collective_broadcast\"(%arg1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = \
                          dense<[[2, 0, 1, 3]]> : tensor<1x4xi64>}> : (tensor<1xf32>) -> tensor<1xf32>
                      %2 = \"stablehlo.collective_broadcast\"(%1) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = \
                          dense<[[1, 0, 2, 3]]> : tensor<1x4xi64>}> : (tensor<1xf32>) -> tensor<1xf32>
                      %3 = \"stablehlo.collective_broadcast\"(%2) <{channel_handle = \
                          #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = \
                          dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}> : (tensor<1xf32>) -> tensor<1xf32>
                      sdy.return %3 : tensor<1xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
    }
}
