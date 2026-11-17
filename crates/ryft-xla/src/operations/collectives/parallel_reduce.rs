use ryft_core::macros::check_count;
use ryft_core::{ArrayType, Operation as CoreOperation, ParallelReduceOperation, ProgramError, ReductionKind};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::lowering::CollectiveLoweringState;
use crate::lowering::types::{lower_f64_constant_splat, lower_tensor_type};
use crate::operations::collectives::{collective_replica_groups, mesh_axis_replica_groups};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

/// Lowers one [`ParallelReduceOperation`] inside a `sdy.manual_computation` region to a `stablehlo.all_reduce` over
/// the device mesh axis the collective names.
///
/// The replica groups are the dense global device-id groups that [`mesh_axis_replica_groups`] derives from the mesh's
/// row-major device linearization, or, when the collective records explicit logical participant groups, those groups
/// expanded inside every fixed coordinate of the mesh's other axes without reordering either the groups or their
/// members. The emitted operation carries a module-unique channel id with a device-to-device channel type and
/// `use_global_device_ids`, the standard SPMD emission for cross-partition collectives.
///
/// Each kind combines with the single StableHLO operation of its reduction (i.e., `add` for `Sum` and `Mean`,
/// `multiply` for `Product`, `maximum` and `minimum` for `Max` and `Min`, and `or` and `and` for `Any` and `All`), and
/// a `Mean` then divides the all-reduced sum by the effective group size. Unlike the `stablehlo.reduce` lowering, the
/// combiner never needs a guard for an identity element because an all-reduce combines only its participants' values,
/// and it must stay a single operation because XLA backends implement all-reduce only for the combiners that they
/// recognize. Floating-point extrema therefore follow the backend's native `maximum` and `minimum`, as
/// `jax.lax.pmax` and `jax.lax.pmin` do, so their NaN and signed-zero behavior is backend-defined. The combiner also
/// accumulates in the input element type, without the `f32` accumulation that [`ReductionKind::Sum`] documents for
/// narrower floating-point `reduce` inputs, so that, e.g., `bf16` gradient all-reduces keep their communication
/// volume. A `LogSumExp` has no single combiner and is rejected, because the
/// [`ParallelReduce`](ryft_core::ParallelReduce) capability composes it from a mesh maximum and a mesh sum instead. A
/// collective lowered outside any manual region, or naming an axis the innermost region does not bind as a manual mesh
/// axis, is an error. A recorded mesh must have the topology of the binding region's mesh and make the same axes manual
/// (i.e., satisfy [`LogicalMesh::has_same_manual_axes`](ryft_core::LogicalMesh::has_same_manual_axes)), so that
/// `auto_axes` and `explicit_axes` may retype the region's free axes, and the replica groups come from the region's
/// mesh.
pub(crate) fn lower_collective_to_all_reduce<'b, 'c: 'b, 't: 'c>(
    operation: &ParallelReduceOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if let Some(mesh) = operation.mesh()
        && collective_state
            .manual_axis_region(operation.axis_name())
            .is_some_and(|region| !mesh.has_same_manual_axes(region.mesh()))
    {
        return Err(ProgramError::MalformedProgram(format!(
            "`{}` mesh does not match its active manual binding",
            operation.name(),
        ))
        .into());
    }
    let (replica_groups, effective_axis_size) = match operation.axis_index_groups() {
        Some(axis_index_groups) => collective_replica_groups(
            collective_state,
            operation.axis_name(),
            operation.axis_size().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "grouped `{}` over axis `{}` does not record the full axis size",
                    operation.name(),
                    operation.axis_name(),
                ))
            })?,
            Some(axis_index_groups),
        )?,
        None => mesh_axis_replica_groups(collective_state, operation.axis_name())?,
    };
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();

    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(output_array_type.data_type()), context, location)?;
    let mut computation = context.region();
    let mut combiner_block =
        computation.append_block(context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]))?;
    let left = combiner_block.argument(0)?.as_ref();
    let right = combiner_block.argument(1)?.as_ref();
    let combined = match operation.kind() {
        ReductionKind::Sum | ReductionKind::Mean => combiner_block
            .append_operation(stable_hlo::add(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Product => combiner_block
            .append_operation(stable_hlo::multiply(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Max => combiner_block
            .append_operation(stable_hlo::maximum(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Min => combiner_block
            .append_operation(stable_hlo::minimum(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::Any => {
            combiner_block.append_operation(stable_hlo::or(left, right, location)?)?.result(0).unwrap().as_ref()
        }
        ReductionKind::All => combiner_block
            .append_operation(stable_hlo::and(left, right, location)?)?
            .result(0)
            .unwrap()
            .as_ref(),
        ReductionKind::LogSumExp => {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` with kind `{}` has no all-reduce combiner; stage it through the `ParallelReduce` \
                     capability, which composes it from primitive mesh reductions",
                    operation.name(),
                    operation.kind(),
                ),
            }
            .into());
        }
    };
    combiner_block.append_operation(stable_hlo::r#return(&[combined], location)?)?;
    let result = block.append_operation(stable_hlo::all_reduce(
        &[input_value],
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        computation,
        location,
    )?)?;
    let reduced = result.result(0).unwrap().as_ref();
    if operation.kind() != ReductionKind::Mean {
        return Ok(reduced);
    }
    // `Mean` is the mean over the effective participant group, so divide the all-reduced sum by the group size.
    let output_tensor_type = lower_tensor_type(output_array_type, context, location)?;
    let divisor = lower_f64_constant_splat(
        effective_axis_size as f64,
        output_array_type,
        output_tensor_type,
        block,
        context,
        location,
    )?;
    let mean = block.append_operation(stable_hlo::divide(reduced, divisor, location)?)?;
    Ok(mean.result(0).unwrap().as_ref())
}

impl LowerableOperation for ParallelReduceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        // This plain dispatch serves nested programs (i.e., control-flow bodies, inlined composite primals, and
        // rematerialized bodies), which can sit inside a `shard_map` manual region. The threaded
        // `CollectiveLoweringState` resolves the collective's mesh axis there and returns an error outside manual
        // regions. A batched axis never reaches this point, because batching consumes it into a `Reduce` when the
        // closure is traced.
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let collective_state = lowerer.collective_state().clone();

        // A mesh reduction tracks manual variation, so its binding must check it. Inside a checked binding, a meshless
        // (e.g., grouped) reduction must keep its recorded output varying along the axis, because otherwise that
        // binding would accept an output that is typed as invariant although the devices can hold different values.
        if let Some(boundary) = collective_state.manual_axis_region(self.axis_name()) {
            let axis_name = self.axis_name();
            let operation_name = self.name();
            if !boundary.checks_manual_variation() && self.mesh().is_some() {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{operation_name}` cannot use tracked mesh metadata inside an unchecked manual binding",
                ))
                .into());
            }
            if boundary.checks_manual_variation()
                && self.mesh().is_none()
                && output_types[0]
                    .sharding()
                    .is_none_or(|sharding| !sharding.varying_manual_axes().contains(axis_name))
            {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{operation_name}` without tracked mesh metadata types its outputs as invariant along manual axis \
                     `{axis_name}` inside a checked manual region",
                ))
                .into());
            }
        }

        let result = lower_collective_to_all_reduce(
            self,
            &collective_state,
            input_values[0],
            &output_types[0],
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        Ok(vec![result])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, Batch, BatchAxis, BatchAxisSpecification, ConditionOperation, Context, DataType, Device,
        DeviceMesh, Differentiate, Dimension as CoreDimension, LogicalMesh, MeshAxis, MeshAxisType, ParallelReduce,
        Placeholder, Reduce, Shape, ShardMapBoundary, ShardMapOptions, Sharding, ShardingDimension, StagingContext,
        Typed, Value as CoreValue, ValueProjection, shard_map,
    };
    use ryft_mlir::Location;
    use ryft_pjrt::{BufferType, Program};

    use crate::lowering::tests::{
        static_sharded_array_type, test_manual_mesh, test_spmd_compilation_options, test_vector_type,
    };
    use crate::operations::{Operation, ProgramBuilder};
    use crate::sessions::tests::session_with_client;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{Array, ArrayTracer, Constant, ExecutionArguments, FromPjrt, TracedProgram, trace};

    use super::*;

    /// Executes `local.parallel_reduce(kind, "x")` inside a `shard_map` over the manual axis `"x"` of four CPU devices,
    /// on the global vector of `4 · shard_size` elements of `data_type` whose device shards hold the raw `shards`, and
    /// returns the raw bytes of every device's output shard in device order.
    pub(crate) fn execute_parallel_reduce_on_cpu(
        kind: ReductionKind,
        data_type: DataType,
        buffer_type: BufferType,
        shard_size: usize,
        shards: [Vec<u8>; 4],
    ) -> Vec<Vec<u8>> {
        let client = cpu_client(4);
        let session = session_with_client(&client);
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(kind, "x"),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new_static(data_type, [4 * shard_size]),
        )
        .unwrap();
        let program = Program::Mlir { bytecode: traced.to_mlir_module("main").unwrap().into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();
        let buffers = devices
            .iter()
            .zip(shards)
            .map(|(device, shard)| {
                client
                    .buffer(shard.as_slice(), buffer_type, [shard_size as u64], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = Array::from_addressable_buffers(
            &session,
            static_sharded_array_type(data_type, &[4 * shard_size], sharding),
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
        executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None, &[])
            .unwrap()
            .block_until_ready()
            .unwrap()
            .into_iter()
            .map(|output| output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap())
            .collect()
    }

    #[test]
    fn test_parallel_reduce_operation_lower() {
        let client = cpu_client(4);
        let session = session_with_client(&client);
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
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        // A sum `parallel_reduce` over the manual mesh axis `"x"` resolves against the seeded body tracing context and
        // lowers to a `stablehlo.all_reduce` whose replica group spans the four devices along `"x"`, so every shard
        // receives the elementwise sum of all four local shards.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
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
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
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
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
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

        // Shards are [1, 2], [3, 4], [5, 6], [7, 8], so every device receives [1+3+5+7, 2+4+6+8] = [16, 20].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [16.0, 20.0]);
        }
    }

    #[test]
    fn test_parallel_reduce_operation_lower_executes_reduction_kinds_on_cpu() {
        // Every primitive kind lowers to a `stablehlo.all_reduce` with the scalar combiner of the matching `reduce`, so
        // every device receives the elementwise reduction of the four local shards.
        let f32_shards = [[1.0f32, -2.0], [3.0, 4.0], [-5.0, 0.5], [2.0, 8.0]].map(|shard| values_to_bytes(&shard));
        for (kind, expected) in [
            (ReductionKind::Sum, [1.0f32, 10.5]),
            (ReductionKind::Product, [-30.0, -32.0]),
            (ReductionKind::Mean, [0.25, 2.625]),
            (ReductionKind::Max, [3.0, 8.0]),
            (ReductionKind::Min, [-5.0, -2.0]),
        ] {
            for output in execute_parallel_reduce_on_cpu(kind, DataType::F32, BufferType::F32, 2, f32_shards.clone()) {
                assert_eq!(values_from_bytes::<f32>(&output), expected, "{kind}");
            }
        }
        let i32_shards = [[1i32, -2], [3, 4], [-5, 7], [2, 8]].map(|shard| values_to_bytes(&shard));
        for (kind, expected) in [
            (ReductionKind::Sum, [1i32, 17]),
            (ReductionKind::Product, [-30, -448]),
            (ReductionKind::Max, [3, 8]),
            (ReductionKind::Min, [-5, -2]),
        ] {
            for output in execute_parallel_reduce_on_cpu(kind, DataType::I32, BufferType::I32, 2, i32_shards.clone()) {
                assert_eq!(values_from_bytes::<i32>(&output), expected, "{kind}");
            }
        }
        let boolean_shards = [[true, false, false], [true, true, false], [true, false, false], [true, false, false]]
            .map(|shard| values_to_bytes(&shard));
        for (kind, expected) in [(ReductionKind::Any, [true, true, false]), (ReductionKind::All, [true, false, false])]
        {
            for output in execute_parallel_reduce_on_cpu(
                kind,
                DataType::Boolean,
                BufferType::Predicate,
                3,
                boolean_shards.clone(),
            ) {
                assert_eq!(values_from_bytes::<bool>(&output), expected, "{kind}");
            }
        }
    }

    #[test]
    fn test_parallel_reduce_operation_lower_executes_log_sum_exp_on_cpu() {
        // A logarithmic sum of exponentials over a manual mesh axis is composed from a mesh maximum and a mesh sum of
        // the shifted exponentials, so it stays finite for large inputs and matches the single-device reduction for
        // infinite inputs. Columns hold large equal inputs, all `-∞`, one `+∞`, mixed magnitudes, and one NaN.
        let shards = [
            [1000.0f32, f32::NEG_INFINITY, f32::INFINITY, -1000.0, 0.0],
            [1000.0, f32::NEG_INFINITY, 1.0, 0.0, f32::NAN],
            [1000.0, f32::NEG_INFINITY, 2.0, 1.0, 0.0],
            [1000.0, f32::NEG_INFINITY, 3.0, 2.0, 0.0],
        ]
        .map(|shard| values_to_bytes(&shard));
        let mixed = (0f64.exp() + 1f64.exp() + 2f64.exp()).ln() as f32;
        for output in
            execute_parallel_reduce_on_cpu(ReductionKind::LogSumExp, DataType::F32, BufferType::F32, 5, shards)
        {
            let values = values_from_bytes::<f32>(&output);
            assert!((values[0] - (1000.0 + 4f32.ln())).abs() <= 1e-4, "{values:?}");
            assert_eq!(values[1..3], [f32::NEG_INFINITY, f32::INFINITY]);
            assert!((values[3] - mixed).abs() <= 1e-6, "{values:?}");
            assert!(values[4].is_nan(), "{values:?}");
        }

        // Complex inputs are shifted by the maximum of their real components, so large real components stay finite
        // and the imaginary components rotate every term before the terms are summed.
        let shards = [[1000.0f32, 0.0], [1000.0, std::f32::consts::FRAC_PI_2], [1000.0, 0.0], [1000.0, 0.0]]
            .map(|shard| values_to_bytes(&shard));
        let expected = num_complex::Complex64::new(3.0, 1.0).ln() + 1000.0;
        for output in
            execute_parallel_reduce_on_cpu(ReductionKind::LogSumExp, DataType::C64, BufferType::C64, 1, shards)
        {
            let values = values_from_bytes::<f32>(&output);
            assert!((values[0] as f64 - expected.re).abs() <= 1e-4, "{values:?}");
            assert!((values[1] as f64 - expected.im).abs() <= 1e-6, "{values:?}");
        }
    }

    #[test]
    fn test_parallel_reduce_operation_lower_executes_log_sum_exp_gradient_on_cpu() {
        let client = cpu_client(4);
        let session = session_with_client(&client);
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The gradient of a logarithmic sum of exponentials is the softmax of its inputs. The composed mesh form
        // differentiates through its mesh sum alone, because its mesh maximum only shifts the exponents.
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
                                    local_x
                                        .reduce(&[0], ReductionKind::Sum)?
                                        .parallel_reduce(ReductionKind::LogSumExp, "x")
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
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = [1000.0f32, 1001.0, 1002.0, f32::NEG_INFINITY];
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
        let normalizer = 0f64.exp() + 1f64.exp() + 2f64.exp();
        let softmax = [0f64.exp() / normalizer, 1f64.exp() / normalizer, 2f64.exp() / normalizer, 0.0];
        assert_eq!(outputs.len(), 4);
        for (output, expected_gradient) in outputs.into_iter().zip(softmax) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap())[0])
                .collect::<Vec<_>>();
            assert!((values[0] as f64 - (1000.0 + normalizer.ln())).abs() <= 1e-4, "{values:?}");
            assert!((values[1] as f64 - expected_gradient).abs() <= 1e-6, "{values:?}");
        }
    }

    #[test]
    fn test_parallel_reduce_operation_lower_mean_divides_by_axis_size() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Mean, "x"),
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

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      %cst = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %3 = stablehlo.divide %1, %2 : tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_parallel_reduce_operation_lower_grouped_mean_preserves_group_order_and_uses_group_divisor() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            local_x.parallel_reduce_with_axis_index_groups(
                                ReductionKind::Mean,
                                "x",
                                vec![vec![0, 2], vec![3, 1]],
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
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %3 = stablehlo.divide %1, %2 : tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_parallel_reduce_operation_lower_inside_condition() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        // The sum `parallel_reduce` sits inside a `condition` branch inside the `shard_map` body, so it lowers through
        // the nested control flow path. The threaded collective lowering state resolves the manual mesh axis inside the
        // `stablehlo.if` region and emits the same `all_reduce` as a sum `parallel_reduce` in the body itself would.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            let local_type = local_x.r#type().into_owned();
                            let parallel_sum_branch = {
                                let mut builder = ProgramBuilder::new();
                                let input = builder.add_input(ArrayIrType::Array(local_type.clone()));
                                let output = builder
                                    .add_instruction(
                                        ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()),
                                        Vec::new(),
                                        vec![input],
                                        None,
                                    )
                                    .unwrap()[0];
                                builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
                            };
                            let identity_branch = {
                                let mut builder = ProgramBuilder::new();
                                let input = builder.add_input(ArrayIrType::Array(local_type));
                                builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
                            };
                            let context = local_x.value().context().clone();
                            let predicate = context.lift(Constant::Boolean(true))?;
                            let mut outputs = context.stage_operation(
                                Operation::Condition(ConditionOperation::new()),
                                vec![parallel_sum_branch, identity_branch],
                                &[predicate, local_x.into_value()],
                            )?;
                            Ok(ValueProjection::<ArrayType>::into_projected(outputs.remove(0)).unwrap())
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

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %c = stablehlo.constant dense<true> : tensor<i1>
                      %1 = "stablehlo.if"(%c) ({
                        %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                          %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                          stablehlo.return %3 : tensor<f32>
                        }) : (tensor<2xf32>) -> tensor<2xf32>
                        stablehlo.return %2 : tensor<2xf32>
                      }, {
                        stablehlo.return %arg1 : tensor<2xf32>
                      }) : (tensor<i1>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_parallel_reduce_operation_lower_inside_named_batch() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        // A named `batch` level inside the `shard_map` body binds `"b"`, while the sum `parallel_reduce` names the
        // *mesh* axis `"x"`. The batching rule forwards the collective through the batch level to the seeded base
        // tracing context, which binds `"x"`, so it lands in the body program on the batched physical value and lowers
        // to the same `all_reduce` as a direct sum `parallel_reduce`. Axis resolution thus composes across binder
        // kinds.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| {
                            let context = local_x.domain();
                            let summed: ArrayTracer = Batch::batch(
                                &context,
                                |item| item.parallel_reduce(ReductionKind::Sum, "x"),
                                local_x,
                                BatchAxis::new(0),
                                BatchAxis::new(0),
                                BatchAxisSpecification::named("b"),
                            )?;
                            Ok(summed)
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

        // The module is identical to the direct sum `parallel_reduce` module, because the batch level forwarded the
        // mesh collective untouched.
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_parallel_reduce_operation_lower_rejects_ordinary_log_sum_exp() {
        // The `ParallelReduce` capability composes a logarithmic sum over a manual mesh axis from primitive mesh
        // reductions, so only a hand-staged ordinary reduction can reach the lowering, which has no combiner for it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                shard_map(
                    |local_x: ArrayTracer| {
                        let context = local_x.value().context().clone();
                        let mut outputs = context.stage_operation(
                            ParallelReduceOperation::new(ReductionKind::LogSumExp, "x".to_string()),
                            Vec::new(),
                            &[local_x.into_value()],
                        )?;
                        Ok(ValueProjection::<ArrayType>::into_projected(outputs.remove(0))?)
                    },
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                    ShardMapOptions::new(),
                )
                .unwrap()
            },
            ArrayType::new_static(DataType::F32, [8]),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap_err().to_string(),
            "`parallel_reduce` with kind `log_sum_exp` has no all-reduce combiner; stage it through the \
             `ParallelReduce` capability, which composes it from primitive mesh reductions",
        );
    }

    #[test]
    fn test_parallel_reduce_operation_lower_revalidates_meshless_bindings() {
        // A meshless (e.g., grouped) reduction inside a checked binding must record outputs that vary along its axis,
        // because an output typed as invariant would be accepted as replicated although the devices can hold
        // different values. A mesh reduction tracks manual variation, so its binding must check it.
        let mesh = test_manual_mesh("x", 2);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();
        let checked = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let unchecked = checked.clone().with_check_manual_variation(false);
        let lower = |operation: &ParallelReduceOperation, boundary: &ShardMapBoundary, sharding: &Sharding| {
            let input_type = test_vector_type(2).with_sharding(sharding.clone()).unwrap();
            let output_types = operation.infer_output_types(std::slice::from_ref(&input_type), &[]).unwrap();
            let state = CollectiveLoweringState::new().enter_manual_region(boundary.clone());
            let context = MlirContext::new();
            let location = context.unknown_location();
            let tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
            let block = context.block(&[(tensor_type, location)]);
            let input = block.argument(0).unwrap().as_ref();
            let mut lowerer: Lowerer<'_, '_, '_, Operation> =
                Lowerer::new(block.as_ref(), &context, location.as_ref()).with_collective_state(state);
            operation.lower(&mut lowerer, &[input], &[input_type], &output_types).map(|_| ())
        };
        let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string());
        let grouped =
            ParallelReduceOperation::grouped(ReductionKind::Sum, "x".to_string(), 2, vec![vec![0], vec![1]]).unwrap();
        let tracked = sum.clone().with_mesh(mesh);

        // Meshless reductions whose recorded outputs vary along the axis and every reduction of an unchecked binding
        // that records no mesh lower.
        lower(&grouped, &checked, &varying).unwrap();
        lower(&sum, &checked, &varying).unwrap();
        lower(&tracked, &checked, &varying).unwrap();
        lower(&sum, &unchecked, &replicated).unwrap();
        lower(&grouped, &unchecked, &replicated).unwrap();

        // Meshless reductions whose recorded outputs are invariant along the axis are rejected in a checked binding.
        for operation in [&sum, &grouped] {
            assert!(matches!(
                lower(operation, &checked, &replicated),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                        if *message == "`parallel_reduce` without tracked mesh metadata types its outputs as invariant \
                                        along manual axis `x` inside a checked manual region",
                ),
            ));
        }

        // Mesh reductions are rejected in an unchecked binding.
        assert!(matches!(
            lower(&tracked, &unchecked, &varying),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_reduce` cannot use tracked mesh metadata inside an unchecked manual binding",
            ),
        ));
    }
}
