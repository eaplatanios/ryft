//! Cross-phase XLA dynamic coverage.

#[cfg(feature = "cuda-13")]
use std::time::Instant;

use pretty_assertions::assert_eq;

use super::*;

#[cfg(feature = "cuda-13")]
#[test]
fn test_cuda_13_plugin_executes_bounded_dynamic_program_at_two_sizes() {
    fn trace_data_derived<'c>(
        extent: DimensionVariable,
        inputs: Vec<CompilationTracer<Domain<'c>>>,
    ) -> Result<CompilationTracer<Domain<'c>>, Error> {
        let [size, value] = inputs.as_slice() else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        };
        let dimension = size.to_dimension(extent)?;
        let broadcast = size
            .context()
            .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[value.clone(), dimension])?
            .remove(0);
        Ok(size
            .context()
            .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[broadcast])?
            .remove(0))
    }

    let plugin = load_cuda_13_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..Default::default()
        }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
    let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
    let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
        |input| input.clone() + input,
        dynamic_type,
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
        staged.lower(&domain).unwrap().compile(&domain).unwrap();
    assert_eq!(domain.program_count(), 1);

    for size in [4usize, 7] {
        let input_type = replicated_vector_type(&mesh, size);
        let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes(values.as_slice()).as_slice())
                .unwrap();
        for _ in 0..2 {
            // Both launches use the same bounded executable. The second also proves that replay does not
            // accidentally specialize or recompile the program for the concrete logical size.
            let output = compiled.executable().execute(&domain, Value::Array(input.clone())).unwrap();
            let Value::Array(output) = output else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            output.block_until_ready().unwrap();
            assert_eq!(output.shape().as_slice(), &[size]);
            assert_eq!(read_f32s(&client, &output), values.iter().map(|value| value * 2.0).collect::<Vec<_>>());
            assert_eq!(domain.program_count(), 1, "executing a loaded executable must not trigger recompilation");
        }
    }

    // The same plugin route also accepts an extent born from device data inside the program. The gateway's
    // ordered bounds assertion and the reduction's dynamic masking both survive compilation, and the scalar input
    // values select different logical extents without producing another retained specialization.
    let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
    let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
    let extent = DimensionVariable::new("data_extent", DimensionBounds::new(1, Some(8)).unwrap());
    let function: CompiledFunctionDispatcher<Domain<'_>, (), Vec<ArrayIrType>, ArrayIrType, _> = domain
        .jit_with_options(move |(), inputs| trace_data_derived(extent.clone(), inputs), Options::new(mesh.clone()));
    let call = |size: i64| {
        let size =
            Array::from_host_buffer(&domain, scalar_i64.clone(), mesh.clone(), size.to_ne_bytes().as_slice()).unwrap();
        let value =
            Array::from_host_buffer(&domain, scalar_f32.clone(), mesh.clone(), 2.0_f32.to_ne_bytes().as_slice())
                .unwrap();
        function.execute((), vec![Value::Array(size), Value::Array(value)])
    };
    assert_eq!(read_f32s(&client, program_array(&call(4).unwrap())), vec![8.0]);
    assert_eq!(read_f32s(&client, program_array(&call(7).unwrap())), vec![14.0]);
    let invalid = call(8).unwrap();
    let error = program_array(&invalid).block_until_ready().unwrap_err();
    assert!(
        error.to_string().contains(
            "`dimension_from_scalar` failed: input dimension `data_extent` = 8 is outside its declared bounds \
             [1, 8)",
        ),
        "{error}",
    );
    assert_eq!(function.specialization_count(), 1);
    assert_eq!(function.statistics().dispatch_misses, 1);
    assert_eq!(function.statistics().dispatch_hits, 2);
}

#[test]
fn test_data_derived_extent_retained_jit_reuses_one_specialization() {
    fn trace<'c>(
        extent: DimensionVariable,
        inputs: Vec<CompilationTracer<Domain<'c>>>,
    ) -> Result<CompilationTracer<Domain<'c>>, Error> {
        let [size, value] = inputs.as_slice() else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        };
        let dimension = size.to_dimension(extent)?;
        let broadcast = size
            .context()
            .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[value.clone(), dimension])?
            .remove(0);
        Ok(size
            .context()
            .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[broadcast])?
            .remove(0))
    }

    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
    let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
    let function: CompiledFunctionDispatcher<Domain<'_>, (), Vec<ArrayIrType>, ArrayIrType, _> =
        domain.jit_with_options(move |(), inputs| trace(extent.clone(), inputs), Options::new(mesh.clone()));

    let call = |size: i64| {
        let size =
            Array::from_host_buffer(&domain, scalar_i64.clone(), mesh.clone(), size.to_ne_bytes().as_slice()).unwrap();
        let value =
            Array::from_host_buffer(&domain, scalar_f32.clone(), mesh.clone(), 2.0_f32.to_ne_bytes().as_slice())
                .unwrap();
        function.execute((), vec![Value::Array(size), Value::Array(value)]).unwrap()
    };

    assert_eq!(read_f32s(&client, program_array(&call(2))), vec![4.0]);
    assert_eq!(read_f32s(&client, program_array(&call(4))), vec![8.0]);
    let statistics = function.statistics();
    assert_eq!(statistics.dispatch_misses, 1);
    assert_eq!(statistics.dispatch_hits, 1);
    assert_eq!(statistics.traces, 1);
    assert_eq!(statistics.lowerings, 1);
    assert_eq!(statistics.compilation_requests, 1);
    assert_eq!(function.specialization_count(), 1);
    assert_eq!(domain.program_count(), 1);
}

#[test]
fn test_data_derived_padding_disciplines_execute_without_observable_padding() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let size_type = replicated_scalar_type(&mesh, DataType::I64);
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
    let program = data_derived_padding_fixture(size_type.clone(), extent);

    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let execute = |size: i64| {
        let input =
            Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice()).unwrap();
        let eager = program
            .interpret_in_context(&domain, vec![Value::Array(input.clone())])
            .unwrap()
            .into_iter()
            .map(|value| match value {
                Value::Array(array) => array,
                Value::Dimension(_) => panic!("padding fixture returned a first-class dimension"),
                Value::Reference(_) => panic!("padding fixture returned a reference"),
            })
            .collect::<Vec<_>>();
        let compiled = domain.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(compiled.len(), eager.len());
        for (compiled, eager) in compiled.iter().zip(&eager) {
            assert_eq!(compiled.shape(), eager.shape());
            assert_eq!(compiled.data_type(), eager.data_type());
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let compiled_bytes = compiled
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let eager_bytes = eager
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            assert_eq!(compiled_bytes, eager_bytes);
        }
        compiled
    };

    let small = execute(2);
    assert_eq!(small[0].shape(), StaticShape::new(vec![2]));
    assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
    assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
    assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
    assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
    assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
    assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
    assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
    assert_eq!(read_booleans(&client, &small[7]), vec![true]);
    assert_eq!(read_booleans(&client, &small[8]), vec![true]);
    assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);
    assert_eq!(read_f32s(&client, &small[12]), vec![2.0]);
    assert_eq!(read_i32s(&client, &small[13]), vec![1]);
    assert_eq!(read_i32s(&client, &small[14]), vec![1]);
    assert_eq!(read_i32s(&client, &small[15]), vec![3, 3]);
    assert_eq!(read_i32s(&client, &small[16]), vec![1, 1, 1, 1]);

    // Row `i` of the `i + j` matrix reduces to `i + ln(1 + e + e² + e³)`, so exactly the live rows appear and a
    // physical suffix row would show up as an extra element. The tolerance is here because the expansion runs
    // `exp` and `log`, not because the discipline is approximate: the byte comparison against the reference
    // kernel inside `execute` is exact.
    assert_eq!(small[10].shape(), StaticShape::new(vec![2]));
    let log_sum_exp = read_f32s(&client, &small[10]);
    assert!((log_sum_exp[0] - 3.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
    assert!((log_sum_exp[1] - 4.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");

    // The prefix scan runs along the static axis, so row `i` accumulates `[i, 2i + 1, 3i + 3, 4i + 6]` while the
    // data-derived extent rides through on the rows it never mixes.
    assert_eq!(small[11].shape(), StaticShape::new(vec![2, 4]));
    assert_eq!(read_f32s(&client, &small[11]), vec![0.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 10.0]);

    // With a single live lane, every physical suffix lane of the data-derived axis is padding, which index
    // reductions must still never select.
    let single = execute(1);
    assert_eq!(read_i32s(&client, &single[13]), vec![0]);
    assert_eq!(read_i32s(&client, &single[14]), vec![0]);
    assert_eq!(read_i32s(&client, &single[15]), vec![3]);
    assert_eq!(read_i32s(&client, &single[16]), vec![0, 0, 0, 0]);

    let at_bound = execute(4);
    assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4]));
    assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
    assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
    assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
    assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
    assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
    assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
    assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
    assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
    assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
    assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
    assert_eq!(read_f32s(&client, &at_bound[12]), vec![24.0]);
    assert_eq!(read_i32s(&client, &at_bound[13]), vec![3]);
    assert_eq!(read_i32s(&client, &at_bound[14]), vec![3]);
    assert_eq!(read_i32s(&client, &at_bound[15]), vec![3, 3, 3, 3]);
    assert_eq!(read_i32s(&client, &at_bound[16]), vec![3, 3, 3, 3]);

    // At the bound the reduction and the scan grow by exactly the two extra live rows, which is what rules out a
    // padded row silently participating at the smaller size.
    assert_eq!(at_bound[10].shape(), StaticShape::new(vec![4]));
    let log_sum_exp = read_f32s(&client, &at_bound[10]);
    assert!((log_sum_exp[0] - 3.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
    assert!((log_sum_exp[1] - 4.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
    assert!((log_sum_exp[2] - 5.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
    assert!((log_sum_exp[3] - 6.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
    assert_eq!(at_bound[11].shape(), StaticShape::new(vec![4, 4]));
    assert_eq!(
        read_f32s(&client, &at_bound[11]),
        vec![0.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 10.0, 2.0, 5.0, 9.0, 14.0, 3.0, 7.0, 12.0, 18.0],
    );
}

#[test]
fn test_data_derived_scaled_dot_and_attention_execute_without_observable_padding() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let size_type = replicated_scalar_type(&mesh, DataType::I64);
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
    let program = data_derived_scaled_dot_attention_fixture(
        size_type.clone(),
        extent,
        DataType::F32,
        DataType::F32,
        AttentionImplementation::Automatic,
    );
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    assert!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""));
    assert!(!lowered.stable_hlo().contains("__cudnn$fmha"));
    let compiled = domain.compile_program(&lowered).unwrap();
    let execute = |size: i64| {
        let input =
            Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice()).unwrap();
        let eager = program
            .interpret_in_context(&domain, vec![Value::Array(input.clone())])
            .unwrap()
            .into_iter()
            .map(|value| match value {
                Value::Array(array) => array,
                Value::Dimension(_) => panic!("attention fixture returned a first-class dimension"),
                Value::Reference(_) => panic!("attention fixture returned a reference"),
            })
            .collect::<Vec<_>>();
        let compiled = domain.execute_program(&compiled, vec![input]).unwrap();
        assert_eq!(compiled.len(), eager.len());
        for (compiled, eager) in compiled.iter().zip(&eager) {
            assert_eq!(compiled.shape(), eager.shape());
            assert_eq!(compiled.data_type(), eager.data_type());
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            assert_eq!(
                compiled
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap(),
                eager
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap(),
            );
        }
        compiled
    };

    let small = execute(2);
    assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
    assert_eq!(read_f32s(&client, &small[0]), vec![16.0; 2]);
    assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
    assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
    assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
    assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
    assert_eq!(
        read_bf16s(&client, &small[4]),
        [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
    );

    let at_bound = execute(4);
    assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
    assert_eq!(read_f32s(&client, &at_bound[0]), vec![16.0; 4]);
    assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
    assert_eq!(
        read_bf16s(&client, &at_bound[1]),
        [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
    );
    assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
    // Three visible keys have equal scores and values 0, 1, 2. Each query contributes
    // `dS[j] = (1 / 3) * 8 * (j - 1)`, so summing three visible queries gives `dK[j] = 8 * (j - 1)`
    // and `dV[j] = 3 * (1 / 3) = 1`. The rounded BF16 residual must not rescale these probabilities.
    assert_eq!(
        read_bf16s(&client, &at_bound[3]),
        [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
    );
    assert_eq!(
        read_bf16s(&client, &at_bound[4]),
        [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
    );

    let zero =
        Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), 0_i64.to_ne_bytes().as_slice()).unwrap();
    let error = domain.execute_program(&compiled, vec![zero]).unwrap_err();
    assert!(
        error.to_string().contains(
            "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
        ),
        "{error}",
    );

    // Eager execution replays the attention calls of the bounded-dynamic program at the exact types of their inputs,
    // so a program that explicitly requests the fused kernels reports that it cannot use them instead of silently
    // computing the portable result.
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
    let fused = data_derived_scaled_dot_attention_fixture(
        size_type.clone(),
        extent,
        DataType::F32,
        DataType::F32,
        AttentionImplementation::Fused,
    );
    let input = Array::from_host_buffer(&domain, size_type, mesh, 2_i64.to_ne_bytes().as_slice()).unwrap();
    let error = fused.interpret_in_context(&domain, vec![Value::Array(input)]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("the fused attention implementation is unavailable because the target platform is not CUDA"),
        "{error}",
    );
}

#[cfg(feature = "cuda-13")]
#[test]
fn test_cuda_data_derived_padding_disciplines_execute_without_observable_padding() {
    let plugin = load_cuda_13_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..Default::default()
        }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let size_type = replicated_scalar_type(&mesh, DataType::I64);
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
    let program = data_derived_padding_fixture(size_type.clone(), extent);
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let execute = |size: i64| {
        let input =
            Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice()).unwrap();
        domain.execute_program(&compiled, vec![input]).unwrap()
    };

    let small = execute(2);
    assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
    assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
    assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
    assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
    assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
    assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
    assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
    assert_eq!(read_booleans(&client, &small[7]), vec![true]);
    assert_eq!(read_booleans(&client, &small[8]), vec![true]);
    assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);
    assert_eq!(read_f32s(&client, &small[12]), vec![2.0]);
    assert_eq!(read_i32s(&client, &small[13]), vec![1]);
    assert_eq!(read_i32s(&client, &small[14]), vec![1]);
    assert_eq!(read_i32s(&client, &small[15]), vec![3, 3]);
    assert_eq!(read_i32s(&client, &small[16]), vec![1, 1, 1, 1]);

    let at_bound = execute(4);
    assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
    assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
    assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
    assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
    assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
    assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
    assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
    assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
    assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
    assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
    assert_eq!(read_f32s(&client, &at_bound[12]), vec![24.0]);
    assert_eq!(read_i32s(&client, &at_bound[13]), vec![3]);
    assert_eq!(read_i32s(&client, &at_bound[14]), vec![3]);
    assert_eq!(read_i32s(&client, &at_bound[15]), vec![3, 3, 3, 3]);
    assert_eq!(read_i32s(&client, &at_bound[16]), vec![3, 3, 3, 3]);
}

#[cfg(feature = "cuda-13")]
#[test]
fn test_cuda_data_derived_scaled_dot_and_attention_execute_fused_without_observable_padding() {
    let plugin = load_cuda_13_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..Default::default()
        }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    for (scaled_element_type, scaled_scale_type) in [
        (DataType::F4E2M1FN, DataType::F8E4M3FN),
        (DataType::F8E4M3FN, DataType::F8E8M0FNU),
        (DataType::F8E5M2, DataType::F8E8M0FNU),
    ] {
        let scaled_dot_value = match scaled_element_type {
            DataType::F8E4M3FN | DataType::F8E5M2 => 32.0,
            _ => 16.0,
        };
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_scaled_dot_attention_fixture(
            size_type.clone(),
            extent,
            scaled_element_type,
            scaled_scale_type,
            AttentionImplementation::Fused,
        );
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        // cuDNN accepts the NVFP4 and `f8e4m3fn` MXFP8 pairs, which lower to the block-scaled-dot custom call, but
        // not `f8e5m2` on both sides, which keeps the composite.
        let uses_custom_call = scaled_element_type != DataType::F8E5M2;
        assert_eq!(lowered.stable_hlo().contains("stablehlo.custom_call @__op$block_scaled_dot"), uses_custom_call);
        assert_eq!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""), !uses_custom_call);
        assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
        assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmaxBackward"));
        let compiled = domain.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            domain.execute_program(&compiled, vec![input]).unwrap()
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
        assert_eq!(read_f32s(&client, &small[0]), vec![scaled_dot_value; 2]);
        assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
        assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
        assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
        assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
        assert_eq!(
            read_bf16s(&client, &small[4]),
            [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![scaled_dot_value; 4]);
        assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
        assert_eq!(
            read_bf16s(&client, &at_bound[1]),
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );
        assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
        let actual_query_gradient = read_bf16s(&client, &at_bound[3]);
        let expected_query_gradient =
            [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
        // cuDNN's fused BF16 backward-attention kernel is not bit-identical to the unfused reference
        // composition, so compare its gradients with an explicit reduced-precision tolerance.
        actual_query_gradient.iter().zip(&expected_query_gradient).for_each(|(actual, expected)| {
            assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
        });
        let actual_value_gradient = read_bf16s(&client, &at_bound[4]);
        let expected_value_gradient =
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
        actual_value_gradient.iter().zip(&expected_value_gradient).for_each(|(actual, expected)| {
            assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
        });

        let zero = Array::from_host_buffer(&domain, size_type, mesh.clone(), 0_i64.to_ne_bytes().as_slice()).unwrap();
        let error = domain.execute_program(&compiled, vec![zero]).unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
            ),
            "{error}",
        );
    }
}

#[test]
fn test_data_derived_prefix_slice_executes_at_multiple_extents() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(4)]))
        .with_sharding(input_sharding.clone())
        .unwrap();
    let values_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
        .with_sharding(input_sharding)
        .unwrap();
    let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

    // This is the compiled counterpart of the Phase 4 prefix-take fixture: the mask determines a device-side
    // count, the checked gateway turns that count into a bounded dimension, and the slice returns that many
    // leading values without a host readback.
    let mut builder = ProgramBuilder::new();
    let mask = builder.add_input(mask_type.into());
    let values = builder.add_input(values_type.into());
    let mask = builder
        .add_instruction(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::I64, false),
            Vec::new(),
            vec![mask],
            None,
        )
        .unwrap()[0];
    let count = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![mask], None)
        .unwrap()[0];
    let count = builder
        .add_instruction(DimensionFromScalarOperation::new(count_variable.into()), Vec::new(), vec![count], None)
        .unwrap()[0];
    let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
    let output = builder
        .add_instruction(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1),
            Vec::new(),
            vec![values, start, count],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
        .unwrap();

    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let values = || f32_vector(&domain, &mesh, &[10.0, 20.0, 30.0, 40.0]);
    let execute = |mask: &[bool]| {
        domain.execute_program(&compiled, vec![boolean_vector(&domain, &mesh, mask), values()]).unwrap()
    };

    let two = execute(&[true, false, true, false]);
    assert_eq!(two[0].shape(), StaticShape::new(vec![2]));
    assert_eq!(read_f32s(&client, &two[0]), vec![10.0, 20.0]);
    let four = execute(&[true, true, true, true]);
    assert_eq!(four[0].shape(), StaticShape::new(vec![4]));
    assert_eq!(read_f32s(&client, &four[0]), vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_data_derived_prefix_slice_executes_with_duplicate_and_equal_outputs() {
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(4)]))
        .with_sharding(input_sharding.clone())
        .unwrap();
    let values_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
        .with_sharding(input_sharding)
        .unwrap();
    let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

    // XLA merges bounded dynamic results that it proves equal and, on CPU, fills the duplicate result with a plain
    // copy that leaves its runtime-size trailer at zero (CUDA preserves the trailer). The program returns the same
    // prefix slice twice and then an equal slice staged separately, which covers duplicate entry results and
    // results merged by XLA's CSE.
    let mut builder = ProgramBuilder::new();
    let mask = builder.add_input(mask_type.into());
    let values = builder.add_input(values_type.into());
    let mask = builder
        .add_instruction(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::I64, false),
            Vec::new(),
            vec![mask],
            None,
        )
        .unwrap()[0];
    let count = builder
        .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![mask], None)
        .unwrap()[0];
    let count = builder
        .add_instruction(DimensionFromScalarOperation::new(count_variable.into()), Vec::new(), vec![count], None)
        .unwrap()[0];
    let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
    let mut prefix = || {
        builder
            .add_instruction(
                DynamicSliceOperation::<ArrayIrType>::from_rank(1),
                Vec::new(),
                vec![values, start, count],
                None,
            )
            .unwrap()[0]
    };
    let output = prefix();
    let equal_output = prefix();
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(
            vec![output, output, equal_output],
            vec![Placeholder, Placeholder],
            vec![Placeholder; 3],
        )
        .unwrap();

    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let values = || f32_vector(&domain, &mesh, &[10.0, 20.0, 30.0, 40.0]);
    let execute = |mask: &[bool]| {
        domain.execute_program(&compiled, vec![boolean_vector(&domain, &mesh, mask), values()]).unwrap()
    };

    let two = execute(&[true, false, true, false]);
    assert_eq!(two.len(), 3);
    for output in &two {
        assert_eq!(output.shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, output), vec![10.0, 20.0]);
    }
    let four = execute(&[true, true, true, true]);
    assert_eq!(four.len(), 3);
    for output in &four {
        assert_eq!(output.shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, output), vec![10.0, 20.0, 30.0, 40.0]);
    }
}

#[test]
fn test_bounded_dynamic_bitcast_element_type_preserves_prefix_extents() {
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let declared_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![CoreDimension::Dynamic(DimensionVariable::new(
            "count",
            DimensionBounds::new(0, Some(5)).unwrap(),
        ))]),
    )
    .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
    .unwrap();
    let mut builder = ProgramBuilder::new();
    let values = builder.add_input(declared_type.into());
    let halves = builder
        .add_instruction(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true),
            Vec::new(),
            vec![values],
            None,
        )
        .unwrap()[0];
    let restored = builder
        .add_instruction(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true),
            Vec::new(),
            vec![halves],
            None,
        )
        .unwrap()[0];
    let bits = builder
        .add_instruction(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::I1, true),
            Vec::new(),
            vec![values],
            None,
        )
        .unwrap()[0];
    let restored_bits = builder
        .add_instruction(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true),
            Vec::new(),
            vec![bits],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(
            vec![halves, restored, bits, restored_bits],
            vec![Placeholder],
            vec![Placeholder; 4],
        )
        .unwrap();
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    // Splitting and joining the static bit-width axis preserves an independent dynamic prefix and its live bytes.
    for count in [2usize, 4] {
        let outputs = domain.execute_program(&compiled, vec![f32_vector(&domain, &mesh, &vec![1.0; count])]).unwrap();
        assert_eq!(outputs[0].shape(), StaticShape::new(vec![count, 2]));
        assert_eq!(outputs[1].shape(), StaticShape::new(vec![count]));
        assert_eq!(read_f32s(&client, &outputs[1]), vec![1.0; count]);
        assert_eq!(outputs[2].shape(), StaticShape::new(vec![count, 32]));
        assert_eq!(outputs[3].shape(), StaticShape::new(vec![count]));
        assert_eq!(read_f32s(&client, &outputs[3]), vec![1.0; count]);
        let bytes = outputs[0]
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(bytes, values_to_bytes(&vec![1.0f32; count]));
    }
}

#[test]
fn test_data_derived_dynamic_slice_checks_runtime_input_and_result_extents() {
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let size_type = replicated_scalar_type(&mesh, DataType::I64);
    let value_type = replicated_scalar_type(&mesh, DataType::F32);
    let input_extent = DimensionVariable::new("input_extent", DimensionBounds::new(1, Some(5)).unwrap());
    let result_extent = DimensionVariable::new("result_extent", DimensionBounds::new(0, Some(5)).unwrap());

    // The result extent is independent of the input extent at the type level. Lowering uses their finite physical
    // bounds for one specialization and preserves the eager `result_extent <= input_extent` contract with an
    // ordered runtime assertion.
    let mut builder = ProgramBuilder::new();
    let input_size = builder.add_input(size_type.clone().into());
    let result_size = builder.add_input(size_type.clone().into());
    let value = builder.add_input(value_type.clone().into());
    let input_dimension = builder
        .add_instruction(DimensionFromScalarOperation::new(input_extent.into()), Vec::new(), vec![input_size], None)
        .unwrap()[0];
    let result_dimension = builder
        .add_instruction(DimensionFromScalarOperation::new(result_extent.into()), Vec::new(), vec![result_size], None)
        .unwrap()[0];
    let input = builder
        .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, input_dimension], None)
        .unwrap()[0];
    let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
    let output = builder
        .add_instruction(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1),
            Vec::new(),
            vec![input, start, result_dimension],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(
            vec![output],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )
        .unwrap();

    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    assert!(lowered.stable_hlo().contains("@ryft.assert"));
    let compiled = domain.compile_program(&lowered).unwrap();
    let scalar =
        |r#type: ArrayType, bytes: &[u8]| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap();
    let inputs = |input_size: i64, result_size: i64| {
        vec![
            scalar(size_type.clone(), input_size.to_ne_bytes().as_slice()),
            scalar(size_type.clone(), result_size.to_ne_bytes().as_slice()),
            scalar(value_type.clone(), 7.0_f32.to_ne_bytes().as_slice()),
        ]
    };

    let eager = program
        .interpret_in_context(&domain, inputs(4, 2).into_iter().map(Value::Array).collect::<Vec<_>>())
        .unwrap();
    let compiled_valid = domain.execute_program(&compiled, inputs(4, 2)).unwrap();
    assert_eq!(read_f32s(&client, program_array(&eager[0])), vec![7.0, 7.0]);
    assert_eq!(read_f32s(&client, &compiled_valid[0]), read_f32s(&client, program_array(&eager[0])));

    let error = match domain.execute_program(&compiled, inputs(2, 4)) {
        Ok(invalid) => invalid[0].block_until_ready().unwrap_err().to_string(),
        Err(error) => error.to_string(),
    };
    assert_eq!(error, "`dynamic_slice` limit 4 exceeds input axis 0 extent 2");
}

#[cfg(feature = "cuda-13")]
#[test]
#[ignore = "hardware throughput benchmark; run explicitly on a CUDA system"]
fn bench_cuda_fused_attention_throughput() {
    let plugin = load_cuda_13_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..Default::default()
        }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let [batch, sequence, heads, head_dimension] = [4, 2048, 32, 128];
    let r#type = ArrayType::new(
        DataType::BF16,
        Shape::new(vec![
            CoreDimension::Static(batch),
            CoreDimension::Static(sequence),
            CoreDimension::Static(heads),
            CoreDimension::Static(head_dimension),
        ]),
    )
    .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 4))
    .unwrap();
    let attention = dot_product_attention::<Domain<'static>>(
        AttentionConfiguration::new().with_scale(1.0 / (head_dimension as f64).sqrt()),
    );
    let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
        |inputs: Vec<Tracer<'static>>| {
            let (output, _) =
                attention.call(AttentionInputs::new(inputs[0].clone(), inputs[1].clone(), inputs[2].clone()))?;
            Ok(vec![output])
        },
        vec![ArrayIrType::from(r#type.clone()); 3],
    )
    .unwrap();
    let program = program.into_flat_program();
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
    let compilation_start = Instant::now();
    let compiled = domain.compile_program(&lowered).unwrap();
    let compilation_duration = compilation_start.elapsed();

    let one = 0x3f80_u16.to_ne_bytes();
    let mut bytes = vec![0_u8; batch * sequence * heads * head_dimension * size_of::<u16>()];
    bytes.chunks_exact_mut(size_of::<u16>()).for_each(|element| element.copy_from_slice(&one));
    let input = Array::from_host_buffer(&domain, r#type, mesh, bytes.as_slice()).unwrap();
    input.block_until_ready().unwrap();
    let execute = || {
        let outputs = domain.execute_program(&compiled, vec![input.clone(), input.clone(), input.clone()])?;
        outputs[0].block_until_ready()?;
        Ok::<_, Error>(())
    };
    // The Spark idles at its lowest power state, so warm the fused kernel long enough for clocks to stabilize
    // before collecting synchronized samples.
    (0..20).for_each(|_| execute().unwrap());
    let mut samples = (0..20)
        .map(|_| {
            let start = Instant::now();
            execute().unwrap();
            start.elapsed().as_nanos()
        })
        .collect::<Vec<_>>();
    samples.sort_unstable();
    let median_nanoseconds = samples[samples.len() / 2];
    let floating_point_operations =
        4.0 * batch as f64 * heads as f64 * sequence as f64 * sequence as f64 * head_dimension as f64;
    let throughput_teraflops = floating_point_operations / (median_nanoseconds as f64 * 1e3);

    eprintln!(
        "fused attention b{batch} h{heads} s{sequence} d{head_dimension}: compile={compilation_duration:?}, \
         median={:.3} ms, throughput={throughput_teraflops:.2} TFLOP/s",
        median_nanoseconds as f64 / 1e6,
    );
}
