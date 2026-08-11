//! Matched lifecycle and execution benchmark for comparison with JAX.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use ryft_core::{
    ArrayIrType, ArrayIrValue, ArrayType, CompilationDomain, CompiledFunction, DataType, Device, DeviceMesh, Dimension,
    DiskCache, JitCacheStatistics, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, Sin, StagedFunction,
    ValueProjection, stage_function,
};
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::{Array, FromPjrt, JittedXlaFunction, XlaCompileTracer, XlaDomain, XlaOptions, jitted};
use serde_json::{Value, json};

type BenchmarkStagedFunction<'c> = StagedFunction<XlaDomain<'c>, ArrayIrType, ArrayIrType>;
type BenchmarkCompiledFunction<'c> = CompiledFunction<XlaDomain<'c>, ArrayIrType, ArrayIrType>;

#[derive(Debug)]
struct Arguments {
    iterations: usize,
    size: usize,
    smoke: bool,
    cache_directory: Option<PathBuf>,
}

impl Default for Arguments {
    fn default() -> Self {
        Self { iterations: 50, size: 1 << 20, smoke: false, cache_directory: None }
    }
}

fn usage() -> &'static str {
    "Usage: compilation_benchmark [OPTIONS]\n\
     \n\
     Options:\n\
       --iterations N     Timed warm iterations (default: 50)\n\
       --size N           Number of f32 elements (default: 1048576)\n\
       --cache-dir PATH   Measure persistent restore using PATH\n\
       --smoke            Use small defaults and assert cache invariants\n\
       -h, --help         Print this help"
}

fn parse_arguments() -> Result<Option<Arguments>, String> {
    let mut arguments = Arguments::default();
    let mut values = env::args().skip(1);
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(None);
            }
            "--iterations" => {
                arguments.iterations = values
                    .next()
                    .ok_or("expected a value after --iterations")?
                    .parse()
                    .map_err(|error| format!("invalid --iterations value: {error}"))?;
            }
            "--size" => {
                arguments.size = values
                    .next()
                    .ok_or("expected a value after --size")?
                    .parse()
                    .map_err(|error| format!("invalid --size value: {error}"))?;
            }
            "--cache-dir" => {
                arguments.cache_directory =
                    Some(PathBuf::from(values.next().ok_or("expected a path after --cache-dir")?));
            }
            "--smoke" => arguments.smoke = true,
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    if arguments.smoke {
        arguments.iterations = arguments.iterations.min(3);
        arguments.size = arguments.size.min(1024);
    }
    if arguments.iterations == 0 || arguments.size == 0 {
        return Err("--iterations and --size must both be greater than zero".into());
    }
    Ok(Some(arguments))
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_string())
}

fn duration_nanoseconds(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn distribution(samples: &[u64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let percentile = |percent: usize| {
        let index = ((sorted.len() - 1) * percent).div_ceil(100);
        sorted[index]
    };
    json!({
        "samples": sorted.len(),
        "minimum_ns": sorted[0],
        "p50_ns": percentile(50),
        "p95_ns": percentile(95),
        "p99_ns": percentile(99),
        "maximum_ns": sorted[sorted.len() - 1],
    })
}

fn jit_statistics(statistics: JitCacheStatistics) -> Value {
    json!({
        "dispatch_hits": statistics.dispatch_hits,
        "dispatch_misses": statistics.dispatch_misses,
        "traces": statistics.traces,
        "lowerings": statistics.lowerings,
        "compilation_requests": statistics.compilation_requests,
        "input_abstractification_duration_ns": statistics.input_abstractification_duration_ns,
        "dispatch_duration_ns": statistics.dispatch_duration_ns,
        "tracing_duration_ns": statistics.tracing_duration_ns,
        "lowering_duration_ns": statistics.lowering_duration_ns,
    })
}

fn compilation_statistics(domain: &XlaDomain<'_>) -> Value {
    let statistics = domain.compilation_context().statistics();
    json!({
        "memory_hits": statistics.memory_hits,
        "persistent_hits": statistics.persistent_hits,
        "misses": statistics.misses,
        "compilations": statistics.compilations,
        "waits": statistics.waits,
        "persistent_errors": statistics.persistent_errors,
        "persistent_lookup_duration_ns": statistics.persistent_lookup_duration_ns,
        "memory_lookup_duration_ns": statistics.memory_lookup_duration_ns,
        "compilation_duration_ns": statistics.compilation_duration_ns,
    })
}

fn mesh(client: &Client<'_>) -> Result<DeviceMesh, Box<dyn std::error::Error>> {
    let device = Device::from_pjrt(&client.addressable_devices()?.remove(0))?;
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
    Ok(DeviceMesh::new(logical_mesh, vec![device])?)
}

fn input_type(mesh: &DeviceMesh, size: usize) -> Result<ArrayType, Box<dyn std::error::Error>> {
    Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))?)
}

fn input_array<'c>(
    client: &'c Client<'c>,
    mesh: &DeviceMesh,
    r#type: ArrayType,
    size: usize,
) -> Result<Array<'c>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(size * size_of::<f32>());
    for index in 0..size {
        bytes.extend_from_slice(&((index % 1024) as f32 / 1024.0).to_ne_bytes());
    }
    Ok(Array::from_host_buffer(client, r#type, mesh.clone(), bytes.as_slice())?)
}

fn stage_workload<'c>(
    domain: &XlaDomain<'c>,
    mesh: &DeviceMesh,
    r#type: ArrayType,
) -> Result<BenchmarkStagedFunction<'c>, Box<dyn std::error::Error>> {
    Ok(stage_function(
        domain,
        |input| {
            let input = ValueProjection::<ArrayType>::into_projected(input).unwrap();
            (input.clone() * input.clone() + input).sin().unwrap().into_value()
        },
        ArrayIrType::Array(r#type),
        XlaOptions::new(mesh.clone()),
    )?)
}

fn call_workload<'c>(
    domain: &XlaDomain<'c>,
    compiled: &BenchmarkCompiledFunction<'c>,
    input: Array<'c>,
) -> Result<Array<'c>, Box<dyn std::error::Error>> {
    match ryft_core::compilation::call_function(domain, compiled.executable_function(), ArrayIrValue::Array(input))? {
        ArrayIrValue::Array(output) => Ok(output),
        ArrayIrValue::Dimension(_) => Err("compilation benchmark produced a first-class dimension".into()),
    }
}

fn persistent_restore(
    client: &Client<'_>,
    mesh: &DeviceMesh,
    r#type: &ArrayType,
    directory: &Path,
) -> Result<Value, Box<dyn std::error::Error>> {
    let producer = XlaDomain::with_configured_disk_cache(
        client,
        DiskCache::open(directory)?.with_write_thresholds(Duration::ZERO, 0),
    );
    let producer_staged = stage_workload(&producer, mesh, r#type.clone())?;
    let producer_lowered = producer.lower(producer_staged)?;
    producer.compile(producer_lowered)?;

    let restored = XlaDomain::with_configured_disk_cache(
        client,
        DiskCache::open(directory)?.with_write_thresholds(Duration::ZERO, 0),
    );
    let restored_staged = stage_workload(&restored, mesh, r#type.clone())?;
    let restored_lowered = restored.lower(restored_staged)?;
    let start = Instant::now();
    restored.compile(restored_lowered)?;
    let duration = duration_nanoseconds(start);
    let statistics = restored.compilation_context().statistics();
    Ok(json!({
        "requested": true,
        "restored": statistics.persistent_hits == 1 && statistics.compilations == 0,
        "duration_ns": duration,
        "statistics": compilation_statistics(&restored),
        "note": "both benchmark contexts use zero compile-duration and entry-size write thresholds",
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(arguments) = parse_arguments().map_err(std::io::Error::other)? else {
        return Ok(());
    };

    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) }))?;
    let mesh = mesh(&client)?;
    let r#type = input_type(&mesh, arguments.size)?;
    let input = input_array(&client, &mesh, r#type.clone(), arguments.size)?;
    input.block_until_ready()?;
    let domain = XlaDomain::new(&client);
    let start = Instant::now();
    let staged = stage_workload(&domain, &mesh, r#type.clone())?;
    let cold_trace_ns = duration_nanoseconds(start);

    let start = Instant::now();
    let lowered = domain.lower(staged)?;
    let cold_lower_ns = duration_nanoseconds(start);

    let start = Instant::now();
    let compiled: BenchmarkCompiledFunction<'_> = domain.compile(lowered)?;
    let cold_compile_ns = duration_nanoseconds(start);

    let warmup = call_workload(&domain, &compiled, input.clone())?;
    warmup.block_until_ready()?;

    let mut enqueue_samples = Vec::with_capacity(arguments.iterations);
    let mut pending_outputs = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        pending_outputs.push(call_workload(&domain, &compiled, input.clone())?);
        enqueue_samples.push(duration_nanoseconds(start));
    }
    for output in pending_outputs {
        output.block_until_ready()?;
    }

    let mut synchronized_samples = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        call_workload(&domain, &compiled, input.clone())?.block_until_ready()?;
        synchronized_samples.push(duration_nanoseconds(start));
    }

    let dispatcher: JittedXlaFunction<'_, _, (), ArrayType, ArrayType> = jitted(
        |(), input: XlaCompileTracer<'_>| (input.clone() * input.clone() + input).sin().unwrap(),
        &domain,
        mesh.clone(),
    );
    dispatcher.call((), input.clone())?.block_until_ready()?;
    let cold_dispatch_statistics = dispatcher.statistics();
    let mut warm_dispatch_samples = Vec::with_capacity(arguments.iterations);
    let mut warm_outputs = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        warm_outputs.push(dispatcher.call((), input.clone())?);
        warm_dispatch_samples.push(duration_nanoseconds(start));
    }
    for output in warm_outputs {
        output.block_until_ready()?;
    }
    let warm_dispatch_statistics = dispatcher.statistics();

    if arguments.smoke {
        assert_eq!(cold_dispatch_statistics.traces, 1, "cold dispatch must trace exactly once");
        assert_eq!(cold_dispatch_statistics.lowerings, 1, "cold dispatch must lower exactly once");
        assert_eq!(cold_dispatch_statistics.compilation_requests, 1, "cold dispatch must request compilation once");
        assert_eq!(warm_dispatch_statistics.traces, 1, "warm dispatch must not retrace");
        assert_eq!(warm_dispatch_statistics.lowerings, 1, "warm dispatch must not relower");
        assert_eq!(warm_dispatch_statistics.compilation_requests, 1, "warm dispatch must not recompile");
        assert_eq!(warm_dispatch_statistics.dispatch_hits, arguments.iterations as u64);
    }

    let persistent = match &arguments.cache_directory {
        Some(directory) => persistent_restore(&client, &mesh, &r#type, directory)?,
        None => json!({"requested": false}),
    };
    if arguments.smoke && arguments.cache_directory.is_some() {
        assert_eq!(persistent["restored"], true, "persistent smoke mode must restore without recompiling");
        assert_eq!(persistent["statistics"]["persistent_hits"], 1);
        assert_eq!(persistent["statistics"]["compilations"], 0);
    }
    let device = client.addressable_devices()?.remove(0);
    let report = json!({
        "schema": "ryft-jax-compilation-benchmark-v1",
        "framework": "ryft",
        "workload": {
            "name": "elementwise_polynomial_sine",
            "expression": "sin(x * x + x)",
            "dtype": "float32",
            "shape": [arguments.size],
            "device_count": 1,
        },
        "configuration": {
            "iterations": arguments.iterations,
            "smoke": arguments.smoke,
            "cache_directory": arguments.cache_directory,
            "synchronization": "Array::block_until_ready",
        },
        "metadata": {
            "ryft_version": env!("CARGO_PKG_VERSION"),
            "git_commit": command_output("git", &["rev-parse", "HEAD"]),
            "rustc_version": command_output("rustc", &["--version"]),
            "operating_system": command_output("uname", &["-a"]),
            "platform_name": client.platform_name()?.into_owned(),
            "platform_version": client.platform_version()?.into_owned(),
            "device_id": device.id()?,
            "device_kind": device.kind()?.into_owned(),
            "xla_flags": env::var("XLA_FLAGS").unwrap_or_default(),
        },
        "lifecycle": {
            "cold_trace_ns": cold_trace_ns,
            "cold_lower_ns": cold_lower_ns,
            "cold_compile_ns": cold_compile_ns,
            "persistent_restore": persistent,
        },
        "execution": {
            "warm_dispatch_call": distribution(&warm_dispatch_samples),
            "enqueue_only": distribution(&enqueue_samples),
            "synchronized": distribution(&synchronized_samples),
        },
        "counters": {
            "after_cold_dispatch": jit_statistics(cold_dispatch_statistics),
            "after_warm_dispatch": jit_statistics(warm_dispatch_statistics),
            "compilation_cache": compilation_statistics(&domain),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
