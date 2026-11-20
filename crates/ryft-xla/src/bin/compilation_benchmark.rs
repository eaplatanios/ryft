//! Compilation lifecycle and execution benchmarks, including a matched lifecycle benchmark for comparison with JAX.
//!
//! The `--scenario` flag selects one of the following scenarios, each of which prints one JSON report:
//!
//!   - **`lifecycle`** (the default): The workload is `sin(x * x + x)` over an `f32` vector on one CPU device. The
//!     scenario measures the cold staging, lowering, and compilation times, the enqueue-only and synchronized execution
//!     times of the compiled function, the warm dispatch time of a [`jit`] dispatcher, and, with `--cache-dir`, the
//!     time to restore the compiled program from a persistent cache. Its report schema is shared with the JAX side of
//!     the comparison.
//!   - **`transforms`**: Measures, on one CPU device, the cold and warm calls of a [`jit`] dispatcher whose function
//!     runs an eager reverse mode gradient, the repeated construction of [`CompiledFunction::gradient`] and
//!     [`CompiledFunction::jvp`], which retain their derived functions after the first construction, and the cost of
//!     transforming one shared `jit_call` callee for several distinct outer functions. The latter is split into the
//!     backend compilation time that the shared compilation cache records and the remaining frontend time, for a
//!     short and a long callee, so that the frontend cost that grows with the callee is visible.
//!   - **`attention`**: Measures the staging, lowering, and compilation times and the synchronized execution time and
//!     throughput of bf16 scaled dot-product attention, on the CPU or, with the `cuda-13` feature and
//!     `RYFT_BENCHMARK_PLATFORM=cuda-13`, on a CUDA GPU, where it lowers to cuDNN's fused attention kernel.
//!
//! Run it with `cargo run --release -p ryft-xla --features performance-benchmarking --bin compilation_benchmark`, and
//! append `-- --scenario <scenario>` to select a scenario and `--smoke` to use small dimensions and iteration counts.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{Value as JsonValue, json};

use ryft_core::{
    ArrayOperation, ArrayType, AttentionConfiguration, AttentionInputs, CompilationTracer, Compile,
    CompiledFunction as CoreCompiledFunction, Cos, DataType, Device, DeviceMesh, Differentiate, Dimension, DiskCache,
    DispatcherStatistics, LogicalMesh, MeshAxis, MeshAxisType, Mul, Reduce, ReductionKind, Shape, Sharding, Sin,
    StagedFunction, Value as CoreValue, ValueProjection, dot_product_attention,
};
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
#[cfg(feature = "cuda-13")]
use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};
use ryft_xla::{
    Array, CompileTracer, CompiledFunction, CompiledFunctionDispatcher, FromPjrt, Operation, Options, Session, Type,
    Value, compile, jit,
};

// TODO(eaplatanios): Review this module.

/// Number of distinct outer functions that stage a `jit_call` of one shared callee in the `transforms` scenario.
const CALLEE_OUTER_SPECIALIZATIONS: usize = 4;

/// Lengths of the short and long callee chains compared by the `transforms` scenario.
const CALLEE_OPERATION_COUNTS: [usize; 2] = [2, 200];

/// Lengths of the short and long callee chains compared by the `transforms` scenario with `--smoke`.
const SMOKE_CALLEE_OPERATION_COUNTS: [usize; 2] = [2, 20];

/// Batch size, sequence length, head count, and head dimension of the `attention` scenario.
const ATTENTION_DIMENSIONS: [usize; 4] = [4, 2048, 32, 128];

/// Batch size, sequence length, head count, and head dimension of the `attention` scenario with `--smoke`.
const SMOKE_ATTENTION_DIMENSIONS: [usize; 4] = [1, 16, 2, 8];

/// Staged benchmark workload.
type BenchmarkStagedFunction = StagedFunction<Session, Type, Type>;

/// Compiled benchmark workload.
type BenchmarkCompiledFunction = CoreCompiledFunction<Session, Type, Type>;

/// Benchmark scenario that `--scenario` selects.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Scenario {
    /// Lifecycle and execution of the workload that is matched with JAX.
    Lifecycle,

    /// Dispatch of a JIT-compiled eager gradient and construction of derived compiled functions.
    Transforms,

    /// Compilation and synchronized execution of scaled dot-product attention.
    Attention,
}

/// Command-line arguments of this benchmark.
#[derive(Debug)]
struct Arguments {
    /// Benchmark scenario to run.
    scenario: Scenario,

    /// Number of timed warm iterations of each execution measurement.
    iterations: usize,

    /// Number of `f32` elements of the workload input.
    size: usize,

    /// Whether to use small dimensions and assert the cache invariants.
    smoke: bool,

    /// Directory of the persistent cache whose restore time is measured, if any.
    cache_directory: Option<PathBuf>,
}

impl Default for Arguments {
    fn default() -> Self {
        Self { scenario: Scenario::Lifecycle, iterations: 50, size: 1 << 20, smoke: false, cache_directory: None }
    }
}

/// Returns the usage message of this benchmark.
fn usage() -> &'static str {
    "Usage: compilation_benchmark [OPTIONS]\n\
     \n\
     Options:\n\
       --scenario NAME    Scenario to run (`lifecycle`, `transforms`, or `attention`; default: `lifecycle`)\n\
       --iterations N     Timed warm iterations (default: 50)\n\
       --size N           Number of f32 elements of the `lifecycle` workload (default: 1048576)\n\
       --cache-dir PATH   Measure persistent restore of the `lifecycle` workload using PATH\n\
       --smoke            Use small defaults and assert cache invariants\n\
       -h, --help         Print this help\n\
     \n\
     The `attention` scenario runs on the platform that `RYFT_BENCHMARK_PLATFORM` selects (`cpu` by default, or\n\
     `cuda-13` with the `cuda-13` feature)."
}

/// Parses the command-line arguments, returning [`None`] when only the usage message was requested.
fn parse_arguments() -> Result<Option<Arguments>, String> {
    let mut arguments = Arguments::default();
    let mut values = env::args().skip(1);
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(None);
            }
            "--scenario" => {
                arguments.scenario = match values.next().ok_or("expected a value after `--scenario`")?.as_str() {
                    "lifecycle" => Scenario::Lifecycle,
                    "transforms" => Scenario::Transforms,
                    "attention" => Scenario::Attention,
                    other => return Err(format!("unknown scenario `{other}`\n\n{}", usage())),
                };
            }
            "--iterations" => {
                arguments.iterations = values
                    .next()
                    .ok_or("expected a value after `--iterations`")?
                    .parse()
                    .map_err(|error| format!("invalid `--iterations` value: {error}"))?;
            }
            "--size" => {
                arguments.size = values
                    .next()
                    .ok_or("expected a value after `--size`")?
                    .parse()
                    .map_err(|error| format!("invalid `--size` value: {error}"))?;
            }
            "--cache-dir" => {
                arguments.cache_directory =
                    Some(PathBuf::from(values.next().ok_or("expected a path after `--cache-dir`")?));
            }
            "--smoke" => arguments.smoke = true,
            other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
        }
    }
    if arguments.smoke {
        arguments.iterations = arguments.iterations.min(3);
        arguments.size = arguments.size.min(1024);
    }
    if arguments.iterations == 0 || arguments.size == 0 {
        return Err("`--iterations` and `--size` must both be greater than zero".into());
    }
    Ok(Some(arguments))
}

/// Returns the trimmed standard output of `program` run with `arguments`, or [`None`] when it fails.
fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_string())
}

/// Returns the number of nanoseconds elapsed since `start`.
fn duration_nanoseconds(start: Instant) -> u64 {
    nanoseconds(start.elapsed())
}

/// Returns `duration` in nanoseconds, saturating at [`u64::MAX`].
fn nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Returns the sample count, the minimum, the 50th, 95th, and 99th percentiles, and the maximum of `samples` as JSON.
/// Percentiles round their rank up to the next sample.
fn distribution(samples: &[u64]) -> JsonValue {
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

/// Returns the counters and durations of `statistics` as JSON.
fn dispatcher_statistics(statistics: DispatcherStatistics) -> JsonValue {
    json!({
        "dispatch_hits": statistics.dispatch_hits,
        "dispatch_misses": statistics.dispatch_misses,
        "traces": statistics.traces,
        "lowerings": statistics.lowerings,
        "compilation_requests": statistics.compilation_requests,
        "dispatch_preparation_duration_ns": nanoseconds(statistics.dispatch_preparation_duration),
        "dispatch_lookup_duration_ns": nanoseconds(statistics.dispatch_lookup_duration),
        "tracing_duration_ns": nanoseconds(statistics.tracing_duration),
        "lowering_duration_ns": nanoseconds(statistics.lowering_duration),
    })
}

/// Returns the counters and durations of the compilation cache of `session` as JSON.
fn compilation_statistics(session: &Session) -> JsonValue {
    let statistics = session.compilation_cache().statistics();
    json!({
        "memory_hits": statistics.memory_hits,
        "persistent_hits": statistics.persistent_hits,
        "misses": statistics.misses,
        "compilations": statistics.compilations,
        "waits": statistics.waits,
        "persistent_errors": statistics.persistent_errors,
        "persistent_lookup_duration_ns": nanoseconds(statistics.persistent_lookup_duration),
        "memory_lookup_duration_ns": nanoseconds(statistics.memory_lookup_duration),
        "compilation_duration_ns": nanoseconds(statistics.compilation_duration),
    })
}

/// Returns a mesh with one auto axis over the first addressable device of `client`.
fn mesh(client: &Client<'_>) -> Result<DeviceMesh, Box<dyn std::error::Error>> {
    let device = Device::from_pjrt(&client.addressable_devices()?.remove(0))?;
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
    Ok(DeviceMesh::new(logical_mesh, vec![device])?)
}

/// Returns the type of an `f32` vector of `size` elements that is replicated over `mesh`.
fn input_type(mesh: &DeviceMesh, size: usize) -> Result<ArrayType, Box<dyn std::error::Error>> {
    Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))?)
}

/// Uploads a deterministic input array of type `r#type` with `size` elements to `mesh`.
fn input_array(
    session: &Session,
    mesh: &DeviceMesh,
    r#type: ArrayType,
    size: usize,
) -> Result<Array, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(size * size_of::<f32>());
    for index in 0..size {
        bytes.extend_from_slice(&((index % 1024) as f32 / 1024.0).to_ne_bytes());
    }
    Ok(Array::from_host_buffer(session, r#type, mesh.clone(), None, bytes.as_slice())?)
}

/// Stages the workload `sin(x * x + x)` for inputs of type `r#type` over `mesh`.
fn stage_workload(
    session: &Session,
    mesh: &DeviceMesh,
    r#type: ArrayType,
) -> Result<BenchmarkStagedFunction, Box<dyn std::error::Error>> {
    Ok(session.stage(
        |input| {
            let input = ValueProjection::<ArrayType>::into_projected(input).unwrap();
            (input.clone() * input.clone() + input).sin().unwrap().into_value()
        },
        Type::Array(r#type),
        Options::new(mesh.clone()),
    )?)
}

/// Executes the compiled workload at `input` and returns its output array.
fn call_workload(
    session: &Session,
    compiled: &BenchmarkCompiledFunction,
    input: Array,
) -> Result<Array, Box<dyn std::error::Error>> {
    match compiled.executable().execute(session, Value::Array(input))? {
        Value::Array(output) => Ok(output),
        Value::Dimension(_) => Err("compilation benchmark produced a first-class dimension".into()),
        Value::Reference(_) => Err("compilation benchmark produced a reference".into()),
    }
}

/// Compiles the workload into the persistent cache in `directory` from one session, then measures how long a second
/// session takes to restore it, and returns the measurement and the compilation statistics of the second session as
/// JSON.
fn persistent_restore(
    client: &'static Client<'static>,
    mesh: &DeviceMesh,
    r#type: &ArrayType,
    directory: &Path,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let disk_cache = || Ok::<_, std::io::Error>(DiskCache::open(directory)?.with_write_thresholds(Duration::ZERO, 0));
    let producer = Session::builder().client(client).disk_cache(disk_cache()?).build()?;
    let producer_staged = stage_workload(&producer, mesh, r#type.clone())?;
    let producer_lowered = producer_staged.lower(&producer)?;
    producer_lowered.compile(&producer)?;

    let restored = Session::builder().client(client).disk_cache(disk_cache()?).build()?;
    let restored_staged = stage_workload(&restored, mesh, r#type.clone())?;
    let restored_lowered = restored_staged.lower(&restored)?;
    let start = Instant::now();
    restored_lowered.compile(&restored)?;
    let duration = duration_nanoseconds(start);
    let statistics = restored.compilation_cache().statistics();
    Ok(json!({
        "requested": true,
        "restored": statistics.persistent_hits == 1 && statistics.compilations == 0,
        "duration_ns": duration,
        "statistics": compilation_statistics(&restored),
        "note": "both benchmark contexts use zero compile-duration and entry-size write thresholds",
    }))
}

/// Returns the type of an `f32` array of `shape` that is replicated over `mesh`.
fn replicated_f32_type(mesh: &DeviceMesh, shape: &[usize]) -> Result<ArrayType, Box<dyn std::error::Error>> {
    Ok(ArrayType::new_static(DataType::F32, shape)
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), shape.len()))?)
}

/// Returns the cold duration of `construct`, the warm durations of `iterations` further calls, and the counters and
/// durations of the compilation cache of `session` across them as JSON. The constructed values are dropped outside
/// the timed region.
fn repeated_construction<T>(
    session: &Session,
    iterations: usize,
    mut construct: impl FnMut() -> Result<T, ryft_xla::Error>,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    session.compilation_cache().clear_statistics();
    let start = Instant::now();
    let constructed = construct()?;
    let cold_nanoseconds = duration_nanoseconds(start);
    drop(constructed);
    let mut warm_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let constructed = construct()?;
        warm_samples.push(duration_nanoseconds(start));
        drop(constructed);
    }
    Ok(json!({
        "cold_ns": cold_nanoseconds,
        "warm": distribution(&warm_samples),
        "compilation_cache": compilation_statistics(session),
        "retained_programs": session.cached_program_count(),
    }))
}

/// Measures the cold call and `iterations` warm calls of a [`jit`] dispatcher whose function runs an eager reverse
/// mode gradient of `x * x` at an `f32` scalar, and returns the measurement as JSON. The JIT boundary stages the eager
/// transform once per specialization, so the retained specialization serves every warm call, which `smoke` asserts
/// together with the computed gradients.
fn jit_eager_gradient(
    client: &'static Client<'static>,
    mesh: &DeviceMesh,
    iterations: usize,
    smoke: bool,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let session = Session::builder().client(client).build()?;
    let r#type = replicated_f32_type(mesh, &[])?;
    let function: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> = jit(
        |_, input: CompileTracer| {
            Ok(input.domain().differentiate_at(input).gradient(|value| Mul::mul(&value, &value))?)
        },
        &session,
        mesh.clone(),
    );
    session.compilation_cache().clear_statistics();
    let input = |value: f32| Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, value.to_ne_bytes());

    let mut samples = Vec::with_capacity(iterations + 1);
    let mut cold_statistics = None;
    for index in 0..=iterations {
        let input = input(index as f32)?;
        let start = Instant::now();
        let output = function.execute((), input)?;
        samples.push(duration_nanoseconds(start));
        let gradient = f32::from_ne_bytes(output.copy_to_host_bytes()?.as_slice().try_into()?);
        if smoke {
            assert_eq!(gradient, 2.0 * index as f32, "the gradient of `x * x` must be `2 * x`");
        }
        cold_statistics.get_or_insert_with(|| function.statistics());
    }
    let statistics = function.statistics();
    if smoke {
        assert_eq!(statistics.dispatch_misses, 1, "only the cold call may miss the dispatch cache");
        assert_eq!(statistics.dispatch_hits, iterations as u64, "every warm call must hit the dispatch cache");
        assert_eq!(statistics.traces, 1, "the eager gradient must be traced exactly once");
        assert_eq!(statistics.lowerings, 1, "the eager gradient must be lowered exactly once");
        assert_eq!(statistics.compilation_requests, 1, "the eager gradient must be compiled exactly once");
    }
    Ok(json!({
        "cold_call_ns": samples[0],
        "warm_call": distribution(&samples[1..]),
        "counters": {
            "after_cold_call": dispatcher_statistics(cold_statistics.unwrap()),
            "after_warm_calls": dispatcher_statistics(statistics),
        },
        "compilation_cache": compilation_statistics(&session),
        "retained_programs": session.cached_program_count(),
    }))
}

/// Measures the forward mode and reverse mode transformation of one shared `jit_call` callee of `callee_operations`
/// alternating `sin` and `cos` operations across [`CALLEE_OUTER_SPECIALIZATIONS`] distinct outer functions, and returns
/// the measurement as JSON.
///
/// Every outer function stages a `jit_call` of the same callee and appends a different number of `sin` operations, so
/// the outer functions are distinct specializations that cannot share the derived functions that a
/// [`CompiledFunction`] retains. Each [`CompiledFunction::jvp`] runs the forward mode rule of the `jit_call` operation
/// over the callee, and each [`CompiledFunction::gradient`] runs both its forward mode rule and its transposition rule.
/// The transform cache of the callee region derives its linearizations and transpositions once and serves them to every
/// later outer function, so the frontend cost that still grows with the callee is the work outside of the retained
/// artifacts (e.g., replaying the derived programs into each outer function). Every construction is reported with its
/// total time, the backend compilation time that the shared compilation cache records during it, and the remaining
/// frontend time. The derived programs of distinct outer functions are distinct compilations that no frontend cache
/// can remove.
fn shared_callee_transforms(
    session: &Session,
    mesh: &DeviceMesh,
    callee_operations: usize,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let r#type = replicated_f32_type(mesh, &[])?;
    let callee: CompiledFunction<ArrayType, ArrayType> = compile(
        move |x| {
            let mut value = x;
            for index in 0..callee_operations {
                value = if index % 2 == 0 { value.sin().unwrap() } else { value.cos().unwrap() };
            }
            value
        },
        r#type.clone(),
        session,
        mesh.clone(),
    )?;

    let mut outers = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
    for index in 0..CALLEE_OUTER_SPECIALIZATIONS {
        let callee = callee.clone();
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            move |x| {
                let mut value = callee.call(x).unwrap();
                for _ in 0..=index {
                    value = value.sin().unwrap();
                }
                value
            },
            r#type.clone(),
            session,
            mesh.clone(),
        )?;

        // The callee must stay behind a `jit_call` boundary for the measurement to be meaningful, so the outer program
        // must hold exactly one `jit_call` and none of the `cos` operations of the callee, which its epilogue of `sin`
        // operations never produces.
        let instructions = outer.source_program().program().instructions();
        let jit_call_count = instructions
            .iter()
            .filter(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
            .count();
        let inlined_cosine_count = instructions
            .iter()
            .filter(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Cos(_))))
            .count();
        assert_eq!(jit_call_count, 1, "each outer function must stage one `jit_call` boundary");
        assert_eq!(inlined_cosine_count, 0, "the outer functions must not inline the callee body");
        outers.push(outer);
    }

    let compilation_duration = || session.compilation_cache().statistics().compilation_duration;
    let mut constructions = serde_json::Map::new();
    let mut mean_frontend_nanoseconds = serde_json::Map::new();
    for transform in ["jvp", "gradient"] {
        let mut rows = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
        let mut frontend_sum = 0u64;
        for outer in &outers {
            let before = compilation_duration();
            let start = Instant::now();
            match transform {
                "jvp" => drop(outer.jvp(session)?),
                _ => drop(outer.gradient(session)?),
            }
            let total = duration_nanoseconds(start);
            let backend = nanoseconds(compilation_duration() - before);
            let frontend = total.saturating_sub(backend);
            frontend_sum += frontend;
            rows.push(json!({ "total_ns": total, "backend_ns": backend, "frontend_ns": frontend }));
        }
        constructions.insert(transform.to_string(), JsonValue::from(rows));
        let mean = frontend_sum / CALLEE_OUTER_SPECIALIZATIONS as u64;
        mean_frontend_nanoseconds.insert(transform.to_string(), JsonValue::from(mean));
    }
    Ok(json!({
        "callee_operations": callee_operations,
        "constructions": constructions,
        "mean_frontend_ns": mean_frontend_nanoseconds,
    }))
}

/// Runs the `transforms` scenario and prints its JSON report.
fn transforms(arguments: &Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))?;
    let client: &'static Client<'static> = Box::leak(Box::new(client));
    let mesh = mesh(client)?;

    let jit_eager_gradient = jit_eager_gradient(client, &mesh, arguments.iterations, arguments.smoke)?;

    // Repeated reverse mode construction of the gradient of `sum(sin(x))` over an `f32[4, 4]` input.
    let session = Session::builder().client(client).build()?;
    let r#type = replicated_f32_type(&mesh, &[4, 4])?;
    let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
        |x| x.sin().unwrap().reduce(&[0, 1], ReductionKind::Sum).unwrap(),
        r#type.clone(),
        &session,
        mesh.clone(),
    )?;
    let gradient_construction = repeated_construction(&session, arguments.iterations, || compiled.gradient(&session))?;

    // Repeated forward mode construction of the JVP of `cos(sin(x))` over an `f32[4, 4]` input.
    let session = Session::builder().client(client).build()?;
    let compiled: CompiledFunction<ArrayType, ArrayType> =
        compile(|x| x.sin().unwrap().cos().unwrap(), r#type, &session, mesh.clone())?;
    let jvp_construction = repeated_construction(&session, arguments.iterations, || compiled.jvp(&session))?;

    // Transformations of a short and a long shared callee, whose difference isolates the frontend cost that grows with
    // the callee.
    let session = Session::builder().client(client).build()?;
    let callee_operation_counts = if arguments.smoke { SMOKE_CALLEE_OPERATION_COUNTS } else { CALLEE_OPERATION_COUNTS };
    let callees = callee_operation_counts
        .into_iter()
        .map(|callee_operations| shared_callee_transforms(&session, &mesh, callee_operations))
        .collect::<Result<Vec<_>, _>>()?;
    let added_operations = (callee_operation_counts[1] - callee_operation_counts[0]) as f64;
    let growth = |transform: &str| {
        let short = callees[0]["mean_frontend_ns"][transform].as_f64().unwrap();
        let long = callees[1]["mean_frontend_ns"][transform].as_f64().unwrap();
        json!({
            "mean_frontend_delta_ns": long - short,
            "mean_frontend_delta_per_callee_operation_ns": (long - short) / added_operations,
        })
    };

    let report = json!({
        "schema": "ryft-compilation-transforms-benchmark-v1",
        "configuration": {
            "iterations": arguments.iterations,
            "smoke": arguments.smoke,
            "outer_specializations": CALLEE_OUTER_SPECIALIZATIONS,
            "callee_operation_counts": callee_operation_counts,
        },
        "metadata": {
            "platform_name": client.platform_name()?.into_owned(),
            "platform_version": client.platform_version()?.into_owned(),
        },
        "jit_eager_gradient": jit_eager_gradient,
        "repeated_gradient_construction": gradient_construction,
        "repeated_jvp_construction": jvp_construction,
        "shared_callee_transforms": {
            "callees": callees,
            "frontend_growth": { "jvp": growth("jvp"), "gradient": growth("gradient") },
            "compilation_cache": compilation_statistics(&session),
            "retained_programs": session.cached_program_count(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Returns a client for the platform that `RYFT_BENCHMARK_PLATFORM` selects (i.e., `cpu`, which is the default, or
/// `cuda-13` with the `cuda-13` feature) and whether that platform is a CUDA GPU.
fn platform_client() -> Result<(&'static Client<'static>, bool), Box<dyn std::error::Error>> {
    match env::var("RYFT_BENCHMARK_PLATFORM").as_deref() {
        #[cfg(feature = "cuda-13")]
        Ok("cuda-13") => {
            let options = GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            };
            Ok((Box::leak(Box::new(load_cuda_13_plugin()?.client(ClientOptions::GPU(options))?)), true))
        }
        Ok("cpu") | Err(_) => {
            let options = CpuClientOptions { device_count: Some(1), ..Default::default() };
            Ok((Box::leak(Box::new(load_cpu_plugin()?.client(ClientOptions::CPU(options))?)), false))
        }
        Ok(platform) => Err(format!("unsupported benchmark platform `{platform}`").into()),
    }
}

/// Runs the `attention` scenario and prints its JSON report.
fn attention(arguments: &Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let (client, cuda) = platform_client()?;
    let mesh = mesh(client)?;
    let session = Session::builder().client(client).build()?;
    let dimensions @ [batch, sequence, heads, head_dimension] =
        if arguments.smoke { SMOKE_ATTENTION_DIMENSIONS } else { ATTENTION_DIMENSIONS };
    let r#type = ArrayType::new_static(DataType::BF16, dimensions)
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))?;
    let attention = dot_product_attention::<Session>(
        AttentionConfiguration::new().with_scale(1.0 / (head_dimension as f64).sqrt()),
    );

    let start = Instant::now();
    let staged: StagedFunction<Session, Vec<Type>, Vec<Type>> = session.stage(
        |inputs: Vec<CompilationTracer<Session>>| {
            let inputs = AttentionInputs::new(inputs[0].clone(), inputs[1].clone(), inputs[2].clone());
            vec![attention.call(inputs).unwrap()]
        },
        vec![Type::from(r#type.clone()); 3],
        Options::new(mesh.clone()),
    )?;
    let staging_nanoseconds = duration_nanoseconds(start);
    let start = Instant::now();
    let lowered = staged.lower(&session)?;
    let lowering_nanoseconds = duration_nanoseconds(start);
    let fused = lowered.lowered_program().stable_hlo().contains("__cudnn$fmhaSoftmax");
    if cuda {
        assert!(fused, "attention must lower to cuDNN's fused attention kernel on CUDA");
    }
    let start = Instant::now();
    let compiled: CoreCompiledFunction<Session, Vec<Type>, Vec<Type>> = lowered.compile(&session)?;
    let compilation_nanoseconds = duration_nanoseconds(start);

    // Every input element is the bf16 encoding of one.
    let element_count = dimensions.iter().product::<usize>();
    let bytes = 0x3f80u16.to_ne_bytes().repeat(element_count);
    let input = Array::from_host_buffer(&session, r#type, mesh, None, bytes.as_slice())?;
    input.block_until_ready()?;
    let inputs = vec![Value::Array(input); 3];
    let execute = || -> Result<(), Box<dyn std::error::Error>> {
        match compiled.executable().execute(&session, inputs.clone())?.remove(0) {
            Value::Array(output) => Ok(output.block_until_ready()?),
            _ => Err("attention benchmark produced a value that is not an array".into()),
        }
    };

    // Accelerators may idle at a low power state, so the warm-up runs as many executions as the timed samples to let
    // their clocks stabilize before the synchronized samples are collected.
    for _ in 0..arguments.iterations {
        execute()?;
    }
    let mut samples = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        execute()?;
        samples.push(duration_nanoseconds(start));
    }
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let median_nanoseconds = sorted[sorted.len() / 2];
    let floating_point_operations =
        4.0 * batch as f64 * heads as f64 * sequence as f64 * sequence as f64 * head_dimension as f64;

    let report = json!({
        "schema": "ryft-compilation-attention-benchmark-v1",
        "workload": {
            "name": "dot_product_attention",
            "dtype": "bfloat16",
            "batch_size": batch,
            "sequence_length": sequence,
            "head_count": heads,
            "head_dimension": head_dimension,
        },
        "configuration": {
            "iterations": arguments.iterations,
            "smoke": arguments.smoke,
            "synchronization": "Array::block_until_ready",
        },
        "metadata": {
            "platform_name": client.platform_name()?.into_owned(),
            "platform_version": client.platform_version()?.into_owned(),
        },
        "lifecycle": {
            "cold_trace_ns": staging_nanoseconds,
            "cold_lower_ns": lowering_nanoseconds,
            "cold_compile_ns": compilation_nanoseconds,
            "fused_kernel": fused,
        },
        "execution": {
            "synchronized": distribution(&samples),
            "median_throughput_tflops": floating_point_operations / (median_nanoseconds as f64 * 1e3),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Runs the selected benchmark scenario and prints its JSON report.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(arguments) = parse_arguments().map_err(std::io::Error::other)? else {
        return Ok(());
    };
    match arguments.scenario {
        Scenario::Lifecycle => {}
        Scenario::Transforms => return transforms(&arguments),
        Scenario::Attention => return attention(&arguments),
    }

    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))?;
    let client: &'static Client<'static> = Box::leak(Box::new(client));
    let mesh = mesh(client)?;
    let r#type = input_type(&mesh, arguments.size)?;
    let session = Session::builder().client(client).build()?;
    let input = input_array(&session, &mesh, r#type.clone(), arguments.size)?;
    input.block_until_ready()?;
    let start = Instant::now();
    let staged = stage_workload(&session, &mesh, r#type.clone())?;
    let cold_staging_nanoseconds = duration_nanoseconds(start);

    let start = Instant::now();
    let lowered = staged.lower(&session)?;
    let cold_lowering_nanoseconds = duration_nanoseconds(start);

    let start = Instant::now();
    let compiled: BenchmarkCompiledFunction = lowered.compile(&session)?;
    let cold_compilation_nanoseconds = duration_nanoseconds(start);

    let warmup = call_workload(&session, &compiled, input.clone())?;
    warmup.block_until_ready()?;

    let mut enqueue_samples = Vec::with_capacity(arguments.iterations);
    let mut pending_outputs = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        pending_outputs.push(call_workload(&session, &compiled, input.clone())?);
        enqueue_samples.push(duration_nanoseconds(start));
    }
    for output in pending_outputs {
        output.block_until_ready()?;
    }

    let mut synchronized_samples = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        call_workload(&session, &compiled, input.clone())?.block_until_ready()?;
        synchronized_samples.push(duration_nanoseconds(start));
    }

    let dispatcher: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
        jit(|(), input: CompileTracer| Ok((input.clone() * input.clone() + input).sin()?), &session, mesh.clone());
    dispatcher.execute((), input.clone())?.block_until_ready()?;
    let cold_dispatch_statistics = dispatcher.statistics();
    let mut warm_dispatch_samples = Vec::with_capacity(arguments.iterations);
    let mut warm_outputs = Vec::with_capacity(arguments.iterations);
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        warm_outputs.push(dispatcher.execute((), input.clone())?);
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
        Some(directory) => persistent_restore(client, &mesh, &r#type, directory)?,
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
            "cold_trace_ns": cold_staging_nanoseconds,
            "cold_lower_ns": cold_lowering_nanoseconds,
            "cold_compile_ns": cold_compilation_nanoseconds,
            "persistent_restore": persistent,
        },
        "execution": {
            "warm_dispatch_call": distribution(&warm_dispatch_samples),
            "enqueue_only": distribution(&enqueue_samples),
            "synchronized": distribution(&synchronized_samples),
        },
        "counters": {
            "after_cold_dispatch": dispatcher_statistics(cold_dispatch_statistics),
            "after_warm_dispatch": dispatcher_statistics(warm_dispatch_statistics),
            "compilation_cache": compilation_statistics(&session),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
