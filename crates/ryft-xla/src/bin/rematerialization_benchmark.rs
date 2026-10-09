//! Staging-cost and runtime benchmark for rematerialization (i.e., gradient checkpointing).
//!
//! Every workload is the gradient of the sum of the outputs of a multilayer perceptron whose layers compute
//! `h ↦ tanh(h · w₁) · w₂` with respect to its parameters, staged, lowered, compiled, and executed through the XLA
//! domain on the CPU (or on a CUDA GPU with the `cuda-13` feature and `RYFT_BENCHMARK_PLATFORM=cuda-13`). The
//! benchmark reports:
//!
//!   - **Staging cost:** the tracing, lowering, and compilation times and the number of StableHLO operations of the
//!     gradients of a model whose layers are called one after the other (i.e., repeated calls) and of a single layer
//!     nested in up to three rematerialized calls, against the same functions without rematerialization.
//!   - **Runtime:** the memory that the compiled gradient of a model that scans over its layers requires (i.e., the
//!     temporary memory and the peak memory from the memory statistics of the compiled executable) and its execution
//!     time, without rematerialization and with rematerialized layers under several policies.
//!
//! Every rematerialized gradient is checked against the gradient without rematerialization. Deterministic centered
//! weights have variance `1 / fan_in`, and every parameter matrix must have a finite, nonzero gradient norm.
//!
//! Run it with `cargo run --release -p ryft-xla --features performance-benchmarking --bin rematerialization_benchmark`.
//! Append `-- --smoke` to use small dimensions and two timed executions while retaining every benchmark variant.

use std::env;
use std::time::{Duration, Instant};

use ryft_core::compilation::{AnalyzableCompilationDomain, call_function};
use ryft_core::{
    ArrayIrType, ArrayType, CompilationDomain, CompiledFunction, Context, DataType, Device, DeviceMesh, Dimension,
    DomainTracer, DomainTracingContext, Dot, DotDimensionNumbers, DotsSavable, LogicalMesh, Memory, MeshAxis,
    MeshAxisType, NothingSavable, OffloadDotsWithNoBatchDimensions, ProgramError, Reduce, ReductionKind,
    RematerializationOptimizationBarrier, ResidualPolicy, ScanOperation, Shape, Sharding, StagedFunction, Tanh, Typed,
    Value, ValueProjection, differentiate_at, rematerialize, stage_function,
};
use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::experimental::ops::XlaOperation;
use ryft_xla::{FromPjrt, XlaArray, XlaDomain, XlaOptions, XlaSession, XlaValue};

type Tracer<'c> = DomainTracer<XlaDomain<'c>>;
type BenchmarkInput = (ArrayIrType, ArrayIrType, ArrayIrType);
type BenchmarkOutput = (ArrayIrType, ArrayIrType);
type BenchmarkStagedFunction<'c> = StagedFunction<XlaDomain<'c>, BenchmarkInput, BenchmarkOutput>;
type BenchmarkCompiledFunction<'c> = CompiledFunction<XlaDomain<'c>, BenchmarkInput, BenchmarkOutput>;

/// Rematerialization of a layer.
#[derive(Copy, Clone, Debug)]
enum Checkpointing {
    /// The layer is not rematerialized.
    None,

    /// The layer is rematerialized and saves nothing.
    NothingSavable,

    /// The layer is rematerialized and saves its dot products.
    DotsSavable,

    /// The layer is rematerialized and offloads its dot products to pinned host memory.
    OffloadDots,
}

impl Checkpointing {
    /// Returns the name of this rematerialization in the benchmark reports.
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NothingSavable => "nothing_savable",
            Self::DotsSavable => "dots_savable",
            Self::OffloadDots => "offload_dots_with_no_batch_dimensions",
        }
    }
}

/// Shape of a benchmarked multilayer perceptron.
#[derive(Copy, Clone, Debug)]
struct Model {
    /// Number of examples in a batch.
    batch_size: usize,

    /// Width of the hidden state.
    width: usize,

    /// Width of the expanded representation inside of each layer.
    expanded_width: usize,

    /// Number of layers.
    layer_count: usize,
}

/// Measurements of one compiled gradient.
struct Measurement<'c> {
    /// Duration of tracing the gradient.
    trace: Duration,

    /// Duration of lowering the traced gradient to StableHLO.
    lower: Duration,

    /// Duration of compiling the lowered gradient.
    compile: Duration,

    /// Number of operations in the lowered StableHLO.
    operation_count: usize,

    /// Temporary device memory of the compiled executable, in bytes.
    temporary_bytes: Option<usize>,

    /// Peak device memory of the compiled executable, in bytes.
    peak_bytes: Option<usize>,

    /// Median execution time.
    execution: Duration,

    /// Computed gradients.
    gradients: Vec<XlaArray<'c>>,
}

/// Returns the dot product dimensions of a matrix multiplication.
fn matrix_multiplication() -> DotDimensionNumbers {
    DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![])
}

/// Applies one layer, `h ↦ tanh(h · w₁) · w₂`, to the hidden state `h`.
fn layer<'c>(hidden: Tracer<'c>, up: Tracer<'c>, down: Tracer<'c>) -> Result<Tracer<'c>, ProgramError> {
    let hidden = ValueProjection::<ArrayType>::into_projected(hidden)?;
    let up = ValueProjection::<ArrayType>::into_projected(up)?;
    let down = ValueProjection::<ArrayType>::into_projected(down)?;
    Ok(hidden
        .dot(&up, &matrix_multiplication())?
        .tanh()?
        .dot(&down, &matrix_multiplication())?
        .into_value())
}

/// Applies [`layer`] rematerialized with `policy` and the provided optimization-barrier flag.
fn rematerialized_layer<'c, P: Clone + ResidualPolicy<ArrayIrType>>(
    policy: P,
    optimization_barrier: bool,
    hidden: Tracer<'c>,
    up: Tracer<'c>,
    down: Tracer<'c>,
) -> Result<Tracer<'c>, ProgramError> {
    rematerialize(|(hidden, up, down): (Tracer<'c>, Tracer<'c>, Tracer<'c>)| layer(hidden, up, down))
        .with_policy(policy)
        .with_optimization_barrier(match optimization_barrier {
            true => RematerializationOptimizationBarrier::All,
            false => RematerializationOptimizationBarrier::None,
        })
        .call((hidden, up, down))
}

/// Applies [`layer`] with the provided rematerialization and optimization-barrier flag.
fn checkpointed_layer<'c>(
    checkpointing: Checkpointing,
    optimization_barrier: bool,
    hidden: Tracer<'c>,
    up: Tracer<'c>,
    down: Tracer<'c>,
) -> Result<Tracer<'c>, ProgramError> {
    match checkpointing {
        Checkpointing::None => layer(hidden, up, down),
        Checkpointing::NothingSavable => rematerialized_layer(NothingSavable, optimization_barrier, hidden, up, down),
        Checkpointing::DotsSavable => rematerialized_layer(DotsSavable, optimization_barrier, hidden, up, down),
        Checkpointing::OffloadDots => {
            let policy = OffloadDotsWithNoBatchDimensions::new(Memory::Host { pinned: true });
            rematerialized_layer(policy, optimization_barrier, hidden, up, down)
        }
    }
}

/// Applies [`layer`] nested in `depth` rematerialized calls that save nothing.
fn nested_layer<'c>(
    depth: usize,
    hidden: Tracer<'c>,
    up: Tracer<'c>,
    down: Tracer<'c>,
) -> Result<Tracer<'c>, ProgramError> {
    if depth == 0 {
        return layer(hidden, up, down);
    }
    rematerialize(move |(hidden, up, down): (Tracer<'c>, Tracer<'c>, Tracer<'c>)| {
        nested_layer(depth - 1, hidden, up, down)
    })
    .call((hidden, up, down))
}

/// Returns the type of an `f32` array of the provided shape.
fn array_type(shape: &[usize]) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(shape.iter().map(|size| Dimension::Static(*size)).collect()))
}

/// Returns the type of an `f32` array of the provided shape that is replicated over `mesh`, which is the type of the
/// inputs of the benchmarked functions (whose bodies are traced over the unannotated types of [`array_type`]).
fn replicated_array_type(mesh: &DeviceMesh, shape: &[usize]) -> ArrayType {
    array_type(shape)
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), shape.len()))
        .unwrap()
}

/// Returns an array replicated over `mesh` with deterministic centered values of variance `1 / fan_in`.
/// Distinct seeds separate inputs and parameter matrices; the generator continues across stacked layers so that
/// their weights differ. Reusing the same seed and shape reproduces identical inputs across benchmark variants.
fn array<'c>(domain: &XlaDomain<'c>, mesh: &DeviceMesh, shape: &[usize], fan_in: usize, mut seed: u64) -> XlaArray<'c> {
    let size = shape.iter().product::<usize>();
    let scale = (3.0 / fan_in as f32).sqrt();
    let bytes = (0..size)
        .flat_map(|_| {
            // SplitMix64 supplies reproducible, decorrelated samples without a benchmark-only dependency.
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut sample = seed;
            sample = (sample ^ (sample >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            sample = (sample ^ (sample >> 27)).wrapping_mul(0x94d049bb133111eb);
            sample ^= sample >> 31;
            let unit = (sample >> 40) as f32 / (1u32 << 24) as f32;
            ((2.0 * unit - 1.0) * scale).to_ne_bytes()
        })
        .collect::<Vec<_>>();
    XlaArray::from_host_buffer(domain, replicated_array_type(mesh, shape), mesh.clone(), bytes.as_slice()).unwrap()
}

/// Returns the values of `array`.
fn values(client: &Client<'_>, array: &XlaArray<'_>) -> Vec<f32> {
    let device = client.addressable_devices().unwrap()[0].id().unwrap();
    let bytes = array.device_shard(device).unwrap().buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
    bytes.chunks_exact(4).map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap())).collect()
}

/// Stages the gradient of the sum of the outputs of `apply` with respect to the parameters `(w₁, w₂)` of shapes
/// `up_shape` and `down_shape`, at a hidden state of `model`.
fn stage_gradient<'c, F>(
    domain: &XlaDomain<'c>,
    mesh: &DeviceMesh,
    model: Model,
    up_shape: &[usize],
    down_shape: &[usize],
    apply: F,
) -> BenchmarkStagedFunction<'c>
where
    F: Fn(Tracer<'c>, Tracer<'c>, Tracer<'c>) -> Result<Tracer<'c>, ProgramError>,
{
    let hidden_type = array_type(&[model.batch_size, model.width]);
    stage_function(
        domain,
        |(hidden, up, down)| {
            // The hidden state is differentiated too, so that every input belongs to the differentiated function, and
            // its gradient is discarded.
            let (_, up, down) = differentiate_at((hidden, up, down))
                .gradient(|(hidden, up, down)| {
                    let (_, apply) = DomainTracingContext::<XlaDomain<'c>>::trace(
                        |(hidden, up, down): (Tracer<'c>, Tracer<'c>, Tracer<'c>)| apply(hidden, up, down),
                        (
                            ArrayIrType::from(hidden_type.clone()),
                            ArrayIrType::from(array_type(up_shape)),
                            ArrayIrType::from(array_type(down_shape)),
                        ),
                    )?;
                    let context = up.domain();
                    let output = apply.into_flat_program().interpret_in_context(&context, vec![hidden, up, down])?;
                    let output = ValueProjection::<ArrayType>::into_projected(output.into_iter().next().unwrap())?;
                    Ok(output.reduce(&[0, 1], ReductionKind::Sum)?.into_value())
                })
                .unwrap();
            (up, down)
        },
        (
            ArrayIrType::from(replicated_array_type(mesh, &[model.batch_size, model.width])),
            ArrayIrType::from(replicated_array_type(mesh, up_shape)),
            ArrayIrType::from(replicated_array_type(mesh, down_shape)),
        ),
        XlaOptions::new(mesh.clone()),
    )
    .unwrap()
}

/// Stages the gradient of a model that scans over its stacked layer parameters, applying each layer with the provided
/// rematerialization. Rematerialized layers disable their optimization barriers, because the scan already separates
/// their recomputations from the forward computation.
fn stage_scanned_gradient<'c>(
    domain: &XlaDomain<'c>,
    mesh: &DeviceMesh,
    model: Model,
    checkpointing: Checkpointing,
) -> BenchmarkStagedFunction<'c> {
    let hidden_type = array_type(&[model.batch_size, model.width]);
    let layer_up_type = array_type(&[model.width, model.expanded_width]);
    let layer_down_type = array_type(&[model.expanded_width, model.width]);
    stage_gradient(
        domain,
        mesh,
        model,
        &[model.layer_count, model.width, model.expanded_width],
        &[model.layer_count, model.expanded_width, model.width],
        move |hidden, up, down| {
            let (_, body) = DomainTracingContext::<XlaDomain<'c>>::trace(
                |(_index, hidden, up, down): (Tracer<'c>, Tracer<'c>, Tracer<'c>, Tracer<'c>)| {
                    checkpointed_layer(checkpointing, false, hidden, up, down)
                },
                (
                    ArrayIrType::from(ArrayType::scalar(DataType::I64)),
                    ArrayIrType::from(hidden_type.clone()),
                    ArrayIrType::from(layer_up_type.clone()),
                    ArrayIrType::from(layer_down_type.clone()),
                ),
            )?;
            let context = hidden.context().clone();
            let scan = XlaOperation::Scan(ScanOperation::new(1, model.layer_count));
            Ok(context.bind(scan, vec![body.into_flat_program()], &[hidden, up, down])?.remove(0))
        },
    )
}

/// Lowers, compiles, analyzes, and executes `staged` at `inputs`, whose staging took `trace`.
fn measure<'c>(
    domain: &XlaDomain<'c>,
    staged: BenchmarkStagedFunction<'c>,
    trace: Duration,
    inputs: (XlaArray<'c>, XlaArray<'c>, XlaArray<'c>),
    execution_count: usize,
) -> Measurement<'c> {
    let start = Instant::now();
    let lowered = domain.lower(staged).unwrap();
    let lower = start.elapsed();
    let operation_count = lowered.lowered_program().stable_hlo().matches(" = stablehlo.").count();
    let start = Instant::now();
    let compiled: BenchmarkCompiledFunction<'c> = domain.compile(lowered).unwrap();
    let compile = start.elapsed();
    let memory = domain.analyze(compiled.executable_function()).unwrap().memory;
    let inputs = (XlaValue::Array(inputs.0), XlaValue::Array(inputs.1), XlaValue::Array(inputs.2));
    let execute = || {
        let (up, down) = call_function(domain, compiled.executable_function(), inputs.clone()).unwrap();
        let gradients = [up, down].map(|gradient| ValueProjection::<ArrayType>::into_projected(gradient).unwrap());
        gradients.iter().for_each(|gradient| gradient.block_until_ready().unwrap());
        gradients
    };
    let gradients = execute();
    let mut executions = (0..execution_count)
        .map(|_| {
            let start = Instant::now();
            execute();
            start.elapsed()
        })
        .collect::<Vec<_>>();
    executions.sort_unstable();
    Measurement {
        trace,
        lower,
        compile,
        operation_count,
        temporary_bytes: memory.map(|memory| memory.device_temporary_size_in_bytes),
        peak_bytes: memory.map(|memory| memory.device_peak_memory_in_bytes),
        execution: executions[execution_count / 2],
        gradients: gradients.into(),
    }
}

/// Returns `bytes` formatted in mebibytes, or `n/a`.
fn mebibytes(bytes: Option<usize>) -> String {
    bytes.map_or_else(|| "n/a".to_owned(), |bytes| format!("{:.1}", bytes as f64 / (1024.0 * 1024.0)))
}

/// Validates gradient structure, finite values, and agreement within `1e-5 + 1e-4 * |expected|`, returning the maximum
/// absolute difference. Reports the element with the largest excess over its tolerance when agreement fails.
/// Each gradient contains `parameter_layer_count` contiguous parameter matrices, each with a nonzero finite norm;
/// scanned parameters use the model's layer count, while shared parameters and single-layer workloads use one.
fn validate_gradients(
    actual: &[(Shape, Vec<f32>)],
    expected: &[(Shape, Vec<f32>)],
    parameter_layer_count: usize,
) -> Result<f64, ProgramError> {
    if actual.len() != expected.len() {
        return Err(ProgramError::InvalidArgument {
            message: format!("gradient count mismatch: expected {}, got {}", expected.len(), actual.len()),
        });
    }
    let mut maximum_difference = 0f64;
    let mut worst_mismatch: Option<(f64, usize, usize, f32, f32)> = None;
    for (gradient_index, ((actual_shape, actual), (expected_shape, expected))) in
        actual.iter().zip(expected).enumerate()
    {
        if actual_shape != expected_shape {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "gradient {gradient_index} shape mismatch: expected `{expected_shape}`, got `{actual_shape}`",
                ),
            });
        }
        if actual.len() != expected.len() || actual_shape.element_count()? != Some(actual.len()) {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "gradient {gradient_index} element count mismatch for shape `{actual_shape}`: \
                     actual {}, expected {}",
                    actual.len(),
                    expected.len(),
                ),
            });
        }
        if parameter_layer_count == 0 || actual.is_empty() || actual.len() % parameter_layer_count != 0 {
            return Err(ProgramError::InvalidArgument {
                message: format!("gradient {gradient_index} cannot be split into {parameter_layer_count} layers"),
            });
        }
        for (element_index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            if !actual.is_finite() || !expected.is_finite() {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "gradient {gradient_index} element {element_index} is not finite: \
                         actual `{actual}`, expected `{expected}`",
                    ),
                });
            }
            let difference = (f64::from(actual) - f64::from(expected)).abs();
            let tolerance = 1e-5 + 1e-4 * f64::from(expected).abs();
            maximum_difference = maximum_difference.max(difference);
            if difference > tolerance && worst_mismatch.is_none_or(|(excess, ..)| difference - tolerance > excess) {
                worst_mismatch = Some((difference - tolerance, gradient_index, element_index, actual, expected));
            }
        }
        for (role, values) in [("actual", actual), ("expected", expected)] {
            for (layer, values) in values.chunks_exact(values.len() / parameter_layer_count).enumerate() {
                let squared_norm = values.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>();
                if !squared_norm.is_finite() || squared_norm == 0.0 {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "{role} gradient {gradient_index} layer {layer} has a zero or nonfinite squared norm",
                        ),
                    });
                }
            }
        }
    }
    if let Some((_, gradient_index, element_index, actual, expected)) = worst_mismatch {
        return Err(ProgramError::InvalidArgument {
            message: format!(
                "gradient {gradient_index} element {element_index} exceeds tolerance: \
                 actual `{actual}`, expected `{expected}`",
            ),
        });
    }
    Ok(maximum_difference)
}

/// Validates `measurement` against `reference` and prints it as a table row. Copies and validation are outside the
/// timed executions. Refer to [`validate_gradients`] for the tolerance and `parameter_layer_count` semantics.
fn report(
    client: &Client<'_>,
    name: &str,
    measurement: &Measurement<'_>,
    reference: &Measurement<'_>,
    parameter_layer_count: usize,
) -> Result<(), ProgramError> {
    let read = |measurement: &Measurement<'_>| {
        measurement
            .gradients
            .iter()
            .map(|array| (array.r#type().shape().clone(), values(client, array)))
            .collect::<Vec<_>>()
    };
    let difference = validate_gradients(&read(measurement), &read(reference), parameter_layer_count)?;
    println!(
        "| {name} | {:.1} | {:.1} | {:.1} | {} | {} | {} | {:.2} | {difference:.1e} |",
        measurement.trace.as_secs_f64() * 1e3,
        measurement.lower.as_secs_f64() * 1e3,
        measurement.compile.as_secs_f64() * 1e3,
        measurement.operation_count,
        mebibytes(measurement.temporary_bytes),
        mebibytes(measurement.peak_bytes),
        measurement.execution.as_secs_f64() * 1e3,
    );
    Ok(())
}

/// Prints the header of a table of measurements.
fn print_header(title: &str) {
    println!("\n{title}\n");
    println!(
        "| Variant | Trace (ms) | Lower (ms) | Compile (ms) | StableHLO operations | Temporary (MiB) | Peak (MiB) \
         | Execution (ms) | Maximum absolute gradient difference |",
    );
    println!("|---|---|---|---|---|---|---|---|---|");
}

/// Returns a client for the benchmarked platform.
fn client() -> Client<'static> {
    match env::var("RYFT_BENCHMARK_PLATFORM").as_deref() {
        #[cfg(feature = "cuda-13")]
        Ok("cuda-13") => {
            use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};
            let options = GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            };
            Box::leak(Box::new(load_cuda_13_plugin().unwrap())).client(ClientOptions::GPU(options)).unwrap()
        }
        Ok("cpu") | Err(_) => Box::leak(Box::new(load_cpu_plugin().unwrap()))
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap(),
        Ok(platform) => panic!("unsupported benchmark platform `{platform}`"),
    }
}

/// Runs every benchmark variant with full or smoke dimensions.
fn main() -> Result<(), ProgramError> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let smoke = match arguments.as_slice() {
        [] => false,
        [argument] if argument == "--smoke" => true,
        _ => {
            return Err(ProgramError::InvalidArgument { message: "expected no arguments or `--smoke`".to_owned() });
        }
    };
    let execution_count = if smoke { 2 } else { 10 };
    let model = if smoke {
        Model { batch_size: 4, width: 8, expanded_width: 16, layer_count: 3 }
    } else {
        Model { batch_size: 512, width: 256, expanded_width: 1024, layer_count: 16 }
    };
    let client = Box::leak(Box::new(client()));
    let device = Device::from_pjrt(&client.addressable_devices().unwrap().remove(0)).unwrap();
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap();
    let mesh = DeviceMesh::new(logical_mesh, vec![device]).unwrap();
    let domain = XlaSession::new(client).domain();
    println!("platform: {}", client.platform_name().unwrap());

    // Runtime of a model that scans over its layers.
    let inputs = || {
        (
            array(&domain, &mesh, &[model.batch_size, model.width], 1, 1),
            array(&domain, &mesh, &[model.layer_count, model.width, model.expanded_width], model.width, 2),
            array(&domain, &mesh, &[model.layer_count, model.expanded_width, model.width], model.expanded_width, 3),
        )
    };
    print_header(&format!("Scan over layers ({model:?})"));
    let mut reference = None;
    for checkpointing in
        [Checkpointing::None, Checkpointing::NothingSavable, Checkpointing::DotsSavable, Checkpointing::OffloadDots]
    {
        let start = Instant::now();
        let staged = stage_scanned_gradient(&domain, &mesh, model, checkpointing);
        let measurement = measure(&domain, staged, start.elapsed(), inputs(), execution_count);
        report(
            client,
            checkpointing.name(),
            &measurement,
            reference.as_ref().unwrap_or(&measurement),
            model.layer_count,
        )?;
        reference.get_or_insert(measurement);
    }

    // Staging cost of repeated calls of a layer.
    let up_shape = [model.width, model.expanded_width];
    let down_shape = [model.expanded_width, model.width];
    let inputs = || {
        (
            array(&domain, &mesh, &[model.batch_size, model.width], 1, 1),
            array(&domain, &mesh, &up_shape, model.width, 2),
            array(&domain, &mesh, &down_shape, model.expanded_width, 3),
        )
    };
    print_header(&format!("Repeated calls ({model:?})"));
    let mut reference = None;
    for (checkpointing, optimization_barrier) in [
        (Checkpointing::None, true),
        (Checkpointing::NothingSavable, true),
        (Checkpointing::NothingSavable, false),
        (Checkpointing::DotsSavable, true),
        (Checkpointing::OffloadDots, true),
    ] {
        let start = Instant::now();
        let staged = stage_gradient(&domain, &mesh, model, &up_shape, &down_shape, |hidden, up, down| {
            (0..model.layer_count).try_fold(hidden, |hidden, _| {
                checkpointed_layer(checkpointing, optimization_barrier, hidden, up.clone(), down.clone())
            })
        });
        let measurement = measure(&domain, staged, start.elapsed(), inputs(), execution_count);
        let name = match (checkpointing, optimization_barrier) {
            (Checkpointing::None, _) | (_, true) => checkpointing.name().to_owned(),
            (_, false) => format!("{} without an optimization barrier", checkpointing.name()),
        };
        report(client, &name, &measurement, reference.as_ref().unwrap_or(&measurement), 1)?;
        reference.get_or_insert(measurement);
    }

    // Staging cost of nested rematerialized calls.
    print_header(&format!("Nested calls ({model:?})"));
    let mut reference = None;
    for depth in 0..=3 {
        let start = Instant::now();
        let staged = stage_gradient(&domain, &mesh, model, &up_shape, &down_shape, |hidden, up, down| {
            nested_layer(depth, hidden, up, down)
        });
        let measurement = measure(&domain, staged, start.elapsed(), inputs(), execution_count);
        report(client, &format!("depth {depth}"), &measurement, reference.as_ref().unwrap_or(&measurement), 1)?;
        reference.get_or_insert(measurement);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_validate_gradients() {
        let shape = Shape::new(vec![2.into()]);
        let expected = vec![(shape.clone(), vec![1.0, 100.0])];
        assert_eq!(validate_gradients(&expected, &expected, 1), Ok(0.0));
        let actual = vec![(shape, vec![1.00001, 100.005])];
        let difference = validate_gradients(&actual, &expected, 1).unwrap();
        assert!((difference - 0.005).abs() < 1e-5);

        let shape = Shape::new(vec![2.into(), 2.into()]);
        let expected = vec![(shape.clone(), vec![1.0, 2.0, 3.0, 4.0])];
        let corrupted = vec![(shape, vec![1.0, 2.01, 3.0, 5.0])];
        assert!(matches!(
            validate_gradients(&corrupted, &expected, 2),
            Err(ProgramError::InvalidArgument { message })
                if message == "gradient 0 element 3 exceeds tolerance: actual `5`, expected `4`"
        ));
    }

    #[test]
    fn test_validate_gradients_structure() {
        let shape = Shape::new(vec![2.into()]);
        let expected = vec![(shape.clone(), vec![1.0, 2.0])];
        let cases = [
            (Vec::new(), "gradient count mismatch: expected 1, got 0", "gradient count mismatch: expected 0, got 1"),
            (
                vec![(Shape::new(vec![1.into(), 2.into()]), vec![1.0, 2.0])],
                "gradient 0 shape mismatch: expected `[2]`, got `[1, 2]`",
                "gradient 0 shape mismatch: expected `[1, 2]`, got `[2]`",
            ),
            (
                vec![(shape.clone(), vec![1.0])],
                "gradient 0 element count mismatch for shape `[2]`: actual 1, expected 2",
                "gradient 0 element count mismatch for shape `[2]`: actual 2, expected 1",
            ),
            (
                vec![(shape.clone(), vec![1.0, 2.0, 3.0])],
                "gradient 0 element count mismatch for shape `[2]`: actual 3, expected 2",
                "gradient 0 element count mismatch for shape `[2]`: actual 2, expected 3",
            ),
        ];
        for (actual, diagnostic, reverse_diagnostic) in cases {
            assert!(matches!(
                validate_gradients(&actual, &expected, 1),
                Err(ProgramError::InvalidArgument { message }) if message == diagnostic
            ));
            assert!(matches!(
                validate_gradients(&expected, &actual, 1),
                Err(ProgramError::InvalidArgument { message }) if message == reverse_diagnostic
            ));
        }
        let truncated = vec![(shape, vec![1.0])];
        assert!(matches!(
            validate_gradients(&truncated, &truncated, 1),
            Err(ProgramError::InvalidArgument { message })
                if message == "gradient 0 element count mismatch for shape `[2]`: actual 1, expected 1"
        ));
        for layer_count in [0, 3] {
            assert!(matches!(
                validate_gradients(&expected, &expected, layer_count),
                Err(ProgramError::InvalidArgument { message })
                    if message == format!("gradient 0 cannot be split into {layer_count} layers")
            ));
        }
    }

    #[test]
    fn test_validate_gradients_nonfinite() {
        let shape = Shape::new(vec![2.into()]);
        let expected = vec![(shape.clone(), vec![1.0, 2.0])];
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let actual = vec![(shape.clone(), vec![1.0, nonfinite])];
            for (actual, expected) in [(&actual, &expected), (&expected, &actual)] {
                assert!(matches!(
                    validate_gradients(actual, expected, 1),
                    Err(ProgramError::InvalidArgument { message })
                        if message == format!(
                            "gradient 0 element 1 is not finite: actual `{}`, expected `{}`",
                            actual[0].1[1], expected[0].1[1],
                        )
                ));
            }
        }
    }

    #[test]
    fn test_validate_gradients_norms() {
        let shape = Shape::new(vec![2.into(), 2.into()]);
        let gradients = vec![(shape.clone(), vec![1.0, 0.0, 0.0, 2.0])];
        assert_eq!(validate_gradients(&gradients, &gradients, 2), Ok(0.0));
        let zero_layer = vec![(shape, vec![1.0, 0.0, 0.0, -0.0])];
        for (actual, expected, role) in [(&zero_layer, &gradients, "actual"), (&gradients, &zero_layer, "expected")] {
            assert!(matches!(
                validate_gradients(actual, expected, 2),
                Err(ProgramError::InvalidArgument { message })
                    if message == format!("{role} gradient 0 layer 1 has a zero or nonfinite squared norm")
            ));
        }
    }
}
