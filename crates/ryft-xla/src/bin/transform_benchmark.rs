use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use ryft_core::ProvidesContext;
use ryft_core::operations::constants::ZeroLike;
use ryft_core::operations::trigonometric::Sin;
use ryft_core::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
use ryft_core::tracing_v2::{
    CoordinateValue, DifferentiableDomainExtension, DifferentialBlock, DifferentiationContext, jacrev, value_and_grad,
};
use ryft_ndarray::{Array, NdArrayDomain};
use serde_json::json;

/// Default number of measured iterations for each transform benchmark.
const DEFAULT_ITERATIONS: usize = 1_000;

/// Default number of warmup iterations for each transform benchmark.
const DEFAULT_WARMUP: usize = 50;

/// One transform runtime benchmark case.
#[derive(Copy, Clone)]
struct TransformBenchmarkCase {
    /// Stable case identifier.
    case_id: &'static str,

    /// High-level category such as `scalar` or `array`.
    category: &'static str,

    /// Transform surface being measured.
    transform: &'static str,

    /// Function that executes this case.
    run: fn(usize, usize) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>>,
}

impl TransformBenchmarkCase {
    /// Creates a benchmark case descriptor.
    const fn new(
        case_id: &'static str,
        category: &'static str,
        transform: &'static str,
        run: fn(usize, usize) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>>,
    ) -> Self {
        Self { case_id, category, transform, run }
    }
}

/// One transform runtime benchmark record.
#[derive(serde::Serialize)]
struct TransformBenchmarkRecord {
    /// Stable case identifier.
    case_id: &'static str,

    /// High-level category such as `scalar` or `array`.
    category: &'static str,

    /// Transform surface being measured.
    transform: &'static str,

    /// Number of warmup iterations that ran before measurement.
    warmup: usize,

    /// Number of measured iterations.
    iterations: usize,

    /// Fastest measured iteration duration.
    min_ns: u128,

    /// Median measured iteration duration.
    median_ns: u128,

    /// Mean measured iteration duration.
    mean_ns: u128,

    /// Slowest measured iteration duration.
    max_ns: u128,

    /// Deterministic checksum derived from benchmark outputs.
    checksum: u64,
}

/// Runs the transform benchmark emitter and prints JSON records to stdout.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut case_ids = Vec::new();
    let mut list_cases = false;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut warmup = DEFAULT_WARMUP;

    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--list" => list_cases = true,
            "--case" => {
                let case_id = arguments.next().ok_or("expected a case ID after --case")?;
                case_ids.push(case_id);
            }
            "--iterations" => {
                iterations = arguments.next().ok_or("expected an iteration count after --iterations")?.parse()?;
            }
            "--warmup" => {
                warmup = arguments.next().ok_or("expected a warmup count after --warmup")?.parse()?;
            }
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }

    let cases = transform_benchmark_cases();
    if list_cases {
        let case_records = cases
            .iter()
            .map(|case| {
                json!({
                    "case_id": case.case_id,
                    "category": case.category,
                    "transform": case.transform,
                })
            })
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&case_records)?);
        return Ok(());
    }

    let selected_cases = if case_ids.is_empty() {
        cases
    } else {
        case_ids
            .iter()
            .map(|case_id| {
                cases
                    .iter()
                    .copied()
                    .find(|case| case.case_id == case_id)
                    .ok_or_else(|| format!("unknown transform benchmark case '{case_id}'"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut records = Vec::new();
    for case in selected_cases {
        records.push((case.run)(iterations, warmup)?);
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

/// Returns the stable transform runtime benchmark case registry.
fn transform_benchmark_cases() -> Vec<TransformBenchmarkCase> {
    vec![
        TransformBenchmarkCase::new("scalar_jvp_direct", "scalar", "jvp", run_scalar_jvp_direct),
        TransformBenchmarkCase::new("scalar_linearize_build", "scalar", "linearize", run_scalar_linearize_build),
        TransformBenchmarkCase::new(
            "scalar_pushforward_apply",
            "scalar",
            "pushforward_apply",
            run_scalar_pushforward_apply,
        ),
        TransformBenchmarkCase::new("scalar_vjp_build", "scalar", "vjp", run_scalar_vjp_build),
        TransformBenchmarkCase::new("scalar_pullback_apply", "scalar", "pullback_apply", run_scalar_pullback_apply),
        TransformBenchmarkCase::new("scalar_value_and_grad", "scalar", "value_and_grad", run_scalar_value_and_grad),
        TransformBenchmarkCase::new("matrix_jvp_matmul", "matrix", "jvp", run_matrix_jvp_matmul),
        TransformBenchmarkCase::new("array_jacfwd_vector", "array", "jacfwd", run_array_jacfwd_vector),
        TransformBenchmarkCase::new("array_jacrev_vector", "array", "jacrev", run_array_jacrev_vector),
        TransformBenchmarkCase::new("array_hessian_scalar", "array", "hessian", run_array_hessian_scalar),
    ]
}

/// Measures `body` for `iterations` after `warmup` executions.
fn measure(
    case_id: &'static str,
    category: &'static str,
    transform: &'static str,
    iterations: usize,
    warmup: usize,
    mut body: impl FnMut() -> Result<u64, Box<dyn std::error::Error>>,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    for _ in 0..warmup {
        black_box(body()?);
    }

    let mut durations = Vec::with_capacity(iterations);
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let start = Instant::now();
        checksum = checksum.wrapping_add(black_box(body()?));
        durations.push(start.elapsed());
    }

    Ok(runtime_record(case_id, category, transform, warmup, iterations, durations, checksum))
}

/// Builds one runtime benchmark record from measured durations.
fn runtime_record(
    case_id: &'static str,
    category: &'static str,
    transform: &'static str,
    warmup: usize,
    iterations: usize,
    mut durations: Vec<Duration>,
    checksum: u64,
) -> TransformBenchmarkRecord {
    durations.sort_unstable();
    let total_ns = durations.iter().map(Duration::as_nanos).sum::<u128>();
    let mean_ns = total_ns / iterations.max(1) as u128;
    TransformBenchmarkRecord {
        case_id,
        category,
        transform,
        warmup,
        iterations,
        min_ns: durations.first().map(Duration::as_nanos).unwrap_or(0),
        median_ns: durations.get(iterations / 2).map(Duration::as_nanos).unwrap_or(0),
        mean_ns,
        max_ns: durations.last().map(Duration::as_nanos).unwrap_or(0),
        checksum,
    }
}

/// Scalar benchmark helper shared by direct and staged scalar AD cases.
fn bilinear_sin<T>(inputs: (T, T)) -> T
where
    T: Clone + Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin()
}

/// Scalar benchmark helper for higher-order scalar AD cases.
fn quartic_plus_sin<T>(x: T) -> T
where
    T: Clone + Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
}

/// Returns a checksum for scalar outputs.
fn scalar_checksum(value: f64) -> u64 {
    value.to_bits()
}

/// Returns a checksum for a differential.
fn differential_checksum<Rows, Partials>(differential: &ryft_core::tracing_v2::Differential<Rows, Partials, f64>) -> u64
where
    Rows: ryft_core::parameters::Parameterized<ryft_core::tracing_v2::DifferentialRow<Partials, f64>>,
    Partials: ryft_core::parameters::Parameterized<DifferentialBlock<f64>>,
{
    differential
        .iter_blocks()
        .flat_map(|(_, _, block)| block.values().iter().copied())
        .fold(0u64, |checksum, value| checksum.rotate_left(1) ^ value.to_bits())
}

/// Returns a checksum for array outputs.
fn array_checksum(value: &Array<f64>) -> u64 {
    value
        .coordinates()
        .into_iter()
        .fold(0u64, |checksum, value| checksum.rotate_left(1) ^ value.to_bits())
}

/// Runs the scalar direct JVP benchmark.
fn run_scalar_jvp_direct(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = ryft_core::scalars::ScalarDomain::<f64>::new();
    measure("scalar_jvp_direct", "scalar", "jvp", iterations, warmup, || {
        let (primal, tangent): (f64, f64) =
            domain.jvp(bilinear_sin, (black_box(2.0), black_box(3.0)), (black_box(1.0), black_box(-1.0)))?;
        Ok(scalar_checksum(primal) ^ scalar_checksum(tangent).rotate_left(1))
    })
}

/// Runs the scalar linearize build benchmark.
fn run_scalar_linearize_build(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = ryft_core::scalars::ScalarDomain::<f64>::new();
    measure("scalar_linearize_build", "scalar", "linearize", iterations, warmup, || {
        let (output, pushforward) = domain.linearize(|x| Ok(quartic_plus_sin(x)), black_box(2.0))?;
        Ok(scalar_checksum(output)
            ^ (pushforward.program().instructions().len() as u64)
            ^ (pushforward.residuals().len() as u64).rotate_left(1))
    })
}

/// Runs the scalar pushforward application benchmark.
fn run_scalar_pushforward_apply(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = ryft_core::scalars::ScalarDomain::<f64>::new();
    let (_, pushforward) = domain.linearize(|x| Ok(quartic_plus_sin(x)), 2.0)?;
    measure("scalar_pushforward_apply", "scalar", "pushforward_apply", iterations, warmup, || {
        Ok(scalar_checksum(pushforward.apply(&domain, black_box(1.0))?))
    })
}

/// Runs the scalar VJP build benchmark.
fn run_scalar_vjp_build(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = ryft_core::scalars::ScalarDomain::<f64>::new();
    measure("scalar_vjp_build", "scalar", "vjp", iterations, warmup, || {
        let (output, pullback) = domain.vjp(|inputs| Ok(bilinear_sin(inputs)), (black_box(2.0), black_box(3.0)))?;
        Ok(scalar_checksum(output) ^ (pullback.instructions().len() as u64).rotate_left(1))
    })
}

/// Runs the scalar pullback application benchmark.
fn run_scalar_pullback_apply(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = ryft_core::scalars::ScalarDomain::<f64>::new();
    let (_, pullback) = domain.vjp(|inputs| Ok(bilinear_sin(inputs)), (2.0, 3.0))?;
    measure("scalar_pullback_apply", "scalar", "pullback_apply", iterations, warmup, || {
        let (left, right): (f64, f64) = pullback.interpret(black_box(1.0))?;
        Ok(scalar_checksum(left) ^ scalar_checksum(right).rotate_left(1))
    })
}

/// Runs the scalar value-and-gradient benchmark.
fn run_scalar_value_and_grad(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = ryft_core::scalars::ScalarDomain::<f64>::new();
    measure("scalar_value_and_grad", "scalar", "value_and_grad", iterations, warmup, || {
        let (value, gradient) = value_and_grad(&domain, |x| quartic_plus_sin(x), black_box(2.0))?;
        Ok(scalar_checksum(value) ^ scalar_checksum(gradient).rotate_left(1))
    })
}

/// Builds the vector input used by dense Jacobian benchmarks.
fn vector_input() -> Result<Array<f64>, Box<dyn std::error::Error>> {
    Ok(Array::from_shape_vec([4], vec![1.0, 2.0, 3.0, 4.0])?)
}

/// Builds the matrix inputs used by the matmul benchmark.
fn matrix_inputs() -> Result<(Array<f64>, Array<f64>), Box<dyn std::error::Error>> {
    Ok((
        Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])?,
        Array::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0])?,
    ))
}

/// Builds the matrix tangent inputs used by the matmul benchmark.
fn matrix_tangents() -> Result<(Array<f64>, Array<f64>), Box<dyn std::error::Error>> {
    Ok((
        Array::from_shape_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0])?,
        Array::from_shape_vec([2, 2], vec![0.0, 1.0, 1.0, 0.0])?,
    ))
}

/// Runs the matrix matmul direct JVP benchmark.
fn run_matrix_jvp_matmul(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = NdArrayDomain::<f64>::new();
    let inputs = matrix_inputs()?;
    let tangents = matrix_tangents()?;
    measure("matrix_jvp_matmul", "matrix", "jvp", iterations, warmup, || {
        let (primal, tangent): (Array<f64>, Array<f64>) = domain.jvp(
            |(left, right)| left.dot(&right, &DotDimensionNumbers::matmul()),
            black_box(inputs.clone()),
            black_box(tangents.clone()),
        )?;
        Ok(array_checksum(&primal) ^ array_checksum(&tangent).rotate_left(1))
    })
}

/// Runs the array `jacfwd` benchmark.
fn run_array_jacfwd_vector(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = NdArrayDomain::<f64>::new();
    let input = vector_input()?;
    measure("array_jacfwd_vector", "array", "jacfwd", iterations, warmup, || {
        let jacobian = domain.jacfwd(|x| Ok(x.clone() * x.clone() + x.sin()), black_box(input.clone()))?;
        Ok(differential_checksum(&jacobian))
    })
}

/// Runs the array `jacrev` benchmark.
fn run_array_jacrev_vector(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = NdArrayDomain::<f64>::new();
    let input = vector_input()?;
    measure("array_jacrev_vector", "array", "jacrev", iterations, warmup, || {
        let jacobian = jacrev(&domain, |x| Ok(x.clone() * x.clone() + x.zero_like()), black_box(input.clone()))?;
        Ok(differential_checksum(&jacobian))
    })
}

/// Runs the scalar-array Hessian benchmark.
fn run_array_hessian_scalar(
    iterations: usize,
    warmup: usize,
) -> Result<TransformBenchmarkRecord, Box<dyn std::error::Error>> {
    let domain = NdArrayDomain::<f64>::new();
    let input = Array::from_shape_vec([], vec![2.0])?;
    measure("array_hessian_scalar", "array", "hessian", iterations, warmup, || {
        let hessian = domain.hessian(|x| x.clone() * x.clone() + x.sin(), black_box(input.clone()))?;
        Ok(differential_checksum(&hessian))
    })
}
