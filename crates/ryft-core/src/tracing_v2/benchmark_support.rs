use std::ops::{Add, Mul, Neg};

use crate::operations::constants::OneLike;
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::operations::trigonometric::{Cos, Sin};
use crate::tracing::domains::{ScalarDomain, Tracer, TracingDomain};
use crate::tracing::{Program, Traceable};
use crate::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program,
};
use crate::tracing_v2::{DifferentiableDomain, value_and_grad};
use crate::types::{DataType, Type};

/// Returns the tracing-only IR benchmark cases.
pub(crate) fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase::new("scalar_bilinear_sin_jit", emit_scalar_bilinear_sin_jit),
        BenchmarkCase::new("scalar_bilinear_sin_jvp", emit_scalar_bilinear_sin_jvp),
        BenchmarkCase::new("scalar_bilinear_sin_vjp_pullback", emit_scalar_bilinear_sin_vjp_pullback),
        BenchmarkCase::new("scalar_quartic_plus_sin_grad", emit_scalar_quartic_plus_sin_grad),
        BenchmarkCase::new("scalar_quartic_plus_sin_value_and_grad", emit_scalar_quartic_plus_sin_value_and_grad),
        BenchmarkCase::new(
            "scalar_quartic_plus_sin_linearize_pushforward",
            emit_scalar_quartic_plus_sin_linearize_pushforward,
        ),
        BenchmarkCase::new("scalar_quartic_plus_sin_hessian_style", emit_scalar_quartic_plus_sin_hessian_style),
    ]
}

/// Summarizes one plain staged `tracing_v2` program.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_tracing_program<
    T: Type,
    V: Traceable<T>,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
    O: Clone + crate::operations::Operation<T>,
>(
    program: &Program<T, V, O, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError> {
    summarize_program(program, |_| Ok(Vec::new()))
}

/// Builds one tracing benchmark record from a staged program.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `surface`: Artifact surface to record.
///   - `program`: Program to render and summarize.
fn tracing_record<
    T: Type,
    V: Traceable<T>
        + Sin
        + Cos
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + crate::operations::constants::ZeroLike
        + crate::operations::constants::OneLike,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
    O: Clone + crate::operations::Operation<T>,
>(
    case_id: &'static str,
    surface: &'static str,
    program: &Program<T, V, O, Input, Output>,
) -> Result<IrBenchmarkRecord, BenchmarkError> {
    Ok(record(case_id, tracing_category(case_id), surface, program.to_string(), summarize_tracing_program(program)?))
}

/// Returns the high-level category string for one tracing case.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
fn tracing_category(case_id: &str) -> &'static str {
    if case_id.starts_with("matrix_") { "matrix" } else { "scalar" }
}

/// Benchmark helper used by the scalar higher-order benchmark family.
///
/// # Parameters
///
///   - `x`: Scalar input.
fn quartic_plus_sin<T>(x: T) -> T
where
    T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
}

fn first_derivative_traced<'domain>(x: Tracer<'domain, ScalarDomain<f64>>) -> Tracer<'domain, ScalarDomain<f64>> {
    ScalarDomain::<f64>::new()
        .value_and_gradient(quartic_plus_sin, x)
        .expect("scalar first traced derivative should succeed")
}

fn hessian_style_second_derivative_traced<'domain>(
    x: Tracer<'domain, ScalarDomain<f64>>,
) -> Tracer<'domain, ScalarDomain<f64>> {
    ScalarDomain::<f64>::new()
        .jvp(first_derivative_traced, x.clone(), x.one_like())
        .expect("scalar Hessian-style benchmark should succeed")
        .1
}

/// Emits the plain JIT scalar bilinear benchmark.
fn emit_scalar_bilinear_sin_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, (f64, f64), f64>) =
        ScalarDomain::<f64>::new()
            .interpret_and_trace(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jit", "jit", &compiled)?])
}

/// Emits the staged scalar bilinear pushforward benchmark.
fn emit_scalar_bilinear_sin_jvp() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
        ScalarDomain::<f64>::new()
            .linearize(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jvp", "jvp_pushforward", &pushforward)?])
}

/// Emits the staged scalar bilinear pullback benchmark.
fn emit_scalar_bilinear_sin_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (f64, Program<DataType, f64, LinearScalarOperation<f64>, f64, (f64, f64)>) =
        ScalarDomain::<f64>::new().vjp(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_vjp_pullback", "vjp_pullback", &pullback)?])
}

/// Emits the staged scalar reverse-mode gradient benchmark.
fn emit_scalar_quartic_plus_sin_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) = ScalarDomain::<f64>::new()
        .interpret_and_trace(
            |x| {
                let gradient: Tracer<'_, ScalarDomain<f64>> =
                    ScalarDomain::<f64>::new().value_and_gradient(quartic_plus_sin, x)?;
                Ok(gradient)
            },
            2.0f64,
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_grad", "grad", &compiled)?])
}

/// Emits the staged scalar value-and-gradient benchmark.
fn emit_scalar_quartic_plus_sin_value_and_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): ((f64, f64), Program<DataType, f64, ScalarOperation<f64>, f64, (f64, f64)>) =
        ScalarDomain::<f64>::new().interpret_and_trace(
            |x| {
                let value_and_gradient: (Tracer<'_, ScalarDomain<f64>>, Tracer<'_, ScalarDomain<f64>>) =
                    value_and_grad(&ScalarDomain::<f64>::new(), quartic_plus_sin, x)?;
                Ok(value_and_gradient)
            },
            2.0f64,
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_value_and_grad", "value_and_grad", &compiled)?])
}

/// Emits the staged scalar linearization benchmark.
fn emit_scalar_quartic_plus_sin_linearize_pushforward() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, f64, f64>) =
        ScalarDomain::<f64>::new().linearize(|x| Ok(quartic_plus_sin(x)), 2.0f64)?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_linearize_pushforward", "linearize_pushforward", &pushforward)?])
}

/// Emits the staged forward-over-reverse scalar benchmark.
fn emit_scalar_quartic_plus_sin_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
        ScalarDomain::<f64>::new().interpret_and_trace(|x| Ok(hessian_style_second_derivative_traced(x)), 2.0f64)?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_hessian_style", "hessian_style", &compiled)?])
}
