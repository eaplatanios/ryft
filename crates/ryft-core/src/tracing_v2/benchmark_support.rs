use std::ops::{Add, Mul, Neg};

use crate::tracing::engines::{Engine, Tracer, TracingEngine};
use crate::tracing::{Program, Traceable, TracingError};
use crate::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program,
};
use crate::tracing_v2::operations::constants::OneLike;
use crate::tracing_v2::{
    ArrayOperation, DifferentiableEngine, DifferentiableTracingEngine, LinearArrayOperation, Sin, grad, jvp,
    jvp_program, value_and_grad, vjp,
};
use crate::types::ArrayType;

#[derive(Copy, Clone, Debug)]
struct ArrayScalarEngine;

impl Engine for ArrayScalarEngine {
    type Type = ArrayType;
    type Value = f64;

    fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
        Ok(0.0)
    }

    fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
        Ok(1.0)
    }
}

impl TracingEngine for ArrayScalarEngine {
    type Operation = ArrayOperation<f64>;
}

impl DifferentiableEngine for ArrayScalarEngine {
    type DifferentiableOperation = ArrayOperation<f64>;
    type LinearOperation = LinearArrayOperation<f64>;
}

impl DifferentiableTracingEngine for ArrayScalarEngine {
    type LinearOperation<'engine>
        = LinearArrayOperation<Tracer<'engine, Self>>
    where
        Self: 'engine;
}

/// Returns the tracing-only IR benchmark cases.
pub(crate) fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase { case_id: "scalar_bilinear_sin_jit", emit: emit_scalar_bilinear_sin_jit },
        BenchmarkCase { case_id: "scalar_bilinear_sin_jvp", emit: emit_scalar_bilinear_sin_jvp },
        BenchmarkCase { case_id: "scalar_bilinear_sin_vjp_pullback", emit: emit_scalar_bilinear_sin_vjp_pullback },
        BenchmarkCase { case_id: "scalar_quartic_plus_sin_grad", emit: emit_scalar_quartic_plus_sin_grad },
        BenchmarkCase {
            case_id: "scalar_quartic_plus_sin_value_and_grad",
            emit: emit_scalar_quartic_plus_sin_value_and_grad,
        },
        BenchmarkCase {
            case_id: "scalar_quartic_plus_sin_linearize_pushforward",
            emit: emit_scalar_quartic_plus_sin_linearize_pushforward,
        },
        BenchmarkCase {
            case_id: "scalar_quartic_plus_sin_hessian_style",
            emit: emit_scalar_quartic_plus_sin_hessian_style,
        },
    ]
}

/// Summarizes one plain staged `tracing_v2` program.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_tracing_program<V, Input, Output, O>(
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError>
where
    V: Traceable<ArrayType>,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
    O: Clone + crate::operations::Operation<ArrayType>,
{
    summarize_program(program, |_| Ok(Vec::new()))
}

/// Builds one tracing benchmark record from a staged program.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `surface`: Artifact surface to record.
///   - `program`: Program to render and summarize.
fn tracing_record<V, Input, Output, O>(
    case_id: &'static str,
    surface: &'static str,
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<IrBenchmarkRecord, BenchmarkError>
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::Sin
        + crate::tracing_v2::Cos
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + crate::tracing_v2::operations::constants::ZeroLike
        + crate::tracing_v2::operations::constants::OneLike
        + crate::tracing_v2::MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
    O: Clone + crate::operations::Operation<ArrayType>,
{
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

fn first_derivative_traced(x: Tracer<ArrayScalarEngine>) -> Tracer<ArrayScalarEngine> {
    grad(&ArrayScalarEngine, quartic_plus_sin, x).expect("scalar first traced derivative should succeed")
}

fn hessian_style_second_derivative_traced(x: Tracer<ArrayScalarEngine>) -> Tracer<ArrayScalarEngine> {
    jvp(&ArrayScalarEngine, first_derivative_traced, x.clone(), x.one_like())
        .expect("scalar Hessian-style benchmark should succeed")
        .1
}

/// Emits the plain JIT scalar bilinear benchmark.
fn emit_scalar_bilinear_sin_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<ArrayType, f64, crate::tracing_v2::ArrayOperation<f64>, (f64, f64), f64>) =
        ArrayScalarEngine
            .interpret_and_trace(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jit", "jit", &compiled)?])
}

/// Emits the staged scalar bilinear pushforward benchmark.
fn emit_scalar_bilinear_sin_jvp() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (
        f64,
        Program<ArrayType, f64, crate::tracing_v2::LinearArrayOperation<f64>, (f64, f64), f64>,
    ) = jvp_program(&ArrayScalarEngine, |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jvp", "jvp_pushforward", &pushforward)?])
}

/// Emits the staged scalar bilinear pullback benchmark.
fn emit_scalar_bilinear_sin_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (f64, Program<ArrayType, f64, crate::tracing_v2::LinearArrayOperation<f64>, f64, (f64, f64)>) =
        vjp(&ArrayScalarEngine, |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_vjp_pullback", "vjp_pullback", &pullback)?])
}

/// Emits the staged scalar reverse-mode gradient benchmark.
fn emit_scalar_quartic_plus_sin_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<ArrayType, f64, crate::tracing_v2::ArrayOperation<f64>, f64, f64>) =
        ArrayScalarEngine.interpret_and_trace(
            |x| {
                let gradient: Tracer<ArrayScalarEngine> = grad(&ArrayScalarEngine, quartic_plus_sin, x)?;
                Ok(gradient)
            },
            2.0f64,
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_grad", "grad", &compiled)?])
}

/// Emits the staged scalar value-and-gradient benchmark.
fn emit_scalar_quartic_plus_sin_value_and_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): ((f64, f64), Program<ArrayType, f64, crate::tracing_v2::ArrayOperation<f64>, f64, (f64, f64)>) =
        ArrayScalarEngine.interpret_and_trace(
            |x| {
                let value_and_gradient: (Tracer<ArrayScalarEngine>, Tracer<ArrayScalarEngine>) =
                    value_and_grad(&ArrayScalarEngine, quartic_plus_sin, x)?;
                Ok(value_and_gradient)
            },
            2.0f64,
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_value_and_grad", "value_and_grad", &compiled)?])
}

/// Emits the staged scalar linearization benchmark.
fn emit_scalar_quartic_plus_sin_linearize_pushforward() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (f64, Program<ArrayType, f64, crate::tracing_v2::LinearArrayOperation<f64>, f64, f64>) =
        jvp_program(&ArrayScalarEngine, |x| Ok(quartic_plus_sin(x)), 2.0f64)?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_linearize_pushforward", "linearize_pushforward", &pushforward)?])
}

/// Emits the staged forward-over-reverse scalar benchmark.
fn emit_scalar_quartic_plus_sin_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<ArrayType, f64, crate::tracing_v2::ArrayOperation<f64>, f64, f64>) =
        ArrayScalarEngine.interpret_and_trace(|x| Ok(hessian_style_second_derivative_traced(x)), 2.0f64)?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_hessian_style", "hessian_style", &compiled)?])
}
