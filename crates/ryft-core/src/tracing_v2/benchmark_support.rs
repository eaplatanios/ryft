use std::ops::{Add, Mul, Neg};

use crate::contexts::Context;
use crate::operations::scalars::ScalarOperation;
use crate::operations::trigonometric::{Cos, Sin};
use crate::programs::{Program, ProgramError, Value};
use crate::scalars::Scalar;
use crate::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program,
};
use crate::tracing_v2::{DifferentiationContext, DifferentiationError};
use crate::types::{DataType, Type};

/// Returns the tracing-only IR benchmark cases.
pub(crate) fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase::new("scalar_bilinear_sin_jit", emit_scalar_bilinear_sin_jit),
        BenchmarkCase::new("scalar_bilinear_sin_vjp_pullback", emit_scalar_bilinear_sin_vjp_pullback),
        BenchmarkCase::new("scalar_quartic_plus_sin_grad", emit_scalar_quartic_plus_sin_grad),
        BenchmarkCase::new("scalar_quartic_plus_sin_value_and_grad", emit_scalar_quartic_plus_sin_value_and_grad),
    ]
}

/// Summarizes one plain staged `tracing_v2` program.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_tracing_program<
    T: Type,
    V: Value<Type = T>,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
    O: Clone + crate::operations::Operation<T>,
>(
    program: &Program<V, O, Input, Output>,
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
    V: Value<Type = T>
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
    program: &Program<V, O, Input, Output>,
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
fn quartic_plus_sin<T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>>(x: T) -> T {
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
}

/// Emits the plain JIT scalar bilinear benchmark.
fn emit_scalar_bilinear_sin_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (Scalar, Program<Scalar, ScalarOperation<Scalar>, (Scalar, Scalar), Scalar>) =
        EagerContext::<Scalar, ScalarOperation<Scalar>>::new().interpret_and_trace(
            |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()?),
            (Scalar::from(2.0), Scalar::from(3.0)),
        )?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jit", "jit", &compiled)?])
}

/// Emits the staged scalar bilinear pullback benchmark.
fn emit_scalar_bilinear_sin_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (Scalar, Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>) =
        EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .vjp(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()?), (Scalar::from(2.0), Scalar::from(3.0)))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_vjp_pullback", "vjp_pullback", &pullback)?])
}

/// Emits the staged scalar reverse-mode gradient benchmark.
fn emit_scalar_quartic_plus_sin_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (Scalar, Program<Scalar, ScalarOperation<Scalar>, Scalar, Scalar>) =
        EagerContext::<Scalar, ScalarOperation<Scalar>>::new().interpret_and_trace(
            |x| {
                let context = x.context().clone();
                // `interpret_and_trace` fixes its closure error to `ProgramError`, so fold the inner gradient's
                // differentiation error into a program error. A non-scalar gradient output cannot occur for this
                // scalar benchmark function.
                let gradient = context.value_and_gradient(quartic_plus_sin, x).map_err(|error| match error {
                    DifferentiationError::Program(error) => error,
                    error => ProgramError::MalformedProgram(error.to_string()),
                })?;
                Ok(gradient)
            },
            Scalar::from(2.0),
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_grad", "grad", &compiled)?])
}

/// Emits the staged scalar value-and-gradient benchmark.
fn emit_scalar_quartic_plus_sin_value_and_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): ((Scalar, Scalar), Program<Scalar, ScalarOperation<Scalar>, Scalar, (Scalar, Scalar)>) =
        EagerContext::<Scalar, ScalarOperation<Scalar>>::new().interpret_and_trace(
            |x| {
                let context = x.context().clone();
                // `interpret_and_trace` fixes its closure error to `ProgramError`, so fold the inner gradient's
                // differentiation error into a program error. A non-scalar gradient output cannot occur for this
                // scalar benchmark function.
                let value_and_gradient = context.value_and_grad(quartic_plus_sin, x).map_err(|error| match error {
                    DifferentiationError::Program(error) => error,
                    error => ProgramError::MalformedProgram(error.to_string()),
                })?;
                Ok(value_and_gradient)
            },
            Scalar::from(2.0),
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_value_and_grad", "value_and_grad", &compiled)?])
}
