//! Internal IR benchmark support for `tracing_v2`.
//!
//! This module owns the Rust-side benchmark cases that stay entirely within the staged
//! `tracing_v2` program IR.

use std::ops::{Add, Mul, Neg};

#[cfg(feature = "ndarray")]
use ndarray::{Array2, arr2};

use crate::tracing_v2::{
    Batch, CompiledFunction, FloatExt, JitTracer, LinearProgram, OneLike, Program, TraceValue,
    benchmarking::{BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_graph},
    grad, jit, jvp, jvp_program, linearize, stack, try_jit, unstack, value_and_grad, vjp, vmap,
};
#[cfg(feature = "ndarray")]
use crate::tracing_v2::{MatrixOps, ZeroLike};
#[cfg(feature = "xla")]
use crate::xla::lowering::to_mlir_module_for_plain_graph;

/// Returns the tracing-only IR benchmark cases.
pub(crate) fn cases() -> Vec<BenchmarkCase> {
    let cases = vec![
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
        BenchmarkCase { case_id: "scalar_grad_of_vmap", emit: emit_scalar_grad_of_vmap },
        BenchmarkCase { case_id: "scalar_vmap_of_grad", emit: emit_scalar_vmap_of_grad },
    ];

    #[cfg(feature = "ndarray")]
    {
        let mut cases = cases;
        cases.push(BenchmarkCase { case_id: "matrix_matmul_jit", emit: emit_matrix_matmul_jit });
        cases.push(BenchmarkCase { case_id: "matrix_matmul_vjp_pullback", emit: emit_matrix_matmul_vjp_pullback });
        cases.push(BenchmarkCase {
            case_id: "matrix_three_matmul_sine_hessian_style",
            emit: emit_matrix_three_matmul_sine_hessian_style,
        });
        return cases;
    }

    #[cfg(not(feature = "ndarray"))]
    {
        cases
    }
}

/// Summarizes one plain staged `tracing_v2` program.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_program<V, Input, Output>(
    program: &Program<V, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError>
where
    V: TraceValue,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
{
    summarize_graph(program.graph(), |_| Ok(Vec::new()))
}

/// Builds one tracing benchmark record from a staged program.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `surface`: Artifact surface to record.
///   - `program`: Program to render and summarize.
#[cfg(feature = "xla")]
fn tracing_record<V, Input, Output>(
    case_id: &'static str,
    surface: &'static str,
    program: &Program<V, Input, Output>,
) -> Result<IrBenchmarkRecord, BenchmarkError>
where
    V: crate::xla::lowering::MlirLowerableValue,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
{
    Ok(record(
        case_id,
        tracing_category(case_id),
        surface,
        to_mlir_module_for_plain_graph(program.graph(), "main")?,
        summarize_program(program)?,
    ))
}

#[cfg(not(feature = "xla"))]
fn tracing_record<V, Input, Output>(
    case_id: &'static str,
    surface: &'static str,
    program: &Program<V, Input, Output>,
) -> Result<IrBenchmarkRecord, BenchmarkError>
where
    V: TraceValue,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
{
    Ok(record(case_id, tracing_category(case_id), surface, program.to_string(), summarize_program(program)?))
}

/// Returns the high-level category string for one tracing case.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
fn tracing_category(case_id: &str) -> &'static str {
    if case_id.starts_with("matrix_") { "matrix" } else { "scalar" }
}

/// Benchmark helper used by the scalar bilinear benchmark family.
///
/// # Parameters
///
///   - `inputs`: Structured scalar inputs.
fn bilinear_sin<T>(inputs: (T, T)) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin()
}

/// Benchmark helper used by the scalar higher-order benchmark family.
///
/// # Parameters
///
///   - `x`: Scalar input.
fn quartic_plus_sin<T>(x: T) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
}

fn first_derivative_traced(x: JitTracer<f64>) -> JitTracer<f64> {
    grad(quartic_plus_sin, x).expect("scalar first traced derivative should succeed")
}

fn hessian_style_second_derivative_traced(x: JitTracer<f64>) -> JitTracer<f64> {
    jvp(first_derivative_traced, x.clone(), x.one_like())
        .expect("scalar Hessian-style benchmark should succeed")
        .1
}

/// Emits the plain JIT scalar bilinear benchmark.
fn emit_scalar_bilinear_sin_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, CompiledFunction<f64, (f64, f64), f64>) = jit(bilinear_sin, (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jit", "jit", compiled.program())?])
}

/// Emits the staged scalar bilinear pushforward benchmark.
fn emit_scalar_bilinear_sin_jvp() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (f64, LinearProgram<f64, (f64, f64), f64>) = jvp_program(bilinear_sin, (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jvp", "jvp_pushforward", pushforward.program())?])
}

/// Emits the staged scalar bilinear pullback benchmark.
fn emit_scalar_bilinear_sin_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (f64, LinearProgram<f64, f64, (f64, f64)>) = vjp(bilinear_sin, (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_vjp_pullback", "vjp_pullback", pullback.program())?])
}

/// Emits the staged scalar reverse-mode gradient benchmark.
fn emit_scalar_quartic_plus_sin_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = try_jit(
        |x: JitTracer<f64>| {
            let gradient: JitTracer<f64> = grad(quartic_plus_sin, x)?;
            Ok(gradient)
        },
        2.0f64,
    )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_grad", "grad", compiled.program())?])
}

/// Emits the staged scalar value-and-gradient benchmark.
fn emit_scalar_quartic_plus_sin_value_and_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): ((f64, f64), CompiledFunction<f64, f64, (f64, f64)>) = try_jit(
        |x: JitTracer<f64>| {
            let value_and_gradient: (JitTracer<f64>, JitTracer<f64>) = value_and_grad(quartic_plus_sin, x)?;
            Ok(value_and_gradient)
        },
        2.0f64,
    )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_value_and_grad", "value_and_grad", compiled.program())?])
}

/// Emits the staged scalar linearization benchmark.
fn emit_scalar_quartic_plus_sin_linearize_pushforward() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (f64, LinearProgram<f64, f64, f64>) = linearize(quartic_plus_sin, 2.0f64)?;
    Ok(vec![tracing_record(
        "scalar_quartic_plus_sin_linearize_pushforward",
        "linearize_pushforward",
        pushforward.program(),
    )?])
}

/// Emits the staged forward-over-reverse scalar benchmark.
fn emit_scalar_quartic_plus_sin_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(hessian_style_second_derivative_traced, 2.0f64)?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_hessian_style", "hessian_style", compiled.program())?])
}

/// Emits the staged reverse-over-batching scalar benchmark.
fn emit_scalar_grad_of_vmap() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = try_jit(
        |x: JitTracer<f64>| {
            let gradient: JitTracer<f64> = grad(
                |y: JitTracer<f64>| {
                    let outputs: Vec<JitTracer<f64>> = vmap(
                        |batch: Batch<JitTracer<f64>>| batch.clone() * batch.clone() + batch.sin(),
                        vec![y.clone(), y],
                    )
                    .unwrap_or_else(|error| {
                        panic!("scalar grad-of-vmap IR benchmark should batch identical tracer inputs: {error}")
                    });
                    outputs[0].clone() + outputs[1].clone()
                },
                x,
            )?;
            Ok(gradient)
        },
        2.0f64,
    )?;
    Ok(vec![tracing_record("scalar_grad_of_vmap", "grad", compiled.program())?])
}

/// Emits the staged batching-over-reverse scalar benchmark.
fn emit_scalar_vmap_of_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = try_jit(
        |x: JitTracer<f64>| {
            let outputs: Vec<JitTracer<f64>> = vmap(
                |batch: Batch<JitTracer<f64>>| {
                    let lanes = unstack::<JitTracer<f64>, JitTracer<f64>>(batch).unwrap_or_else(|error| {
                        panic!("scalar vmap-of-grad IR benchmark should unstack the batch: {error}")
                    });
                    let gradients = lanes
                        .into_iter()
                        .map(|lane| {
                            grad(quartic_plus_sin, lane).unwrap_or_else(|error| {
                                panic!("scalar vmap-of-grad IR benchmark should trace each lane gradient: {error}")
                            })
                        })
                        .collect::<Vec<_>>();
                    stack::<JitTracer<f64>, JitTracer<f64>>(gradients).unwrap_or_else(|error| {
                        panic!("scalar vmap-of-grad IR benchmark should restack lane gradients: {error}")
                    })
                },
                vec![x.clone(), x],
            )
            .unwrap_or_else(|error| panic!("scalar vmap-of-grad IR benchmark should batch the gradients: {error}"));
            Ok(outputs[0].clone() + outputs[1].clone())
        },
        2.0f64,
    )?;
    Ok(vec![tracing_record("scalar_vmap_of_grad", "vmap_of_grad", compiled.program())?])
}

/// Returns the fixed matrix inputs used by the matrix benchmark cases.
#[cfg(feature = "ndarray")]
fn matrix_inputs() -> (Array2<f64>, Array2<f64>) {
    (arr2(&[[1.0f64, 2.0], [3.0, 4.0]]), arr2(&[[5.0f64, 6.0], [7.0, 8.0]]))
}

/// Benchmark helper used by the matrix benchmark family.
///
/// # Parameters
///
///   - `inputs`: Structured matrix inputs.
#[cfg(feature = "ndarray")]
fn bilinear_matmul<M>(inputs: (M, M)) -> M
where
    M: Clone + FloatExt + MatrixOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>,
{
    inputs.0.matmul(inputs.1)
}

#[cfg(feature = "ndarray")]
fn three_matmul_sine<M>(inputs: (M, M, M, M)) -> M
where
    M: Clone + FloatExt + MatrixOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>,
{
    let (x, a, b, c) = inputs;
    x.matmul(a).sin().matmul(b).matmul(c)
}

#[cfg(feature = "ndarray")]
fn hessian_style_matrix_inputs() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    (arr2(&[[0.7f64]]), arr2(&[[2.0f64]]), arr2(&[[-1.5f64]]), arr2(&[[4.0f64]]))
}

#[cfg(feature = "ndarray")]
fn first_matrix_gradient_traced(
    inputs: (JitTracer<Array2<f64>>, JitTracer<Array2<f64>>, JitTracer<Array2<f64>>, JitTracer<Array2<f64>>),
) -> JitTracer<Array2<f64>> {
    let (x_bar, _, _, _) = grad(three_matmul_sine, inputs).expect("nested matrix gradient benchmark should stage");
    x_bar
}

#[cfg(feature = "ndarray")]
fn matrix_hessian_style_second_derivative(
    inputs: (JitTracer<Array2<f64>>, JitTracer<Array2<f64>>, JitTracer<Array2<f64>>, JitTracer<Array2<f64>>),
) -> JitTracer<Array2<f64>> {
    let seeds = (inputs.0.one_like(), inputs.1.zero_like(), inputs.2.zero_like(), inputs.3.zero_like());
    jvp(first_matrix_gradient_traced, inputs, seeds)
        .expect("matrix Hessian-style benchmark should succeed")
        .1
}

/// Emits the staged matrix JIT benchmark.
#[cfg(feature = "ndarray")]
fn emit_matrix_matmul_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (Array2<f64>, CompiledFunction<Array2<f64>, (Array2<f64>, Array2<f64>), Array2<f64>>) =
        jit(bilinear_matmul, matrix_inputs())?;
    Ok(vec![tracing_record("matrix_matmul_jit", "jit", compiled.program())?])
}

/// Emits the staged matrix pullback benchmark.
#[cfg(feature = "ndarray")]
fn emit_matrix_matmul_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (Array2<f64>, LinearProgram<Array2<f64>, Array2<f64>, (Array2<f64>, Array2<f64>)>) =
        vjp(bilinear_matmul, matrix_inputs())?;
    Ok(vec![tracing_record("matrix_matmul_vjp_pullback", "vjp_pullback", pullback.program())?])
}

/// Emits the staged matrix Hessian-style benchmark.
#[cfg(feature = "ndarray")]
fn emit_matrix_three_matmul_sine_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (
        Array2<f64>,
        CompiledFunction<Array2<f64>, (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), Array2<f64>>,
    ) = jit(matrix_hessian_style_second_derivative, hessian_style_matrix_inputs())?;
    Ok(vec![tracing_record("matrix_three_matmul_sine_hessian_style", "hessian_style", compiled.program())?])
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ndarray")]
    use pretty_assertions::assert_eq;

    #[cfg(feature = "ndarray")]
    use super::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_emit_scalar_grad_of_vmap_keeps_dynamic_cosine() {
        let records = emit_scalar_grad_of_vmap().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].raw_ir.contains("stablehlo.cosine %arg0"));
        assert!(!records[0].raw_ir.contains("-0.41614683654714241"));
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_emit_matrix_three_matmul_sine_hessian_style_surfaces_sine_and_negate() {
        let records = emit_matrix_three_matmul_sine_hessian_style().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].summary.op_histogram.get("sin"), Some(&1));
        assert_eq!(records[0].summary.op_histogram.get("neg"), Some(&1));
    }
}
