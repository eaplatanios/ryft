use std::ops::{Add, Mul, Neg};

#[cfg(feature = "ndarray")]
use ndarray::{Array2, arr2};

use crate::tracing::{Program, Traceable};
#[cfg(feature = "ndarray")]
use crate::tracing_v2::{
    MatrixOps,
    operations::{constants::ZeroLike, matrix::ndarray_support::Array2Engine},
};
use crate::tracing_v2::{
    Sin, Tracer,
    benchmarking::{BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program},
    engines::ScalarEngine,
    grad, interpret_and_trace, jvp, jvp_program,
    operations::constants::OneLike,
    value_and_grad, vjp,
};
use crate::types::ArrayType;

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
fn summarize_tracing_program<V, Input, Output, O>(
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError>
where
    V: Traceable<ArrayType>,
    Input: crate::parameters::Parameterized<V>,
    Output: crate::parameters::Parameterized<V>,
    O: Clone + crate::tracing::Operation<ArrayType>,
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
    O: Clone + crate::tracing::Operation<ArrayType>,
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

/// Benchmark helper used by the scalar bilinear benchmark family.
///
/// # Parameters
///
///   - `inputs`: Structured scalar inputs.
fn bilinear_sin<T>(inputs: (T, T)) -> T
where
    T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
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
    T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
}

fn first_derivative_traced(x: Tracer<ScalarEngine<f64>>) -> Tracer<ScalarEngine<f64>> {
    grad(&ScalarEngine::<f64>::new(), quartic_plus_sin, x).expect("scalar first traced derivative should succeed")
}

fn hessian_style_second_derivative_traced(x: Tracer<ScalarEngine<f64>>) -> Tracer<ScalarEngine<f64>> {
    jvp(&ScalarEngine::<f64>::new(), first_derivative_traced, x.clone(), x.one_like())
        .expect("scalar Hessian-style benchmark should succeed")
        .1
}

/// Emits the plain JIT scalar bilinear benchmark.
fn emit_scalar_bilinear_sin_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<ArrayType, f64, crate::tracing_v2::PrimitiveOperation<f64>, (f64, f64), f64>) =
        interpret_and_trace(&ScalarEngine::<f64>::new(), |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jit", "jit", &compiled)?])
}

/// Emits the staged scalar bilinear pushforward benchmark.
fn emit_scalar_bilinear_sin_jvp() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (
        f64,
        Program<ArrayType, f64, crate::tracing_v2::LinearPrimitiveOperation<f64>, (f64, f64), f64>,
    ) = jvp_program(&ScalarEngine::<f64>::new(), |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_jvp", "jvp_pushforward", &pushforward)?])
}

/// Emits the staged scalar bilinear pullback benchmark.
fn emit_scalar_bilinear_sin_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (
        f64,
        Program<ArrayType, f64, crate::tracing_v2::LinearPrimitiveOperation<f64>, f64, (f64, f64)>,
    ) = vjp(&ScalarEngine::<f64>::new(), |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64))?;
    Ok(vec![tracing_record("scalar_bilinear_sin_vjp_pullback", "vjp_pullback", &pullback)?])
}

/// Emits the staged scalar reverse-mode gradient benchmark.
fn emit_scalar_quartic_plus_sin_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<ArrayType, f64, crate::tracing_v2::PrimitiveOperation<f64>, f64, f64>) =
        interpret_and_trace(
            &ScalarEngine::<f64>::new(),
            |x| {
                let gradient: Tracer<ScalarEngine<f64>> =
                    grad(&ScalarEngine::<f64>::new(), quartic_plus_sin, x)?;
                Ok(gradient)
            },
            2.0f64,
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_grad", "grad", &compiled)?])
}

/// Emits the staged scalar value-and-gradient benchmark.
fn emit_scalar_quartic_plus_sin_value_and_grad() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (
        (f64, f64),
        Program<ArrayType, f64, crate::tracing_v2::PrimitiveOperation<f64>, f64, (f64, f64)>,
    ) = interpret_and_trace(
        &ScalarEngine::<f64>::new(),
        |x| {
            let value_and_gradient: (Tracer<ScalarEngine<f64>>, Tracer<ScalarEngine<f64>>) =
                value_and_grad(&ScalarEngine::<f64>::new(), quartic_plus_sin, x)?;
            Ok(value_and_gradient)
        },
        2.0f64,
    )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_value_and_grad", "value_and_grad", &compiled)?])
}

/// Emits the staged scalar linearization benchmark.
fn emit_scalar_quartic_plus_sin_linearize_pushforward() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pushforward): (f64, Program<ArrayType, f64, crate::tracing_v2::LinearPrimitiveOperation<f64>, f64, f64>) =
        jvp_program(&ScalarEngine::<f64>::new(), |x| Ok(quartic_plus_sin(x)), 2.0f64)?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_linearize_pushforward", "linearize_pushforward", &pushforward)?])
}

/// Emits the staged forward-over-reverse scalar benchmark.
fn emit_scalar_quartic_plus_sin_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (f64, Program<ArrayType, f64, crate::tracing_v2::PrimitiveOperation<f64>, f64, f64>) =
        interpret_and_trace(
            &ScalarEngine::<f64>::new(),
            |x| Ok(hessian_style_second_derivative_traced(x)),
            2.0f64,
        )?;
    Ok(vec![tracing_record("scalar_quartic_plus_sin_hessian_style", "hessian_style", &compiled)?])
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
    M: Clone + MatrixOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>,
{
    inputs.0.matmul(inputs.1)
}

#[cfg(feature = "ndarray")]
fn three_matmul_sine<M>(inputs: (M, M, M, M)) -> M
where
    M: Clone + Sin + MatrixOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>,
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
    inputs: (
        Tracer<Array2Engine<f64>>,
        Tracer<Array2Engine<f64>>,
        Tracer<Array2Engine<f64>>,
        Tracer<Array2Engine<f64>>,
    ),
) -> Tracer<Array2Engine<f64>> {
    let (x_bar, _, _, _) = grad(&Array2Engine::<f64>::new(), three_matmul_sine, inputs)
        .expect("nested matrix gradient benchmark should stage");
    x_bar
}

#[cfg(feature = "ndarray")]
fn matrix_hessian_style_second_derivative(
    inputs: (
        Tracer<Array2Engine<f64>>,
        Tracer<Array2Engine<f64>>,
        Tracer<Array2Engine<f64>>,
        Tracer<Array2Engine<f64>>,
    ),
) -> Tracer<Array2Engine<f64>> {
    let seeds = (inputs.0.one_like(), inputs.1.zero_like(), inputs.2.zero_like(), inputs.3.zero_like());
    jvp(&Array2Engine::<f64>::new(), first_matrix_gradient_traced, inputs, seeds)
        .expect("matrix Hessian-style benchmark should succeed")
        .1
}

/// Emits the staged matrix JIT benchmark.
#[cfg(feature = "ndarray")]
fn emit_matrix_matmul_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (
        Array2<f64>,
        Program<
            ArrayType,
            Array2<f64>,
            crate::tracing_v2::PrimitiveOperation<Array2<f64>>,
            (Array2<f64>, Array2<f64>),
            Array2<f64>,
        >,
    ) = interpret_and_trace(&Array2Engine::<f64>::new(), |inputs| Ok(bilinear_matmul(inputs)), matrix_inputs())?;
    Ok(vec![tracing_record("matrix_matmul_jit", "jit", &compiled)?])
}

/// Emits the staged matrix pullback benchmark.
#[cfg(feature = "ndarray")]
fn emit_matrix_matmul_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (
        Array2<f64>,
        Program<
            ArrayType,
            Array2<f64>,
            crate::tracing_v2::LinearPrimitiveOperation<Array2<f64>>,
            Array2<f64>,
            (Array2<f64>, Array2<f64>),
        >,
    ) = vjp(&Array2Engine::<f64>::new(), |inputs| Ok(bilinear_matmul(inputs)), matrix_inputs())?;
    Ok(vec![tracing_record("matrix_matmul_vjp_pullback", "vjp_pullback", &pullback)?])
}

/// Emits the staged matrix Hessian-style benchmark.
#[cfg(feature = "ndarray")]
fn emit_matrix_three_matmul_sine_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (
        Array2<f64>,
        Program<
            ArrayType,
            Array2<f64>,
            crate::tracing_v2::PrimitiveOperation<Array2<f64>>,
            (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>),
            Array2<f64>,
        >,
    ) = interpret_and_trace(
        &Array2Engine::<f64>::new(),
        |inputs| Ok(matrix_hessian_style_second_derivative(inputs)),
        hessian_style_matrix_inputs(),
    )?;
    Ok(vec![tracing_record("matrix_three_matmul_sine_hessian_style", "hessian_style", &compiled)?])
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ndarray")]
    use pretty_assertions::assert_eq;

    #[cfg(feature = "ndarray")]
    use super::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_emit_matrix_three_matmul_sine_hessian_style_surfaces_sine_and_negate() {
        let records = emit_matrix_three_matmul_sine_hessian_style().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].summary.op_histogram.get("sin"), Some(&1));
        assert_eq!(records[0].summary.op_histogram.get("neg"), Some(&1));
    }
}
