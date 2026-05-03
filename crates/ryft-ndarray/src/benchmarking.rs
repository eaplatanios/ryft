use std::ops::{Add, Mul, Neg};

use ryft_core::operations::Operation;
use ryft_core::operations::constants::{OneLike, ZeroLike};
use ryft_core::parameters::Parameterized;
use ryft_core::tracing::engines::{Tracer, TracingEngine};
use ryft_core::tracing::{Program, Traceable};
use ryft_core::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program,
};
use ryft_core::tracing_v2::{MatrixOps, Sin, jvp, vjp};
use ryft_core::types::ArrayType;

use crate::{Array, LinearNdarrayOperation, NdArrayEngine, NdarrayOperation};

type Matrix = Array<f64>;
type MatrixPair = (Matrix, Matrix);
type MatrixQuad = (Matrix, Matrix, Matrix, Matrix);
type MatrixTracer<'engine> = Tracer<'engine, NdArrayEngine<f64>>;
type MatrixProgram<Input, Output> = Program<ArrayType, Matrix, NdarrayOperation<Matrix>, Input, Output>;
type MatrixLinearProgram<Input, Output> = Program<ArrayType, Matrix, LinearNdarrayOperation<Matrix>, Input, Output>;

/// Returns the `ndarray`-backed IR benchmark cases.
pub fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase { case_id: "matrix_matmul_jit", emit: emit_matrix_matmul_jit },
        BenchmarkCase { case_id: "matrix_matmul_vjp_pullback", emit: emit_matrix_matmul_vjp_pullback },
        BenchmarkCase {
            case_id: "matrix_three_matmul_sine_hessian_style",
            emit: emit_matrix_three_matmul_sine_hessian_style,
        },
    ]
}

/// Summarizes one plain staged `tracing_v2` program.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_tracing_program<
    V: Traceable<ArrayType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + Operation<ArrayType>,
>(
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError> {
    summarize_program(program, |_| Ok(Vec::new()))
}

/// Builds one ndarray benchmark record from a staged program.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `surface`: Artifact surface to record.
///   - `program`: Program to render and summarize.
fn ndarray_record<
    V: Traceable<ArrayType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + Operation<ArrayType>,
>(
    case_id: &'static str,
    surface: &'static str,
    program: &Program<ArrayType, V, O, Input, Output>,
) -> Result<IrBenchmarkRecord, BenchmarkError> {
    Ok(record(case_id, "matrix", surface, program.to_string(), summarize_tracing_program(program)?))
}

/// Returns the fixed matrix inputs used by the matrix benchmark cases.
fn matrix_inputs() -> MatrixPair {
    (
        Array::from_shape_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap(),
        Array::from_shape_vec([2, 2], vec![5.0f64, 6.0, 7.0, 8.0]).unwrap(),
    )
}

/// Benchmark helper used by the matrix benchmark family.
///
/// # Parameters
///
///   - `inputs`: Structured matrix inputs.
fn bilinear_matmul<M>(inputs: (M, M)) -> M
where
    M: Clone + MatrixOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>,
{
    inputs.0.matmul(inputs.1)
}

/// Benchmark helper used by the higher-order matrix benchmark.
///
/// # Parameters
///
///   - `inputs`: Structured matrix inputs.
fn three_matmul_sine<M>(inputs: (M, M, M, M)) -> M
where
    M: Clone + Sin + MatrixOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>,
{
    let (x, a, b, c) = inputs;
    x.matmul(a).sin().matmul(b).matmul(c)
}

/// Returns the compact matrix inputs used by the higher-order benchmark.
fn hessian_style_matrix_inputs() -> MatrixQuad {
    (
        Array::from_shape_vec([1, 1], vec![0.7f64]).unwrap(),
        Array::from_shape_vec([1, 1], vec![2.0f64]).unwrap(),
        Array::from_shape_vec([1, 1], vec![-1.5f64]).unwrap(),
        Array::from_shape_vec([1, 1], vec![4.0f64]).unwrap(),
    )
}

/// Stages one matrix JVP for the higher-order benchmark.
///
/// # Parameters
///
///   - `inputs`: Traced matrix inputs.
fn first_matrix_jvp_traced<'engine>(
    inputs: (MatrixTracer<'engine>, MatrixTracer<'engine>, MatrixTracer<'engine>, MatrixTracer<'engine>),
) -> MatrixTracer<'engine> {
    let seeds = (inputs.0.one_like(), inputs.1.zero_like(), inputs.2.zero_like(), inputs.3.zero_like());
    jvp(&NdArrayEngine::<f64>::new(), three_matmul_sine, inputs, seeds)
        .expect("nested matrix JVP benchmark should stage")
        .1
}

/// Stages a second matrix JVP to exercise higher-order matrix IR.
///
/// # Parameters
///
///   - `inputs`: Traced matrix inputs.
fn matrix_hessian_style_second_derivative<'engine>(
    inputs: (MatrixTracer<'engine>, MatrixTracer<'engine>, MatrixTracer<'engine>, MatrixTracer<'engine>),
) -> MatrixTracer<'engine> {
    let seeds = (inputs.0.one_like(), inputs.1.zero_like(), inputs.2.zero_like(), inputs.3.zero_like());
    jvp(&NdArrayEngine::<f64>::new(), first_matrix_jvp_traced, inputs, seeds)
        .expect("matrix Hessian-style benchmark should succeed")
        .1
}

/// Emits the staged matrix JIT benchmark.
fn emit_matrix_matmul_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (Matrix, MatrixProgram<MatrixPair, Matrix>) =
        NdArrayEngine::<f64>::new().interpret_and_trace(|inputs| Ok(bilinear_matmul(inputs)), matrix_inputs())?;
    Ok(vec![ndarray_record("matrix_matmul_jit", "jit", &compiled)?])
}

/// Emits the staged matrix pullback benchmark.
fn emit_matrix_matmul_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (Matrix, MatrixLinearProgram<Matrix, MatrixPair>) =
        vjp(&NdArrayEngine::<f64>::new(), |inputs| Ok(bilinear_matmul(inputs)), matrix_inputs())?;
    Ok(vec![ndarray_record("matrix_matmul_vjp_pullback", "vjp_pullback", &pullback)?])
}

/// Emits the staged matrix Hessian-style benchmark.
fn emit_matrix_three_matmul_sine_hessian_style() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (Matrix, MatrixProgram<MatrixQuad, Matrix>) = NdArrayEngine::<f64>::new().interpret_and_trace(
        |inputs| Ok(matrix_hessian_style_second_derivative(inputs)),
        hessian_style_matrix_inputs(),
    )?;
    Ok(vec![ndarray_record("matrix_three_matmul_sine_hessian_style", "hessian_style", &compiled)?])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::{cases, emit_matrix_three_matmul_sine_hessian_style};

    #[test]
    fn test_cases_contains_matrix_benchmarks() {
        let case_ids = cases().into_iter().map(|case| case.case_id).collect::<Vec<_>>();

        assert_eq!(
            case_ids,
            vec!["matrix_matmul_jit", "matrix_matmul_vjp_pullback", "matrix_three_matmul_sine_hessian_style"]
        );
    }

    #[test]
    fn test_emit_matrix_three_matmul_sine_hessian_style_surfaces_sine_and_negate() {
        let records = emit_matrix_three_matmul_sine_hessian_style().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].summary.op_histogram.get("sin"), Some(&2));
        assert_eq!(records[0].summary.op_histogram.get("neg"), Some(&1));
    }
}
