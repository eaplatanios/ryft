use std::ops::{Add, Mul, Neg};

use ryft_core::operations::Operation;
use ryft_core::parameters::Parameterized;
use ryft_core::tracing::contexts::TracingContext;
use ryft_core::tracing::domains::TracingDomain;
use ryft_core::tracing::{Program, Traceable};
use ryft_core::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program,
};
use ryft_core::tracing_v2::operations::dot::DotDimensionNumbers;
use ryft_core::tracing_v2::{DifferentiableDomain, DotOps};
use ryft_core::types::ArrayType;

use crate::{Array, LinearNdarrayOperation, NdArrayDomain, NdarrayOperation};

type Matrix = Array<f64>;
type MatrixPair = (Matrix, Matrix);
type MatrixProgram<Input, Output> = Program<ArrayType, Matrix, NdarrayOperation<Matrix>, Input, Output>;
type MatrixLinearProgram<Input, Output> = Program<ArrayType, Matrix, LinearNdarrayOperation<Matrix>, Input, Output>;

/// Returns the `ndarray`-backed IR benchmark cases.
pub fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase::new("matrix_matmul_jit", emit_matrix_matmul_jit),
        BenchmarkCase::new("matrix_matmul_vjp_pullback", emit_matrix_matmul_vjp_pullback),
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
fn bilinear_matmul<M: Clone + DotOps + Add<Output = M> + Mul<Output = M> + Neg<Output = M>>(inputs: (M, M)) -> M {
    inputs.0.dot(inputs.1, &DotDimensionNumbers::matmul())
}

/// Emits the staged matrix JIT benchmark.
fn emit_matrix_matmul_jit() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, compiled): (Matrix, MatrixProgram<MatrixPair, Matrix>) = TracingContext::interpret_and_trace(
        &NdArrayDomain::<f64>::new(),
        |inputs| Ok(bilinear_matmul(inputs)),
        matrix_inputs(),
    )?;
    Ok(vec![ndarray_record("matrix_matmul_jit", "jit", &compiled)?])
}

/// Emits the staged matrix pullback benchmark.
fn emit_matrix_matmul_vjp_pullback() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let (_, pullback): (Matrix, MatrixLinearProgram<Matrix, MatrixPair>) =
        NdArrayDomain::<f64>::new().vjp(|inputs| Ok(bilinear_matmul(inputs)), matrix_inputs())?;
    Ok(vec![ndarray_record("matrix_matmul_vjp_pullback", "vjp_pullback", &pullback)?])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::cases;

    #[test]
    fn test_cases_contains_matrix_benchmarks() {
        let case_ids = cases().into_iter().map(|case| case.case_id()).collect::<Vec<_>>();

        assert_eq!(case_ids, vec!["matrix_matmul_jit", "matrix_matmul_vjp_pullback"]);
    }
}
