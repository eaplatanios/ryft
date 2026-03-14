//! Matrix-specific tracing extensions built on top of the core `tracing_v2` primitives.
//!
//! This module keeps the non-commutative details of matrix multiplication localized by introducing explicit left and
//! right linear actions for transposition, while reusing the same JVP, batching, and JIT infrastructure as scalar
//! values.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::Parameter,
    tracing_v2::{
        FloatExt, ScalarAbstract, TraceError, TraceValue, ZeroLike,
        batch::Batch as BatchedValue,
        forward::{JvpTracer, TangentSpace},
        graph::{AtomId, GraphBuilder},
        jit::JitTracer,
        linear::LinearTerm,
        ops::{BatchOp, LinearOp, LinearOpRef, Op},
    },
};

/// Abstract matrix metadata used by staged tracing.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Parameter)]
pub struct MatrixAbstract {
    /// Scalar dtype of the matrix entries.
    pub scalar: ScalarAbstract,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl Display for MatrixAbstract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let scalar = match self.scalar {
            ScalarAbstract::F32 => "f32",
            ScalarAbstract::F64 => "f64",
        };
        write!(f, "{scalar}[{},{}]", self.rows, self.cols)
    }
}
/// Matrix operations required by the tracing prototype.
pub trait MatrixOps: Sized {
    /// Matrix multiplication.
    fn matmul(self, rhs: Self) -> Self;

    /// Matrix transpose.
    fn transpose_matrix(self) -> Self;
}

/// Convenience trait for traceable matrix leaves.
pub trait MatrixValue: TraceValue<Abstract = MatrixAbstract> + MatrixOps {}

impl<T> MatrixValue for T where T: TraceValue<Abstract = MatrixAbstract> + MatrixOps {}

/// Tangent representation for matrix-valued primals.
pub trait MatrixTangentSpace<V>: TangentSpace<V>
where
    V: MatrixValue,
{
    /// Applies the linear map `tangent -> factor @ tangent`.
    fn matmul_left(factor: V, tangent: Self) -> Self;

    /// Applies the linear map `tangent -> tangent @ factor`.
    fn matmul_right(tangent: Self, factor: V) -> Self;

    /// Transposes a tangent value.
    fn transpose_matrix(value: Self) -> Self;
}

impl<V> MatrixTangentSpace<V> for V
where
    V: MatrixValue + FloatExt + ZeroLike,
{
    #[inline]
    fn matmul_left(factor: V, tangent: Self) -> Self {
        factor.matmul(tangent)
    }

    #[inline]
    fn matmul_right(tangent: Self, factor: V) -> Self {
        tangent.matmul(factor)
    }

    #[inline]
    fn transpose_matrix(value: Self) -> Self {
        value.transpose_matrix()
    }
}

fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

fn expect_batch_sizes_match<V>(left: &BatchedValue<V>, right: &BatchedValue<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

fn matmul_abstract(lhs: &MatrixAbstract, rhs: &MatrixAbstract, op: &'static str) -> Result<MatrixAbstract, TraceError> {
    if lhs.scalar != rhs.scalar || lhs.cols != rhs.rows {
        return Err(TraceError::IncompatibleAbstractValues { op });
    }
    Ok(MatrixAbstract { scalar: lhs.scalar, rows: lhs.rows, cols: rhs.cols })
}

fn transpose_abstract(input: &MatrixAbstract) -> MatrixAbstract {
    MatrixAbstract { scalar: input.scalar, rows: input.cols, cols: input.rows }
}

fn single_batch_output<V>(mut outputs: Vec<BatchedValue<V>>, op: &'static str) -> BatchedValue<V>
where
    V: MatrixValue,
{
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single batched output");
    outputs.pop().expect("single-output matrix primitive should return one batched output")
}

/// Primitive representing matrix multiplication.
#[derive(Clone, Default)]
pub struct MatMulOp;

impl Debug for MatMulOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MatMul")
    }
}

impl Display for MatMulOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "matmul")
    }
}

impl<V> Op<V> for MatMulOp
where
    V: MatrixValue,
{
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![matmul_abstract(&inputs[0], &inputs[1], "matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<V> BatchOp<V> for MatMulOp
where
    V: MatrixValue,
{
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![BatchedValue::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left.matmul(right))
                .collect(),
        )])
    }
}

/// Primitive representing matrix transposition.
#[derive(Clone, Default)]
pub struct MatrixTransposeOp;

impl Debug for MatrixTransposeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MatrixTranspose")
    }
}

impl Display for MatrixTransposeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "matrix_transpose")
    }
}

impl<V> Op<V> for MatrixTransposeOp
where
    V: MatrixValue,
{
    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0])])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V> BatchOp<V> for MatrixTransposeOp
where
    V: MatrixValue,
{
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}

#[derive(Clone)]
struct LeftMatMulOp<V>
where
    V: MatrixValue,
{
    factor: V,
}

impl<V> LeftMatMulOp<V>
where
    V: MatrixValue,
{
    #[inline]
    fn new(factor: V) -> Self {
        Self { factor }
    }
}

impl<V> Debug for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LeftMatMul")
    }
}

impl<V> Display for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "left_matmul")
    }
}

impl<V> Op<V> for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![matmul_abstract(&self.factor.abstract_value(), &inputs[0], "left_matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }
}

impl<V> LinearOp<V> for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(
            Arc::new(LeftMatMulOp::new(self.factor.clone().transpose_matrix())),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }
}

#[derive(Clone)]
struct RightMatMulOp<V>
where
    V: MatrixValue,
{
    factor: V,
}

impl<V> RightMatMulOp<V>
where
    V: MatrixValue,
{
    #[inline]
    fn new(factor: V) -> Self {
        Self { factor }
    }
}

impl<V> Debug for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RightMatMul")
    }
}

impl<V> Display for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "right_matmul")
    }
}

impl<V> Op<V> for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn name(&self) -> &'static str {
        "right_matmul"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![matmul_abstract(&inputs[0], &self.factor.abstract_value(), "right_matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().matmul(self.factor.clone())])
    }
}

impl<V> LinearOp<V> for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(
            Arc::new(RightMatMulOp::new(self.factor.clone().transpose_matrix())),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }
}

#[derive(Clone, Default)]
struct LinearMatrixTransposeOp;

impl Debug for LinearMatrixTransposeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LinearMatrixTranspose")
    }
}

impl Display for LinearMatrixTransposeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "linear_matrix_transpose")
    }
}

impl<V> Op<V> for LinearMatrixTransposeOp
where
    V: MatrixValue,
{
    fn name(&self) -> &'static str {
        "linear_matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0])])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V> LinearOp<V> for LinearMatrixTransposeOp
where
    V: MatrixValue,
{
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(Arc::new(LinearMatrixTransposeOp), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
    }
}

impl<V, T> MatrixOps for JvpTracer<V, T>
where
    V: MatrixValue,
    T: MatrixTangentSpace<V>,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        JvpTracer {
            primal: self.primal.clone().matmul(rhs.primal.clone()),
            tangent: T::add(T::matmul_right(self.tangent, rhs.primal), T::matmul_left(self.primal, rhs.tangent)),
        }
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        JvpTracer { primal: self.primal.transpose_matrix(), tangent: T::transpose_matrix(self.tangent) }
    }
}

impl<V> MatrixOps for JitTracer<V>
where
    V: MatrixValue,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self.binary(rhs, Arc::new(MatMulOp), MatrixOps::matmul)
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        self.unary(Arc::new(MatrixTransposeOp), MatrixOps::transpose_matrix)
    }
}

impl<V> MatrixOps for BatchedValue<V>
where
    V: MatrixValue,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        single_batch_output(MatMulOp.batch(&[self, rhs]).expect("batched matmul rule should succeed"), "matmul")
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        single_batch_output(
            MatrixTransposeOp.batch(&[self]).expect("batched transpose rule should succeed"),
            "matrix_transpose",
        )
    }
}

impl<V> MatrixTangentSpace<V> for LinearTerm<V>
where
    V: MatrixValue + FloatExt + ZeroLike,
{
    #[inline]
    fn matmul_left(factor: V, tangent: Self) -> Self {
        tangent.apply_linear_op(Arc::new(LeftMatMulOp::new(factor)))
    }

    #[inline]
    fn matmul_right(tangent: Self, factor: V) -> Self {
        tangent.apply_linear_op(Arc::new(RightMatMulOp::new(factor)))
    }

    #[inline]
    fn transpose_matrix(value: Self) -> Self {
        value.apply_linear_op(Arc::new(LinearMatrixTransposeOp))
    }
}

#[cfg(any(feature = "ndarray", test))]
mod ndarray_support {
    use ndarray::Array2;

    use super::{MatrixAbstract, MatrixOps};
    use crate::{
        parameters::Parameter,
        tracing_v2::{CoordinateValue, FloatExt, OneLike, ScalarAbstract, TraceLeaf, ZeroLike},
    };

    impl Parameter for Array2<f32> {}
    impl Parameter for Array2<f64> {}

    impl FloatExt for Array2<f32> {
        #[inline]
        fn sin(self) -> Self {
            self.mapv(f32::sin)
        }

        #[inline]
        fn cos(self) -> Self {
            self.mapv(f32::cos)
        }
    }

    impl FloatExt for Array2<f64> {
        #[inline]
        fn sin(self) -> Self {
            self.mapv(f64::sin)
        }

        #[inline]
        fn cos(self) -> Self {
            self.mapv(f64::cos)
        }
    }

    impl TraceLeaf for Array2<f32> {
        type Abstract = MatrixAbstract;

        #[inline]
        fn abstract_value(&self) -> Self::Abstract {
            MatrixAbstract { scalar: ScalarAbstract::F32, rows: self.nrows(), cols: self.ncols() }
        }
    }

    impl TraceLeaf for Array2<f64> {
        type Abstract = MatrixAbstract;

        #[inline]
        fn abstract_value(&self) -> Self::Abstract {
            MatrixAbstract { scalar: ScalarAbstract::F64, rows: self.nrows(), cols: self.ncols() }
        }
    }

    impl ZeroLike for Array2<f32> {
        #[inline]
        fn zero_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 0.0)
        }
    }

    impl ZeroLike for Array2<f64> {
        #[inline]
        fn zero_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 0.0)
        }
    }

    impl OneLike for Array2<f32> {
        #[inline]
        fn one_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 1.0)
        }
    }

    impl OneLike for Array2<f64> {
        #[inline]
        fn one_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 1.0)
        }
    }

    impl CoordinateValue for Array2<f32> {
        type Coordinate = f32;

        #[inline]
        fn coordinate_count(&self) -> usize {
            self.len()
        }

        fn coordinate_basis(&self) -> Vec<Self> {
            let mut basis = Vec::with_capacity(self.len());
            for row in 0..self.nrows() {
                for col in 0..self.ncols() {
                    let mut tangent = Array2::from_elem(self.raw_dim(), 0.0);
                    tangent[(row, col)] = 1.0;
                    basis.push(tangent);
                }
            }
            basis
        }

        #[inline]
        fn coordinates(&self) -> Vec<Self::Coordinate> {
            self.iter().copied().collect::<Vec<_>>()
        }
    }

    impl CoordinateValue for Array2<f64> {
        type Coordinate = f64;

        #[inline]
        fn coordinate_count(&self) -> usize {
            self.len()
        }

        fn coordinate_basis(&self) -> Vec<Self> {
            let mut basis = Vec::with_capacity(self.len());
            for row in 0..self.nrows() {
                for col in 0..self.ncols() {
                    let mut tangent = Array2::from_elem(self.raw_dim(), 0.0);
                    tangent[(row, col)] = 1.0;
                    basis.push(tangent);
                }
            }
            basis
        }

        #[inline]
        fn coordinates(&self) -> Vec<Self::Coordinate> {
            self.iter().copied().collect::<Vec<_>>()
        }
    }

    impl MatrixOps for Array2<f32> {
        #[inline]
        fn matmul(self, rhs: Self) -> Self {
            self.dot(&rhs)
        }

        #[inline]
        fn transpose_matrix(self) -> Self {
            self.reversed_axes()
        }
    }

    impl MatrixOps for Array2<f64> {
        #[inline]
        fn matmul(self, rhs: Self) -> Self {
            self.dot(&rhs)
        }

        #[inline]
        fn transpose_matrix(self) -> Self {
            self.reversed_axes()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2};

    use super::{MatrixOps, MatrixValue};
    use crate::{
        parameters::{Parameterized, ParameterizedFamily},
        tracing_v2::{FloatExt, Linearized, OneLike, ZeroLike, grad, hessian, jit, jvp, test_support, vjp, vmap},
    };

    fn approx_eq_matrix(left: &Array2<f64>, right: &Array2<f64>) {
        assert_eq!(left.shape(), right.shape(), "matrix shapes differ: {:?} vs {:?}", left.shape(), right.shape());
        for (left_value, right_value) in left.iter().zip(right.iter()) {
            let delta = (left_value - right_value).abs();
            assert!(delta <= 1e-9, "expected {left_value} ~= {right_value}; absolute error {delta} exceeded tolerance");
        }
    }

    fn bilinear_matmul<Context, M>(_: &mut Context, inputs: (M, M)) -> M
    where
        M: MatrixOps,
    {
        inputs.0.matmul(inputs.1)
    }

    fn three_matmul_sine<Context, M>(_: &mut Context, inputs: (M, M, M, M)) -> M
    where
        M: MatrixOps + FloatExt,
    {
        let (x, a, b, c) = inputs;
        x.matmul(a).sin().matmul(b).matmul(c)
    }

    fn first_matrix_gradient<Context, V>(context: &mut Context, inputs: (V, V, V, V)) -> V
    where
        V: MatrixValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>, To = Linearized<V>>,
    {
        let (x_bar, _, _, _) = grad(context, three_matmul_sine, inputs).expect("matrix gradient should succeed");
        x_bar
    }

    fn full_matrix_gradient<Context, V>(context: &mut Context, inputs: (V, V, V, V)) -> (V, V, V, V)
    where
        V: MatrixValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>, To = Linearized<V>>,
    {
        grad(context, three_matmul_sine, inputs).expect("matrix gradient should succeed")
    }

    #[test]
    fn forward_mode_linearizes_matrix_multiplication() {
        let mut context = ();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let da = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let db = arr2(&[[1.0, 1.0], [1.0, 1.0]]);

        let (primal, tangent) =
            jvp(&mut context, bilinear_matmul, (a.clone(), b.clone()), (da.clone(), db.clone())).unwrap();

        approx_eq_matrix(&primal, &a.clone().matmul(b.clone()));
        approx_eq_matrix(&tangent, &(da.matmul(b.clone()) + a.matmul(db)));
        test_support::assert_matrix_pushforward_rendering();
    }

    #[test]
    fn reverse_mode_transposes_left_and_right_matrix_actions() {
        let mut context = ();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[2.0, 1.0], [0.0, 3.0]]);
        let cotangent = arr2(&[[1.0, -1.0], [2.0, 0.5]]);

        let (output, pullback) = vjp(&mut context, bilinear_matmul, (a.clone(), b.clone())).unwrap();
        approx_eq_matrix(&output, &a.clone().matmul(b.clone()));

        let (a_bar, b_bar) = pullback.call(cotangent.clone()).unwrap();
        approx_eq_matrix(&a_bar, &cotangent.clone().matmul(b.clone().transpose_matrix()));
        approx_eq_matrix(&b_bar, &a.transpose_matrix().matmul(cotangent));
        test_support::assert_matrix_pullback_rendering();
    }

    #[test]
    fn hessian_works_for_three_matrix_multiplications_with_sine_inside() {
        let mut context = ();
        let x = arr2(&[[0.7f64]]);
        let a = arr2(&[[2.0f64]]);
        let b = arr2(&[[-1.5f64]]);
        let c = arr2(&[[4.0f64]]);

        let (_, hessian_times_ones) = jvp(
            &mut context,
            first_matrix_gradient,
            (x.clone(), a.clone(), b.clone(), c.clone()),
            (x.one_like(), a.zero_like(), b.zero_like(), c.zero_like()),
        )
        .expect("matrix Hessian should succeed");

        let expected = arr2(&[[24.0 * (2.0f64 * 0.7f64).sin()]]);
        approx_eq_matrix(&hessian_times_ones, &expected);
        test_support::assert_matrix_hessian_style_jit_rendering();
    }

    #[test]
    fn dense_hessian_materializes_for_three_matrix_multiplications_with_sine_inside() {
        let mut context = ();
        let x = arr2(&[[0.7f64]]);
        let a = arr2(&[[2.0f64]]);
        let b = arr2(&[[-1.5f64]]);
        let c = arr2(&[[4.0f64]]);
        let xa = 2.0f64 * 0.7f64;
        let cosine = xa.cos();
        let sine = xa.sin();

        let dense_hessian = hessian(&mut context, full_matrix_gradient, (x, a, b, c)).unwrap().to_array2();
        let expected = arr2(&[
            [24.0 * sine, -6.0 * (cosine - 1.4 * sine), 8.0 * cosine, -3.0 * cosine],
            [-6.0 * (cosine - 1.4 * sine), 2.94 * sine, 2.8 * cosine, -1.05 * cosine],
            [8.0 * cosine, 2.8 * cosine, 0.0, sine],
            [-3.0 * cosine, -1.05 * cosine, sine, 0.0],
        ]);

        approx_eq_matrix(&dense_hessian, &expected);
        test_support::assert_matrix_hessian_style_jit_rendering();
    }

    #[test]
    fn jit_stages_matrix_multiplication_programs() {
        let mut context = ();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);

        let (output, compiled) = jit(&mut context, bilinear_matmul, (a.clone(), b.clone())).unwrap();
        approx_eq_matrix(&output, &a.matmul(b));
        let replay_left = arr2(&[[0.0, 1.0], [2.0, 3.0]]);
        let replay_right = arr2(&[[1.0, 4.0], [2.0, 5.0]]);
        let replayed = compiled.call(&mut context, (replay_left.clone(), replay_right.clone())).unwrap();
        approx_eq_matrix(&replayed, &replay_left.matmul(replay_right));
        test_support::assert_matrix_jit_rendering();
    }

    #[test]
    fn batching_vectorizes_matrix_multiplication_lane_wise() {
        let mut context = ();
        let inputs = vec![
            (arr2(&[[1.0, 2.0], [0.0, 1.0]]), arr2(&[[2.0, 0.0], [1.0, 2.0]])),
            (arr2(&[[0.0, 1.0], [3.0, 4.0]]), arr2(&[[1.0, 1.0], [0.0, 2.0]])),
        ];

        let outputs = vmap(&mut context, bilinear_matmul, inputs.clone()).unwrap();
        assert_eq!(outputs.len(), inputs.len());
        for (output, (left, right)) in outputs.into_iter().zip(inputs) {
            approx_eq_matrix(&output, &left.matmul(right));
        }
        test_support::assert_matrix_jit_rendering();
    }
}
