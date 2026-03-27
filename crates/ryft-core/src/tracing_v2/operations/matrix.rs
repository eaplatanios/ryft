//! Matrix-specific tracing extensions built on top of the core `tracing_v2` primitives.
//!
//! This module keeps the non-commutative details of matrix multiplication localized by introducing explicit left and
//! right linear actions for transposition, while reusing the same JVP, batching, and JIT infrastructure as scalar
//! values.

use std::sync::Arc;

use crate::{
    tracing_v2::{
        FloatExt, TraceError, TraceValue, TransformLeaf, ZeroLike,
        batch::Batch as BatchedValue,
        forward::{JvpTracer, TangentSpace},
        jit::JitTracer,
        linear::LinearTerm,
        ops::BatchOp,
    },
    types::{ArrayType, DataType, Shape, Size, Typed},
};

use super::{LeftMatMulOp, LinearMatrixTransposeOp, MatMulOp, MatrixTransposeOp, RightMatMulOp};

/// Matrix operations required by the tracing prototype.
pub trait MatrixOps: Sized {
    /// Matrix multiplication.
    fn matmul(self, rhs: Self) -> Self;

    /// Matrix transpose.
    fn transpose_matrix(self) -> Self;
}

/// Convenience trait for traceable matrix leaves.
///
/// Matrix values use [`ArrayType`] as their staged descriptor. The matrix-specific primitives in this module expect
/// those array types to describe rank-2 matrices with static dimensions and floating-point element types.
pub trait MatrixValue: TraceValue + MatrixOps {}

impl<T> MatrixValue for T where T: TraceValue + MatrixOps {}

impl MatrixOps for f32 {
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        self
    }
}

impl MatrixOps for f64 {
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        self
    }
}

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

fn matrix_array_type(data_type: DataType, rows: usize, cols: usize) -> ArrayType {
    ArrayType::new(data_type, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None)
}

fn matrix_parts(r#type: &ArrayType, op: &'static str) -> Result<(DataType, usize, usize), TraceError> {
    if !matches!(r#type.data_type, DataType::F32 | DataType::F64) || r#type.rank() != 2 {
        return Err(TraceError::IncompatibleAbstractValues { op });
    }

    let Size::Static(rows) = r#type.dimension(0) else {
        return Err(TraceError::IncompatibleAbstractValues { op });
    };
    let Size::Static(cols) = r#type.dimension(1) else {
        return Err(TraceError::IncompatibleAbstractValues { op });
    };
    Ok((r#type.data_type, rows, cols))
}

/// Computes the abstract output type of one matrix multiplication.
pub(crate) fn matmul_abstract(lhs: &ArrayType, rhs: &ArrayType, op: &'static str) -> Result<ArrayType, TraceError> {
    let (lhs_data_type, lhs_rows, lhs_cols) = matrix_parts(lhs, op)?;
    let (rhs_data_type, rhs_rows, rhs_cols) = matrix_parts(rhs, op)?;
    if lhs_data_type != rhs_data_type || lhs_cols != rhs_rows {
        return Err(TraceError::IncompatibleAbstractValues { op });
    }
    Ok(matrix_array_type(lhs_data_type, lhs_rows, rhs_cols))
}

/// Computes the abstract output type of one matrix transpose.
pub(crate) fn transpose_abstract(input: &ArrayType, op: &'static str) -> Result<ArrayType, TraceError> {
    let (data_type, rows, cols) = matrix_parts(input, op)?;
    Ok(matrix_array_type(data_type, cols, rows))
}

fn matrix_transpose_is_identity_type(r#type: &ArrayType) -> bool {
    matches!(r#type.shape.dimensions.as_slice(), [Size::Static(1), Size::Static(1)])
}

fn single_batch_output<V>(mut outputs: Vec<BatchedValue<V>>, op: &'static str) -> BatchedValue<V>
where
    V: MatrixValue,
{
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single batched output");
    outputs.pop().expect("single-output matrix primitive should return one batched output")
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
        if matrix_transpose_is_identity_type(&self.primal.tpe()) {
            return self;
        }
        JvpTracer { primal: self.primal.transpose_matrix(), tangent: T::transpose_matrix(self.tangent) }
    }
}

impl<V> MatrixOps for JitTracer<V>
where
    V: TransformLeaf,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self.binary(rhs, Arc::new(MatMulOp), MatrixOps::matmul)
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        if matrix_transpose_is_identity_type(&self.tpe()) {
            return self;
        }
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
        if self.lanes().first().map(|lane| matrix_transpose_is_identity_type(&lane.tpe())).unwrap_or(false) {
            return self;
        }
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

    use super::{MatrixOps, matrix_array_type};
    use crate::{
        parameters::Parameter,
        tracing_v2::{CoordinateValue, FloatExt, OneLike, TraceValue, ZeroLike},
        types::{ArrayType, DataType, Typed},
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

    impl Typed<ArrayType> for Array2<f32> {
        #[inline]
        fn tpe(&self) -> ArrayType {
            matrix_array_type(DataType::F32, self.nrows(), self.ncols())
        }
    }

    impl TraceValue for Array2<f32> {}

    impl Typed<ArrayType> for Array2<f64> {
        #[inline]
        fn tpe(&self) -> ArrayType {
            matrix_array_type(DataType::F64, self.nrows(), self.ncols())
        }
    }

    impl TraceValue for Array2<f64> {}

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
            if self.nrows() == 1 && self.ncols() == 1 {
                return self;
            }
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
            if self.nrows() == 1 && self.ncols() == 1 {
                return self;
            }
            self.reversed_axes()
        }
    }
}
