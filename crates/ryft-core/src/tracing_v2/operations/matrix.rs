use std::{
    collections::BTreeSet,
    ops::{Add, Mul, Neg},
};

use crate::{
    sharding::{Sharding, ShardingDimension},
    tracing::Traceable,
    tracing_v2::{
        forward::{JvpTracer, TangentSpace},
        jit::Tracer,
        linear::LinearTerm,
        operations::constants::ZeroLike,
    },
    types::{ArrayType, DataType, Shape, Size, TypeError, Typed},
};

use super::{
    LinearAddOperation, LinearLeftMatMulOperation, LinearMatrixTransposeOperation, LinearNegOperation,
    LinearRightMatMulOperation, LinearScaleOperation, MatMulTracingOperation, MatrixTransposeTracingOperation,
    Operation,
};

/// Matrix operations required by the tracing prototype.
///
/// This is the value-level capability trait that generic user code and primitive replay rely on
/// when they want to treat a leaf as a matrix.
pub trait MatrixOps: Sized {
    /// Matrix multiplication.
    fn matmul(self, rhs: Self) -> Self;

    /// Matrix transpose.
    fn transpose_matrix(self) -> Self;
}

/// Convenience trait for traceable matrix leaves.
///
/// Matrix values use [`ArrayType`] as their staged descriptor. The matrix-specific primitives in
/// this module expect those array types to describe rank-2 matrices with static dimensions and
/// floating-point element types.
pub trait MatrixValue: Traceable<ArrayType> + MatrixOps {}

impl<T: Traceable<ArrayType> + MatrixOps> MatrixValue for T {}

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
///
/// This extends [`TangentSpace`](crate::tracing_v2::TangentSpace) with the additional linear
/// actions needed by matrix-valued JVP and transpose rules.
pub trait MatrixTangentSpace<V: MatrixValue>: TangentSpace<ArrayType, V> {
    /// Applies the linear map `tangent -> factor @ tangent`.
    fn matmul_left(factor: V, tangent: Self) -> Self;

    /// Applies the linear map `tangent -> tangent @ factor`.
    fn matmul_right(tangent: Self, factor: V) -> Self;

    /// Transposes a tangent value.
    fn transpose_matrix(value: Self) -> Self;
}

impl<V: MatrixValue + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + ZeroLike> MatrixTangentSpace<V> for V {
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

fn matrix_array_type(data_type: DataType, rows: usize, cols: usize, sharding: Option<Sharding>) -> ArrayType {
    ArrayType::new(data_type, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None, sharding)
        .expect("matrix abstract evaluation should preserve rank-2 sharding")
}

fn matrix_parts(r#type: &ArrayType, op: &'static str) -> Result<(DataType, usize, usize), TypeError> {
    if !matches!(r#type.data_type, DataType::F32 | DataType::F64) || r#type.rank() != 2 {
        return Err(TypeError { message: format!("{op} expects rank-2 f32 or f64 matrix inputs") });
    }

    let Size::Static(rows) = r#type.dimension(0) else {
        return Err(TypeError { message: format!("{op} requires statically shaped matrix inputs") });
    };
    let Size::Static(cols) = r#type.dimension(1) else {
        return Err(TypeError { message: format!("{op} requires statically shaped matrix inputs") });
    };
    Ok((r#type.data_type, rows, cols))
}

fn is_replicated_sharding(sharding: &Sharding) -> bool {
    sharding.dimensions.iter().all(|dimension| matches!(dimension, ShardingDimension::Replicated))
}

fn merge_unique_axes(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

fn transpose_array_sharding(input: &ArrayType) -> Option<Sharding> {
    let sharding = input.sharding.clone()?;
    if sharding.rank() != 2 {
        return None;
    }
    Sharding::with_manual_axes(
        sharding.mesh.clone(),
        vec![sharding.dimensions[1].clone(), sharding.dimensions[0].clone()],
        sharding.unreduced_axes.clone(),
        sharding.reduced_manual_axes.clone(),
        sharding.varying_manual_axes.clone(),
    )
    .map(|sharding| sharding.without_auto_axes())
    .ok()
}

fn matmul_array_sharding(lhs: &ArrayType, rhs: &ArrayType) -> Option<Sharding> {
    let left = lhs.sharding.clone()?;
    let right = rhs.sharding.clone()?;
    if left == right && is_replicated_sharding(&left) {
        return Some(left);
    }
    if left.rank() != 2 || right.rank() != 2 {
        return None;
    }
    if !matches!(left.dimensions[1], ShardingDimension::Replicated)
        || !matches!(right.dimensions[0], ShardingDimension::Replicated)
    {
        return None;
    }
    Sharding::with_manual_axes(
        left.mesh.clone(),
        vec![left.dimensions[0].clone(), right.dimensions[1].clone()],
        merge_unique_axes(&left.unreduced_axes, &right.unreduced_axes),
        merge_unique_axes(&left.reduced_manual_axes, &right.reduced_manual_axes),
        merge_unique_axes(&left.varying_manual_axes, &right.varying_manual_axes),
    )
    .map(|sharding| sharding.without_auto_axes())
    .ok()
}

/// Computes the abstract output type of one matrix multiplication.
///
/// This is the shared shape-and-sharding rule used by matrix multiplication across tracing,
/// simplification, and backend wrappers.
pub fn matmul_abstract(lhs: &ArrayType, rhs: &ArrayType, op: &'static str) -> Result<ArrayType, TypeError> {
    let (lhs_data_type, lhs_rows, lhs_cols) = matrix_parts(lhs, op)?;
    let (rhs_data_type, rhs_rows, rhs_cols) = matrix_parts(rhs, op)?;
    if lhs_data_type != rhs_data_type || lhs_cols != rhs_rows {
        return Err(TypeError { message: format!("{op} input matrix dimensions or element types are incompatible") });
    }
    let sharding = matmul_array_sharding(lhs, rhs);
    Ok(matrix_array_type(lhs_data_type, lhs_rows, rhs_cols, sharding))
}

/// Computes the abstract output type of one matrix transpose.
///
/// This centralizes the matrix-transpose metadata rule so both the core primitive and any backend
/// wrappers agree on how shapes and sharding should propagate.
pub fn transpose_abstract(input: &ArrayType, op: &'static str) -> Result<ArrayType, TypeError> {
    let (data_type, rows, cols) = matrix_parts(input, op)?;
    let sharding = transpose_array_sharding(input);
    Ok(matrix_array_type(data_type, cols, rows, sharding))
}

fn matrix_transpose_is_identity_type(r#type: &ArrayType) -> bool {
    matches!(r#type.shape.dimensions.as_slice(), [Size::Static(1), Size::Static(1)])
}

impl<V: MatrixValue, T: MatrixTangentSpace<V>> MatrixOps for JvpTracer<V, T> {
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        JvpTracer {
            primal: self.primal.clone().matmul(rhs.primal.clone()),
            tangent: T::add(T::matmul_right(self.tangent, rhs.primal), T::matmul_left(self.primal, rhs.tangent)),
        }
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        if matrix_transpose_is_identity_type(&self.primal.r#type()) {
            return self;
        }
        JvpTracer { primal: self.primal.transpose_matrix(), tangent: T::transpose_matrix(self.tangent) }
    }
}

impl<'engine, V: Traceable<ArrayType>, E: crate::tracing_v2::Engine<Type = ArrayType, Value = V> + ?Sized> MatrixOps
    for Tracer<'engine, E>
where
    E::TracingOperation: MatMulTracingOperation<ArrayType, V> + MatrixTransposeTracingOperation<ArrayType, V>,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self.binary(rhs, E::TracingOperation::matmul_op())
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        if matrix_transpose_is_identity_type(&self.r#type()) {
            return self;
        }
        self.unary(E::TracingOperation::matrix_transpose_op())
    }
}

impl<
    V: MatrixValue + ZeroLike,
    O: LinearLeftMatMulOperation<ArrayType, V>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + Operation<ArrayType>
        + LinearRightMatMulOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>
        + LinearMatrixTransposeOperation<ArrayType, V>,
> MatrixTangentSpace<V> for LinearTerm<ArrayType, V, O>
{
    #[inline]
    fn matmul_left(factor: V, tangent: Self) -> Self {
        tangent.apply_linear_op(O::linear_left_matmul_op(factor))
    }

    #[inline]
    fn matmul_right(tangent: Self, factor: V) -> Self {
        tangent.apply_linear_op(O::linear_right_matmul_op(factor))
    }

    #[inline]
    fn transpose_matrix(value: Self) -> Self {
        value.apply_linear_op(O::linear_matrix_transpose_op())
    }
}

#[cfg(any(feature = "ndarray", test))]
pub mod ndarray_support {
    use std::borrow::Cow;
    use std::marker::PhantomData;

    use ndarray::Array2;

    use super::{MatrixOps, matrix_array_type};
    use crate::{
        parameters::Parameter,
        tracing::{Traceable, TracingError, Value},
        tracing_v2::{
            CoordinateValue, Cos, LinearPrimitiveOperation, PrimitiveOperation, Sin,
            engine::Engine,
            operations::constants::{OneLike, ZeroLike},
        },
        types::{ArrayType, DataType, TypeError, Typed},
    };

    /// Stateless engine that synthesizes [`Array2`] values from [`ArrayType`] metadata.
    ///
    /// [`Array2Engine<V>`] is a zero-sized type used whenever a matrix pipeline needs an engine
    /// whose [`Type`](Engine::Type) is [`ArrayType`] and whose [`Value`](Engine::Value) is an
    /// [`Array2<V>`]. The engine reads the rank-2 shape off the supplied [`ArrayType`] metadata
    /// and returns a uniformly filled matrix of the requested shape.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Array2Engine<V> {
        /// Phantom marker tying the zero-sized engine to its matrix element type.
        marker: PhantomData<fn() -> V>,
    }

    impl<V> Array2Engine<V> {
        /// Returns a new [`Array2Engine<V>`]. This is a no-op at runtime since the engine is
        /// zero-sized.
        #[inline]
        pub const fn new() -> Self {
            Self { marker: PhantomData }
        }
    }

    fn matrix_extent(r#type: &ArrayType) -> Result<(usize, usize), TracingError> {
        if r#type.rank() != 2 {
            return Err(
                TypeError { message: format!("array2 engine requires rank-2 array type but got {}", r#type) }.into()
            );
        }
        let row_size = r#type.dimension(0);
        let col_size = r#type.dimension(1);
        let rows = row_size.value().ok_or_else(|| TypeError {
            message: format!("array2 engine requires a static row count but got {row_size}"),
        })?;
        let cols = col_size.value().ok_or_else(|| TypeError {
            message: format!("array2 engine requires a static column count but got {col_size}"),
        })?;
        Ok((rows, cols))
    }

    impl Engine for Array2Engine<f32> {
        type Type = ArrayType;
        type Value = Array2<f32>;
        type TracingOperation = PrimitiveOperation<Array2<f32>>;
        type LinearOperation = LinearPrimitiveOperation<Array2<f32>>;

        #[inline]
        fn zero(&self, r#type: &ArrayType) -> Result<Array2<f32>, TracingError> {
            Ok(Array2::from_elem(matrix_extent(r#type)?, 0.0))
        }

        #[inline]
        fn one(&self, r#type: &ArrayType) -> Result<Array2<f32>, TracingError> {
            Ok(Array2::from_elem(matrix_extent(r#type)?, 1.0))
        }
    }

    impl Engine for Array2Engine<f64> {
        type Type = ArrayType;
        type Value = Array2<f64>;
        type TracingOperation = PrimitiveOperation<Array2<f64>>;
        type LinearOperation = LinearPrimitiveOperation<Array2<f64>>;

        #[inline]
        fn zero(&self, r#type: &ArrayType) -> Result<Array2<f64>, TracingError> {
            Ok(Array2::from_elem(matrix_extent(r#type)?, 0.0))
        }

        #[inline]
        fn one(&self, r#type: &ArrayType) -> Result<Array2<f64>, TracingError> {
            Ok(Array2::from_elem(matrix_extent(r#type)?, 1.0))
        }
    }

    impl Parameter for Array2<f32> {}
    impl Parameter for Array2<f64> {}

    impl Sin for Array2<f32> {
        #[inline]
        fn sin(self) -> Self {
            self.mapv(f32::sin)
        }
    }

    impl Cos for Array2<f32> {
        #[inline]
        fn cos(self) -> Self {
            self.mapv(f32::cos)
        }
    }

    impl Sin for Array2<f64> {
        #[inline]
        fn sin(self) -> Self {
            self.mapv(f64::sin)
        }
    }

    impl Cos for Array2<f64> {
        #[inline]
        fn cos(self) -> Self {
            self.mapv(f64::cos)
        }
    }

    impl Typed<ArrayType> for Array2<f32> {
        #[inline]
        fn r#type(&self) -> Cow<'_, ArrayType> {
            Cow::Owned(matrix_array_type(DataType::F32, self.nrows(), self.ncols(), None))
        }
    }

    impl Traceable<ArrayType> for Array2<f32> {
        fn is_zero(&self) -> bool {
            self.iter().all(|&x| x == 0.0)
        }

        fn is_one(&self) -> bool {
            self.iter().all(|&x| x == 1.0)
        }
    }

    impl Value<ArrayType> for Array2<f32> {}

    impl Typed<ArrayType> for Array2<f64> {
        #[inline]
        fn r#type(&self) -> Cow<'_, ArrayType> {
            Cow::Owned(matrix_array_type(DataType::F64, self.nrows(), self.ncols(), None))
        }
    }

    impl Traceable<ArrayType> for Array2<f64> {
        fn is_zero(&self) -> bool {
            self.iter().all(|&x| x == 0.0)
        }

        fn is_one(&self) -> bool {
            self.iter().all(|&x| x == 1.0)
        }
    }

    impl Value<ArrayType> for Array2<f64> {}

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
