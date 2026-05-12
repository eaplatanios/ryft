use std::fmt::Display;

use half::{bf16, f16};

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::domains::{Tracer, TracingDomain};
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
use crate::types::{ArrayType, Size, Type, TypeError, Typed};

use super::matrix::{MatrixValue, transpose_abstract};

/// Trait that represents [`Operation`] carrier types that support/include [`MatrixTransposeOperation`]. Backend-owned
/// closed [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this
/// trait so that generic transform code can stage [`MatrixTransposeOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsMatrixTranspose<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the matrix transposition [`Operation`].
    fn matrix_transpose_operation() -> Self;
}

/// Value-level matrix transposition capability.
///
/// [`MatrixTranspose`] is the receiver-style entry point for staging or executing [`MatrixTransposeOperation`].
pub trait MatrixTranspose: Sized {
    /// Computes the rank-2 matrix transpose of `self`.
    fn transpose_matrix(self) -> Self;
}

macro_rules! impl_matrix_transpose_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl MatrixTranspose for $ty {
                #[inline]
                fn transpose_matrix(self) -> Self {
                    self
                }
            }
        )*
    };
}

impl_matrix_transpose_for_scalar!(bf16, f16, f32, f64);

impl<'domain, D> MatrixTranspose for Tracer<'domain, D>
where
    D: TracingDomain<Type = ArrayType>,
    D::OperationCarrier: SupportsMatrixTranspose<ArrayType, D::Value>,
{
    #[inline]
    fn transpose_matrix(self) -> Self {
        if matrix_transpose_is_identity_type(&self.r#type()) {
            return self;
        }
        self.unary(D::OperationCarrier::matrix_transpose_operation())
    }
}

/// Symbolic-zero-aware tangent matrix transpose. `Zero[m, n].transpose_matrix() -> Zero[n, m]`,
/// rewriting the carried type's shape so downstream consumers see the post-transpose dimensions.
impl<V> MatrixTranspose for crate::differentiation::Tangent<ArrayType, V>
where
    V: crate::tracing::Traceable<ArrayType> + MatrixTranspose,
{
    fn transpose_matrix(self) -> Self {
        match self {
            Self::Zero(mut r#type) => {
                if !matrix_transpose_is_identity_type(&r#type) {
                    if let [first, second] = r#type.shape.dimensions.as_mut_slice() {
                        std::mem::swap(first, second);
                    }
                }
                Self::Zero(r#type)
            }
            Self::Value(value) => Self::Value(value.transpose_matrix()),
        }
    }
}

fn matrix_transpose_is_identity_type(r#type: &ArrayType) -> bool {
    matches!(r#type.shape.dimensions.as_slice(), [Size::Static(1), Size::Static(1)])
}

/// Primitive representing matrix transposition.
///
/// [`MatrixTransposeOperation`] is stored directly in traced programs whenever a matrix leaf is
/// transposed symbolically.
#[derive(Clone, Debug, Default)]
pub struct MatrixTransposeOperation;

impl Display for MatrixTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for MatrixTransposeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![transpose_abstract(&input_types[0], "matrix_transpose")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatrixTransposeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V, O> LinearOperation<ArrayType, V, O> for MatrixTransposeOperation
where
    V: MatrixValue,
    O: Clone + Operation<ArrayType> + SupportsMatrixTranspose<ArrayType, V>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().transpose_matrix())]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D> DifferentiableOperation<D> for MatrixTransposeOperation
where
    D: DifferentiableDomain<Type = ArrayType>,
    D::Value: MatrixValue + Differentiable<ArrayType>,
    D::LinearOperationCarrier: SupportsMatrixTranspose<ArrayType, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let primal = inputs[0].primal.clone().transpose_matrix();
        let tangent = inputs[0].tangent.clone().transpose_matrix();
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
