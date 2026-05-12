use std::fmt::Display;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::ZeroLike;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::domains::{RuntimeDomain, Tracer, TracingContext, TracingDomain};
use crate::tracing::{ProgramTracingContext, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation, DifferentiableTracingDomain};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::matmul::{MatMul, SupportsMatMul};
use super::matrix::{MatrixOps, MatrixValue, matmul_abstract};

/// Trait that represents [`Operation`] carrier types that support/include [`LeftMatMulOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`LeftMatMulOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsLeftMatMul<T: Type, V: Traceable<T>, F: Traceable<T> = V> {
    /// Constructs the carrier-specific representation of the left matrix multiplication [`Operation`].
    fn left_matmul_operation(factor: F) -> Self;
}

/// Value-level left matrix multiplication by a captured factor.
///
/// [`LeftMatMul`] fills the same role for [`LeftMatMulOperation`] that
/// [`crate::operations::arithmetic::Scale`] fills for scalar scaling: the receiver is the linear input and `factor`
/// is closed over by the staged operation.
pub trait LeftMatMul<Factor = Self>: Sized {
    /// Computes `factor @ self`.
    fn left_matmul(self, factor: Factor) -> Self;
}

impl<'domain, D, F> LeftMatMul<F> for Tracer<'domain, D>
where
    D: TracingDomain<Type = ArrayType>,
    D::OperationCarrier: SupportsLeftMatMul<ArrayType, D::Value, F>,
    F: Traceable<ArrayType>,
{
    #[inline]
    fn left_matmul(self, factor: F) -> Self {
        self.unary(D::OperationCarrier::left_matmul_operation(factor))
    }
}

/// Symbolic-zero-aware tangent left-matrix-multiplication. `Zero.left_matmul(_) -> Zero`. JVP
/// rules use the tangent's `.left_matmul(primal)` to scale a tangent by a captured primal factor;
/// the symbolic-zero variant short-circuits without staging the underlying matmul.
impl<T, V, F> LeftMatMul<F> for crate::differentiation::Tangent<T, V>
where
    T: crate::types::Type,
    V: crate::tracing::Traceable<T> + LeftMatMul<F>,
{
    #[inline]
    fn left_matmul(self, factor: F) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type),
            Self::Value(value) => Self::Value(value.left_matmul(factor)),
        }
    }
}

/// Linear map `tangent -> factor @ tangent`.
///
/// [`LeftMatMulOperation`] is the matrix-valued analogue of
/// [`ScaleOperation`](crate::operations::arithmetic::ScaleOperation): it captures one factor in the op object and
/// applies that factor to every input it is replayed on.
#[derive(Clone, Debug)]
pub struct LeftMatMulOperation<V: MatrixValue> {
    /// Matrix factor multiplied on the left of every input.
    pub factor: V,
}

impl<V: MatrixValue> LeftMatMulOperation<V> {
    /// Creates one left multiplication op capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }
}

/// Validates abstract inputs using the factor's abstract type without needing a concrete instance.
///
/// Backend carriers use this helper when they need the metadata rule for a captured left-matmul
/// operation without first constructing a concrete [`LeftMatMulOperation`].
pub fn left_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
    check_count!("input", inputs, 1, TypeError);
    Ok(vec![matmul_abstract(factor_type, &inputs[0], "left_matmul")?])
}

impl<V: MatrixValue> Display for LeftMatMulOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: MatrixValue> Operation<ArrayType> for LeftMatMulOperation<V> {
    #[inline]
    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        left_matmul_abstract_eval(&<V as Typed<ArrayType>>::r#type(&self.factor), input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", &self.factor))
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for LeftMatMulOperation<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }
}

impl<V, O> LinearOperation<ArrayType, V, O> for LeftMatMulOperation<V>
where
    V: MatrixValue,
    O: Clone + Operation<ArrayType> + SupportsLeftMatMul<ArrayType, V>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, ArrayType, V, O>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                Ok(vec![Cotangent::Staged(cotangent.clone().left_matmul(self.factor.clone().transpose_matrix()))])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<V, D> DifferentiableOperation<D> for LeftMatMulOperation<V>
where
    V: MatrixValue + ZeroLike,
    D: DifferentiableDomain<Type = ArrayType, Value = V>,
    D::LinearOperationCarrier: SupportsLeftMatMul<ArrayType, D::Tangent, V>,
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
        let primal = self.factor.clone().matmul(inputs[0].primal.clone());
        let tangent = inputs[0].tangent.clone().left_matmul(self.factor.clone());
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

/// JVP rule for `LeftMatMulOperation` under
/// [`TracingContext`].
impl<'domain, D, V, O> DifferentiableOperation<TracingContext<'domain, D>> for LeftMatMulOperation<V>
where
    D: DifferentiableTracingDomain<Type = ArrayType, Value = V, OperationCarrier = O> + RuntimeDomain + 'domain,
    V: MatrixValue + Value<ArrayType>,
    O: SupportsAdd<ArrayType, V> + SupportsMatMul<ArrayType, V> + 'domain,
    <TracingContext<'domain, D> as DifferentiableDomain>::LinearOperationCarrier:
        SupportsLeftMatMul<ArrayType, Tracer<'domain, D>>,
    Tracer<'domain, D>: MatrixOps,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, D>>,
        inputs: &[JvpTracer<Tracer<'domain, D>, D::Type, Tracer<'jvp, TracingContext<'domain, D>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'domain, D>, D::Type, Tracer<'jvp, TracingContext<'domain, D>>>>, TracingError>
    where
        TracingContext<'domain, D>: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let factor_tracer = context.domain.constant(self.factor.clone());
        let primal = factor_tracer.clone().matmul(inputs[0].primal.clone());
        let tangent = inputs[0].tangent.clone().left_matmul(factor_tracer);
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
