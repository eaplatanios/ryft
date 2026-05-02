use std::fmt::Display;

use crate::macros::check_input_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::engines::Tracer;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::operations::constants::ZeroLike;
use crate::tracing_v2::{DifferentiableOperation, DifferentiableTracingEngine, LinearizableEngine};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::SupportsAdd;
use super::matrix::{MatrixOps, MatrixValue, matmul_abstract};
use super::primitive::LinearArrayOperation;

/// Trait that represents [`Operation`] carrier types that support/include [`LeftMatMulOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`LeftMatMulOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsLeftMatMul<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the left matrix multiplication [`Operation`].
    fn left_matmul_operation(factor: V) -> Self;
}

/// Linear map `tangent -> factor @ tangent`.
///
/// [`LeftMatMulOperation`] is the matrix-valued analogue of [`super::ScaleOperation`]: it captures one factor in
/// the op object and applies that factor to every input it is replayed on.
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
    check_input_count!(inputs, 1, TypeError);
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
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V, LinearArrayOperation<V>> for LeftMatMulOperation<V> {
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .stage(
                        LinearArrayOperation::LeftMatMul { factor: self.factor.clone().transpose_matrix() },
                        &[atom],
                    )?
                    .into_iter()
                    .next()
                    .expect("left matmul should produce one cotangent contribution"),
            )]),
            None => Ok(vec![None]),
        }
    }
}

impl<V, E> DifferentiableOperation<E> for LeftMatMulOperation<V>
where
    V: MatrixValue + ZeroLike + Differentiable<ArrayType, Tangent = V>,
    E: LinearizableEngine<Type = ArrayType, Value = V> + ?Sized,
    E::LinearOperationCarrier: SupportsLeftMatMul<ArrayType, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let primal = self.factor.clone().matmul(inputs[0].primal.clone());
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperationCarrier as SupportsLeftMatMul<ArrayType, V>>::left_matmul_operation(
                    self.factor.clone(),
                ),
                1,
            )?
            .into_iter()
            .next()
            .expect("left matmul jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

/// JVP rule for `LeftMatMulOperation` under
/// [`TracingContext`](crate::tracing::engines::TracingContext).
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for LeftMatMulOperation<V>
where
    V: MatrixValue + Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V> + ?Sized,
    EInner::OperationCarrier: SupportsAdd<ArrayType, V>,
    EInner::LinearOperationCarrier<'engine>: SupportsLeftMatMul<ArrayType, Tracer<'engine, EInner>>,
    Tracer<'engine, EInner>: MatrixOps,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let factor_tracer = context.engine.constant(self.factor.clone());
        let primal = factor_tracer.clone().matmul(inputs[0].primal.clone());
        let tangent =
            context
                .apply_operation(
                    &[inputs[0].tangent],
                    <EInner::LinearOperationCarrier<'engine> as SupportsLeftMatMul<
                        ArrayType,
                        Tracer<'engine, EInner>,
                    >>::left_matmul_operation(factor_tracer),
                    1,
                )?
                .into_iter()
                .next()
                .expect("left matmul jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
