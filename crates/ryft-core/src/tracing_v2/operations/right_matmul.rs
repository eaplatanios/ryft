use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::engines::Tracer;
use crate::tracing::{AtomId, OperationFormatter, Traceable, TracingError, Value};
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::operations::constants::ZeroLike;
use crate::tracing_v2::{DifferentiableEngine, DifferentiableTracingEngine};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::matrix::{MatrixOps, MatrixValue, matmul_abstract};
use super::primitive::LinearArrayOperation;
use super::{DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, SupportsAdd};

/// Hidden carrier capability for staging the right matrix-multiplication primitive.
#[doc(hidden)]
pub trait SupportsRightMatMul<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the right matrix-multiplication primitive
    /// with a captured factor.
    fn right_matmul_operation(factor: V) -> Self;
}

/// Linear map `tangent -> tangent @ factor`.
///
/// [`RightMatMulOperation`] is the right-acting sibling of [`super::LeftMatMulOperation`].
#[derive(Clone)]
pub struct RightMatMulOperation<V: MatrixValue> {
    /// Matrix factor multiplied on the right of every input.
    factor: V,
}

impl<V: MatrixValue> RightMatMulOperation<V> {
    /// Creates one right multiplication op capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }

    /// Returns the captured matrix factor.
    #[inline]
    pub fn factor(&self) -> &V {
        &self.factor
    }
}

/// Validates abstract inputs using the factor's abstract type without needing a concrete instance.
///
/// Backend carriers use this helper when they need the metadata rule for a captured right-matmul
/// operation without first constructing a concrete [`RightMatMulOperation`].
pub fn right_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
    if inputs.len() != 1 {
        return Err(TypeError { message: format!("right_matmul expected 1 input type but got {}", inputs.len()) });
    }
    Ok(vec![matmul_abstract(&inputs[0], factor_type, "right_matmul")?])
}

impl<V: MatrixValue> Debug for RightMatMulOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "RightMatMul")
    }
}

impl<V: MatrixValue> Display for RightMatMulOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "right_matmul")
    }
}

impl<V: MatrixValue> Operation<ArrayType> for RightMatMulOperation<V> {
    fn name(&self) -> &'static str {
        "right_matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        right_matmul_abstract_eval(&<V as Typed<ArrayType>>::r#type(&self.factor), input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", self.factor()))
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for RightMatMulOperation<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().matmul(self.factor.clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V, LinearArrayOperation<V>> for RightMatMulOperation<V> {
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<'_, ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .apply_operation(
                        &[atom],
                        LinearArrayOperation::RightMatMul { factor: self.factor.clone().transpose_matrix() },
                        1,
                    )?
                    .into_iter()
                    .next()
                    .expect("right matmul should produce one cotangent contribution"),
            )]),
            None => Ok(vec![None]),
        }
    }
}

impl<V, E> DifferentiableOperation<E> for RightMatMulOperation<V>
where
    V: MatrixValue + ZeroLike + Differentiable<ArrayType, Tangent = V>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    E::LinearOperation: SupportsRightMatMul<ArrayType, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let primal = inputs[0].primal.clone().matmul(self.factor().clone());
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsRightMatMul<ArrayType, V>>::right_matmul_operation(
                    self.factor().clone(),
                ),
                1,
            )?
            .into_iter()
            .next()
            .expect("right matmul jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

/// JVP rule for `RightMatMulOperation` under
/// [`TracingContext`](crate::tracing::engines::TracingContext).
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for RightMatMulOperation<V>
where
    V: MatrixValue + Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V> + ?Sized,
    EInner::Operation: SupportsAdd<ArrayType, V>,
    EInner::LinearOperation<'engine>: SupportsRightMatMul<ArrayType, Tracer<'engine, EInner>>,
    Tracer<'engine, EInner>: MatrixOps,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let factor_tracer = context.engine.constant(self.factor().clone());
        let primal = inputs[0].primal.clone().matmul(factor_tracer.clone());
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <EInner::LinearOperation<'engine> as SupportsRightMatMul<
                    ArrayType,
                    Tracer<'engine, EInner>,
                >>::right_matmul_operation(factor_tracer),
                1,
            )?
            .into_iter()
            .next()
            .expect("right matmul jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
