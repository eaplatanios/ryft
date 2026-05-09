use std::fmt::Display;

use crate::differentiation::LinearOperation;
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::ZeroLike;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::engines::{Tracer, TracingContext, TracingEngine};
use crate::tracing::{Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::matmul::MatMul;
use super::matrix::{MatrixOps, MatrixValue, matmul_abstract};
use super::primitive::LinearArrayOperation;

/// Trait that represents [`Operation`] carrier types that support/include [`RightMatMulOperation`]. Backend-owned
/// closed [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this
/// trait so that generic transform code can stage [`RightMatMulOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsRightMatMul<T: Type, V: Traceable<T>, F: Traceable<T> = V> {
    /// Constructs the carrier-specific representation of the right matrix multiplication [`Operation`].
    fn right_matmul_operation(factor: F) -> Self;
}

/// Value-level right matrix multiplication by a captured factor.
///
/// [`RightMatMul`] is the receiver-style entry point for staging or executing [`RightMatMulOperation`]. The receiver is
/// the linear input and `factor` is closed over by the staged operation.
pub trait RightMatMul<Factor = Self>: Sized {
    /// Computes `self @ factor`.
    fn right_matmul(self, factor: Factor) -> Self;
}

impl<'engine, E, F> RightMatMul<F> for Tracer<'engine, E>
where
    E: TracingEngine<Type = ArrayType>,
    F: Traceable<ArrayType>,
    E::OperationCarrier: SupportsRightMatMul<ArrayType, E::Value, F>,
{
    #[inline]
    fn right_matmul(self, factor: F) -> Self {
        self.unary(E::OperationCarrier::right_matmul_operation(factor))
    }
}

/// Linear map `tangent -> tangent @ factor`.
///
/// [`RightMatMulOperation`] is the right-acting sibling of [`super::LeftMatMulOperation`].
#[derive(Clone, Debug)]
pub struct RightMatMulOperation<V: MatrixValue> {
    /// Matrix factor multiplied on the right of every input.
    pub factor: V,
}

impl<V: MatrixValue> RightMatMulOperation<V> {
    /// Creates one right multiplication op capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }
}

/// Validates abstract inputs using the factor's abstract type without needing a concrete instance.
///
/// Backend carriers use this helper when they need the metadata rule for a captured right-matmul
/// operation without first constructing a concrete [`RightMatMulOperation`].
pub fn right_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
    check_count!("input", inputs, 1, TypeError);
    Ok(vec![matmul_abstract(&inputs[0], factor_type, "right_matmul")?])
}

impl<V: MatrixValue> Display for RightMatMulOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: MatrixValue> Operation<ArrayType> for RightMatMulOperation<V> {
    #[inline]
    fn name(&self) -> &'static str {
        "right_matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        right_matmul_abstract_eval(&<V as Typed<ArrayType>>::r#type(&self.factor), input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", &self.factor))
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for RightMatMulOperation<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().matmul(self.factor.clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>> for RightMatMulOperation<V> {
    fn transpose(
        &self,
        context: &mut crate::differentiation::TranspositionContext<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let cotangent_outputs = context.stage(
                    LinearArrayOperation::RightMatMul { factor: self.factor.clone().transpose_matrix() },
                    &[atom],
                )?;
                check_count!("output", cotangent_outputs, 1, TracingError);
                Ok(vec![Some(cotangent_outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

impl<V, E> DifferentiableOperation<E> for RightMatMulOperation<V>
where
    V: MatrixValue + ZeroLike + Differentiable<ArrayType>,
    E: DifferentiableEngine<Type = ArrayType, Value = V>,
    E::LinearOperationCarrier: SupportsRightMatMul<ArrayType, E::Tangent, V>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>],
    ) -> Result<Vec<JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let primal = inputs[0].primal.clone().matmul(self.factor.clone());
        let tangent = inputs[0].tangent.clone().right_matmul(self.factor.clone());
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

/// JVP rule for `RightMatMulOperation` under
/// [`TracingContext`](crate::tracing::engines::TracingContext).
impl<'engine, V, EInner> DifferentiableOperation<TracingContext<'engine, EInner>> for RightMatMulOperation<V>
where
    V: MatrixValue + Value<ArrayType> + Differentiable<ArrayType>,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V>,
    EInner::OperationCarrier: SupportsAdd<ArrayType, V>,
    EInner::LinearOperationCarrier<'engine>: SupportsRightMatMul<ArrayType, Tracer<'engine, EInner>>,
    Tracer<'engine, EInner>: MatrixOps,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, Tracer<'jvp, TracingContext<'engine, EInner>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, Tracer<'jvp, TracingContext<'engine, EInner>>>>, TracingError>
    {
        check_count!("input", inputs, 1, TracingError);
        let factor_tracer = context.engine.constant(self.factor.clone());
        let primal = inputs[0].primal.clone().matmul(factor_tracer.clone());
        let tangent = inputs[0].tangent.clone().right_matmul(factor_tracer);
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
