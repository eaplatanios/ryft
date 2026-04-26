use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::{AtomId, OperationFormatter, Traceable, TracingError, Value};
use crate::tracing_v2::{
    engines::{DifferentiableEngine, Engine},
    forward::{Differentiable, JvpContext, JvpTracer},
    jit::Tracer,
    operations::constants::ZeroLike,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, TracedLinearizationCarrier,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
    primitive::LinearPrimitiveOperation,
};

/// Hidden carrier capability for staging the left matrix-multiplication primitive.
#[doc(hidden)]
pub trait SupportsLeftMatMul<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the left matrix-multiplication primitive
    /// with a captured factor.
    fn left_matmul_operation(factor: V) -> Self;
}

/// Linear map `tangent -> factor @ tangent`.
///
/// [`LeftMatMulOperation`] is the matrix-valued analogue of [`super::ScaleOperation`]: it captures one factor in
/// the op object and applies that factor to every input it is replayed on.
#[derive(Clone)]
pub struct LeftMatMulOperation<V: MatrixValue> {
    /// Matrix factor multiplied on the left of every input.
    factor: V,
}

impl<V: MatrixValue> LeftMatMulOperation<V> {
    /// Creates one left multiplication op capturing the provided factor.
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
/// Backend carriers use this helper when they need the metadata rule for a captured left-matmul
/// operation without first constructing a concrete [`LeftMatMulOperation`].
pub fn left_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
    if inputs.len() != 1 {
        return Err(TypeError { message: format!("left_matmul expected 1 input type but got {}", inputs.len()) });
    }
    Ok(vec![matmul_abstract(factor_type, &inputs[0], "left_matmul")?])
}

impl<V: MatrixValue> Debug for LeftMatMulOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LeftMatMul")
    }
}

impl<V: MatrixValue> Display for LeftMatMulOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "left_matmul")
    }
}

impl<V: MatrixValue> Operation<ArrayType> for LeftMatMulOperation<V> {
    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        left_matmul_abstract_eval(&<V as Typed<ArrayType>>::r#type(&self.factor), input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", self.factor()))
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for LeftMatMulOperation<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V> for LeftMatMulOperation<V> {
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<
            '_,
            ArrayType,
            V,
            LinearPrimitiveOperation<V>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .apply_operation(
                        &[atom],
                        LinearPrimitiveOperation::LeftMatMul { factor: self.factor.clone().transpose_matrix() },
                        1,
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
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    E::LinearOperation: SupportsLeftMatMul<ArrayType, V>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, V, E::LinearOperation>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
        check_input_count!(inputs, 1);
        let primal = self.factor().clone().matmul(inputs[0].primal.clone());
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsLeftMatMul<ArrayType, V>>::left_matmul_operation(self.factor().clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("left matmul jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

/// JVP rule for `LeftMatMulOperation` under
/// [`LinearizationEngine`](crate::tracing_v2::LinearizationEngine).
impl<'engine, V, EInner, OInner>
    DifferentiableOperation<crate::tracing_v2::LinearizationEngine<'engine, EInner, OInner>> for LeftMatMulOperation<V>
where
    V: MatrixValue + Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    EInner: Engine<Type = ArrayType, Value = V> + ?Sized,
    OInner: TracedLinearizationCarrier<V>,
    Tracer<'engine, EInner, OInner>: MatrixOps,
{
    fn jvp(
        &self,
        engine: &crate::tracing_v2::LinearizationEngine<'engine, EInner, OInner>,
        context: &mut JvpContext<
            '_,
            Tracer<'engine, EInner, OInner>,
            LinearPrimitiveOperation<Tracer<'engine, EInner, OInner>>,
        >,
        inputs: &[JvpTracer<Tracer<'engine, EInner, OInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner, OInner>, AtomId>>, TracingError> {
        check_input_count!(inputs, 1);
        let factor_tracer = engine.lift_constant(self.factor().clone());
        let primal = factor_tracer.clone().matmul(inputs[0].primal.clone());
        let tangent = context
            .apply_operation(&[inputs[0].tangent], LinearPrimitiveOperation::LeftMatMul { factor: factor_tracer }, 1)?
            .into_iter()
            .next()
            .expect("left matmul jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
