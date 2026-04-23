use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::{
    engine::Engine,
    forward::{Differentiable, EngineTangent, JvpTracer, TangentSpace},
    jit::Tracer,
    linear::LinearTerm,
    operations::constants::ZeroLike,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation,
    add::LinearAddOperation,
    lift_jit_constant,
    matmul::MatMulTracingOperation,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
    matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeTracingOperation},
    neg::LinearNegOperation,
    primitive::LinearPrimitiveOperation,
    right_matmul::LinearRightMatMulOperation,
    scale::LinearScaleOperation,
};

/// Hidden staging trait for the left matrix-multiplication primitive.
#[doc(hidden)]
pub trait LeftMatMulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the left matrix-multiplication primitive
    /// with a captured factor.
    fn left_matmul_op(factor: V) -> Self;
}

/// Hidden staging trait for the left matrix-multiplication primitive in linear programs.
#[doc(hidden)]
pub trait LinearLeftMatMulOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear left matrix-multiplication
    /// primitive with a captured factor.
    fn linear_left_matmul_op(factor: V) -> Self;
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

impl<V: MatrixValue> Operation for LeftMatMulOperation<V> {
    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        left_matmul_abstract_eval(&<V as Typed<ArrayType>>::r#type(&self.factor), input_types)
    }

    fn try_simplify(
        &self,
        inputs: &[AtomId],
        _is_zero_constant: &dyn Fn(AtomId) -> bool,
        _is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        if self.factor.is_one() { Some(inputs.to_vec()) } else { None }
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
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                output_cotangents[0].builder.clone(),
                std::slice::from_ref(&output_cotangents[0]),
                LinearPrimitiveOperation::LeftMatMul { factor: self.factor.clone().transpose_matrix() },
                1,
            )?
            .into_iter()
            .next()
            .expect("left matmul should produce one cotangent contribution"),
        )])
    }
}

impl<
    'engine,
    V: Value<ArrayType> + MatrixOps + ZeroLike,
    O: MatMulTracingOperation<ArrayType, V> + MatrixTransposeTracingOperation<ArrayType, V> + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
    InnerLinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + LinearScaleOperation<ArrayType, Tracer<'engine, E>>
        + LinearLeftMatMulOperation<ArrayType, Tracer<'engine, E>>
        + LinearRightMatMulOperation<ArrayType, Tracer<'engine, E>>
        + LinearMatrixTransposeOperation<ArrayType, Tracer<'engine, E>>,
> InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>
    for LeftMatMulOperation<V>
where
    O: Operation<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>, TracingError>
    {
        check_input_count!(inputs, 1);
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }
}

impl<V, E> DifferentiableOperation<E> for LeftMatMulOperation<V>
where
    V: MatrixValue + ZeroLike + Differentiable<ArrayType>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized,
    E::LinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
    EngineTangent<E>: super::matrix::MatrixTangentSpace<V>,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<V, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<V, EngineTangent<E>>>, TracingError> {
        check_input_count!(inputs, 1);
        let factor = JvpTracer {
            primal: self.factor().clone(),
            tangent: TangentSpace::zero_like(&self.factor, &inputs[0].tangent),
        };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }
}
