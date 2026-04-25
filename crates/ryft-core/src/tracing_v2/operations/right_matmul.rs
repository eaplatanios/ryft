use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::{OperationFormatter, Traceable, TracingError, Value};
use crate::tracing_v2::{
    engines::{DifferentiableEngine, Engine},
    forward::{Differentiable, EngineTangent, JvpTracer, TangentSpace},
    jit::Tracer,
    linear::LinearTerm,
    operations::constants::ZeroLike,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation,
    add::LinearAddOperation,
    left_matmul::LinearLeftMatMulOperation,
    lift_jit_constant,
    matmul::MatMulTracingOperation,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
    matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeTracingOperation},
    neg::LinearNegOperation,
    primitive::LinearPrimitiveOperation,
    scale::LinearScaleOperation,
};

/// Hidden staging trait for the right matrix-multiplication primitive.
#[doc(hidden)]
pub trait RightMatMulTracingOperation<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the right matrix-multiplication primitive
    /// with a captured factor.
    fn right_matmul_op(factor: V) -> Self;
}

/// Hidden staging trait for the right matrix-multiplication primitive in linear programs.
#[doc(hidden)]
pub trait LinearRightMatMulOperation<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear right matrix-multiplication
    /// primitive with a captured factor.
    fn linear_right_matmul_op(factor: V) -> Self;
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
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().matmul(self.factor.clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V> for RightMatMulOperation<V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                output_cotangents[0].builder.clone(),
                std::slice::from_ref(&output_cotangents[0]),
                LinearPrimitiveOperation::RightMatMul { factor: self.factor.clone().transpose_matrix() },
                1,
            )?
            .into_iter()
            .next()
            .expect("right matmul should produce one cotangent contribution"),
        )])
    }
}

impl<
    'engine,
    V: Value<ArrayType> + MatrixOps + ZeroLike,
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    InnerLinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + LinearScaleOperation<ArrayType, Tracer<'engine, E>>
        + LinearLeftMatMulOperation<ArrayType, Tracer<'engine, E>>
        + LinearRightMatMulOperation<ArrayType, Tracer<'engine, E>>
        + LinearMatrixTransposeOperation<ArrayType, Tracer<'engine, E>>,
> InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>
    for RightMatMulOperation<V>
where
    E::TracingOperation: MatMulTracingOperation<ArrayType, V> + MatrixTransposeTracingOperation<ArrayType, V> + 'static,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>, TracingError>
    {
        check_input_count!(inputs, 1);
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![inputs[0].clone().matmul(factor)])
    }
}

impl<V, E> DifferentiableOperation<E> for RightMatMulOperation<V>
where
    V: MatrixValue + ZeroLike + Differentiable<ArrayType>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    E::LinearOperation:
        LinearAddOperation<ArrayType, V> + LinearNegOperation<ArrayType, V> + LinearScaleOperation<ArrayType, V>,
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
        Ok(vec![inputs[0].clone().matmul(factor)])
    }
}
