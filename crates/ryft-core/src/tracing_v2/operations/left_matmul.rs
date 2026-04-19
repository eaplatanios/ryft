//! Left matrix-multiplication primitive for [`crate::tracing_v2`].
//!
//! This module specializes matrix multiplication to the common linear-map form `factor @ input`.
//! It is used heavily by matrix transpose rules because cotangent propagation naturally produces
//! captured left and right linear actions rather than only free-standing binary matmuls.

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    TraceError, Traceable, Value, ZeroLike,
    batch::Batch as BatchedValue,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::Tracer,
    linear::LinearTerm,
};
use crate::types::{ArrayType, Type, Typed};

use super::{
    DifferentiableOp, InterpretableOp, LinearOperation, Op, VectorizableOp,
    add::LinearAddOperation,
    expect_input_count, lift_jit_constant,
    matmul::MatMulTracingOperation,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
    matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeTracingOperation},
    neg::LinearNegOperation,
    primitive::LinearPrimitiveOp,
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
/// [`LeftMatMulOp`] is the matrix-valued analogue of [`super::ScaleOp`]: it captures one factor in
/// the op object and applies that factor to every input it is replayed on.
#[derive(Clone)]
pub struct LeftMatMulOp<V: MatrixValue> {
    /// Matrix factor multiplied on the left of every input.
    factor: V,
}

impl<V: MatrixValue> LeftMatMulOp<V> {
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
/// operation without first constructing a concrete [`LeftMatMulOp`].
pub fn left_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(vec![matmul_abstract(factor_type, &inputs[0], "left_matmul")?])
}

impl<V: MatrixValue> Debug for LeftMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LeftMatMul")
    }
}

impl<V: MatrixValue> Display for LeftMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "left_matmul")
    }
}

impl<V: MatrixValue> Op for LeftMatMulOp<V> {
    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        left_matmul_abstract_eval(&<V as Typed<ArrayType>>::tpe(&self.factor), inputs)
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        _is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        if crate::tracing_v2::is_identity_one(&self.factor) { Some(inputs.to_vec()) } else { None }
    }
}

impl<V: MatrixValue> InterpretableOp<ArrayType, V> for LeftMatMulOp<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V> for LeftMatMulOp<V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                std::slice::from_ref(&output_cotangents[0]),
                LinearPrimitiveOp::LeftMatMul { factor: self.factor.clone().transpose_matrix() },
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
        + LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + LinearScaleOperation<ArrayType, Tracer<'engine, E>>
        + LinearLeftMatMulOperation<ArrayType, Tracer<'engine, E>>
        + LinearRightMatMulOperation<ArrayType, Tracer<'engine, E>>
        + LinearMatrixTransposeOperation<ArrayType, Tracer<'engine, E>>,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>
    for LeftMatMulOp<V>
where
    O: Op<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }
}

impl<
    V: MatrixValue + ZeroLike,
    O: Clone,
    L: Clone
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>
        + LinearLeftMatMulOperation<ArrayType, V>
        + LinearRightMatMulOperation<ArrayType, V>
        + LinearMatrixTransposeOperation<ArrayType, V>,
> DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L> for LeftMatMulOp<V>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V, L>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V, L>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = JvpTracer {
            primal: self.factor().clone(),
            tangent: TangentSpace::zero_like(&self.factor, &inputs[0].tangent),
        };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue> VectorizableOp<ArrayType, V> for LeftMatMulOp<V> {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| self.factor.clone().matmul(lane)).collect(),
        )])
    }
}
