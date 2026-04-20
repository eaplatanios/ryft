//! Right matrix-multiplication primitive for [`crate::tracing_v2`].
//!
//! This module specializes matrix multiplication to the linear-map form `input @ factor`. Together
//! with [`super::left_matmul`], it gives transpose and linearization code explicit building blocks
//! for non-commutative matrix actions.

use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing_v2::{
    AtomId, Traceable, TracingError, Value, ZeroLike,
    batch::Batch as BatchedValue,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::Tracer,
    linear::LinearTerm,
};
use crate::types::{ArrayType, Type, Typed};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, VectorizableOperation,
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
pub trait RightMatMulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the right matrix-multiplication primitive
    /// with a captured factor.
    fn right_matmul_op(factor: V) -> Self;
}

/// Hidden staging trait for the right matrix-multiplication primitive in linear programs.
#[doc(hidden)]
pub trait LinearRightMatMulOperation<T: Type + Display, V: Traceable<T>>: Clone {
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
pub fn right_matmul_abstract_eval(
    factor_type: &ArrayType,
    inputs: &[ArrayType],
) -> Result<Vec<ArrayType>, TracingError> {
    check_input_count!(inputs, 1);
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

impl<V: MatrixValue> Operation for RightMatMulOperation<V> {
    fn name(&self) -> &'static str {
        "right_matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        right_matmul_abstract_eval(&<V as Typed<ArrayType>>::r#type(&self.factor), input_types)
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
    for RightMatMulOperation<V>
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
        Ok(vec![inputs[0].clone().matmul(factor)])
    }
}

impl<
    V: MatrixValue + ZeroLike,
    O: Clone,
    L: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>
        + LinearLeftMatMulOperation<ArrayType, V>
        + LinearRightMatMulOperation<ArrayType, V>
        + LinearMatrixTransposeOperation<ArrayType, V>,
> DifferentiableOperation<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L> for RightMatMulOperation<V>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V, L>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V, L>>>, TracingError> {
        check_input_count!(inputs, 1);
        let factor = JvpTracer {
            primal: self.factor().clone(),
            tangent: TangentSpace::zero_like(&self.factor, &inputs[0].tangent),
        };
        Ok(vec![inputs[0].clone().matmul(factor)])
    }
}

impl<V: MatrixValue> VectorizableOperation<ArrayType, V> for RightMatMulOperation<V> {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| lane.matmul(self.factor.clone())).collect(),
        )])
    }
}
