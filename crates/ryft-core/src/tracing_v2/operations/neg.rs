//! Negation primitive for [`crate::tracing_v2`].
//!
//! This module provides the unary sign-flip primitive used throughout staged arithmetic. Like the
//! other core scalar primitives, it demonstrates the full lifecycle of one semantic operation
//! across abstract evaluation, replay, linear transposition, forward-mode AD, and batching.

use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::batching::Batch;
use crate::macros::check_input_count;
use crate::tracing_v2::{
    AtomId, Traceable, TracingError, ZeroLike,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    linear::LinearTerm,
};
use crate::types::{ArrayType, Type};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, VectorizableOperation, unary_abstract,
};

/// Hidden staging trait for the negation primitive.
#[doc(hidden)]
pub trait NegTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the negation primitive.
    fn neg_op() -> Self;
}

/// Hidden staging trait for the negation primitive in linear programs.
#[doc(hidden)]
pub trait LinearNegOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear negation primitive.
    fn linear_neg_op() -> Self;
}

/// Elementwise negation primitive.
///
/// [`NegOperation`] is the canonical example of a shape-preserving unary primitive with a nontrivial
/// transpose rule.
#[derive(Clone, Default)]
pub struct NegOperation;

impl Debug for NegOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Neg")
    }
}

impl Display for NegOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "neg")
    }
}

impl Operation for NegOperation {
    fn name(&self) -> &'static str {
        "neg"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        Ok(vec![unary_abstract(input_types)?])
    }

    fn try_simplify(
        &self,
        inputs: &[AtomId],
        is_zero_constant: &dyn Fn(AtomId) -> bool,
        _is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        if inputs.len() == 1 && is_zero_constant(inputs[0]) { Some(vec![inputs[0]]) } else { None }
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>> InterpretableOperation<ArrayType, V> for NegOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike> LinearOperation<ArrayType, V> for NegOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        Ok(vec![Some(output_cotangents[0].clone().neg())])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOperation<ArrayType, V, T, O, L> for NegOperation
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: T::neg(inputs[0].tangent.clone()) }])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>> VectorizableOperation<ArrayType, V> for NegOperation {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| -lane).collect())])
    }
}
