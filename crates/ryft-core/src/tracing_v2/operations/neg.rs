use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::macros::check_input_count;
use crate::tracing::{Traceable, TracingError};
use crate::tracing_v2::{
    engine::Engine,
    forward::{Differentiable, EngineTangent, JvpTracer, TangentSpace},
    linear::LinearTerm,
    operations::constants::ZeroLike,
};
use crate::types::{ArrayType, Type, TypeError};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearAddOperation, LinearOperation, LinearScaleOperation,
    Operation, unary_abstract,
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

impl Operation<ArrayType> for NegOperation {
    fn name(&self) -> &'static str {
        "neg"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(vec![unary_abstract(input_types)?])
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

impl<E> DifferentiableOperation<E> for NegOperation
where
    E: Engine<Type = ArrayType> + ?Sized,
    E::Value: Traceable<ArrayType> + Neg<Output = E::Value> + Differentiable<ArrayType>,
    E::LinearOperation: LinearAddOperation<ArrayType, E::Value>
        + LinearNegOperation<ArrayType, E::Value>
        + LinearScaleOperation<ArrayType, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![JvpTracer {
            primal: -inputs[0].primal.clone(),
            tangent: EngineTangent::<E>::neg(inputs[0].tangent.clone()),
        }])
    }
}
