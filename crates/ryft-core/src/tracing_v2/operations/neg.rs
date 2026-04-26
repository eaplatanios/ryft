use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::macros::check_input_count;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::{
    LinearPrimitiveOperation,
    engines::DifferentiableEngine,
    forward::{Differentiable, JvpContext, JvpTracer},
    operations::constants::ZeroLike,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::{DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, unary_abstract};

/// Hidden carrier capability for staging the negation primitive.
#[doc(hidden)]
pub trait SupportsNeg<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the negation primitive.
    fn neg_operation() -> Self;
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

impl<V: Typed<ArrayType> + Clone + Neg<Output = V>> InterpretableOperation<ArrayType, V> for NegOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike> LinearOperation<ArrayType, V> for NegOperation {
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
                    .apply_operation(&[atom], LinearPrimitiveOperation::Neg, 1)?
                    .into_iter()
                    .next()
                    .expect("neg transpose should produce one cotangent contribution"),
            )]),
            None => Ok(vec![None]),
        }
    }
}

impl<E> DifferentiableOperation<E> for NegOperation
where
    E: DifferentiableEngine + ?Sized,
    NegOperation: Operation<E::Type>,
    E::Value: Neg<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsNeg<E::Type, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1);
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsNeg<E::Type, E::Value>>::neg_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("neg jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent }])
    }
}
