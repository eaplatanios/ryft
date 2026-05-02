use std::fmt::{Debug, Display};
use std::ops::Neg;

use crate::macros::check_input_count;
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::operations::constants::ZeroLike;
use crate::tracing_v2::{DifferentiableEngine, LinearArrayOperation};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::{DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Hidden carrier capability for staging the negation primitive.
#[doc(hidden)]
pub trait SupportsNeg<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the negation primitive.
    fn neg_operation() -> Self;
}

impl<'engine, E: TracingEngine + ?Sized> Neg for Tracer<'engine, E>
where
    E::Operation: SupportsNeg<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(E::Operation::neg_operation())
    }
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

impl<T: Type> Operation<T> for NegOperation {
    fn name(&self) -> &'static str {
        "neg"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<V: Typed<ArrayType> + Clone + Neg<Output = V>> InterpretableOperation<ArrayType, V> for NegOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: Typed<DataType> + Clone + Neg<Output = V>> InterpretableOperation<DataType, V> for NegOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike> LinearOperation<ArrayType, V, LinearArrayOperation<V>>
    for NegOperation
{
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<'_, ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .apply_operation(&[atom], LinearArrayOperation::Neg, 1)?
                    .into_iter()
                    .next()
                    .expect("neg transpose should produce one cotangent contribution"),
            )]),
            None => Ok(vec![None]),
        }
    }
}

impl<V: Traceable<DataType> + crate::parameters::Parameter + Neg<Output = V> + ZeroLike>
    LinearOperation<DataType, V, LinearArrayOperation<V, DataType>> for NegOperation
{
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<
            '_,
            DataType,
            V,
            LinearArrayOperation<V, DataType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .apply_operation(&[atom], LinearArrayOperation::<V, DataType>::Neg, 1)?
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
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
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
