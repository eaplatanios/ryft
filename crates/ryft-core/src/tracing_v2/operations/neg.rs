use std::fmt::Display;
use std::ops::Neg;

use crate::macros::check_input_count;
use crate::operations::constants::ZeroLike;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearArrayOperation, LinearizableEngine};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Trait that represents [`Operation`] carrier types that support/include [`NegOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`NegOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsNeg<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the negation [`Operation`].
    fn neg_operation() -> Self;
}

impl<'engine, E: TracingEngine + ?Sized> Neg for Tracer<'engine, E>
where
    E::OperationCarrier: SupportsNeg<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(E::OperationCarrier::neg_operation())
    }
}

/// Elementwise negation primitive.
///
/// [`NegOperation`] is the canonical example of a shape-preserving unary primitive with a nontrivial
/// transpose rule.
#[derive(Clone, Debug, Default)]
pub struct NegOperation;

impl Display for NegOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl<T: Type> Operation<T> for NegOperation {
    #[inline]
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

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike>
    LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>> for NegOperation
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            V,
            LinearArrayOperation<V, ArrayType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .stage(LinearArrayOperation::Neg, &[atom])?
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
        context: &mut crate::tracing::transposition::TranspositionContext<
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
                    .stage(LinearArrayOperation::<V, DataType>::Neg, &[atom])?
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
    E: LinearizableEngine + ?Sized,
    NegOperation: Operation<E::Type>,
    E::Value: Neg<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsNeg<E::Type, E::Value>,
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
                <E::LinearOperationCarrier as SupportsNeg<E::Type, E::Value>>::neg_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("neg jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent }])
    }
}
