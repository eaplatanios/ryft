use std::ops::Neg;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{NegOperation, SupportsNeg};
use crate::tracing::engines::Tracer;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError, TranspositionContext};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::Type;

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsNeg<T, V>> LinearOperation<T, V, O> for NegOperation
where
    NegOperation: Operation<T>,
{
    #[inline]
    fn transpose(
        &self,
        context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let cotangent_outputs = context.stage(O::neg_operation(), &[atom])?;
                check_count!("output", cotangent_outputs, 1, TracingError);
                Ok(vec![Some(cotangent_outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

impl<E: DifferentiableEngine> DifferentiableOperation<E> for NegOperation
where
    E::Value: Neg<Output = E::Value> + Differentiable<E::Type>,
    E::LinearOperationCarrier: SupportsNeg<E::Type, E::Tangent>,
    NegOperation: Operation<E::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>],
    ) -> Result<Vec<JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: -inputs[0].tangent.clone() }])
    }
}
