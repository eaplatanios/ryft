use std::ops::Add;

use crate::differentiation::{LinearOperation, TranspositionContext};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::tracing::engines::Tracer;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::Type;

impl<T: PartialEq + Type, V: Traceable<T>, O: Clone + Operation<T>> LinearOperation<T, V, O> for AddOperation
where
    AddOperation: Operation<T>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<E: DifferentiableEngine> DifferentiableOperation<E> for AddOperation
where
    E::Value: Add<Output = E::Value> + Differentiable<E::Type>,
    AddOperation: Operation<E::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>],
    ) -> Result<Vec<JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.clone() + inputs[1].primal.clone(),
            tangent: inputs[0].tangent.clone() + inputs[1].tangent.clone(),
        }])
    }
}
