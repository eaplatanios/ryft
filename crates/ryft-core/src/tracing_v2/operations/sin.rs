use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, SupportsScale};
use crate::operations::trigonometric::{Cos, Sin, SinOperation};
use crate::tracing::TracingError;
use crate::tracing::engines::Tracer;
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};

impl<E> DifferentiableOperation<E> for SinOperation
where
    E: DifferentiableEngine,
    SinOperation: Operation<E::Type>,
    E::Value: Sin + Cos + Differentiable<E::Type>,
    E::LinearOperationCarrier: SupportsScale<E::Type, E::Tangent, E::Value>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>],
    ) -> Result<Vec<JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: input.tangent.clone().scale(input.primal.clone().cos()),
        }])
    }
}
