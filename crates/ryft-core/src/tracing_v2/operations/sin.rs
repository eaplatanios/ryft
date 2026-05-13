use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, SupportsScale};
use crate::operations::trigonometric::{Cos, Sin, SinOperation};
use crate::tracing::TracingError;
use crate::tracing::domains::Tracer;
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};

impl<D> DifferentiableOperation<D> for SinOperation
where
    D: DifferentiableDomain,
    SinOperation: Operation<D::Type>,
    D::Value: Sin + Cos,
    D::LinearOperationCarrier: SupportsScale<D::Type, D::Tangent, D::Value>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        Ok(vec![JvpTracer::new(
            input.primal().clone().sin(),
            input.tangent().clone().scale(input.primal().clone().cos()),
        )])
    }
}
