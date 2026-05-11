use std::ops::Neg;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, SupportsNeg, SupportsScale};
use crate::operations::trigonometric::{Cos, CosOperation, Sin};
use crate::tracing::TracingError;
use crate::tracing::domains::Tracer;
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};

impl<D> DifferentiableOperation<D> for CosOperation
where
    D: DifferentiableDomain,
    CosOperation: Operation<D::Type>,
    D::Value: Cos + Sin + Neg<Output = D::Value> + Differentiable<D::Type>,
    D::LinearOperationCarrier: SupportsNeg<D::Type, D::Tangent> + SupportsScale<D::Type, D::Tangent, D::Value>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().cos(),
            tangent: -input.tangent.clone().scale(input.primal.clone().sin()),
        }])
    }
}
