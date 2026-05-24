use std::ops::Neg;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{NegOperation, SupportsNeg};
use crate::parameters::Parameter;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{Differentiable, DifferentiableOperation};
use crate::types::Type;

impl<T: Parameter + Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsNeg<T, V>> LinearOperation<T, V, O>
    for NegOperation
where
    NegOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D: Differentiable> DifferentiableOperation<D> for NegOperation
where
    D::Value: Neg<Output = D::Value>,
    D::LinearOperationCarrier: SupportsNeg<D::Type, D::Tangent>,
    NegOperation: Operation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer::new(-inputs[0].primal().clone(), -inputs[0].tangent().clone())])
    }
}
