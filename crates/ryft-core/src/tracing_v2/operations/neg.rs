use std::ops::Neg;

use crate::differentiation::LinearOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{NegOperation, SupportsNeg};
use crate::parameters::Parameter;
use crate::tracing::domains::{ProgramTracer, Tracer};
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
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
        output_cotangents: &[Option<ProgramTracer<'transpose, T, V, O>>],
    ) -> Result<Vec<Option<ProgramTracer<'transpose, T, V, O>>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Some(cotangent) => Ok(vec![Some(-cotangent.clone())]),
            None => Ok(vec![None]),
        }
    }
}

impl<D: DifferentiableDomain> DifferentiableOperation<D> for NegOperation
where
    D::Value: Neg<Output = D::Value> + Differentiable<D::Type>,
    D::LinearOperationCarrier: SupportsNeg<D::Type, D::Tangent>,
    NegOperation: Operation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: -inputs[0].tangent.clone() }])
    }
}
