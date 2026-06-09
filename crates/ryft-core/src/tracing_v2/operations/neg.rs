use std::ops::Neg;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{NegOperation, SupportsNeg};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::Type;

impl<T: Parameter + Type, V: Value<T>, O: Operation<T> + SupportsNeg<T>> TransposableOperation<T, V, O> for NegOperation
where
    NegOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D: DifferentiationContext> DifferentiableOperation<D> for NegOperation
where
    D::Value: Neg<Output = D::Value>,
    LinearOperationOf<D>: SupportsNeg<D::Type>,
    NegOperation: Operation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(-inputs[0].primal().clone(), -inputs[0].tangent().clone())])
    }
}
