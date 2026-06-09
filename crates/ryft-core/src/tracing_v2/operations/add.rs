use std::ops::Add;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::Type;

impl<T: Parameter + PartialEq + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for AddOperation
where
    AddOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
    }
}

impl<D: DifferentiationContext> DifferentiableOperation<D> for AddOperation
where
    D::Value: Add<Output = D::Value>,
    LinearOperationOf<D>: SupportsAdd<D::Type>,
    AddOperation: Operation<D::Type>,
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
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![JvpTracer::new(
            inputs[0].primal().clone() + inputs[1].primal().clone(),
            inputs[0].tangent().clone() + inputs[1].tangent().clone(),
        )])
    }
}
