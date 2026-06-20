use std::ops::Add;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::Type;

impl<T: Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for AddOperation
where
    AddOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
    }
}

impl<D: DifferentiationContext> DifferentiableOperation<D> for AddOperation
where
    D::Value: Add<Output = D::Value>,
    LinearOperationOf<D>: From<AddOperation>,
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

#[cfg(test)]
mod tests {
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::{DifferentiationContext, value_and_grad};

    #[test]
    fn test_add_jvp_and_gradient_are_linear() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) = domain.jvp(|(left, right)| left + right, (2.0f64, 5.0f64), (3.0f64, -1.0f64)).unwrap();
        assert_eq!(primal, 7.0);
        assert_eq!(tangent, 2.0);

        let (value, gradient) = value_and_grad(&domain, |(left, right)| left + right, (2.0f64, 5.0f64)).unwrap();
        assert_eq!(value, 7.0);
        assert_eq!(gradient, (1.0, 1.0));
    }
}
