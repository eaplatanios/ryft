use std::ops::Neg;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::NegOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::Type;

impl<T: Type, V: Value<T>, O: Operation<T> + From<NegOperation>> TransposableOperation<T, V, O> for NegOperation
where
    NegOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
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
    LinearOperationOf<D>: From<NegOperation>,
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

#[cfg(test)]
mod tests {
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::{DifferentiationContext, value_and_grad};

    #[test]
    fn test_neg_jvp_and_gradient_negate() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) = domain.jvp(|x| -x, 2.0f64, 3.0f64).unwrap();
        assert_eq!(primal, -2.0);
        assert_eq!(tangent, -3.0);

        let (value, gradient) = value_and_grad(&domain, |x| -x, 2.0f64).unwrap();
        assert_eq!(value, -2.0);
        assert_eq!(gradient, -1.0);
    }
}
