use std::ops::Add;

use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::tracing_v2::differentiation::{DifferentiableOperation, combine_terms};

impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for AddOperation
where
    AddOperation: Operation<V::Type>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![outputs[0].clone(), outputs[0].clone()])
    }
}

impl<C: Context> DifferentiableOperation<C> for AddOperation
where
    C::Operation: Clone,
    C::Value: Add<Output = C::Value>,
    AddOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().clone() + inputs[1].primal().clone();
        // Structural zeros are dropped so the tangent program never stages `Add(zero, ..)`, which the
        // straight-line tangent transposition would reject for having a known (non-linear) operand.
        let left = inputs[0].tangent().as_value().cloned();
        let right = inputs[1].tangent().as_value().cloned();
        let tangent = combine_terms(left, right, &primal);
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::EagerContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::scalars::Scalar;
    use crate::tracing_v2::{Differentiate, value_and_grad};

    #[test]
    fn test_add_jvp_and_gradient_are_linear() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain
            .jvp(
                |(left, right)| Ok(left + right),
                (Scalar::from(2.0), Scalar::from(5.0)),
                (Scalar::from(3.0), Scalar::from(-1.0)),
            )
            .unwrap();
        assert_eq!(primal, 7.0);
        assert_eq!(tangent, 2.0);

        let (value, gradient) =
            value_and_grad(&domain, |(left, right)| left + right, (Scalar::from(2.0), Scalar::from(5.0))).unwrap();
        assert_eq!(value, 7.0);
        assert_eq!(gradient, (Scalar::from(1.0), Scalar::from(1.0)));
    }
}
