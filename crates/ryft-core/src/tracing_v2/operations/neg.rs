use std::ops::Neg;

use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::NegOperation;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;

impl<V: Value, O: Operation<V::Type> + From<NegOperation>> TransposableOperation<V, O> for NegOperation
where
    NegOperation: Operation<V::Type>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Value(cotangent) => Ok(vec![MaybeZero::Value(-cotangent.clone())]),
            MaybeZero::Zero(r#type) => Ok(vec![MaybeZero::Zero(r#type.clone())]),
        }
    }
}

impl<C: Context> DifferentiableOperation<C> for NegOperation
where
    C::Operation: Clone,
    C::Value: Neg<Output = C::Value>,
    NegOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = -inputs[0].primal().clone();
        // A negated structural zero stays a structural zero, keeping `Neg(zero)` out of the tangent program.
        let tangent = inputs[0].tangent().clone().map(|tangent| -tangent);
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::EagerContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::scalars::Scalar;
    use crate::tracing_v2::{Differentiate, value_and_gradient};

    #[test]
    fn test_neg_jvp_and_gradient_negate() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain.jvp(|x| Ok(-x), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_eq!(primal, -2.0);
        assert_eq!(tangent, -3.0);

        let (value, gradient) = value_and_gradient(&domain, |x| -x, Scalar::from(2.0)).unwrap();
        assert_eq!(value, -2.0);
        assert_eq!(gradient, -1.0);
    }
}
