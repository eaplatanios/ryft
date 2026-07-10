use std::ops::{Neg, Sub};

use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{NegOperation, SubOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::types::Typed;

impl<V: Value, O: Operation<V::Type> + From<NegOperation>> TransposableOperation<V, O> for SubOperation
where
    SubOperation: Operation<V::Type>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Value(cotangent) => {
                Ok(vec![MaybeZero::Value(cotangent.clone()), MaybeZero::Value(-cotangent.clone())])
            }
            MaybeZero::Zero(r#type) => Ok(vec![MaybeZero::Zero(r#type.clone()), MaybeZero::Zero(r#type.clone())]),
        }
    }
}

impl<C: Context> DifferentiableOperation<C> for SubOperation
where
    C::Operation: Clone,
    C::Value: Sub<Output = C::Value> + Neg<Output = C::Value>,
    SubOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().clone() - inputs[1].primal().clone();
        // Structural zeros are dropped; a zero minuend collapses to the negated subtrahend so the tangent
        // program never stages `Sub(zero, ..)` or `Sub(.., zero)`.
        let left = inputs[0].tangent().as_value().cloned();
        let right = inputs[1].tangent().as_value().cloned();
        let tangent = match (left, right) {
            (Some(left), Some(right)) => MaybeZero::Value(left - right),
            (Some(term), None) => MaybeZero::Value(term),
            (None, Some(term)) => MaybeZero::Value(-term),
            (None, None) => MaybeZero::Zero(primal.r#type().into_owned()),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::scalars::Scalar;
    use crate::tracing_v2::ForwardModeDifferentiate;

    #[test]
    fn test_sub_jvp_matches_the_difference_rule() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent): (Scalar, Scalar) = domain
            .jvp(
                |(left, right)| Ok(left - right),
                (Scalar::from(5.0), Scalar::from(2.0)),
                (Scalar::from(3.0), Scalar::from(1.0)),
            )
            .unwrap();

        assert_eq!(primal, 3.0);
        assert_eq!(tangent, 2.0);
    }
}
