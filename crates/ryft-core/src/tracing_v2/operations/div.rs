use std::ops::{Add, Div, Mul, Neg};

use crate::contexts::Context;
use crate::differentiation::DifferentiationDual;
use crate::differentiation::TransposableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::DivOperation;
use crate::operations::constants::OneLike;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::differentiation::DifferentiableOperation;
use crate::types::Typed;

impl<C: Context> DifferentiableOperation<C> for DivOperation
where
    C::Operation: Clone,
    C::Value:
        OneLike + Add<Output = C::Value> + Div<Output = C::Value> + Mul<Output = C::Value> + Neg<Output = C::Value>,
    DivOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone() / right.primal().clone();
        // Quotient rule `d(a/b) = (1/b)*da - (a/b²)*db`, with each coefficient built as a fresh primal operation on the
        // primal tracers and then multiplied by the input tangent. Expressing each tangent term as a primal-coefficient
        // `Mul` (rather than a `Div` of the tangent) keeps the tangent map in the bilinear-`Mul` form that both the
        // scalar and array tangent transposers fold into closed constant factors. Zero terms are dropped so the
        // tangent stays minimal.
        let left_term = left.tangent().as_value().map(|tangent| {
            let reciprocal = right.primal().one_like() / right.primal().clone();
            reciprocal * tangent.clone()
        });
        let right_term = right.tangent().as_value().map(|tangent| {
            let denominator = right.primal().clone() * right.primal().clone();
            -(left.primal().clone() / denominator) * tangent.clone()
        });
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(primal.r#type().into_owned()), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`DivOperation`]: division is nonlinear in its operands, so a tangent program never contains a
/// primal `div` on a linear operand (the quotient-rule forward stages bilinear `mul` coefficients instead) and the
/// rule reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for DivOperation
where
    DivOperation: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no partition-aware transpose rule", self.name()),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::EagerContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::scalars::Scalar;
    use crate::tracing_v2::Differentiate;
    use crate::tracing_v2::test_util::assert_scalar_close;

    #[test]
    fn test_div_jvp_matches_the_quotient_rule() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent): (Scalar, Scalar) = domain
            .jvp(
                |(left, right)| Ok(left / right),
                (Scalar::from(6.0), Scalar::from(2.0)),
                (Scalar::from(3.0), Scalar::from(4.0)),
            )
            .unwrap();

        assert_scalar_close(primal, 3.0);
        assert_scalar_close(tangent, -4.5);
    }
}
