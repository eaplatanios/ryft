use std::ops::{Neg, Sub};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::math::{NegOperation, SubOperation};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::programs::types::Typed;
use crate::tracing_v2::operations::broadcasting::ElementwiseDifferentiableValue;

impl<V: Value, O: Operation<V::Type> + From<NegOperation>> TransposableOperation<V, O> for SubOperation
where
    SubOperation: Operation<V::Type>,
    V::Type: DifferentiableType,
    Tracer<TracingContext<V, O>>: ElementwiseDifferentiableValue<V::Type>,
{
    #[inline]
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Value(cotangent) => inputs
                .iter()
                .enumerate()
                .map(|(input_index, input)| {
                    let target = input.r#type().cotangent();
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "'sub' input has no cotangent space".to_string(),
                        }
                        .into());
                    }
                    let contribution = if input_index == 0 { cotangent.clone() } else { -cotangent.clone() };
                    Ok(MaybeZero::Value(contribution.unbroadcast_elementwise_cotangent(&target)?))
                })
                .collect(),
            MaybeZero::Zero(_) => inputs
                .iter()
                .map(|input| {
                    let target = input.r#type().cotangent();
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "'sub' input has no cotangent space".to_string(),
                        });
                    }
                    Ok(MaybeZero::Zero(target))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(Into::into),
        }
    }
}

impl<C: Context> DifferentiableOperation<C> for SubOperation
where
    SubOperation: Operation<C::Type>,
    C::Type: DifferentiableType,
    C::Value: Sub<Output = C::Value> + Neg<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().clone() - inputs[1].primal().clone();
        // Structural zeros are dropped; a zero minuend collapses to the negated subtrahend so the tangent
        // program never stages `Sub(zero, ..)` or `Sub(.., zero)`.
        let left = inputs[0].tangent().as_value().cloned();
        let right = inputs[1].tangent().as_value().cloned();
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'sub' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        let tangent = match (left, right) {
            (Some(left), Some(right)) => MaybeZero::Value(
                left.normalize_elementwise_tangent(&target)? - right.normalize_elementwise_tangent(&target)?,
            ),
            (Some(tangent), None) => MaybeZero::Value(tangent.normalize_elementwise_tangent(&target)?),
            (None, Some(tangent)) => {
                let tangent = -tangent;
                MaybeZero::Value(tangent.normalize_elementwise_tangent(&target)?)
            }
            (None, None) => MaybeZero::Zero(target),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
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
