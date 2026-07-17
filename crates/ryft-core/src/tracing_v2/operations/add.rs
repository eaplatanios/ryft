use std::ops::Add;

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::math::AddOperation;
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::programs::types::Typed;
use crate::tracing_v2::operations::broadcasting::ElementwiseDifferentiableValue;

impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for AddOperation
where
    AddOperation: Operation<V::Type>,
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
            MaybeZero::Zero(_) => inputs
                .iter()
                .map(|input| {
                    let target = input.r#type().cotangent();
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "'add' input has no cotangent space".to_string(),
                        });
                    }
                    Ok(MaybeZero::Zero(target))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(Into::into),
            MaybeZero::Value(cotangent) => inputs
                .iter()
                .map(|input| {
                    let target = input.r#type().cotangent();
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "'add' input has no cotangent space".to_string(),
                        }
                        .into());
                    }
                    Ok(MaybeZero::Value(cotangent.unbroadcast_elementwise_cotangent(&target)?))
                })
                .collect(),
        }
    }
}

impl<C: Context> DifferentiableOperation<C> for AddOperation
where
    AddOperation: Operation<C::Type>,
    C::Type: DifferentiableType,
    C::Value: Add<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().clone() + inputs[1].primal().clone();
        // Structural zeros are dropped so the tangent program never stages `Add(zero, ..)`, which the
        // straight-line tangent transposition would reject for having a known (non-linear) operand.
        let left = inputs[0].tangent().as_value().cloned();
        let right = inputs[1].tangent().as_value().cloned();
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'add' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        let tangent = match (left, right) {
            (Some(left), Some(right)) => MaybeZero::Value(
                left.normalize_elementwise_tangent(&target)? + right.normalize_elementwise_tangent(&target)?,
            ),
            (Some(tangent), None) | (None, Some(tangent)) => {
                MaybeZero::Value(tangent.normalize_elementwise_tangent(&target)?)
            }
            (None, None) => MaybeZero::Zero(target),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};

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

        let (value, gradient) = domain
            .value_and_gradient(|(left, right)| left + right, (Scalar::from(2.0), Scalar::from(5.0)))
            .unwrap();
        assert_eq!(value, 7.0);
        assert_eq!(gradient, (Scalar::from(1.0), Scalar::from(1.0)));
    }
}
