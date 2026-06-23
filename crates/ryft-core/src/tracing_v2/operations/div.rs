use std::ops::{Div, Mul, Neg};

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{AddOperation, DivOperation, Scalable, ScaleOperation};
use crate::operations::constants::{MaybeZeroOperation, OneLike, ZeroOperation};
use crate::payloads::Input;
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ValueOrCapture};
use crate::types::Typed;

impl<D> DifferentiableOperation<D> for DivOperation
where
    D: DifferentiationContext,
    DivOperation: Operation<D::Type>,
    D::Value: Clone + Div<Output = D::Value> + Mul<Output = D::Value> + Neg<Output = D::Value> + OneLike,
    LinearOperationOf<D>: MaybeZeroOperation<D::Type>
        + From<AddOperation>
        + From<ScaleOperation<D::Type, ValueOrCapture<D::Type, D::Value>, Input>>
        + From<ZeroOperation<D::Type>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone() / right.primal().clone();
        let left_factor = right.primal().one_like() / right.primal().clone();
        let right_factor = -(left.primal().clone() / (right.primal().clone() * right.primal().clone()));
        let left_term = if context.is_zero(left.tangent())? {
            None
        } else {
            let factor = context.factor(left_factor);
            Some(left.tangent().scale(factor)?)
        };
        let right_term = if context.is_zero(right.tangent())? {
            None
        } else {
            let factor = context.factor(right_factor);
            Some(right.tangent().scale(factor)?)
        };
        let tangent = match (left_term, right_term) {
            (Some(left_term), Some(right_term)) => left_term + right_term,
            (Some(term), None) | (None, Some(term)) => term,
            (None, None) => {
                let mut tangent_outputs =
                    context.stage_nullary_operation(ZeroOperation::new(primal.r#type().into_owned()))?;
                check_count!("output", tangent_outputs, 1, ProgramError);
                tangent_outputs.remove(0)
            }
        };
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::DifferentiationContext;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_div_jvp_matches_the_quotient_rule() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent): (f64, f64) =
            domain.jvp(|(left, right)| left / right, (6.0f64, 2.0f64), (3.0f64, 4.0f64)).unwrap();

        approx_eq(primal, 3.0);
        approx_eq(tangent, -4.5);

        let (_, pushforward) = domain.linearize(|inputs| Ok(inputs.0 / inputs.1), (6.0f64, 2.0f64)).unwrap();
        let pushforward = pushforward.instantiate_program().unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=0.5] %0
                    %3:f64 = scale [factor=-1.5] %1
                    %4:f64 = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }
}
