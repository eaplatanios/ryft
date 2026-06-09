use std::ops::{Div, Mul, Neg};

use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{DivOperation, Scale, SupportsAdd, SupportsScale};
use crate::operations::constants::OneLike;
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, ResidualFactor, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};

impl<D> DifferentiableOperation<D> for DivOperation
where
    D: DifferentiationContext,
    DivOperation: Operation<D::Type>,
    D::Value: Clone + Div<Output = D::Value> + Mul<Output = D::Value> + Neg<Output = D::Value> + OneLike,
    LinearOperationOf<D>: SupportsAdd<D::Type> + SupportsScale<D::Type, ResidualFactor<D::Type, D::Value>>,
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
        let left_factor = right.primal().one_like() / right.primal().clone();
        let right_factor = -(left.primal().clone() / (right.primal().clone() * right.primal().clone()));
        let left_term = match left.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(tangent.scale(context.factor(left_factor))),
        };
        let right_term = match right.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(tangent.scale(context.factor(right_factor))),
        };
        Ok(vec![JvpTracer::new(left.primal().clone() / right.primal().clone(), left_term + right_term)])
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

        let linearized = domain.linearize(|inputs| Ok(inputs.0 / inputs.1), (6.0f64, 2.0f64)).unwrap();
        let (_, pushforward) = linearized.into_parts();
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
