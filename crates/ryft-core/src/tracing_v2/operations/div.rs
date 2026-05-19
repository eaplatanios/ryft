use std::ops::{Div, Mul, Neg};

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{DivOperation, Scale, SupportsScale};
use crate::operations::constants::OneLike;
use crate::tracing::TracingError;
use crate::tracing::domains::Tracer;
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};

impl<D> DifferentiableOperation<D> for DivOperation
where
    D: DifferentiableDomain,
    DivOperation: Operation<D::Type>,
    D::Value: Clone + Div<Output = D::Value> + Mul<Output = D::Value> + Neg<Output = D::Value> + OneLike,
    D::LinearOperationCarrier: SupportsScale<D::Type, D::Tangent, D::Value>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let left_factor = right.primal().one_like() / right.primal().clone();
        let right_factor = -(left.primal().clone() / (right.primal().clone() * right.primal().clone()));
        Ok(vec![JvpTracer::new(
            left.primal().clone() / right.primal().clone(),
            left.tangent().clone().scale(left_factor) + right.tangent().clone().scale(right_factor),
        )])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::scalars::LinearScalarOperation;
    use crate::tracing::Program;
    use crate::tracing::domains::ScalarDomain;
    use crate::tracing_v2::DifferentiableDomain;
    use crate::types::DataType;

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

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
            domain.linearize(|inputs| Ok(inputs.0 / inputs.1), (6.0f64, 2.0f64)).unwrap();

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
