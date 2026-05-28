use std::ops::Mul;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{MulOperation, Scale, SupportsAdd, SupportsScale};
use crate::tracing::TracingError;
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer, LinearOperationCarrier};
use crate::tracing_v2::{Differentiable, DifferentiableOperation};

impl<D> DifferentiableOperation<D> for MulOperation
where
    D: Differentiable,
    MulOperation: Operation<D::Type>,
    D::Value: Mul<Output = D::Value>,
    LinearOperationCarrier<D>: SupportsAdd<D::Type, D::Tangent> + SupportsScale<D::Type, D::Tangent, D::Value>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        Ok(vec![JvpTracer::new(
            left.primal().clone() * right.primal().clone(),
            left.tangent().clone().scale(right.primal().clone()) + right.tangent().clone().scale(left.primal().clone()),
        )])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::scalars::LinearScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::tracing::Program;
    use crate::tracing::domains::ScalarDomain;
    use crate::tracing_v2::DifferentiableDomain;
    use crate::types::DataType;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) = domain.jvp(|(left, right)| left * right, (2.0f64, 5.0f64), (3.0f64, -1.0f64)).unwrap();

        approx_eq(primal, 10.0);
        approx_eq(tangent, 13.0);

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) = domain
            .linearize(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))
            .unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=3] %0
                    %3:f64 = scale [factor=2] %1
                    %4:f64 = add %2 %3
                    %5:f64 = scale [factor=-0.4161468365471424] %0
                    %6:f64 = add %4 %5
                in (%6)
            "}
            .trim_end(),
        );
    }
}
