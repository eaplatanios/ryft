use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scalable, ScaleOperation};
use crate::operations::constants::MaybeZeroOperation;
use crate::operations::trigonometric::{Cos, Sin, SinOperation};
use crate::payloads::Input;
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ValueOrCapture};

impl<D> DifferentiableOperation<D> for SinOperation
where
    D: DifferentiationContext,
    SinOperation: Operation<D::Type>,
    D::Value: Sin + Cos,
    LinearOperationOf<D>: From<ScaleOperation<D::Type, ValueOrCapture<D::Type, D::Value>, Input>>,
    LinearOperationOf<D>: MaybeZeroOperation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let tangent = if context.is_zero(input.tangent())? {
            input.tangent().clone()
        } else {
            let factor = context.factor(input.primal().clone().cos());
            input.tangent().scale(factor)?
        };
        Ok(vec![JvpTracer::new(input.primal().clone().sin(), tangent)])
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::trigonometric::Sin;
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::{DifferentiationContext, value_and_grad};

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_sin_jvp_and_gradient_scale_by_cosine() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) = domain.jvp(|x| x.sin(), 2.0f64, 3.0f64).unwrap();
        approx_eq(primal, 2.0f64.sin());
        approx_eq(tangent, 3.0 * 2.0f64.cos());

        let (value, gradient) = value_and_grad(&domain, |x| x.sin(), 2.0f64).unwrap();
        approx_eq(value, 2.0f64.sin());
        approx_eq(gradient, 2.0f64.cos());
    }
}
