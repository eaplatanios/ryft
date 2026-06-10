use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::stop_gradient::StopGradientOperation;
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::Typed;

/// JVP rule for [`StopGradientOperation`]: the primal passes through unchanged and the tangent is
/// replaced with a symbolic [`Tangent::Zero`], severing derivative flow in both forward and
/// reverse mode. The rule stages no linear operation, so `stop_gradient` never appears in a
/// pushforward program and needs no transpose rule.
impl<D: DifferentiationContext> DifferentiableOperation<D> for StopGradientOperation
where
    StopGradientOperation: Operation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone();
        let tangent = Tangent::Zero(primal.r#type().into_owned());
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::stop_gradient::StopGradient;
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::{DifferentiationContext, value_and_grad};

    #[test]
    fn test_stop_gradient_jvp_severs_the_tangent() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) = domain.jvp(|x| x.stop_gradient(), 2.0f64, 3.0f64).unwrap();
        assert_eq!(primal, 2.0);
        assert_eq!(tangent, 0.0);
    }

    #[test]
    fn test_stop_gradient_composes_with_batch() {
        use crate::tracing_v2::Batch;
        use crate::tracing_v2::test_util::{TestArray, TestArrayDomain};

        let output: TestArray = TestArrayDomain
            .batch(
                |x| Ok(x.clone() * x.stop_gradient()),
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_stop_gradient_treats_the_marked_value_as_a_constant() {
        // The JAX documentation example: `f(x) = x * stop_gradient(x)` differentiates like
        // `x * c` with `c` frozen at the primal value, so `f'(x) = stop_gradient(x)`.
        let domain = ScalarDomain::<f64>::new();
        let (value, gradient) = value_and_grad(&domain, |x| x.clone() * x.stop_gradient(), 3.0f64).unwrap();
        assert_eq!(value, 9.0);
        assert_eq!(gradient, 3.0);
    }
}
