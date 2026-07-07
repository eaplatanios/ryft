use std::ops::{Mul, Neg};

use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::trigonometric::{Cos, CosOperation, Sin};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiableOperation, JvpTracer};

impl<C: Context> DifferentiableOperation<C> for CosOperation
where
    C::Operation: Clone,
    C::Value: Sin + Cos + Mul<Output = C::Value> + Neg<Output = C::Value>,
    CosOperation: Operation<C::Type>,
{
    fn jvp(&self, _context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().cos()?;
        // d(cos x) = -sin(x) * dx, staging a fresh `Sin` primal operation as the coefficient. A structural zero
        // tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(-(input.primal().sin()? * tangent.clone())),
        };
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Transpose rule for [`CosOperation`]: the cosine is nonlinear in its operand, so a tangent program never contains a
/// primal `cos` on a linear operand (the chain-rule forward stages a bilinear `mul` by a fresh negated `sin`
/// coefficient instead) and the rule reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for CosOperation
where
    CosOperation: Operation<V::Type>,
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
    use crate::operations::trigonometric::Cos;
    use crate::scalars::Scalar;
    use crate::tracing_v2::test_util::assert_scalar_close;
    use crate::tracing_v2::{DifferentiationContext, value_and_grad};

    #[test]
    fn test_cos_jvp_and_gradient_scale_by_negated_sine() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain.jvp(|x| x.cos(), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_scalar_close(primal, 2.0f64.cos());
        assert_scalar_close(tangent, -3.0 * 2.0f64.sin());

        let (value, gradient) = value_and_grad(&domain, |x| x.cos().unwrap(), Scalar::from(2.0)).unwrap();
        assert_scalar_close(value, 2.0f64.cos());
        assert_scalar_close(gradient, -2.0f64.sin());
    }
}
