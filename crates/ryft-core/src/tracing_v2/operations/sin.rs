use std::ops::Mul;

use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::differentiation::{DifferentiableOperation, DifferentiationDual};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::trigonometric::{Cos, Sin, SinOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

impl<C: Context> DifferentiableOperation<C> for SinOperation
where
    C::Operation: Clone,
    C::Value: Sin + Cos + Mul<Output = C::Value>,
    SinOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().sin()?;
        // d(sin x) = cos(x) * dx, staging a fresh `Cos` primal operation as the coefficient. A structural zero
        // tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(input.primal().cos()? * tangent.clone()),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`SinOperation`]: the sine is nonlinear in its operand, so a tangent program never contains a
/// primal `sin` on a linear operand (the chain-rule forward stages a bilinear `mul` by a fresh `cos` coefficient
/// instead) and the rule reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for SinOperation
where
    SinOperation: Operation<V::Type>,
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
    use crate::operations::trigonometric::Sin;
    use crate::scalars::Scalar;
    use crate::tracing_v2::test_util::assert_scalar_close;
    use crate::tracing_v2::{Differentiate, value_and_gradient};

    #[test]
    fn test_sin_jvp_and_gradient_scale_by_cosine() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain.jvp(|x| x.sin(), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_scalar_close(primal, 2.0f64.sin());
        assert_scalar_close(tangent, 3.0 * 2.0f64.cos());

        let (value, gradient) = value_and_gradient(&domain, |x| x.sin().unwrap(), Scalar::from(2.0)).unwrap();
        assert_scalar_close(value, 2.0f64.sin());
        assert_scalar_close(gradient, 2.0f64.cos());
    }
}
