use std::ops::{Mul, Neg};

use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::differentiation::{DifferentiableOperation, DifferentiationDual, DifferentiationError};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::trigonometric::{Cos, CosOperation, Sin};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

impl<C: Context> DifferentiableOperation<C> for CosOperation
where
    C::Operation: Clone,
    C::Value: Sin + Cos + Mul<Output = C::Value> + Neg<Output = C::Value>,
    CosOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().cos()?;
        // d(cos x) = -sin(x) * dx, staging a fresh `Sin` primal operation as the coefficient. A structural zero
        // tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(-(input.primal().sin()? * tangent.clone())),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
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
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no partition-aware transpose rule", self.name()),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::backends::scalars::Scalar;
    use crate::backends::scalars::ScalarOperation;
    use crate::contexts::EagerContext;
    use crate::operations::trigonometric::Cos;
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};

    #[test]
    fn test_cos_jvp_and_gradient_scale_by_negated_sine() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain.jvp(|x| x.cos(), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_abs_diff_eq!(primal, 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, -3.0 * 2.0f64.sin(), epsilon = 1e-9);

        let (value, gradient) = domain.value_and_gradient(|x| x.cos().unwrap(), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, -2.0f64.sin(), epsilon = 1e-9);
    }
}
