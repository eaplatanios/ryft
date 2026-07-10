use std::ops::{Div, Mul};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDual, DifferentiationError, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Abs, AbsOperation};
use crate::operations::complex::{Conjugate, Real};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{Type, Typed};

impl<C: Context> DifferentiableOperation<C> for AbsOperation
where
    C::Operation: Clone,
    C::Value: Abs + Conjugate + Real + Mul<Output = C::Value> + Div<Output = C::Value>,
    AbsOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().abs()?;
        // The real derivative is d|x| = (x / |x|) · dx (i.e., sign(x) · dx, undefined at zero), and the complex
        // magnitude is a ℂ → ℝ map with d|z| = Re(z̄ · dz) / |z|, so the complex branch routes the tangent through
        // `conjugate` and `real` while the real branch reuses the primal directly. A structural zero tangent stays
        // symbolic, retyped to the (real) output type.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => {
                let numerator = if input.primal().r#type().is_complex() {
                    (input.primal().conjugate()? * tangent.clone()).real()?
                } else {
                    input.primal().clone() * tangent.clone()
                };
                MaybeZero::Value(numerator / primal.clone())
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`AbsOperation`]: the absolute value is nonlinear in its operand, so a tangent
/// program never contains a primal `abs` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for AbsOperation
where
    AbsOperation: Operation<V::Type>,
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
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::operations::arithmetic::Abs;
    use crate::scalars::Scalar;
    use crate::tracing_v2::{gradient, value_and_gradient};

    #[test]
    fn test_abs_gradient_is_the_sign_for_real_operands() {
        // d|x| = sign(x): +1 above zero and -1 below.
        let gradient_value = gradient(|x| x.abs().unwrap(), Scalar::from(0.7f64)).unwrap();
        assert_abs_diff_eq!(gradient_value, 1.0, epsilon = 1e-9);
        let gradient_value = gradient(|x| x.abs().unwrap(), Scalar::from(-0.7f64)).unwrap();
        assert_abs_diff_eq!(gradient_value, -1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_abs_gradient_of_complex_operands_is_the_normalized_conjugate() {
        // |z| is a ℂ → ℝ function, so it flows through the plain gradient entry point. With d|z| = Re(z̄ · dz) / |z|,
        // the bilinear-pairing gradient is z̄ / |z| (the unit-magnitude conjugate direction) — the reverse-mode
        // counterpart of `∇|z|² = 2z̄` after the chain rule through the square root.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) = value_and_gradient(|z| z.abs().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.norm()));
        let expected = z.conj() / z.norm();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - expected).norm() < 1e-12, "expected {expected} but got {actual}");
    }
}
