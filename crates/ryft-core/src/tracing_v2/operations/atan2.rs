use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError, TransposableOperation,
    TranspositionDriver,
};
use crate::macros::check_count;
use crate::operations::math::{Atan2, Atan2Operation};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::Typed;

impl<C: Context> DifferentiableOperation<C> for Atan2Operation
where
    C::Value: Atan2
        + Add<Output = C::Value>
        + Sub<Output = C::Value>
        + Mul<Output = C::Value>
        + Div<Output = C::Value>
        + Neg<Output = C::Value>,
    Atan2Operation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let y = &inputs[0];
        let x = &inputs[1];
        let primal = y.primal().atan2(x.primal())?;
        // d(atan2(y, x)) = (x · dy - y · dx) / (x² + y²). Zero terms are dropped so the staged numerator stays as
        // small as its live tangents, and the tangent stays a symbolic zero when both input tangents are zeros.
        let y_term = y.tangent().as_value().map(|tangent| x.primal().clone() * tangent.clone());
        let x_term = x.tangent().as_value().map(|tangent| y.primal().clone() * tangent.clone());
        let numerator = match (y_term, x_term) {
            (None, None) => None,
            (Some(y_term), None) => Some(y_term),
            (None, Some(x_term)) => Some(-x_term),
            (Some(y_term), Some(x_term)) => Some(y_term - x_term),
        };
        let tangent = match numerator {
            None => MaybeZero::Zero(primal.r#type().into_owned()),
            Some(numerator) => {
                let denominator = x.primal().clone() * x.primal().clone() + y.primal().clone() * y.primal().clone();
                MaybeZero::Value(numerator / denominator)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`Atan2Operation`]: the two-argument arc tangent is nonlinear in its operands, so a
/// tangent program never contains a primal `atan2` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for Atan2Operation
where
    Atan2Operation: Operation<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
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

    use crate::backends::scalars::Scalar;
    use crate::operations::complex::{Imaginary, Real};
    use crate::operations::math::Atan2;
    use crate::tracing_v2::{gradient, value_and_gradient};

    #[test]
    fn test_atan2_gradient_matches_the_analytic_derivatives() {
        // d(atan2(y, x))/dy = x / (x² + y²) and d(atan2(y, x))/dx = -y / (x² + y²).
        let (y, x): (f64, f64) = (0.7, -0.3);
        let (value, gradient_value) =
            value_and_gradient(|(y, x)| y.atan2(&x).unwrap(), (Scalar::from(y), Scalar::from(x))).unwrap();
        assert_abs_diff_eq!(value, y.atan2(x), epsilon = 1e-9);
        let (y_gradient, x_gradient) = gradient_value;
        assert_abs_diff_eq!(y_gradient, x / (x * x + y * y), epsilon = 1e-9);
        assert_abs_diff_eq!(x_gradient, -y / (x * x + y * y), epsilon = 1e-9);
    }

    #[test]
    fn test_angle_is_the_atan_of_the_complex_parts() {
        // The angle (i.e., argument) of a complex value is not a primitive: it is the composition
        // `atan2(imaginary(z), real(z))`, which is a ℂ → ℝ function and so flows through the plain gradient
        // entry point with the same conjugate convention as `∇|z|² = 2z̄`.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let angle = |z: crate::differentiation::LinearizationTracer<
            crate::contexts::EagerContext<Scalar, crate::backends::scalars::ScalarOperation<Scalar>>,
        >| { z.imaginary().unwrap().atan2(&z.real().unwrap()).unwrap() };
        let (value, gradient_value) = value_and_gradient(angle, Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.im.atan2(z.re)));
        // With ∂θ/∂re = -im/|z|² and ∂θ/∂im = re/|z|², the `real`/`imaginary` transposes assemble the complex input
        // cotangent as `complex(∂θ/∂re, -∂θ/∂im) = (-im - re·i) / |z|² = -i·z̄ / |z|²` (the imaginary-part transpose
        // carries the negation of the bilinear pairing).
        let expected = ComplexNumber::new(-z.im, -z.re) / z.norm_sqr();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - expected).norm() < 1e-12, "expected {expected} but got {actual}");

        // Gradient-only form agrees.
        let gradient_value = gradient(angle, Scalar::from(z)).unwrap();
        assert_eq!(gradient_value, Scalar::C128(actual));
    }
}
