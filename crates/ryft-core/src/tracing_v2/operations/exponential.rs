use std::ops::{Add, Div, Mul};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError, TransposableOperation,
    TranspositionDriver,
};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::math::{Exp, ExpOperation, Log, LogOperation, Sqrt, SqrtOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

impl<C: Context> DifferentiableOperation<C> for ExpOperation
where
    C::Value: Exp + Mul<Output = C::Value>,
    ExpOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().exp()?;
        // d(eˣ) = eˣ · dx, reusing the primal result as the coefficient (this also holds for the complex analytic
        // continuation). A structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(primal.clone() * tangent.clone()),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`ExpOperation`]: the exponential is nonlinear in its operand, so a tangent program
/// never contains a primal `exp` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for ExpOperation
where
    ExpOperation: Operation<V::Type>,
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

impl<C: Context> DifferentiableOperation<C> for LogOperation
where
    C::Value: Log + Div<Output = C::Value>,
    LogOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().log()?;
        // d(ln x) = dx / x (this also holds for the principal branch of the complex logarithm away from its cut). A
        // structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.clone() / input.primal().clone()),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`LogOperation`]: the logarithm is nonlinear in its operand, so a tangent program never
/// contains a primal `log` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for LogOperation
where
    LogOperation: Operation<V::Type>,
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

impl<C: Context> DifferentiableOperation<C> for SqrtOperation
where
    C::Value: Sqrt + Add<Output = C::Value> + Div<Output = C::Value>,
    SqrtOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().sqrt()?;
        // d(√x) = dx / (2 · √x), reusing the primal result in the denominator (this also holds for the principal
        // branch of the complex square root away from its cut). A structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.clone() / (primal.clone() + primal.clone())),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`SqrtOperation`]: the square root is nonlinear in its operand, so a tangent program
/// never contains a primal `sqrt` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for SqrtOperation
where
    SqrtOperation: Operation<V::Type>,
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
    use crate::operations::math::{Exp, Log, Sqrt};
    use crate::tracing_v2::{ReverseModeDifferentiate, gradient, value_and_gradient_holomorphic};

    #[test]
    fn test_exponential_family_gradients() {
        // Real gradients: d(eˣ) = eˣ, d(ln x) = 1/x, and d(√x) = 1/(2√x).
        let x: f64 = 0.7;
        let gradient_value = gradient(|x| x.exp().unwrap(), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(gradient_value, x.exp(), epsilon = 1e-9);
        let gradient_value = gradient(|x| x.log().unwrap(), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(gradient_value, 1.0 / x, epsilon = 1e-9);
        let gradient_value = gradient(|x| x.sqrt().unwrap(), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(gradient_value, 0.5 / x.sqrt(), epsilon = 1e-9);

        // Holomorphic gradients at a genuinely complex point: the same rules compute the analytic continuations
        // d(e^z) = e^z, d(ln z) = 1/z, and d(√z) = 1/(2√z).
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) = value_and_gradient_holomorphic(|x| x.exp().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.exp()));
        assert_eq!(gradient_value, Scalar::from(z.exp()));
        let gradient_value = crate::tracing_v2::gradient_holomorphic(|x| x.log().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(gradient_value, Scalar::from(ComplexNumber::new(1.0, 0.0) / z));
        let gradient_value = crate::tracing_v2::gradient_holomorphic(|x| x.sqrt().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(gradient_value, Scalar::from(ComplexNumber::new(1.0, 0.0) / (z.sqrt() + z.sqrt())));

        // The composition log(exp(z)) has unit derivative, exercising the chain rule through both rules under an
        // explicit context (up to the rounding of the pullback's `e/e` complex division).
        let domain = crate::contexts::EagerContext::<Scalar, crate::backends::scalars::ScalarOperation<Scalar>>::new();
        let gradient_value = domain.gradient_holomorphic(|x| x.exp().unwrap().log().unwrap(), Scalar::from(z)).unwrap();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - ComplexNumber::new(1.0, 0.0)).norm() < 1e-12, "expected a unit derivative but got {actual}",);
    }
}
