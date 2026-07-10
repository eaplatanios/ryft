use std::ops::{Add, Div, Mul};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDual, DifferentiationError, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::exponential::{
    Exponential, ExponentialOperation, Logarithm, LogarithmOperation, SquareRoot, SquareRootOperation,
};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

impl<C: Context> DifferentiableOperation<C> for ExponentialOperation
where
    C::Operation: Clone,
    C::Value: Exponential + Mul<Output = C::Value>,
    ExponentialOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().exponential()?;
        // d(eˣ) = eˣ · dx, reusing the primal result as the coefficient (this also holds for the complex analytic
        // continuation). A structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(primal.clone() * tangent.clone()),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`ExponentialOperation`]: the exponential is nonlinear in its operand, so a tangent program
/// never contains a primal `exponential` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for ExponentialOperation
where
    ExponentialOperation: Operation<V::Type>,
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

impl<C: Context> DifferentiableOperation<C> for LogarithmOperation
where
    C::Operation: Clone,
    C::Value: Logarithm + Div<Output = C::Value>,
    LogarithmOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().logarithm()?;
        // d(ln x) = dx / x (this also holds for the principal branch of the complex logarithm away from its cut). A
        // structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.clone() / input.primal().clone()),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`LogarithmOperation`]: the logarithm is nonlinear in its operand, so a tangent program never
/// contains a primal `logarithm` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for LogarithmOperation
where
    LogarithmOperation: Operation<V::Type>,
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

impl<C: Context> DifferentiableOperation<C> for SquareRootOperation
where
    C::Operation: Clone,
    C::Value: SquareRoot + Add<Output = C::Value> + Div<Output = C::Value>,
    SquareRootOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().square_root()?;
        // d(√x) = dx / (2 · √x), reusing the primal result in the denominator (this also holds for the principal
        // branch of the complex square root away from its cut). A structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.clone() / (primal.clone() + primal.clone())),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`SquareRootOperation`]: the square root is nonlinear in its operand, so a tangent program
/// never contains a primal `square_root` on a linear operand and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for SquareRootOperation
where
    SquareRootOperation: Operation<V::Type>,
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

    use crate::backends::scalars::Scalar;
    use crate::operations::exponential::{Exponential, Logarithm, SquareRoot};
    use crate::tracing_v2::{ReverseModeDifferentiate, gradient, value_and_gradient_holomorphic};

    #[test]
    fn test_exponential_family_gradients() {
        // Real gradients: d(eˣ) = eˣ, d(ln x) = 1/x, and d(√x) = 1/(2√x).
        let x: f64 = 0.7;
        let gradient_value = gradient(|x| x.exponential().unwrap(), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(gradient_value, x.exp(), epsilon = 1e-9);
        let gradient_value = gradient(|x| x.logarithm().unwrap(), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(gradient_value, 1.0 / x, epsilon = 1e-9);
        let gradient_value = gradient(|x| x.square_root().unwrap(), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(gradient_value, 0.5 / x.sqrt(), epsilon = 1e-9);

        // Holomorphic gradients at a genuinely complex point: the same rules compute the analytic continuations
        // d(e^z) = e^z, d(ln z) = 1/z, and d(√z) = 1/(2√z).
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) =
            value_and_gradient_holomorphic(|x| x.exponential().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.exp()));
        assert_eq!(gradient_value, Scalar::from(z.exp()));
        let gradient_value =
            crate::tracing_v2::gradient_holomorphic(|x| x.logarithm().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(gradient_value, Scalar::from(ComplexNumber::new(1.0, 0.0) / z));
        let gradient_value =
            crate::tracing_v2::gradient_holomorphic(|x| x.square_root().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(gradient_value, Scalar::from(ComplexNumber::new(1.0, 0.0) / (z.sqrt() + z.sqrt())));

        // The composition log(exp(z)) has unit derivative, exercising the chain rule through both rules under an
        // explicit context (up to the rounding of the pullback's `e/e` complex division).
        let domain =
            crate::contexts::EagerContext::<Scalar, crate::operations::scalars::ScalarOperation<Scalar>>::new();
        let gradient_value = domain
            .gradient_holomorphic(|x| x.exponential().unwrap().logarithm().unwrap(), Scalar::from(z))
            .unwrap();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - ComplexNumber::new(1.0, 0.0)).norm() < 1e-12, "expected a unit derivative but got {actual}",);
    }
}
