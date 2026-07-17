use std::ops::{Add, Div, Mul};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::check_count;
use crate::operations::math::{Exp, ExpOperation, Log, LogOperation, Sqrt, SqrtOperation};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use super::broadcasting::ElementwiseDifferentiableValue;

impl<C: Context> DifferentiableOperation<C> for ExpOperation
where
    C::Type: DifferentiableType,
    C::Value: Exp + Mul<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
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
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'exp' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        // d(eˣ) = eˣ · dx (this also holds for the complex analytic continuation). A structural zero tangent
        // stays symbolic. Compute the coefficient from the input normalized to the output tangent descriptor before
        // multiplying so a wider differential representation does not inherit primal rounding or range limitations.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(target),
            MaybeZero::Value(tangent) => {
                let coefficient = input.primal().normalize_elementwise_tangent(&target)?.exp()?;
                MaybeZero::Value(coefficient * tangent.normalize_elementwise_tangent(&target)?)
            }
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
    C::Type: DifferentiableType,
    C::Value: Log + Div<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
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
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'log' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        // d(ln x) = dx / x (this also holds for the principal branch of the complex logarithm away from its cut). A
        // structural zero tangent stays symbolic. Normalize the tangent and primal denominator before dividing because
        // the output tangent descriptor may be wider than the primal representation.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(target),
            MaybeZero::Value(tangent) => MaybeZero::Value(
                tangent.normalize_elementwise_tangent(&target)?
                    / input.primal().normalize_elementwise_tangent(&target)?,
            ),
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
    C::Type: DifferentiableType,
    C::Value: Sqrt + Add<Output = C::Value> + Div<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
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
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'sqrt' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        // d(√x) = dx / (2 · √x) (this also holds for the principal branch of the complex square root away from
        // its cut). A structural zero tangent stays symbolic. Compute the denominator from the input normalized to the
        // output tangent descriptor so a wider differential representation does not inherit primal rounding.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(target),
            MaybeZero::Value(tangent) => {
                let denominator = input.primal().normalize_elementwise_tangent(&target)?.sqrt()?;
                MaybeZero::Value(tangent.normalize_elementwise_tangent(&target)? / (denominator.clone() + denominator))
            }
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
    use crate::contexts::EagerContext;
    use crate::operations::math::{Exp, Log, Sqrt};
    use crate::programs::types::Typed;
    use crate::tests::TestArray;
    use crate::tracing_v2::{
        ArrayOperation, ForwardModeDifferentiate, ReverseModeDifferentiate, gradient, value_and_gradient_holomorphic,
    };
    use crate::types::{ArrayType, DataType};

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

    #[test]
    fn test_exponential_family_jvps_compute_in_widened_tangent_type() {
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primal = TestArray::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]);

        let (_, exponential_tangent) = context.jvp(|input| input.exp(), primal.clone(), input_tangent.clone()).unwrap();
        assert_eq!(exponential_tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(exponential_tangent.values()[0], 3.0 * 2.0f64.exp(), epsilon = 1e-9);

        let (_, logarithm_tangent) = context.jvp(|input| input.log(), primal.clone(), input_tangent.clone()).unwrap();
        assert_eq!(logarithm_tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(logarithm_tangent.values()[0], 1.5, epsilon = 1e-9);

        let (_, square_root_tangent) = context.jvp(|input| input.sqrt(), primal, input_tangent).unwrap();
        assert_eq!(square_root_tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(square_root_tangent.values()[0], 3.0 / (2.0 * 2.0f64.sqrt()), epsilon = 1e-9);
    }
}
