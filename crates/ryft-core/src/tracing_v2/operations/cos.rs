use std::ops::{Mul, Neg};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::check_count;
use crate::operations::math::{Cos, CosOperation, Sin};
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use super::broadcasting::ElementwiseDifferentiableValue;

impl<C: Context> DifferentiableOperation<C> for CosOperation
where
    C::Type: DifferentiableType,
    C::Value: Sin + Cos + Mul<Output = C::Value> + Neg<Output = C::Value> + ElementwiseDifferentiableValue<C::Type>,
    CosOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().cos()?;
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'cos' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        // d(cos x) = -sin(x) * dx, staging a fresh `Sin` primal operation as the coefficient. A structural zero
        // tangent stays symbolic. Normalize the coefficient and live tangent before multiplying because the output
        // tangent descriptor may be wider than the primal representation.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(target),
            MaybeZero::Value(tangent) => {
                let coefficient = input.primal().normalize_elementwise_tangent(&target)?.sin()?;
                MaybeZero::Value(-(coefficient * tangent.normalize_elementwise_tangent(&target)?))
            }
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
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::operations::math::Cos;
    use crate::programs::types::Typed;
    use crate::tests::TestArray;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::types::{ArrayType, DataType};

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

    #[test]
    fn test_cos_jvp_computes_widened_coefficient_in_tangent_type() {
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primal = TestArray::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![4.0]);
        let input_tangent = TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]);

        let (_, tangent) = context.jvp(|input| input.cos(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.values()[0], -3.0 * 4.0f64.sin(), epsilon = 1e-9);
    }
}
