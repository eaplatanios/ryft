use std::ops::{Add, Div, Mul, Neg};

use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::macros::check_count;
use crate::operations::constants::OneLike;
use crate::operations::math::DivOperation;
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use super::broadcasting::ElementwiseDifferentiableValue;

impl<C: Context> DifferentiableOperation<C> for DivOperation
where
    C::Value: OneLike
        + Add<Output = C::Value>
        + Div<Output = C::Value>
        + Mul<Output = C::Value>
        + Neg<Output = C::Value>
        + ElementwiseDifferentiableValue<C::Type>,
    C::Type: DifferentiableType,
    DivOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone() / right.primal().clone();
        let target = primal.r#type().tangent();
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'div' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        // Quotient rule `d(a/b) = (1/b)*da - (a/b²)*db`, with each coefficient built as a fresh primal operation on the
        // primal tracers and then multiplied by the input tangent. Expressing each tangent term as a primal-coefficient
        // `Mul` (rather than a `Div` of the tangent) keeps the tangent map in the bilinear-`Mul` form that both the
        // scalar and array tangent transposers fold into closed constant factors. Zero terms are dropped so the
        // tangent stays minimal.
        let left_term = left
            .tangent()
            .as_value()
            .map(|tangent| -> Result<_, DifferentiationError> {
                let right_primal = right.primal().normalize_elementwise_tangent(&target)?;
                let reciprocal = right_primal.one_like() / right_primal;
                Ok(reciprocal * tangent.normalize_elementwise_tangent(&target)?)
            })
            .transpose()?;
        let right_term = right
            .tangent()
            .as_value()
            .map(|tangent| -> Result<_, DifferentiationError> {
                let left_primal = left.primal().normalize_elementwise_tangent(&target)?;
                let right_primal = right.primal().normalize_elementwise_tangent(&target)?;
                let denominator = right_primal.clone() * right_primal;
                let coefficient = -(left_primal / denominator);
                Ok(coefficient * tangent.normalize_elementwise_tangent(&target)?)
            })
            .transpose()?;
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(target), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Partition-aware transpose rule for [`DivOperation`]. Division is linear in its numerator but nonlinear in its
/// denominator, so in a valid pushforward the numerator is the linear operand and the denominator is a known runtime
/// value (rules such as the logarithm, square-root, and absolute-value forward-mode rules stage exactly this
/// `tangent / known` shape). The transpose of `x ↦ x / k` is `x̄ ↦ x̄ / k` — dividing by a known factor is
/// self-adjoint, like scaling by one — with the known denominator's value read from its pullback value atom. A
/// linear denominator reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error because `k / x`
/// is not a linear map.
impl<V: Value, O: Operation<V::Type> + From<DivOperation>> TransposableOperation<V, O> for DivOperation
where
    DivOperation: Operation<V::Type>,
    V::Type: DifferentiableType,
    Tracer<TracingContext<V, O>>: ElementwiseDifferentiableValue<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        if inputs[1].is_unknown() {
            return Err(ProgramError::UnsupportedOperation {
                message: "`div` with a linear denominator is nonlinear and cannot be transposed".to_string(),
            }
            .into());
        }
        let numerator_type = inputs[0].r#type().cotangent();
        if numerator_type.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: "'div' numerator has no cotangent space".to_string(),
            }
            .into());
        }
        let numerator_contribution = match &outputs[0] {
            MaybeZero::Zero(_) => MaybeZero::Zero(numerator_type),
            MaybeZero::Value(output_cotangent) => {
                // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                let denominator =
                    inputs[1].as_known().expect("dispatch guarantees a known operand carries its pullback value");
                let denominator = denominator.normalize_elementwise_tangent(output_cotangent.r#type().as_ref())?;
                MaybeZero::Value(
                    output_cotangent
                        .binary(&denominator, DivOperation)
                        .unbroadcast_elementwise_cotangent(&numerator_type)?,
                )
            }
        };
        let denominator_type = inputs[1].r#type();
        Ok(vec![numerator_contribution, MaybeZero::Zero(denominator_type.cotangent())])
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::operations::manipulation::ConvertElementType;
    use crate::tracing_v2::ForwardModeDifferentiate;

    #[test]
    fn test_div_jvp_matches_the_quotient_rule() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent): (Scalar, Scalar) = domain
            .jvp(
                |(left, right)| Ok(left / right),
                (Scalar::from(6.0), Scalar::from(2.0)),
                (Scalar::from(3.0), Scalar::from(4.0)),
            )
            .unwrap();

        assert_abs_diff_eq!(primal, 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, -4.5, epsilon = 1e-9);
    }

    #[test]
    fn test_div_jvp_computes_widened_coefficients_in_tangent_type() {
        let left = Scalar::from(4.0f32).convert_element_type(crate::types::DataType::F8E8M0FNU).unwrap();
        let right = Scalar::from(2.0f32).convert_element_type(crate::types::DataType::F8E8M0FNU).unwrap();
        let (primal, tangent): (Scalar, Scalar) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .jvp(|(left, right)| Ok(left / right), (left, right), (Scalar::from(1.0f32), Scalar::from(1.0f32)))
            .unwrap();

        assert_eq!(primal, Scalar::from(2.0f32).convert_element_type(crate::types::DataType::F8E8M0FNU).unwrap());
        assert_eq!(tangent, Scalar::from(-0.5f32));
    }
}
