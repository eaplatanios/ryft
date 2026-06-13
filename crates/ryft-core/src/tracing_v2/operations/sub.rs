use std::ops::Sub;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{SubOperation, SupportsNeg, SupportsSub};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::Type;

impl<T: Parameter + Type, V: Value<T>, O: Operation<T> + SupportsNeg<T>> TransposableOperation<T, V, O> for SubOperation
where
    SubOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
        }
    }
}

impl<D: DifferentiationContext> DifferentiableOperation<D> for SubOperation
where
    D::Value: Sub<Output = D::Value>,
    LinearOperationOf<D>: SupportsSub<D::Type> + SupportsNeg<D::Type>,
    SubOperation: Operation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![JvpTracer::new(
            inputs[0].primal().clone() - inputs[1].primal().clone(),
            inputs[0].tangent().clone() - inputs[1].tangent().clone(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::DifferentiationContext;

    #[test]
    fn test_sub_jvp_matches_the_difference_rule() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent): (f64, f64) =
            domain.jvp(|(left, right)| left - right, (5.0f64, 2.0f64), (3.0f64, 1.0f64)).unwrap();

        assert_eq!(primal, 3.0);
        assert_eq!(tangent, 2.0);

        let (_, pushforward) = domain.linearize(|inputs| Ok(inputs.0 - inputs.1), (5.0f64, 2.0f64)).unwrap();
        let pushforward = pushforward.instantiate_program().unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = sub %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
