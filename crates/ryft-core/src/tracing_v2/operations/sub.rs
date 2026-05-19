use std::ops::Sub;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{SubOperation, SupportsNeg, SupportsSub};
use crate::parameters::Parameter;
use crate::tracing::domains::Tracer;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
use crate::types::Type;

impl<T: Parameter + Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsNeg<T, V>> LinearOperation<T, V, O>
    for SubOperation
where
    SubOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                Ok(vec![Cotangent::Staged(cotangent.clone()), Cotangent::Staged(-cotangent.clone())])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
        }
    }
}

impl<D: DifferentiableDomain> DifferentiableOperation<D> for SubOperation
where
    D::Value: Sub<Output = D::Value>,
    D::LinearOperationCarrier: SupportsSub<D::Type, D::Tangent> + SupportsNeg<D::Type, D::Tangent>,
    SubOperation: Operation<D::Type>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Type, D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Type, D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, TracingError);
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

    use crate::operations::scalars::LinearScalarOperation;
    use crate::tracing::Program;
    use crate::tracing::domains::ScalarDomain;
    use crate::tracing_v2::DifferentiableDomain;
    use crate::types::DataType;

    #[test]
    fn test_sub_jvp_matches_the_difference_rule() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent): (f64, f64) =
            domain.jvp(|(left, right)| left - right, (5.0f64, 2.0f64), (3.0f64, 1.0f64)).unwrap();

        assert_eq!(primal, 3.0);
        assert_eq!(tangent, 2.0);

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
            domain.linearize(|inputs| Ok(inputs.0 - inputs.1), (5.0f64, 2.0f64)).unwrap();

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
