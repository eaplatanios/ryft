use std::ops::Sub;

use crate::TracingEngine;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{SubOperation, SupportsNeg, SupportsSub};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError, TranspositionContext};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::Type;

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsNeg<T, V>> LinearOperation<T, V, O> for SubOperation
where
    SubOperation: Operation<T>,
{
    #[inline]
    fn transpose(
        &self,
        context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let negated_outputs = context.stage(O::neg_operation(), &[atom])?;
                check_count!("output", negated_outputs, 1, TracingError);
                Ok(vec![Some(atom), Some(negated_outputs[0])])
            }
            None => Ok(vec![None, None]),
        }
    }
}

impl<E: DifferentiableEngine> DifferentiableOperation<E> for SubOperation
where
    E::Value: Sub<Output = E::Value> + Differentiable<E::Type>,
    <E::LinearEngine as TracingEngine>::OperationCarrier: SupportsSub<E::Type, E::Tangent>,
    SubOperation: Operation<E::Type>,
{
    #[inline]
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        let tangent_inputs = &[inputs[0].tangent, inputs[1].tangent];
        let tangent_outputs = context.stage(
            <<E::LinearEngine as TracingEngine>::OperationCarrier as SupportsSub<E::Type, E::Tangent>>::sub_operation(),
            tangent_inputs,
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone() - inputs[1].primal.clone(), tangent: tangent_outputs[0] }])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::scalars::LinearScalarOperation;
    use crate::tracing::Program;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::{DifferentiableEngine, linearize};
    use crate::types::DataType;

    #[test]
    fn test_sub_jvp_matches_the_difference_rule() {
        let engine = ScalarEngine::<f64>::new();
        let (primal, tangent): (f64, f64) =
            engine.jvp(|(left, right)| left - right, (5.0f64, 2.0f64), (3.0f64, 1.0f64)).unwrap();

        assert_eq!(primal, 3.0);
        assert_eq!(tangent, 2.0);

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
            linearize(&engine, |inputs| Ok(inputs.0 - inputs.1), (5.0f64, 2.0f64)).unwrap();

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
