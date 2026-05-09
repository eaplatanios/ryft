use std::ops::{Div, Mul, Neg};

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{DivOperation, SupportsAdd, SupportsScale};
use crate::operations::constants::OneLike;
use crate::tracing::engines::TracingEngine;
use crate::tracing::{AtomId, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};

impl<E> DifferentiableOperation<E> for DivOperation
where
    E: DifferentiableEngine,
    DivOperation: Operation<E::Type>,
    E::Value: Clone
        + Div<Output = E::Value>
        + Mul<Output = E::Value>
        + Neg<Output = E::Value>
        + OneLike
        + Differentiable<E::Type>,
    <E::LinearEngine as TracingEngine>::OperationCarrier: SupportsScale<E::Type, E::Tangent, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let left_factor = right.primal.one_like() / right.primal.clone();
        let right_factor = -(left.primal.clone() / (right.primal.clone() * right.primal.clone()));
        let left_term_outputs = context.stage(
            <E::LinearEngine as TracingEngine>::OperationCarrier::scale_operation(left_factor),
            &[left.tangent],
        )?;
        check_count!("output", left_term_outputs, 1, TracingError);
        let right_term_outputs = context.stage(
            <E::LinearEngine as TracingEngine>::OperationCarrier::scale_operation(right_factor),
            &[right.tangent],
        )?;
        check_count!("output", right_term_outputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <E::LinearEngine as TracingEngine>::OperationCarrier::add_operation(),
            &[left_term_outputs[0], right_term_outputs[0]],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: left.primal.clone() / right.primal.clone(), tangent: tangent_outputs[0] }])
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

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_div_jvp_matches_the_quotient_rule() {
        let engine = ScalarEngine::<f64>::new();
        let (primal, tangent): (f64, f64) =
            engine.jvp(|(left, right)| left / right, (6.0f64, 2.0f64), (3.0f64, 4.0f64)).unwrap();

        approx_eq(primal, 3.0);
        approx_eq(tangent, -4.5);

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
            linearize(&engine, |inputs| Ok(inputs.0 / inputs.1), (6.0f64, 2.0f64)).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=0.5] %0
                    %3:f64 = scale [factor=-1.5] %1
                    %4:f64 = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }
}
