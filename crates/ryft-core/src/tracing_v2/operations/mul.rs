use std::ops::Mul;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{MulOperation, SupportsAdd};
use crate::tracing::{AtomId, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};

use super::SupportsScale;

impl<E> DifferentiableOperation<E> for MulOperation
where
    E: DifferentiableEngine,
    MulOperation: Operation<E::Type>,
    E::Value: Mul<Output = E::Value> + Differentiable<E::Type>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier:
        SupportsScale<E::Type, E::Tangent, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let left_term_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsScale<
                E::Type,
                E::Tangent,
                E::Value,
            >>::scale_operation(right.primal.clone()),
            &[left.tangent],
        )?;
        check_count!("output", left_term_outputs, 1, TracingError);
        let right_term_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsScale<
                E::Type,
                E::Tangent,
                E::Value,
            >>::scale_operation(left.primal.clone()),
            &[right.tangent],
        )?;
        check_count!("output", right_term_outputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsAdd<
                E::Type,
                E::Tangent,
            >>::add_operation(),
            &[left_term_outputs[0], right_term_outputs[0]],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: left.primal.clone() * right.primal.clone(), tangent: tangent_outputs[0] }])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::tracing::Program;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::{DifferentiableEngine, LinearScalarOperation, Sin, linearize};
    use crate::types::DataType;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let engine = ScalarEngine::<f64>::new();
        let (primal, tangent) = engine.jvp(|(left, right)| left * right, (2.0f64, 5.0f64), (3.0f64, -1.0f64)).unwrap();

        approx_eq(primal, 10.0);
        approx_eq(tangent, 13.0);

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
            linearize(&engine, |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64)).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=3] %0
                    %3:f64 = scale [factor=2] %1
                    %4:f64 = add %2 %3
                    %5:f64 = scale [factor=-0.4161468365471424] %0
                    %6:f64 = add %4 %5
                in (%6)
            "}
            .trim_end(),
        );
    }
}
