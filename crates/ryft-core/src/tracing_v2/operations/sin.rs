use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::SupportsScale;
use crate::operations::trigonometric::{Cos, Sin, SinOperation};
use crate::tracing::{AtomId, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};

impl<E> DifferentiableOperation<E> for SinOperation
where
    E: DifferentiableEngine,
    SinOperation: Operation<E::Type>,
    E::Value: Sin + Cos + Differentiable<E::Type>,
    E::LinearOperationCarrier: SupportsScale<E::Type, E::Tangent, E::Value>,
{
    #[inline]
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let tangent_outputs =
            context.stage(E::LinearOperationCarrier::scale_operation(input.primal.clone().cos()), &[input.tangent])?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: input.primal.clone().sin(), tangent: tangent_outputs[0] }])
    }
}
