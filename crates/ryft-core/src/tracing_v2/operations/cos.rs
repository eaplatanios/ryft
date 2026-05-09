use std::ops::Neg;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{SupportsNeg, SupportsScale};
use crate::operations::trigonometric::{Cos, CosOperation, Sin};
use crate::tracing::{AtomId, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};

impl<E> DifferentiableOperation<E> for CosOperation
where
    E: DifferentiableEngine,
    CosOperation: Operation<E::Type>,
    E::Value: Cos + Sin + Neg<Output = E::Value> + Differentiable<E::Type>,
    E::LinearOperationCarrier: SupportsNeg<E::Type, E::Tangent> + SupportsScale<E::Type, E::Tangent, E::Value>,
{
    #[inline]
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let scaled_outputs =
            context.stage(E::LinearOperationCarrier::scale_operation(input.primal.clone().sin()), &[input.tangent])?;
        check_count!("output", scaled_outputs, 1, TracingError);
        let tangent_outputs = context.stage(E::LinearOperationCarrier::neg_operation(), &[scaled_outputs[0]])?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: input.primal.clone().cos(), tangent: tangent_outputs[0] }])
    }
}
