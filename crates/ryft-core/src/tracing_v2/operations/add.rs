use std::ops::Add;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::tracing::TranspositionContext;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::Type;

impl<T: PartialEq + Type, V: Traceable<T>, O: Clone + Operation<T>> LinearOperation<T, V, O> for AddOperation
where
    AddOperation: Operation<T>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<E: DifferentiableEngine> DifferentiableOperation<E> for AddOperation
where
    E::Value: Add<Output = E::Value> + Differentiable<E::Type>,
    <E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier:
        SupportsAdd<E::Type, E::Tangent>,
    AddOperation: Operation<E::Type>,
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
            <<E::LinearEngine as crate::tracing_v2::LinearizableEngine>::LinearOperationCarrier as SupportsAdd<
                E::Type,
                E::Tangent,
            >>::add_operation(),
            tangent_inputs,
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone() + inputs[1].primal.clone(), tangent: tangent_outputs[0] }])
    }
}
