use std::ops::Add;

use crate::macros::check_input_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::ZeroLike;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearArrayOperation, LinearizableEngine};
use crate::types::{ArrayType, DataType};
use crate::{Parameter, TranspositionContext};

impl<V: Traceable<ArrayType> + Add<Output = V> + ZeroLike>
    LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>> for AddOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<V: Traceable<DataType> + Parameter + Add<Output = V> + ZeroLike>
    LinearOperation<DataType, V, LinearArrayOperation<V, DataType>> for AddOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<DataType, V, LinearArrayOperation<V, DataType>>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<
    E: LinearizableEngine<
            Value: Add<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
            LinearOperationCarrier: SupportsAdd<E::Type, E::Value>,
        > + ?Sized,
> DifferentiableOperation<E> for AddOperation
where
    AddOperation: Operation<E::Type>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.clone() + inputs[1].primal.clone(),
            tangent: context
                .stage(E::LinearOperationCarrier::add_operation(), &[inputs[0].tangent, inputs[1].tangent])?
                .into_iter()
                .next()
                .expect("add jvp should produce one tangent"),
        }])
    }
}
