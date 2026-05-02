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

impl<V: Traceable<ArrayType> + Add<Output = V> + ZeroLike>
    LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>> for AddOperation
{
    fn transpose(
        &self,
        _context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            V,
            LinearArrayOperation<V, ArrayType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<V: Traceable<DataType> + crate::parameters::Parameter + Add<Output = V> + ZeroLike>
    LinearOperation<DataType, V, LinearArrayOperation<V, DataType>> for AddOperation
{
    fn transpose(
        &self,
        _context: &mut crate::tracing::transposition::TranspositionContext<
            DataType,
            V,
            LinearArrayOperation<V, DataType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<E> DifferentiableOperation<E> for AddOperation
where
    E: LinearizableEngine + ?Sized,
    AddOperation: Operation<E::Type>,
    E::Value: Add<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsAdd<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent, inputs[1].tangent],
                <E::LinearOperationCarrier as SupportsAdd<E::Type, E::Value>>::add_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("add jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone() + inputs[1].primal.clone(), tangent }])
    }
}
