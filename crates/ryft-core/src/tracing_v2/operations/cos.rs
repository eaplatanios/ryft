use std::ops::Neg;

use crate::macros::check_count;
use crate::operations::arithmetic::{Scale, SupportsNeg, SupportsScale};
use crate::operations::trigonometric::{Cos, CosOperation, Sin};
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::domains::Tracer;
use crate::tracing::{Traceable, TracingError};
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingOutput, batch_elementwise, lift_elementwise_output,
};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
use crate::types::ArrayType;

impl<V> BatchableOperation<V> for CosOperation
where
    V: Traceable<ArrayType>,
    CosOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        batch_elementwise(self, inputs)
    }

    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingOutput<Self>, TracingError> {
        lift_elementwise_output(self, input_types, input_axes, axis_size)
    }
}

impl<D> DifferentiableOperation<D> for CosOperation
where
    D: DifferentiableDomain,
    CosOperation: Operation<D::Type>,
    D::Value: Cos + Sin + Neg<Output = D::Value>,
    D::LinearOperationCarrier: SupportsNeg<D::Type, D::Tangent> + SupportsScale<D::Type, D::Tangent, D::Value>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        Ok(vec![JvpTracer::new(
            input.primal().clone().cos(),
            -input.tangent().clone().scale(input.primal().clone().sin()),
        )])
    }
}
