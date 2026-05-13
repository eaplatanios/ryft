use std::ops::Neg;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::{NegOperation, SupportsNeg};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::Parameter;
use crate::tracing::domains::Tracer;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingOutput, batch_elementwise, lift_elementwise_output,
};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
use crate::types::{ArrayType, Type};

impl<V> BatchableOperation<V> for NegOperation
where
    V: Traceable<ArrayType>,
    NegOperation: InterpretableOperation<ArrayType, V>,
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

impl<T: Parameter + Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsNeg<T, V>> LinearOperation<T, V, O>
    for NegOperation
where
    NegOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D: DifferentiableDomain> DifferentiableOperation<D> for NegOperation
where
    D::Value: Neg<Output = D::Value>,
    D::LinearOperationCarrier: SupportsNeg<D::Type, D::Tangent>,
    NegOperation: Operation<D::Type>,
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
        Ok(vec![JvpTracer::new(-inputs[0].primal().clone(), -inputs[0].tangent().clone())])
    }
}
