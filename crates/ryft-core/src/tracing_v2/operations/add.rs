use std::ops::Add;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
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

impl<T: Parameter + PartialEq + Type, V: Traceable<T>, O: Clone + Operation<T>> LinearOperation<T, V, O>
    for AddOperation
where
    AddOperation: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
    }
}

impl<V> BatchableOperation<V> for AddOperation
where
    V: Traceable<ArrayType>,
    AddOperation: InterpretableOperation<ArrayType, V>,
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

impl<D: DifferentiableDomain> DifferentiableOperation<D> for AddOperation
where
    D::Value: Add<Output = D::Value>,
    AddOperation: Operation<D::Type>,
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
        check_count!("input", inputs, 2, TracingError);
        Ok(vec![JvpTracer::new(
            inputs[0].primal().clone() + inputs[1].primal().clone(),
            inputs[0].tangent().clone() + inputs[1].tangent().clone(),
        )])
    }
}
