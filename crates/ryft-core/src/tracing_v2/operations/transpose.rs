use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::manipulation::{Transpose, TransposeOperation, inverse_permutation};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ValueOrCapture};
use crate::types::ArrayType;

impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for TransposeOperation
where
    O: Operation<ArrayType> + From<TransposeOperation>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        let inverse = inverse_permutation(self.permutation());
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.transpose(inverse)?)]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D> DifferentiableOperation<D> for TransposeOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Transpose,
    D::Tangent: Transpose,
    D::LinearOperation<D::Tangent, ValueOrCapture<D::Type, D::Value>>: From<TransposeOperation>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().transpose(self.permutation().to_vec())?;
        let tangent = inputs[0].tangent().clone().transpose(self.permutation().to_vec())?;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Lifts an axis `permutation` through one batching level inserted at `batch_axis`.
///
/// The returned permutation has length `permutation.len() + 1`, places the batch axis at the
/// same output position as it appears in the input (so the output's batch axis stays at the
/// input's `batch_axis`), and shifts every other axis index `i` to `i + 1` when `i >= batch_axis`.
pub fn lift_permutation(permutation: &[usize], batch_axis: usize) -> Vec<usize> {
    let mut lifted = Vec::with_capacity(permutation.len() + 1);
    for output_axis in 0..=permutation.len() {
        if output_axis == batch_axis {
            lifted.push(batch_axis);
        } else {
            let original_output_axis = if output_axis < batch_axis { output_axis } else { output_axis - 1 };
            let input_axis = permutation[original_output_axis];
            lifted.push(if input_axis >= batch_axis { input_axis + 1 } else { input_axis });
        }
    }
    lifted
}

impl<V: Value<ArrayType>> crate::tracing_v2::batching::BatchableOperation<V, V::InterpretationContext>
    for TransposeOperation
where
    TransposeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let (lifted_permutation, output_axis) = match input_axes[0] {
            Some(batch_axis) => (lift_permutation(self.permutation(), batch_axis), Some(batch_axis)),
            None => (self.permutation().to_vec(), None),
        };
        let lifted_op = TransposeOperation::new(lifted_permutation);
        crate::tracing_v2::batching::apply_with_axes(context, &lifted_op, inputs, &[output_axis])
    }
}
