use crate::batching::InterpretableBatchableOperation;
use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::manipulation::{Transpose, TransposeOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::tracing_v2::differentiation::{DifferentiableOperation, JvpTracer};
use crate::types::{ArrayType, Typed};

impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for TransposeOperation
where
    O: Operation<ArrayType> + From<TransposeOperation>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        let inverse = self.permutation().inverse();
        match &outputs[0] {
            MaybeZero::Value(cotangent) => Ok(vec![MaybeZero::Value(cotangent.transpose(inverse)?)]),
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
        }
    }
}

/// Forward-mode rule for [`TransposeOperation`]: `transpose` is structural-linear, so the tangent is the same
/// transpose applied to the operand tangent. The shared all-zero fast path handles a zero operand tangent before this
/// rule is consulted, so the operand tangent reaching here is always live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for TransposeOperation
where
    C::Operation: Clone + From<TransposeOperation>,
    C::Value: Transpose,
{
    fn jvp(&self, _context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().transpose(self.permutation())?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.transpose(self.permutation())?),
        };
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

impl<V: Value<Type = ArrayType>, C> crate::batching::BatchableOperation<V, C> for TransposeOperation
where
    TransposeOperation: InterpretableOperation<V, C>,
{
    fn batch(
        &self,
        context: &C,
        inputs: &[crate::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::batching::ArrayBatch<V>>, crate::batching::BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        crate::batching::ArrayBatch::common_batch_size(inputs)?;
        let (lifted_permutation, output_axis) = match inputs[0].batch_axis().axis() {
            Some(batch_axis) => (lift_permutation(self.permutation(), batch_axis), Some(batch_axis)),
            None => (self.permutation().to_vec(), None),
        };
        let lifted_op = TransposeOperation::new(lifted_permutation);
        lifted_op.interpret_with_batch_axes(context, inputs, &[output_axis.into()])
    }
}
