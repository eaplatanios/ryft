use std::fmt::Display;

use crate::batching::ArrayBatch;
use crate::batching::BatchableOperation;
use crate::batching::BatchingError;
use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::types::{ArrayType, TypeError, Typed};

/// Linear-program payload for recomputing a primal operation without differentiating through it.
///
/// [`RecomputeOperation`] is used by linear control-flow programs when a value from the primal computation is
/// reconstructed inside the linear program instead of being carried as a saved residual. Interpreting and lowering
/// the wrapper delegates to the wrapped operation, but automatic differentiation treats the recomputed value as a
/// non-differentiated residual: its JVP has symbolic zero tangents and its transpose contributes zero cotangents to
/// its inputs.
#[derive(Clone, Debug)]
pub struct RecomputeOperation<O> {
    /// Wrapped primal operation to replay.
    operation: O,
}

impl<O> RecomputeOperation<O> {
    /// Creates a recomputed-primal wrapper around `operation`.
    #[inline]
    pub fn new(operation: O) -> Self {
        Self { operation }
    }

    /// Returns the wrapped primal operation.
    #[inline]
    pub fn operation(&self) -> &O {
        &self.operation
    }

    /// Consumes this wrapper and returns the wrapped primal operation.
    #[inline]
    pub fn into_operation(self) -> O {
        self.operation
    }
}

impl<O> From<O> for RecomputeOperation<O> {
    #[inline]
    fn from(operation: O) -> Self {
        Self::new(operation)
    }
}

impl<O: Operation<ArrayType>> Display for RecomputeOperation<O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<O: Operation<ArrayType>> Operation<ArrayType> for RecomputeOperation<O> {
    #[inline]
    fn name(&self) -> &'static str {
        self.operation.name()
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        self.operation.infer_output_types(input_types)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.operation.render(formatter, indentation)
    }
}

impl<V, O, C> InterpretableOperation<V, C> for RecomputeOperation<O>
where
    V: Value<Type = ArrayType>,
    O: InterpretableOperation<V, C>,
{
    #[inline]
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        self.operation.interpret(context, inputs)
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`RecomputeOperation`].
impl<RecomputedOperation: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for RecomputeOperation<RecomputedOperation>
where
    C::Operation: From<RecomputeOperation<RecomputedOperation>>,
{
}

impl<V, O, Target> TransposableOperation<V, Target> for RecomputeOperation<O>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType>,
    Target: Operation<ArrayType>,
{
    #[inline]
    fn transpose(
        &self,
        _context: &mut TracingContext<V, Target>,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_types = self.infer_output_types(input_types.as_slice())?;
        check_count!("output", outputs, output_types.len(), ProgramError);
        Ok(input_types.into_iter().map(MaybeZero::Zero).collect())
    }
}

impl<V, O, C> BatchableOperation<V, C> for RecomputeOperation<O>
where
    V: Value<Type = ArrayType>,
    O: BatchableOperation<V, C>,
{
    #[inline]
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        self.operation.batch(context, inputs)
    }
}

#[cfg(test)]
mod tests {

    use crate::operations::arithmetic::AddOperation;
    use crate::tests::TestArray;
    use crate::tracing_v2::operations::ArrayOperation;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_recompute_transpose_returns_zero_input_cotangents() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut context = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let operation = RecomputeOperation::new(ArrayOperation::<TestArray>::Add(AddOperation));

        let cotangents = operation
            .transpose(
                &mut context,
                &[PartialValue::Unknown(scalar_type.clone()), PartialValue::Unknown(scalar_type.clone())],
                &[MaybeZero::Zero(scalar_type.clone())],
            )
            .unwrap();

        assert_eq!(cotangents.len(), 2);
        assert!(cotangents.iter().all(MaybeZero::is_zero));
    }
}
