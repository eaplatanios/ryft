use crate::batching::{ArrayBatch, BatchableOperation, BatchingContext, BatchingError, BatchingTracer};
use crate::contexts::{Context, Domain, EagerContext};
use crate::operations::Operation;
use crate::programs::{Program, ProgramError, Value};
use crate::tracing_v2::differentiation::DifferentiationContext;
use crate::types::ArrayType;

// TODO(eaplatanios): Review this module.

impl<C: Context<Type = ArrayType>> BatchingContext<C> {
    /// Replays a captured flat program by binding each instruction's [`BatchableOperation`] rule against this batching
    /// context, threading the batch-carrying inputs through. Constants are lifted in the parent context and replicated
    /// across the batch. Higher-order batching rules use this to batch a captured sub-program without concretizing any
    /// batch-item values, so batched control-flow structure composes into the enclosing computation (executing under an
    /// eager parent, staging under a live trace).
    ///
    /// This only requires the program's own operation family `O` to be batchable; it deliberately does not require the
    /// enclosing context's [`Operation`](DispatchDomain::Operation) to be batchable, so higher-order batching rules can replay
    /// a captured sub-program through a [`BatchingContext`] whose [`Context`] impl is not yet in scope.
    pub(crate) fn interpret_program<O>(
        &self,
        program: &Program<C::Constant, O, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
    where
        O: BatchableOperation<C::Value, Self>,
    {
        program.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::replicated(self.parent().lift(constant.clone())?)),
            |instruction, instruction_inputs| instruction.operation().batch(self, instruction_inputs),
        )
    }
}

/// Replays `program` over packed batch values, dispatching every instruction through its value-level batching rule.
pub(crate) fn batch_program_inline<V, O>(
    context: &EagerContext<V, O>,
    program: &Program<V, O, Vec<V>, Vec<V>>,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<V, O>>,
{
    program.interpret_with(
        inputs.to_vec(),
        |_, constant: &V| Ok(ArrayBatch::replicated(constant.clone())),
        |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
    )
}

impl<C> DifferentiationContext for BatchingContext<C>
where
    C: Context<Type = ArrayType> + DifferentiationContext,
    C::Operation: BatchableOperation<<C as Domain>::Value, Self>,
{
    /// A batched primal is valid exactly when the parent context accepts the value it packs.
    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        self.parent().validate_primal(primal.batch().value())
    }
}
