use std::fmt::Display;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::ZeroOperation;
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ZeroTangentOperation};
use crate::types::{ArrayType, TypeError};

/// Linear-program payload for recomputing a primal operation without differentiating through it.
///
/// [`RecomputeOperation`] is used by fused linear control-flow programs when a value from the primal computation is
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

impl<V: Value<ArrayType>, O: InterpretableOperation<ArrayType, V>> InterpretableOperation<ArrayType, V>
    for RecomputeOperation<O>
{
    #[inline]
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        self.operation.interpret(context, inputs)
    }
}

impl<D: DifferentiationContext<Type = ArrayType>, O: InterpretableOperation<ArrayType, D::Value>>
    ZeroTangentOperation<D> for RecomputeOperation<O>
{
}

impl<D: DifferentiationContext<Type = ArrayType>, O: InterpretableOperation<ArrayType, D::Value>>
    DifferentiableOperation<D> for RecomputeOperation<O>
where
    LinearOperationOf<D>: From<ZeroOperation<ArrayType>>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        self.zero_tangent_jvp(context, inputs)
    }
}

impl<V, O, Target> TransposableOperation<ArrayType, V, Target> for RecomputeOperation<O>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType>,
    Target: Operation<ArrayType>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, Target>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, Target>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, Target>>, ProgramError> {
        let input_types = input_types.iter().map(|input_type| (*input_type).clone()).collect::<Vec<_>>();
        let output_types = self.infer_output_types(input_types.as_slice())?;
        check_count!("output", output_cotangents, output_types.len(), ProgramError);
        Ok(vec![Cotangent::Zero; input_types.len()])
    }
}

impl<V, O, C> BatchableOperation<V, C> for RecomputeOperation<O>
where
    V: Value<ArrayType>,
    O: BatchableOperation<V, C>,
{
    #[inline]
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        self.operation.batch(context, inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::contexts::StagingContext;
    use crate::domains::AbstractDomain;
    use crate::operations::arithmetic::AddOperation;
    use crate::programs::ProgramBuilder;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::LinearOperationOf;
    use crate::tracing_v2::operations::{ArrayOperation, LinearArrayOperation};
    use crate::types::{DataType, Typed};

    use super::*;

    #[test]
    fn test_recompute_jvp_replays_primal_with_zero_tangent() {
        type LinearOperation = LinearOperationOf<TestArrayDomain>;

        let scalar_type = ArrayType::scalar(DataType::F64);
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, LinearOperation>::new()));
        let mut context = TangentContext::new(&TestArrayDomain, builder.clone());
        let tangent = context.input(scalar_type.clone());
        let operation = RecomputeOperation::new(ArrayOperation::<TestArray>::Add(AddOperation));

        let outputs = operation
            .jvp(
                &mut context,
                &[
                    JvpTracer::from_value(TestArray::scalar(2.0), tangent.clone()),
                    JvpTracer::from_value(TestArray::scalar(3.0), tangent),
                ],
            )
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestArray::scalar(5.0));
        assert_eq!(outputs[0].tangent().r#type().into_owned(), scalar_type);
        assert!(context.is_zero(outputs[0].tangent()).unwrap());
        assert_eq!(builder.borrow().instructions().len(), 1);
    }

    #[test]
    fn test_recompute_transpose_returns_zero_input_cotangents() {
        type LinearOperation = LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>;

        let scalar_type = ArrayType::scalar(DataType::F64);
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, LinearOperation>::new()));
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, builder);
        let operation = RecomputeOperation::new(ArrayOperation::<TestArray>::Add(AddOperation));

        let cotangents = operation.transpose(&mut context, &[&scalar_type, &scalar_type], &[Cotangent::Zero]).unwrap();

        assert_eq!(cotangents.len(), 2);
        assert!(cotangents.iter().all(Cotangent::is_zero));
    }
}
