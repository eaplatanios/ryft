use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, BatchingTracer,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::forward::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::{DifferentiationError, TransposableOperation, TranspositionDriver};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation};
use crate::partial::{PartialEvaluationContext, PartialTracer, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::tracing::{Tracer, TracingContext};
use crate::types::ArrayType;

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &str = "zero";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with zeros.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a new [`ZeroOperation`].
    #[inline]
    pub fn new(r#type: T) -> Self {
        Self { r#type }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }
}

impl<T: Type> Display for ZeroOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation<T> for ZeroOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        ZERO_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 0
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ZERO_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, C: Domain<Type = T> + Zero<C::Value>> InterpretableOperation<C> for ZeroOperation<T> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.zero(&self.r#type)?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<ZeroOperation<T>>>> PartiallyEvaluatableOperation<C>
    for ZeroOperation<T>
{
}

impl_non_differentiable_operation!(ZeroOperation<C::Type>);

impl<T: Type, V: Value<Type = T>, O: Operation<T>> TransposableOperation<V, O> for ZeroOperation<T> {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

/// [`ZeroOperation`] takes no inputs and produces a constant of its captured type. The same
/// constant is the right value for every batch item, so the rule interprets the operation once
/// under the parent context — constructing the constant eagerly under an eager parent and
/// staging a nullary operation under a staging parent — and wraps each output as a replicated
/// [`ArrayBatch`] (`batch_axis = None`). Downstream elementwise consumers that need the constant
/// materialized at the batched physical shape will broadcast it through the internal elementwise
/// batching rule.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> BatchableOperation<C> for ZeroOperation<ArrayType> {
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let outputs = self.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::replicated).collect())
    }
}

/// Represents the ability to synthesize a _zero_ value for a given [`Type`] in an interpretation context. [`Zero`] is
/// the [`Type`]-driven counterpart to [`ZeroLike`](super::ZeroLike). It is what [`ZeroOperation`] needs for its
/// [`InterpretableOperation`] implementation, and it lives on the context because producing an eager value can be
/// backend- or context-dependent.
pub trait Zero<V: Typed> {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(&self, r#type: &V::Type) -> Result<V, ProgramError>;
}

impl<C: StagingContext<Operation: From<ZeroOperation<C::Type>>>> Zero<Tracer<C>> for C {
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(ZeroOperation::new(r#type.clone()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> Zero<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<ZeroOperation<C::Type>>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Zero<C::Value>> Zero<BatchingTracer<C>> for BatchingContext<C> {
    #[inline]
    fn zero(&self, r#type: &ArrayType) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().zero(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> Zero<DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().zero(r#type)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::batching::{Batch, BatchAxis};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_zero() {
        let context = EagerContext::<Scalar, ZeroOperation<DataType>>::new();
        assert_eq!(context.zero(&DataType::Boolean), Ok(Scalar::from(false)));
        assert_eq!(context.zero(&DataType::I8), Ok(Scalar::from(0i8)));
        assert_eq!(context.zero(&DataType::I16), Ok(Scalar::from(0i16)));
        assert_eq!(context.zero(&DataType::I32), Ok(Scalar::from(0i32)));
        assert_eq!(context.zero(&DataType::I64), Ok(Scalar::from(0i64)));
        assert_eq!(context.zero(&DataType::U8), Ok(Scalar::from(0u8)));
        assert_eq!(context.zero(&DataType::U16), Ok(Scalar::from(0u16)));
        assert_eq!(context.zero(&DataType::U32), Ok(Scalar::from(0u32)));
        assert_eq!(context.zero(&DataType::U64), Ok(Scalar::from(0u64)));
        assert_eq!(context.zero(&DataType::BF16), Ok(Scalar::from(bf16::ZERO)));
        assert_eq!(context.zero(&DataType::F16), Ok(Scalar::from(f16::ZERO)));
        assert_eq!(context.zero(&DataType::F32), Ok(Scalar::from(0.0f32)));
        assert_eq!(context.zero(&DataType::F64), Ok(Scalar::from(0.0f64)));

        let operation = ZeroOperation::new(DataType::F64);
        assert_eq!(Operation::<DataType>::name(&operation), ZERO_OPERATION_NAME);
        assert!(Operation::<DataType>::is_zero(&operation, 0));
        assert!(!Operation::<DataType>::is_zero(&operation, 1));
        assert_eq!(format!("{operation:?}"), "ZeroOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "zero [type=f64]");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[], &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Ok(vec![Scalar::from(0.0)]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.5)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &ZeroOperation::new(DataType::F32),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Scalar::from(0.0f32)]),
        );

        let mut builder = ProgramBuilder::<Scalar, ZeroOperation<DataType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Scalar>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = zero [type=f64]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_zero_batching_yields_replicated_output() {
        // End-to-end: a batched function that stages `ZeroOperation` produces a replicated zero
        // value at the per-item scalar type. Verifies that the trace-time stage hook accepts a
        // zero-input operation and that the post-trace replay materializes the same zero for
        // every batch item through the replicated broadcast path.
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| {
                    let zero_op = ArrayOperation::<Array>::Zero(ZeroOperation::new(ArrayType::scalar(DataType::F64)));
                    let zero = x.context().bind(zero_op, Vec::new(), &[])?.into_iter().next().unwrap();
                    Ok(x + zero)
                },
                Array::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0]);
    }
}
