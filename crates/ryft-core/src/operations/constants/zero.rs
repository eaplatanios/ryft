use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingTracer};
use crate::contexts::Context;
use crate::contexts::StagingContext;
use crate::differentiation::DifferentiationDual;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::tracing_v2::differentiation::{DifferentiableOperation, DifferentiationContext, DifferentiationTracer};
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &'static str = "zero";

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
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ZERO_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, V: Value<Type = T>, C: Zero<V>> InterpretableOperation<V, C> for ZeroOperation<T> {
    #[inline]
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.zero(&self.r#type)?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<ZeroOperation<T>>>> PartiallyEvaluatableOperation<C>
    for ZeroOperation<T>
{
}

/// Represents [`Operation`]s that may be or may be carrying [`ZeroOperation`] payloads. [`MaybeZeroOperation`] says
/// that a borrowed operation value can be inspected to determine whether it is a [`ZeroOperation`] (or a wrapper of
/// one), without cloning, moving, allocating, or manufacturing placeholder operations. Structural zero-ness ordinarily
/// flows *symbolically* through the differentiation transforms as [`MaybeZero`](crate::MaybeZero) values and never
/// needs to be recognized from staged instructions. The one place zero-ness can be lost is an opaque program splice
/// (i.e., a user-authored program replayed into an active trace, such as a `custom_vjp` backward program whose outputs
/// include canonical zeros for non-differentiated inputs) and the splicing rule uses this trait to recover it with one
/// local pass over the spliced program's output producers.
pub trait MaybeZeroOperation<T: Type> {
    /// Returns `true` if `self` is a [`ZeroOperation`] (or a wrapper of one).
    fn is_zero_operation(&self) -> bool;
}

impl<T: Type, O> MaybeZeroOperation<T> for O
where
    for<'operation> &'operation ZeroOperation<T>: TryFrom<&'operation O>,
{
    #[inline]
    fn is_zero_operation(&self) -> bool {
        <&ZeroOperation<T>>::try_from(self).is_ok()
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

impl<C: Context<Type = ArrayType, Operation: BatchableOperation<C::Value, BatchingContext<C>>> + Zero<C::Value>>
    Zero<BatchingTracer<C>> for BatchingContext<C>
{
    #[inline]
    fn zero(&self, r#type: &ArrayType) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().zero(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Operation: Clone + DifferentiableOperation<C>> + Zero<C::Value>> Zero<DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.context().zero(r#type)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::scalars::Scalar;
    use crate::types::{DataType, TypeError};

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
        assert_eq!(format!("{operation:?}"), "ZeroOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "zero [type=f64]");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<Scalar, crate::EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[]
            ),
            Ok(vec![Scalar::from(0.0)]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<Scalar, crate::EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(2.5)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<Scalar, crate::EagerContext<Scalar>>::interpret(
                &ZeroOperation::new(DataType::F32),
                &EagerContext::new(),
                &[],
            ),
            Ok(vec![Scalar::from(0.0f32)]),
        );

        let mut builder = ProgramBuilder::<Scalar, ZeroOperation<DataType>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
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
}
