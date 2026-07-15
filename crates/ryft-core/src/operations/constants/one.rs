use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::tracing::{Tracer, TracingContext};
use crate::types::ArrayType;

/// Canonical operation name for [`OneOperation`].
pub const ONE_OPERATION_NAME: &str = "one";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _one_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with ones.
#[derive(Clone, Debug)]
pub struct OneOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> OneOperation<T> {
    /// Creates a new [`OneOperation`].
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

impl<T: Type> Display for OneOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation<T> for OneOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        ONE_OPERATION_NAME
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
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ONE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, C: Domain<Type = T> + One<C::Value>> InterpretableOperation<C> for OneOperation<T> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.one(&self.r#type)?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<OneOperation<T>>>> PartiallyEvaluatableOperation<C>
    for OneOperation<T>
{
}

/// Represents the ability to synthesize a _one_ value for a given [`Type`] in an interpretation context. [`One`]
/// is the [`Type`]-driven counterpart to [`OneLike`](super::OneLike). It is what [`OneOperation`] needs for its
/// [`InterpretableOperation`] implementation, and it lives on the context because producing an eager value can be
/// backend- or context-dependent.
pub trait One<V: Typed> {
    /// Returns a _one_ value for the provided [`Type`].
    fn one(&self, r#type: &V::Type) -> Result<V, ProgramError>;
}

impl<C: StagingContext<Operation: From<OneOperation<C::Type>>>> One<Tracer<C>> for C {
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(OneOperation::new(r#type.clone()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> One<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<OneOperation<C::Type>>,
{
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(OneOperation::new(r#type.clone()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + One<C::Value>> One<BatchingTracer<C>> for BatchingContext<C> {
    #[inline]
    fn one(&self, r#type: &ArrayType) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().one(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context + One<C::Value>> One<DifferentiationTracer<C>> for DifferentiationContext<C> {
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().one(r#type)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
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
    fn test_one() {
        let context = EagerContext::<Scalar, OneOperation<DataType>>::new();
        assert_eq!(context.one(&DataType::Boolean), Ok(Scalar::from(true)));
        assert_eq!(context.one(&DataType::I8), Ok(Scalar::from(1i8)));
        assert_eq!(context.one(&DataType::I16), Ok(Scalar::from(1i16)));
        assert_eq!(context.one(&DataType::I32), Ok(Scalar::from(1i32)));
        assert_eq!(context.one(&DataType::I64), Ok(Scalar::from(1i64)));
        assert_eq!(context.one(&DataType::U8), Ok(Scalar::from(1u8)));
        assert_eq!(context.one(&DataType::U16), Ok(Scalar::from(1u16)));
        assert_eq!(context.one(&DataType::U32), Ok(Scalar::from(1u32)));
        assert_eq!(context.one(&DataType::U64), Ok(Scalar::from(1u64)));
        assert_eq!(context.one(&DataType::BF16), Ok(Scalar::from(bf16::ONE)));
        assert_eq!(context.one(&DataType::F16), Ok(Scalar::from(f16::ONE)));
        assert_eq!(context.one(&DataType::F32), Ok(Scalar::from(1.0f32)));
        assert_eq!(context.one(&DataType::F64), Ok(Scalar::from(1.0f64)));

        let operation = OneOperation::new(DataType::F64);
        assert_eq!(Operation::<DataType>::name(&operation), ONE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "OneOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "one [type=f64]");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[], &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Scalar::from(1.0)]),
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
                &OneOperation::new(DataType::F32),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Scalar::from(1.0f32)]),
        );

        let mut builder = ProgramBuilder::<Scalar, OneOperation<DataType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Scalar>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = one [type=f64]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
