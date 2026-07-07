use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingTracer};
use crate::contexts::Context;
use crate::contexts::StagingContext;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Type, TypeError};

/// Canonical operation name for [`FillOperation`].
pub const FILL_OPERATION_NAME: &'static str = "fill";

/// [`Operation`] that has no inputs and that produces a single output equal to the [`Type`] it holds (i.e., its
/// `r#type` field) filled with a captured scalar `V` value. [`FillOperation`] is the scalar-broadcast counterpart of
/// [`ConstantOperation`](super::ConstantOperation). Rather than carrying a fully typed value, it carries a target
/// [`Type`] plus a scalar `V` and synthesizes its output value through the [`Fill`] trait when interpreted. For arrays,
/// this corresponds to an array of the held type and shape with every element set to the captured scalar. It mirrors
/// [`ZeroOperation`](super::ZeroOperation) and [`OneOperation`](super::OneOperation), generalizing the fixed `zero` or
/// `one` value to an arbitrary captured scalar value.
#[derive(Copy, Clone, Debug)]
pub struct FillOperation<T: Type, V> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,

    /// Captured scalar value used to fill the produced value when this operation is interpreted.
    value: V,
}

impl<T: Type, V> FillOperation<T, V> {
    /// Creates a new [`FillOperation`] with the provided output type and fill value.
    #[inline]
    pub fn new(r#type: T, value: V) -> Self {
        Self { r#type, value }
    }

    /// Returns the type of the value produced by this [`FillOperation`].
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }

    /// Returns the captured scalar value used to fill the produced value for this [`FillOperation`].
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

impl<T: Type, V: Display> Display for FillOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Display> Operation<T> for FillOperation<T, V> {
    #[inline]
    fn name(&self) -> &'static str {
        FILL_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, FILL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("type", &self.r#type)?;
            operation.field("value", &self.value)
        })
    }
}

impl<T: Type, V: Value<Type = T>, S: Clone + Display, C: Fill<S, V>> InterpretableOperation<V, C>
    for FillOperation<T, S>
{
    #[inline]
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.fill(&self.r#type, self.value.clone())?])
    }
}

impl<T: Type, Constant: Clone + Display, C: Context<Type = T, Operation: From<FillOperation<T, Constant>>>>
    PartiallyEvaluatableOperation<C> for FillOperation<T, Constant>
{
}

/// Represents the ability to synthesize a value for a given [`Type`] filled with a captured scalar in an interpretation
/// context. [`Fill`] is the [`Type`]-driven counterpart needed by [`FillOperation`] for its [`InterpretableOperation`]
/// implementation. It sits alongside [`Zero`](super::Zero) and [`One`](super::One) in the same type-driven family, but
/// generalizes the fixed `zero` or `one` value to an arbitrary scalar `S` value supplied at the call site.
pub trait Fill<S, V: Value> {
    /// Returns a value of `type` with every element set to `value`.
    fn fill(&self, r#type: &V::Type, value: S) -> Result<V, ProgramError>;
}

impl<V: Clone + Display, C: StagingContext<Operation: From<FillOperation<C::Type, V>>>> Fill<V, Tracer<C>> for C {
    #[inline]
    fn fill(&self, r#type: &C::Type, value: V) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(FillOperation::new(r#type.clone(), value))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType, Operation: BatchableOperation<C::Value, BatchingContext<C>>> + Fill<S, C::Value>, S>
    Fill<S, BatchingTracer<C>> for BatchingContext<C>
{
    #[inline]
    fn fill(&self, r#type: &ArrayType, value: S) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().fill(r#type, value)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_fill() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        let context = EagerContext::<TestArray, FillOperation<ArrayType, f64>>::new();
        assert_eq!(context.fill(&r#type, 3.5), Ok(TestArray::new(r#type.clone(), vec![3.5, 3.5])));

        // Filling a dynamically sized type cannot materialize element data and surfaces an error.
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        assert_eq!(
            context.fill(&dynamic_type, 3.5),
            Err(ProgramError::Type(TypeError {
                message: "cannot materialize a value of dynamically sized type f64[*]".to_string()
            })),
        );

        let operation = FillOperation::new(r#type.clone(), 3.5);

        assert_eq!(Operation::<ArrayType>::name(&operation), FILL_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "fill [type=f64[2], value=3.5]");
        assert_eq!(operation.r#type(), &r#type);
        assert_eq!(operation.value(), &3.5);
        assert_eq!(Operation::<ArrayType>::infer_output_types(&operation, &[]), Ok(vec![r#type.clone()]));
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[]
            ),
            Ok(vec![TestArray::new(r#type.clone(), vec![3.5, 3.5])]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[r#type.clone()]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::new(r#type.clone(), vec![0.0, 0.0])],
            ),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );

        let mut builder = ProgramBuilder::<TestArray, FillOperation<ArrayType, f64>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
        let program = builder.build::<(), TestArray>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[2] = fill [type=f64[2], value=3.5]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
