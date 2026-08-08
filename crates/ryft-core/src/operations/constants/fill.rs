use crate::arrays::{ArrayBatch, ArrayBatching, ArrayElement, ArrayType};
use crate::backends::Array;
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::manipulation::broadcasting::{LegacyBroadcast, LegacyBroadcastOperation};
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::memory::TransferToMemory;
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::{Operation, OperationProjection, ProgramError, Type, TypeError, Typed, Value, ValueProjection};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Represents the ability to synthesize a value filled with one typed host literal. Array implementations encode the
/// literal as a rank-zero [`Array`], convert it to the requested element type and memory, and use ordinary broadcasting
/// for every rank-positive result. This keeps the fill value explicit in Static Single Assignment (SSA) dataflow and
/// avoids a separate array-fill operation.
pub trait Fill<S, V: Typed> {
    /// Returns a value of [`Type`] `type` with every element it holds set to `value`.
    fn fill(&self, r#type: &V::Type, value: S) -> Result<V, ProgramError>;
}

impl<S: ArrayElement, O: Operation<Type = ArrayType>> Fill<S, Array> for EagerContext<Array, O> {
    fn fill(&self, r#type: &ArrayType, value: S) -> Result<Array, ProgramError> {
        if r#type.static_shape().is_none() {
            return Err(TypeError::invalid(
                format!("cannot materialize a value of dynamically sized type {}", r#type,),
            )
            .into());
        }
        Array::scalar(value)
            .convert_element_type(r#type.data_type())?
            .transfer_to_memory(r#type.memory())
            .legacy_broadcast(r#type.clone(), &[])
    }
}

/// Context-level implementation used by the generic transform forwarding below.
pub(crate) trait FillContext<S, T: Type>: Context<Type = T> {
    /// Materializes one value with `type` filled by `value`.
    fn fill_literal(&self, r#type: &T, value: S) -> Result<Self::Value, ProgramError>;
}

impl<
    S: ArrayElement,
    C: Context<
            Type = ArrayType,
            Value: LegacyBroadcast,
            Operation: From<ConstantOperation<Array>> + From<LegacyBroadcastOperation>,
        >,
> FillContext<S, ArrayType> for C
{
    fn fill_literal(&self, r#type: &ArrayType, value: S) -> Result<Self::Value, ProgramError> {
        let literal =
            Array::scalar(value).convert_element_type(r#type.data_type())?.transfer_to_memory(r#type.memory());
        let scalar = self.bind(ConstantOperation::new(literal), Vec::new(), &[])?.remove(0);
        scalar.legacy_broadcast(r#type.clone(), &[])
    }
}

impl<S, C: Context, T: Type> Fill<S, <C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
    ProjectedContext<C, T>: Context<Type = T, Value = <C::Value as ValueProjection<T>>::Projected> + FillContext<S, T>,
{
    #[inline]
    fn fill(&self, r#type: &T, value: S) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        <ProjectedContext<C, T> as FillContext<S, T>>::fill_literal(self, r#type, value)
    }
}

impl<S, T: Type, C: StagingContext<Type = T> + FillContext<S, T>> Fill<S, Tracer<C>> for C {
    #[inline]
    fn fill(&self, r#type: &C::Type, value: S) -> Result<Tracer<C>, ProgramError> {
        self.fill_literal(r#type, value)
    }
}

impl<S, T: Type, C: Context<Type = T>> Fill<S, PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    PartialEvaluationContext<C>: Context<Type = T, Value = PartialTracer<C>> + FillContext<S, T>,
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: S) -> Result<PartialTracer<C>, ProgramError> {
        <PartialEvaluationContext<C> as FillContext<S, T>>::fill_literal(self, r#type, value)
    }
}

impl<S, C: Context<Type = ArrayType> + Fill<S, C::Value>> Fill<S, BatchingTracer<C, ArrayBatching>>
    for BatchingContext<C, ArrayBatching>
{
    #[inline]
    fn fill(&self, r#type: &ArrayType, value: S) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().fill(r#type, value)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<S, C: Context<Type: DifferentiableType> + Fill<S, C::Value>> Fill<S, DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: S) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().fill(r#type, value)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayOperation, ArrayType, DataType, Dimension, Memory, Shape};
    use crate::backends::Array;
    use crate::parameters::Placeholder;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_staged_array_fill_is_literal_constant_plus_broadcast() {
        for (memory, expected) in [
            (
                Memory::Device,
                indoc! {"
                    lambda  .
                    let %0:f32[] = constant [value=[2.5]]
                        %1:f32[2, 3] = broadcast [output_type=f32[2, 3], output_axes=[]] %0
                    in (%1)
                "}
                .trim_end(),
            ),
            (
                Memory::Host { pinned: true },
                indoc! {"
                    lambda  .
                    let %0:f32[]@Host[Pinned] = constant [value=[2.5]]
                        %1:f32[2, 3]@Host[Pinned] = broadcast \
                            [output_type=f32[2, 3]@Host[Pinned], output_axes=[]] %0
                    in (%1)
                "}
                .trim_end(),
            ),
            (
                Memory::Host { pinned: false },
                indoc! {"
                    lambda  .
                    let %0:f32[]@Host[Unpinned] = constant [value=[2.5]]
                        %1:f32[2, 3]@Host[Unpinned] = broadcast \
                            [output_type=f32[2, 3]@Host[Unpinned], output_axes=[]] %0
                    in (%1)
                "}
                .trim_end(),
            ),
        ] {
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let output_type =
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_memory(memory);
            let output = context.fill(&output_type, 2.5f64).unwrap();
            assert_eq!(*output.r#type(), output_type);
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
                .unwrap();
            assert_eq!(program.to_string(), expected);
        }
    }

    #[test]
    fn test_staged_rank_zero_array_fill_is_literal_constant() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output = context.fill(&ArrayType::scalar(DataType::U32), 2.5f64).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:u32[] = constant [value=[2]]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
