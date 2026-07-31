use crate::backends::arrays::Array;
use crate::backends::scalars::Scalar;
use crate::batching::{ArrayBatch, ArrayBatchingPolicy, BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, ProjectedContext, StagingContext};
use crate::differentiation::forward::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::differentiation::types::DifferentiableType;
use crate::operations::constants::ConstantOperation;
use crate::operations::manipulation::{ConvertElementType, LegacyBroadcast, LegacyBroadcastOperation};
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::operations::OperationProjection;
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType};

// TODO(eaplatanios): Review this module.

/// Represents the ability to synthesize a value filled with a host scalar. Staged scalar implementations bind a
/// literal [`ConstantOperation`]. Staged array implementations bind the same rank-zero literal in the requested
/// [`Memory`](crate::Memory) space and use ordinary broadcasting for every rank-positive result, keeping the fill
/// value and output extents explicit in Static Signal Assignment (SSA) dataflow.
pub trait Fill<S, V: Typed> {
    /// Returns a value of [`Type`] `type` with every element it holds set to `value`.
    fn fill(&self, r#type: &V::Type, value: S) -> Result<V, ProgramError>;
}

/// Private context-level dispatch for the two concrete type families that support scalar filling.
trait FillContext<T: Type>: Context<Type = T> {
    /// Materializes one value with `type` filled by `value`.
    fn fill_scalar(&self, r#type: &T, value: Scalar) -> Result<Self::Value, ProgramError>;
}

impl<C: Context<Type = DataType, Operation: From<ConstantOperation<Scalar>>>> FillContext<DataType> for C {
    #[inline]
    fn fill_scalar(&self, r#type: &DataType, value: Scalar) -> Result<Self::Value, ProgramError> {
        let value = value.convert_element_type(*r#type)?;
        Ok(self.bind(ConstantOperation::new(value), Vec::new(), &[])?.remove(0))
    }
}

impl<
    C: Context<
            Type = ArrayType,
            Value: LegacyBroadcast,
            Operation: From<ConstantOperation<Array>> + From<LegacyBroadcastOperation>,
        >,
> FillContext<ArrayType> for C
{
    #[inline]
    fn fill_scalar(&self, r#type: &ArrayType, value: Scalar) -> Result<Self::Value, ProgramError> {
        let value = value.convert_element_type(r#type.data_type())?;
        let literal = Array::new(ArrayType::scalar(r#type.data_type()).with_memory(r#type.memory()), vec![value])?;
        let scalar = self.bind(ConstantOperation::new(literal), Vec::new(), &[])?.remove(0);
        scalar.legacy_broadcast(r#type.clone(), &[])
    }
}

impl<C: Context, T: Type> Fill<Scalar, <C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
    ProjectedContext<C, T>: Context<Type = T, Value = <C::Value as ValueProjection<T>>::Projected> + FillContext<T>,
{
    #[inline]
    fn fill(&self, r#type: &T, value: Scalar) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        <ProjectedContext<C, T> as FillContext<T>>::fill_scalar(self, r#type, value)
    }
}

impl<T: Type, C: StagingContext<Type = T> + FillContext<T>> Fill<Scalar, Tracer<C>> for C {
    #[inline]
    fn fill(&self, r#type: &C::Type, value: Scalar) -> Result<Tracer<C>, ProgramError> {
        self.fill_scalar(r#type, value)
    }
}

impl<T: Type, C: Context<Type = T>> Fill<Scalar, PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    PartialEvaluationContext<C>: Context<Type = T, Value = PartialTracer<C>> + FillContext<T>,
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: Scalar) -> Result<PartialTracer<C>, ProgramError> {
        <PartialEvaluationContext<C> as FillContext<T>>::fill_scalar(self, r#type, value)
    }
}

impl<C: Context<Type = ArrayType> + Fill<Scalar, C::Value>> Fill<Scalar, BatchingTracer<C, ArrayBatchingPolicy>>
    for BatchingContext<C, ArrayBatchingPolicy>
{
    #[inline]
    fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<BatchingTracer<C, ArrayBatchingPolicy>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().fill(r#type, value)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Fill<Scalar, C::Value>> Fill<Scalar, DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: Scalar) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().fill(r#type, value)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::parameters::Placeholder;
    use crate::tracing::TracingContext;
    use crate::types::{ArrayType, DataType, Dimension, Memory, Shape};

    use super::*;

    #[test]
    fn test_staged_scalar_fill_is_literal_constant() {
        let context = TracingContext::<Scalar, ScalarOperation<Scalar>>::new();
        let output = context.fill(&DataType::F32, Scalar::F64(2.5)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f32 = constant [value=2.5]
                in (%0)
            "}
            .trim_end(),
        );
    }

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
            let output = context.fill(&output_type, Scalar::F64(2.5)).unwrap();
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
        let output = context.fill(&ArrayType::scalar(DataType::U32), Scalar::F64(2.5)).unwrap();
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
