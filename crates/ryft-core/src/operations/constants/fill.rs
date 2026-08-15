use crate::arrays::{Array, ArrayBatch, ArrayBatching, ArrayElement, ArrayType};
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::manipulation::broadcasting::{BROADCAST_OPERATION_NAME, Broadcast};
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::memory::TransferToMemory;
use crate::partial::{PartialEvaluationContext, PartialTracer};
use crate::programs::{Operation, ProgramError, Type, TypeError, Typed, Value, ValueProjection};
use crate::tracing::Tracer;

/// Represents the ability to synthesize a value filled with one typed host literal. [`ArrayType`] implementations
/// encode the literal as a rank-zero array, convert it to the requested element [`DataType`](crate::DataType) and
/// [`Memory`](crate::Memory), and use ordinary broadcasting for every rank-positive result. This keeps the fill value
/// explicit in Static Single Assignment (SSA) dataflow and avoids the need for a separate array fill operation type.
pub trait Fill<L, V: Typed> {
    /// Returns a value of [`Type`] `type` with every element it holds set to `value`.
    fn fill(&self, r#type: &V::Type, value: L) -> Result<V, ProgramError>;
}

impl<L: ArrayElement, O: Operation<Type = ArrayType>> Fill<L, Array> for EagerContext<Array, O> {
    fn fill(&self, r#type: &ArrayType, value: L) -> Result<Array, ProgramError> {
        if r#type.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "cannot materialize a value of dynamically sized type {}; stage a rank-zero fill \
                 and expand it with a dynamic `{BROADCAST_OPERATION_NAME}` operation instead",
                r#type,
            ))
            .into());
        }
        Array::scalar(value)
            .convert_element_type(r#type.data_type())?
            .transfer_to_memory(r#type.memory())
            .broadcast(r#type.clone(), &[])
    }
}

impl<L, C: Context, T: Type> Fill<L, <C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    ProjectedContext<C, T>: Context<Type = T, Value = <C::Value as ValueProjection<T>>::Projected> + FillLiteral<L, T>,
{
    #[inline]
    fn fill(&self, r#type: &T, value: L) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        <ProjectedContext<C, T> as FillLiteral<L, T>>::fill_literal(self, r#type, value)
    }
}

impl<L, T: Type, C: StagingContext<Type = T> + FillLiteral<L, T>> Fill<L, Tracer<C>> for C {
    #[inline]
    fn fill(&self, r#type: &C::Type, value: L) -> Result<Tracer<C>, ProgramError> {
        self.fill_literal(r#type, value)
    }
}

impl<L, T: Type, C: Context<Type = T>> Fill<L, PartialTracer<C>> for PartialEvaluationContext<C>
where
    PartialEvaluationContext<C>: Context<Type = T, Value = PartialTracer<C>> + FillLiteral<L, T>,
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: L) -> Result<PartialTracer<C>, ProgramError> {
        <PartialEvaluationContext<C> as FillLiteral<L, T>>::fill_literal(self, r#type, value)
    }
}

impl<L, C: Context<Type = ArrayType> + Fill<L, C::Value>> Fill<L, BatchingTracer<C, ArrayBatching>>
    for BatchingContext<C, ArrayBatching>
{
    #[inline]
    fn fill(&self, r#type: &ArrayType, value: L) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().fill(r#type, value)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<L, C: Context<Type: DifferentiableType> + Fill<L, C::Value>> Fill<L, DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: L) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().fill(r#type, value)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

/// Internal trait used for implementing [`Fill`] by embedding a typed host literal in the active [`Context`]. For
/// arrays, the implementation converts `value` into a rank-zero [`Array`] and binds it as a [`ConstantOperation`],
/// then broadcasts that scalar to the requested type. The [`Array`] is only a portable literal payload that a backend
/// context interprets or lowers the constant into its own runtime value. It is distinct from the context's
/// [`Domain::Constant`](crate::Domain::Constant) representation, which stores lifted constants and program captures.
trait FillLiteral<L, T: Type>: Context<Type = T> {
    /// Embeds `value` as a host literal and expands it to a value of `type` in this [`Context`].
    fn fill_literal(&self, r#type: &T, value: L) -> Result<Self::Value, ProgramError>;
}

impl<L: ArrayElement, C: Context<Type = ArrayType>> FillLiteral<L, ArrayType> for C
where
    C::Value: Broadcast,
    C::Operation: From<ConstantOperation<Array>>,
{
    fn fill_literal(&self, r#type: &ArrayType, value: L) -> Result<Self::Value, ProgramError> {
        let value = Array::scalar(value).convert_element_type(r#type.data_type())?.transfer_to_memory(r#type.memory());
        self.bind(ConstantOperation::new(value), Vec::new(), &[])?.remove(0).broadcast(r#type.clone(), &[])
    }
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatching, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionVariable, Memory, Shape, f6e2m3fn, u4,
    };
    use crate::parameters::Placeholder;
    use crate::programs::MaybeZero;
    use crate::tracing::TracingContext;

    use super::*;

    fn array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().copied().map(Dimension::Static).collect()))
    }

    #[test]
    fn test_fill_eager_context() {
        let context = EagerContext::<Array>::new();

        // Rank-zero and rank-positive fills both apply the requested element conversion exactly once.
        assert_eq!(context.fill(&ArrayType::scalar(DataType::U32), 2.5f64), Ok(Array::scalar(2u32)));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            context.fill(&output_type, 2.5f64).unwrap(),
            Array::from_elements(output_type, &[2.5f32; 6]).unwrap(),
        );
        assert_eq!(
            context.fill(&array_type(DataType::F32, &[2]), ComplexNumber::new(1.0f32, 2.0)),
            Ok(Array::vector(vec![1.0f32; 2])),
        );
        assert_eq!(
            context.fill(&array_type(DataType::Boolean, &[2]), ComplexNumber::new(0.0f32, 2.0)),
            Array::from_elements(array_type(DataType::Boolean, &[2]), &[true; 2]).map_err(Into::into),
        );

        // Fill uses the same checked element codecs for low-precision and sub-byte values as direct array creation.
        let low_precision = f6e2m3fn::from_bits(0x08).unwrap();
        assert_eq!(
            context.fill(&array_type(DataType::F6E2M3FN, &[2]), low_precision),
            Array::from_elements(array_type(DataType::F6E2M3FN, &[2]), &[low_precision; 2]).map_err(Into::into),
        );
        assert_eq!(
            context.fill(&array_type(DataType::U4, &[2]), 2.5f64).unwrap().elements::<u4>(),
            Ok(vec![u4::new(2).unwrap(); 2]),
        );

        // Eager arrays have no runtime extent operands from which to materialize a dynamically shaped result.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            context.fill(&dynamic_type, 1.0f32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f32[size]; stage a rank-zero \
                               fill and expand it with a dynamic `broadcast` operation instead",
        ));
    }

    #[test]
    fn test_fill_projected_context() {
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ProjectedContext::<_, ArrayType>::new(parent.clone());
        let output_type = array_type(DataType::F32, &[2]);
        let output = context.fill(&output_type, 1.5f32).unwrap();
        assert_eq!(output.r#type().as_ref(), &output_type);

        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.into_value().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[] = constant [value=1.5]
                    %1:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_fill_staging_context() {
        // A rank-zero fill is represented by its literal constant alone.
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
                let %0:u32[] = constant [value=2]
                in (%0)
            "}
            .trim_end(),
        );

        // A rank-positive fill broadcasts that literal, retaining the requested memory placement in both operations.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_type = array_type(DataType::F32, &[2, 3]).with_memory(Memory::Host { pinned: false });
        let output = context.fill(&output_type, 2.5f64).unwrap();
        assert_eq!(output.r#type().as_ref(), &output_type);
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
                let %0:f32[]@Host[Unpinned] = constant [value=2.5]
                    %1:f32[2, 3]@Host[Unpinned] = broadcast \
                        [output_type=f32[2, 3]@Host[Unpinned], output_axes=[]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_fill_partial_evaluation_context() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = array_type(DataType::F32, &[2]);
        let output = context.fill(&output_type, 3.5f32).unwrap();
        let expected = Array::from_elements(output_type, &[3.5f32; 2]).unwrap();

        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_fill_batching_context() {
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = array_type(DataType::F32, &[2]);
        let output = context.fill(&output_type, 4.5f32).unwrap();

        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[4.5f32; 2]).unwrap(),);
    }

    #[test]
    fn test_fill_differentiation_context() {
        let context = DifferentiationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = array_type(DataType::F32, &[2]);
        let output = context.fill(&output_type, 5.5f32).unwrap();

        assert_eq!(output.primal(), &Array::from_elements(output_type.clone(), &[5.5f32; 2]).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }
}
