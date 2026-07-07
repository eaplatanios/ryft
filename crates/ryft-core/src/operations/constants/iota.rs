use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingTracer};
use crate::contexts::Context;
use crate::contexts::StagingContext;
use crate::differentiation::DifferentiationDual;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::constants::Zero;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::tracing_v2::differentiation::{DifferentiableOperation, DifferentiationContext, DifferentiationTracer};
use crate::types::{ArrayType, Type, TypeError, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`IotaOperation`].
pub const IOTA_OPERATION_NAME: &'static str = "iota";

/// [`Operation`] that has no inputs and that produces a single output of the [`Type`] it holds (i.e., its `r#type`
/// field) whose elements increase from `0` along a chosen dimension. Along
/// [`iota_dimension`](Self::iota_dimension) the element at index `k` is `k` (cast to the output element type), and the
/// value is constant along every other dimension. It is the index-generating counterpart of
/// [`FillOperation`](super::FillOperation): rather than filling every element with one captured scalar, it synthesizes
/// the per-position index through the [`Iota`] trait when interpreted. It mirrors StableHLO's
/// [`iota`](https://openxla.org/stablehlo/spec#iota).
#[derive(Copy, Clone, Debug)]
pub struct IotaOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,

    /// Dimension of `type` along which the produced values increase from `0`.
    iota_dimension: usize,
}

impl<T: Type> IotaOperation<T> {
    /// Creates a new [`IotaOperation`] with the provided output type and iota dimension.
    #[inline]
    pub fn new(r#type: T, iota_dimension: usize) -> Self {
        Self { r#type, iota_dimension }
    }

    /// Returns the type of the value produced by this [`IotaOperation`].
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }

    /// Returns the dimension along which the produced values increase from `0`.
    #[inline]
    pub fn iota_dimension(&self) -> usize {
        self.iota_dimension
    }
}

impl<T: Type> Display for IotaOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation<T> for IotaOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        IOTA_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, IOTA_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("type", &self.r#type)?;
            operation.field("dimension", &self.iota_dimension)
        })
    }
}

impl<T: Type, V: Value<Type = T>, C: Iota<V>> InterpretableOperation<V, C> for IotaOperation<T> {
    #[inline]
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.iota(&self.r#type, self.iota_dimension)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<T: Type, C: Context<Type = T>> PartiallyEvaluatableOperation<C> for IotaOperation<T> where
    C::Operation: From<IotaOperation<T>>
{
}

/// Represents the ability to synthesize a value for a given [`Type`] whose elements increase from `0` along a chosen
/// dimension in an interpretation context. [`Iota`] is the [`Type`]-driven capability needed by [`IotaOperation`] for
/// its [`InterpretableOperation`] implementation, sitting alongside [`Zero`](super::Zero), [`One`](super::One), and
/// [`Fill`](super::Fill) in the same type-driven family.
pub trait Iota<V: Typed> {
    /// Returns a value of `type` whose elements increase from `0` along `dimension` and are constant along every other
    /// dimension.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Type of the value to produce.
    ///   - `dimension`: Dimension of `type` along which the produced values increase from `0`.
    fn iota(&self, r#type: &V::Type, dimension: usize) -> Result<V, ProgramError>;
}

impl<C: StagingContext<Operation: From<IotaOperation<C::Type>>>> Iota<Tracer<C>> for C {
    #[inline]
    fn iota(&self, r#type: &C::Type, dimension: usize) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(IotaOperation::new(r#type.clone(), dimension))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType, Operation: BatchableOperation<C::Value, BatchingContext<C>>> + Iota<C::Value>>
    Iota<BatchingTracer<C>> for BatchingContext<C>
{
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().iota(r#type, dimension)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Operation: Clone + DifferentiableOperation<C>> + Zero<C::Value> + Iota<C::Value>>
    Iota<DifferentiationTracer<C>> for DifferentiationContext<C>
{
    #[inline]
    fn iota(&self, r#type: &C::Type, dimension: usize) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.context().iota(r#type, dimension)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_iota() {
        // A rank-2 iota along dimension 1 increases across columns and repeats down rows.
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let context = EagerContext::<TestArray, IotaOperation<ArrayType>>::new();
        assert_eq!(context.iota(&r#type, 1), Ok(TestArray::new(r#type.clone(), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),);
        // Along dimension 0 the index increases down rows and repeats across columns.
        assert_eq!(context.iota(&r#type, 0), Ok(TestArray::new(r#type.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])),);

        // An out-of-bounds dimension surfaces an error rather than mis-indexing.
        assert!(matches!(context.iota(&r#type, 2), Err(ProgramError::Type(_))));

        let operation = IotaOperation::new(r#type.clone(), 1);
        assert_eq!(Operation::<ArrayType>::name(&operation), IOTA_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "iota [type=f64[2, 3], dimension=1]");
        assert_eq!(operation.r#type(), &r#type);
        assert_eq!(operation.iota_dimension(), 1);
        assert_eq!(Operation::<ArrayType>::infer_output_types(&operation, &[]), Ok(vec![r#type.clone()]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[r#type.clone()]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );

        let mut builder = ProgramBuilder::<TestArray, IotaOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
        let program = builder.build::<(), TestArray>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[2, 3] = iota [type=f64[2, 3], dimension=1]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
