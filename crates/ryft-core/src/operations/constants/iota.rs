use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::forward::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation,
};
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::tracing::{Tracer, TracingContext};
use crate::types::ArrayType;

/// Canonical operation name for [`IotaOperation`].
pub const IOTA_OPERATION_NAME: &str = "iota";

/// [`Operation`] that has no inputs and that produces a single output of the [`Type`] it holds (i.e., its `r#type`
/// field) whose elements increase from `0` along a dimension chosen by [`dimension`](Self::dimension). Along that
/// dimension, the element at index `k` is `k`, and the value is constant along every other dimension. It is the
/// index-generating counterpart of [`FillOperation`](super::FillOperation). Rather than filling every element with one
/// captured scalar value, it synthesizes the per-position index through the [`Iota`] trait when interpreted. It mirrors
/// StableHLO's [`iota`](https://openxla.org/stablehlo/spec#iota).
#[derive(Copy, Clone, Debug)]
pub struct IotaOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,

    /// Dimension of `type` along which the produced values increase from `0`.
    dimension: usize,
}

impl<T: Type> IotaOperation<T> {
    /// Creates a new [`IotaOperation`] with the provided output type and iota dimension.
    #[inline]
    pub fn new(r#type: T, dimension: usize) -> Self {
        Self { r#type, dimension }
    }

    /// Returns the type of the value produced by this [`IotaOperation`].
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }

    /// Returns the dimension along which the produced values increase from `0`.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<T: Type> Display for IotaOperation<T> {
    #[inline]
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
        OperationFormatter::new(formatter, indentation, IOTA_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("type", &self.r#type)?;
            operation.field("dimension", &self.dimension)
        })
    }
}

impl<T: Type, C: Domain<Type = T> + Iota<C::Value>> InterpretableOperation<C> for IotaOperation<T> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.iota(&self.r#type, self.dimension)?])
    }
}

impl<T: Type, C: Context<Type = T>> PartiallyEvaluatableOperation<C> for IotaOperation<T> where
    C::Operation: From<IotaOperation<T>>
{
}

impl_non_differentiable_operation!(IotaOperation<C::Type>);
impl_nullary_transposable_operation!(IotaOperation<T>);
impl_nullary_batchable_operation!(@replicated IotaOperation<ArrayType>);

/// Represents the ability to synthesize a value for a given [`Type`] whose elements increase from `0` along a chosen
/// dimension in an interpretation context. [`Iota`] is the [`Type`]-driven capability needed by [`IotaOperation`] for
/// its [`InterpretableOperation`] implementation, sitting alongside [`Zero`](crate::Zero), [`One`](super::One), and
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

impl<C: Context> Iota<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<IotaOperation<C::Type>>,
{
    #[inline]
    fn iota(&self, r#type: &C::Type, dimension: usize) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(IotaOperation::new(r#type.clone(), dimension), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Iota<C::Value>> Iota<BatchingTracer<C>> for BatchingContext<C> {
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().iota(r#type, dimension)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Iota<C::Value>> Iota<DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn iota(&self, r#type: &C::Type, dimension: usize) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().iota(r#type, dimension)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::types::{ArrayType, DataType, Dimension, Shape};

    use super::*;

    #[test]
    fn test_iota() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let context = EagerContext::<Array, IotaOperation<ArrayType>>::new();

        // Axis zero varies between rows, while an axis outside the rank is rejected.
        assert_eq!(context.iota(&r#type, 0), Ok(Array::from_f64s(r#type.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])),);
        assert!(matches!(context.iota(&r#type, 2), Err(ProgramError::Type(_))));

        // Verify the operation's stored type and axis, identity, and rendering.
        let operation = IotaOperation::new(r#type.clone(), 1);
        assert_eq!(operation.name(), IOTA_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "iota [type=f64[2, 3], dimension=1]");
        assert_eq!(operation.r#type(), &r#type);
        assert_eq!(operation.dimension(), 1);
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![r#type.clone()]));

        // Eager interpretation along axis one varies between columns and repeats across rows.
        let expected = Array::from_f64s(r#type.clone(), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, IotaOperation<ArrayType>>>::interpret(
                &operation,
                &context,
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![expected.clone()]),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, IotaOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
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
