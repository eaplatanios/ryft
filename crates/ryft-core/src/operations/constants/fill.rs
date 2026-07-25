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

/// Canonical operation name for [`FillOperation`].
pub const FILL_OPERATION_NAME: &str = "fill";

/// [`Operation`] that has no inputs and that produces a single output equal to the [`Type`] it holds (i.e., its
/// `r#type` field) filled with a captured scalar `V` value. [`FillOperation`] is the scalar-broadcast counterpart
/// of [`ConstantOperation`](crate::ConstantOperation). Rather than carrying a fully typed value, it carries a target
/// [`Type`] plus a scalar `V` and synthesizes its output value through the [`Fill`] trait when interpreted. For arrays,
/// this corresponds to an array of the held type and shape with every element set to the captured scalar. It mirrors
/// [`ZeroOperation`](crate::ZeroOperation) and [`OneOperation`](crate::OneOperation), generalizing the fixed `zero` or
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

impl<T: Type, V: Clone + Display> Display for FillOperation<T, V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Clone + Display> Operation<T> for FillOperation<T, V> {
    #[inline]
    fn name(&self) -> &'static str {
        FILL_OPERATION_NAME
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
        OperationFormatter::new(formatter, indentation, FILL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("type", &self.r#type)?;
            operation.field("value", &self.value)
        })
    }
}

impl<T: Type, S: Clone + Display, C: Domain<Type = T> + Fill<S, C::Value>> InterpretableOperation<C>
    for FillOperation<T, S>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.fill(&self.r#type, self.value.clone())?])
    }
}

impl<T: Type, Constant: Clone + Display, C: Context<Type = T, Operation: From<FillOperation<T, Constant>>>>
    PartiallyEvaluatableOperation<C> for FillOperation<T, Constant>
{
}

impl_non_differentiable_operation!(<F> FillOperation<C::Type, F>);
impl_nullary_transposable_operation!(<F> FillOperation<T, F>);
impl_nullary_batchable_operation!(@replicated <F> FillOperation<ArrayType, F>);

/// Represents the ability to synthesize a value for a given [`Type`] filled with a captured scalar in an interpretation
/// context. [`Fill`] is the [`Type`]-driven counterpart needed by [`FillOperation`] for its [`InterpretableOperation`]
/// implementation. It sits alongside [`Zero`](crate::Zero) and [`One`](crate::One) in the same type-driven family, but
/// generalizes the fixed `zero` or `one` value to an arbitrary scalar `S` value supplied at the call site.
pub trait Fill<S, V: Typed> {
    /// Returns a value of [`Type`] `type` with every element it holds set to `value`.
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

impl<S: Clone + Display, C: Context> Fill<S, PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<FillOperation<C::Type, S>>,
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: S) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(FillOperation::new(r#type.clone(), value), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Fill<S, C::Value>, S> Fill<S, BatchingTracer<C>> for BatchingContext<C> {
    #[inline]
    fn fill(&self, r#type: &ArrayType, value: S) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().fill(r#type, value)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Fill<S, C::Value>, S> Fill<S, DifferentiationTracer<C>>
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

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::types::{ArrayType, DataType, Dimension, Shape};

    use super::*;

    #[test]
    fn test_fill() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let context = EagerContext::<Array, FillOperation<ArrayType, Scalar>>::new();

        // Dynamically sized outputs cannot be materialized by the eager array backend.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(crate::types::dimensions::DimensionVariable::new(
                "dynamic",
                crate::types::dimensions::DimensionBounds::unbounded(),
            ))]),
        );
        assert_eq!(
            context.fill(&dynamic_type, Scalar::from(3.5)),
            Err(ProgramError::Type(TypeError::invalid(
                "cannot materialize a value of dynamically sized type f64[dynamic]".to_string(),
            ))),
        );

        // Verify the operation's stored type and value, identity, and rendering.
        let operation = FillOperation::new(r#type.clone(), Scalar::from(3.5));
        assert_eq!(operation.name(), FILL_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "fill [type=f64[2], value=3.5]");
        assert_eq!(operation.r#type(), &r#type);
        assert_eq!(operation.value(), &Scalar::from(3.5));
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![r#type.clone()]));

        // Eager interpretation fills every element with the stored scalar value.
        let expected = Array::from_f64s(r#type.clone(), vec![3.5, 3.5]);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, FillOperation<ArrayType, Scalar>>>::interpret(
                &operation,
                &context,
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![expected.clone()]),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, FillOperation<ArrayType, Scalar>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
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
