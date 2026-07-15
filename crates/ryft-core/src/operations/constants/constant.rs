use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::batching::{ArrayBatch, BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_builders, check_count};
use crate::partial::PartiallyEvaluatableOperation;
use crate::payloads::{Captured, Input};
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::Tracer;
use crate::types::ArrayType;

/// Canonical operation name for [`ConstantOperation`].
pub const CONSTANT_OPERATION_NAME: &str = "constant";

/// [`Operation`] that has no inputs and produces a single output equal to a captured typed value. [`ConstantOperation`]
/// is a true literal constant. It carries a `V` value that is [`Typed`], and so its output type is exactly the value's
/// type, and interpreting it simply clones the captured value. Unlike [`FillOperation`](super::FillOperation), it does
/// not synthesize a value from a scalar; it returns the value the caller already provided when constructing it.
///
/// The `Payload` type parameter is a zero-sized semantic tag that tells interpretation how the stored value should be
/// treated. The default [`Captured`] payload means the value is a literal carried by the operation, such as an eager
/// array stored in a program. The [`Input`] payload is used when the value is already part of the active runtime or
/// staging context, such as a [`Tracer`] captured while rewriting a program. In that case, interpretation forwards the
/// existing value after validating that it belongs to the same builder. Keeping this distinction in the operation type
/// lets both forms share the same operation struct without adding runtime fields or ambiguous interpretation
/// implementations.
#[derive(Copy, Clone)]
pub struct ConstantOperation<V: Clone + Typed, Payload = Captured> {
    /// Captured value produced by this [`Operation`] when interpreted.
    value: V,

    /// [`PhantomData`] marker tying the captured value to its payload role. The `fn() -> ...` form indexes by type
    /// without owning one, and so this operation's `Send` and `Sync` depend only on the captured value (as well as
    /// any trait implementations derived using `#[derive]`).
    marker: PhantomData<fn() -> Payload>,
}

impl<V: Clone + Typed, Payload> ConstantOperation<V, Payload> {
    /// Creates a new [`ConstantOperation`] capturing the provided typed value.
    #[inline]
    pub fn new(value: V) -> Self {
        Self { value, marker: PhantomData }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> Cow<'_, V::Type> {
        self.value.r#type()
    }

    /// Returns the captured value produced by this operation.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

impl<V: Clone + Debug + Typed, Payload> Debug for ConstantOperation<V, Payload> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ConstantOperation").field("value", &self.value).finish()
    }
}

impl<V: Clone + Display + Typed, Payload: Clone> Display for ConstantOperation<V, Payload> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Clone + Display + Typed, Payload: Clone> Operation<V::Type> for ConstantOperation<V, Payload> {
    #[inline]
    fn name(&self) -> &'static str {
        CONSTANT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[V::Type],
        _region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<V::Type>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.value.r#type().into_owned()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONSTANT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("value", &self.value))
    }
}

impl<
    Stored: Clone + Display + Typed,
    Payload: Clone,
    C: Domain<Type = Stored::Type> + Constant<C::Value, Stored, Payload>,
> InterpretableOperation<C> for ConstantOperation<Stored, Payload>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.constant(self.value.clone())?])
    }
}

impl<V: Clone + Typed, Payload: Clone, C: Context<Type = V::Type, Operation: From<ConstantOperation<V, Payload>>>>
    PartiallyEvaluatableOperation<C> for ConstantOperation<V, Payload>
{
}

/// Represents the ability to materialize a captured [`ConstantOperation`] payload and is typically implemented
/// by [`Context`]s. [`Constant`] is the literal value counterpart to [`Zero`](crate::Zero),
/// [`One`](crate::One), and [`Fill`](crate::Fill). It typically lives on [`Context`]s because producing
/// a runtime value from a captured payload can be context-dependent. For example, [`EagerContext`]s can return the
/// value directly, ordinary [`StagingContext`]s record a builder constant, and nested [`StagingContext`]s may receive
/// an already-staged [`Tracer`] that should be forwarded after builder validation.
///
/// The `Payload` parameter mirrors [`ConstantOperation`]'s payload tag and selects the context-specific materialization
/// path. [`Captured`] payloads are values carried by the operation and may need to be inserted into the context, while
/// [`Input`] payloads are already context values and should be reused rather than re-materialized. This type-level tag
/// keeps those semantics explicit even when the payload value type itself is otherwise the same.
pub trait Constant<V, C, Payload = Captured> {
    /// Returns the runtime value represented by `value`.
    fn constant(&self, value: C) -> Result<V, ProgramError>;
}

impl<V: Value, O: Operation<V::Type>, Payload> Constant<V, V, Payload> for EagerContext<V, O> {
    #[inline]
    fn constant(&self, value: V) -> Result<V, ProgramError> {
        Ok(value)
    }
}

impl<C: StagingContext> Constant<Tracer<C>, C::Constant, Captured> for C {
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<Tracer<C>, ProgramError> {
        Ok(StagingContext::constant(self, value))
    }
}

impl<C: StagingContext> Constant<Tracer<C>, Tracer<C>, Input> for C {
    #[inline]
    fn constant(&self, value: Tracer<C>) -> Result<Tracer<C>, ProgramError> {
        check_builders!(self.builder(), value.context().builder()).map_err(|error| self.error(error))?;
        Ok(value)
    }
}

impl<C: Context<Type = ArrayType> + Constant<C::Value, Stored, Payload>, Stored, Payload>
    Constant<BatchingTracer<C>, Stored, Payload> for BatchingContext<C>
{
    fn constant(&self, value: Stored) -> Result<BatchingTracer<C>, ProgramError> {
        let parent_value = self.parent().constant(value)?;
        let physical_type = parent_value.r#type().into_owned();
        let batch = ArrayBatch::new(physical_type, parent_value, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context> Constant<DifferentiationTracer<C>, C::Constant> for DifferentiationContext<C> {
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().lift(value)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::atoms::Atom;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::tracing::{DomainTracingContext, Tracer};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_constant() {
        let operation = ConstantOperation::<Scalar>::new(Scalar::from(3.5));

        assert_eq!(Operation::<DataType>::name(&operation), CONSTANT_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ConstantOperation { value: F64(3.5) }");
        assert_eq!(format!("{operation}"), "constant [value=3.5]");
        assert_eq!(operation.value(), &Scalar::from(3.5));
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[], &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::<Scalar>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Scalar::from(3.5)]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::<Scalar>::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.0)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );

        let mut builder = ProgramBuilder::<Scalar, ConstantOperation<Scalar>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Scalar>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = constant [value=3.5]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_constant_captured_interpretation() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let builder = context.builder().clone();
        let operation = ConstantOperation::<Scalar>::new(Scalar::from(3.5));
        let outputs =
            InterpretableOperation::<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>::interpret(
                &operation,
                &context.clone(),
                &EmptyRegionDriver,
                &[],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let builder = builder.borrow();
        assert!(builder.instructions().is_empty());
        assert!(matches!(&builder.atoms()[0], Atom::Constant(value) if *value == 3.5));
    }

    #[test]
    fn test_constant_input_interpretation() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);
        let operation = ConstantOperation::<
            Tracer<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
            Input,
        >::new(input.clone());
        let outputs =
            InterpretableOperation::<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>::interpret(
                &operation,
                &context.clone(),
                &EmptyRegionDriver,
                &[],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].atom_id(), input.atom_id());
        assert!(builder.borrow().instructions().is_empty());

        // Test that interpretation rejects a foreign builder.
        let foreign_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let foreign_builder = foreign_context.builder().clone();
        assert!(matches!(
            InterpretableOperation::<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>::interpret(
                &operation,
                &foreign_context.clone(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder.borrow().error().cloned(), None);
        assert_eq!(foreign_builder.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }
}
