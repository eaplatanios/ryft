use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::{EagerContext, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::payloads::{Captured, Input};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{Type, TypeError, Typed};

/// Canonical operation name for [`ConstantOperation`].
pub const CONSTANT_OPERATION_NAME: &'static str = "constant";

/// [`Operation`] that has no inputs and produces a single output equal to a captured typed value. [`ConstantOperation`]
/// is a true literal constant. It carries a `V` value that is [`Typed`] against the operation's [`Type`] `T`, and so
/// its output type is exactly the value's type, and interpreting it simply clones the captured value. Unlike
/// [`FillOperation`](super::FillOperation), it does not synthesize a value from a scalar; it returns the value the
/// caller already provided when constructing it.
#[derive(Copy, Clone)]
pub struct ConstantOperation<T: Type, V: Clone + Typed<T>, Payload = Captured> {
    /// Captured value produced by this [`Operation`] when interpreted.
    value: V,

    /// [`PhantomData`] marker tying the captured value to the [`Type`] it is typed against and to its payload role.
    /// The `fn() -> ...` form indexes by type without owning one, and so this operation's `Send` and `Sync` depend
    /// only on the captured value (as well as any trait implementations derived using `#[derive]`).
    marker: PhantomData<fn() -> (T, Payload)>,
}

impl<T: Type, V: Clone + Typed<T>, Payload> ConstantOperation<T, V, Payload> {
    /// Creates a new [`ConstantOperation`] capturing the provided typed value.
    #[inline]
    pub fn new(value: V) -> Self {
        Self { value, marker: PhantomData }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> Cow<'_, T> {
        self.value.r#type()
    }

    /// Returns the captured value produced by this operation.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

impl<T: Type, V: Clone + Debug + Typed<T>, Payload> Debug for ConstantOperation<T, V, Payload> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ConstantOperation").field("value", &self.value).finish()
    }
}

impl<T: Type, V: Clone + Display + Typed<T>, Payload> Display for ConstantOperation<T, V, Payload> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Clone + Display + Typed<T>, Payload> Operation<T> for ConstantOperation<T, V, Payload> {
    #[inline]
    fn name(&self) -> &'static str {
        CONSTANT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.value.r#type().into_owned()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONSTANT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("value", &self.value))
    }
}

impl<T: Type, V: Value<T, InterpretationContext: Constant<T, V, C, Payload>>, C: Clone + Display + Typed<T>, Payload>
    InterpretableOperation<T, V> for ConstantOperation<T, C, Payload>
{
    #[inline]
    fn interpret(
        &self,
        context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.constant(self.value.clone())?])
    }
}

/// Represents the ability to materialize a captured [`ConstantOperation`] payload and is typically implemented
/// by [`Context`](crate::Context)s. [`Constant`] is the literal value counterpart to [`Zero`](crate::Zero),
/// [`One`](crate::One), and [`Fill`](crate::Fill). It typically lives on [`Context`](crate::Context)s because producing
/// a runtime value from a captured payload can be context-dependent. For example, [`EagerContext`]s can return the
/// value directly, ordinary [`StagingContext`]s record a builder constant, and nested [`StagingContext`]s may receive
/// an already-staged [`Tracer`] that should be forwarded after builder validation.
pub trait Constant<T: Type, V: Value<T>, C, Payload = Captured> {
    /// Returns the runtime value represented by `value`.
    fn constant(&self, value: C) -> Result<V, ProgramError>;
}

impl<T: Type, V: Value<T>, O: Operation<T>, Payload> Constant<T, V, V, Payload> for EagerContext<T, V, O> {
    #[inline]
    fn constant(&self, value: V) -> Result<V, ProgramError> {
        Ok(value)
    }
}

impl<C: StagingContext> Constant<C::Type, Tracer<C>, C::Constant, Captured> for C {
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<Tracer<C>, ProgramError> {
        Ok(StagingContext::constant(self, value))
    }
}

impl<C: StagingContext> Constant<C::Type, Tracer<C>, Tracer<C>, Input> for C {
    #[inline]
    fn constant(&self, value: Tracer<C>) -> Result<Tracer<C>, ProgramError> {
        check_builders!(self.builder(), value.context().builder()).map_err(|error| self.error(error))?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, ProgramBuilder, ProgramError};
    use crate::scalars::ScalarDomain;
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{DataType, TypeError};

    use super::*;

    #[test]
    fn test_constant() {
        let operation = ConstantOperation::<DataType, f64>::new(3.5);

        assert_eq!(Operation::<DataType>::name(&operation), CONSTANT_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ConstantOperation { value: 3.5 }");
        assert_eq!(format!("{operation}"), "constant [value=3.5]");
        assert_eq!(operation.value(), &3.5);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &EagerContext::new(), &[]),
            Ok(vec![3.5]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &EagerContext::new(), &[0.0]),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );

        let mut builder = ProgramBuilder::<DataType, f64, ConstantOperation<DataType, f64>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
        let program = builder.build::<(), f64>(vec![output], (), Placeholder).unwrap();
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

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_literal_constant_interpretation_stages_builder_constant() {
        type TestContext<'domain> = TracingContext<'domain, ScalarDomain<f64>>;

        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder.clone());
        let operation = ConstantOperation::<DataType, f64>::new(3.5);

        let outputs =
            InterpretableOperation::<DataType, Tracer<TestContext<'_>>>::interpret(&operation, &context, &[]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let builder = builder.borrow();
        assert!(builder.instructions().is_empty());
        assert!(matches!(&builder.atoms()[0], Atom::Constant(value) if *value == 3.5));
    }

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_value_constant_interpretation_forwards_same_builder_tracer() {
        type TestContext<'domain> = TracingContext<'domain, ScalarDomain<f64>>;

        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder.clone());
        let input = context.input(DataType::F64);
        let operation = ConstantOperation::<DataType, Tracer<TestContext<'_>>, Input>::new(input.clone());

        let outputs =
            InterpretableOperation::<DataType, Tracer<TestContext<'_>>>::interpret(&operation, &context, &[]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].atom_id(), input.atom_id());
        assert!(builder.borrow().instructions().is_empty());
    }

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_value_constant_interpretation_rejects_foreign_builder_tracer() {
        type TestContext<'domain> = TracingContext<'domain, ScalarDomain<f64>>;

        let domain = ScalarDomain::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let foreign_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let context = TracingContext::new(&domain, builder.clone());
        let foreign_context = TracingContext::new(&domain, foreign_builder);
        let foreign_input = foreign_context.input(DataType::F64);
        let operation = ConstantOperation::<DataType, Tracer<TestContext<'_>>, Input>::new(foreign_input);

        assert!(matches!(
            InterpretableOperation::<DataType, Tracer<TestContext<'_>>>::interpret(&operation, &context, &[]),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }
}
