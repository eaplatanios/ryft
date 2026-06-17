use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::ProgramError;
use crate::types::{Type, TypeError, Typed};

/// Canonical operation name for [`ConstantOperation`].
pub const CONSTANT_OPERATION_NAME: &'static str = "constant";

/// [`Operation`] that has no inputs and produces a single output equal to a captured typed value. [`ConstantOperation`]
/// is a true literal constant. It carries a `V` value that is [`Typed`] against the operation's [`Type`] `T`, and so
/// its output type is exactly the value's type, and interpreting it simply clones the captured value. Unlike
/// [`FillOperation`](super::FillOperation), it does not synthesize a value from a scalar; it returns the value the
/// caller already provided when constructing it.
#[derive(Copy, Clone, Debug)]
pub struct ConstantOperation<T: Type, V: Clone + Typed<T>> {
    /// Captured value produced by this [`Operation`] when interpreted.
    value: V,

    /// [`PhantomData`] marker tying the captured value to the [`Type`] it is typed against. The `fn() -> T` form
    /// indexes by `T` without owning one, and so this operation's `Send` and `Sync` depend only on the captured value
    /// (as well as any trait implementations derived using `#[derive]`).
    marker: PhantomData<fn() -> T>,
}

impl<T: Type, V: Clone + Typed<T>> ConstantOperation<T, V> {
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

impl<T: Type, V: Clone + Display + Typed<T>> Display for ConstantOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Clone + Display + Typed<T>> Operation<T> for ConstantOperation<T, V> {
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

impl<T: Type, V: Clone + Display + Typed<T>> InterpretableOperation<T, V> for ConstantOperation<T, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![self.value.clone()])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::types::{DataType, TypeError};

    use super::*;

    #[test]
    fn test_constant() {
        let operation = ConstantOperation::<DataType, f64>::new(3.5);

        assert_eq!(Operation::<DataType>::name(&operation), CONSTANT_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            "ConstantOperation { value: 3.5, marker: PhantomData<fn() -> ryft_core::types::data_types::DataType> }"
        );
        assert_eq!(format!("{operation}"), "constant [value=3.5]");
        assert_eq!(operation.value(), &3.5);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[]), Ok(vec![3.5]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[0.0]),
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
}
