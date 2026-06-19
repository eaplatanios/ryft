use std::fmt::Display;
use std::ops::Not;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::ArrayType;

/// Canonical operation name for [`NotOperation`].
pub const NOT_OPERATION_NAME: &'static str = "not";

/// [`Operation`] that computes the elementwise negation (i.e., `!input`) of one value. This operation covers both
/// logical (i.e., Boolean) and bitwise negation: the two semantics coincide on Boolean element types, and StableHLO's
/// [`not`](https://openxla.org/stablehlo/spec#not) operation likewise serves both. Value types provide the
/// elementwise behavior through the standard [`Not`] operator trait.
#[derive(Clone, Debug, Default)]
pub struct NotOperation;

impl Display for NotOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl ElementwiseOperation for NotOperation {
    #[inline]
    fn name(&self) -> &'static str {
        NOT_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Value<ArrayType> + Not<Output = V>> InterpretableOperation<ArrayType, V> for NotOperation {
    #[inline]
    fn interpret(
        &self,
        _context: &mut <V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![!inputs[0].clone()])
    }
}

impl<C: StagingContext<Operation: From<NotOperation>>> Not for Tracer<C> {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        self.unary(NotOperation)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_not() {
        let operation = NotOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), NOT_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NotOperation");
        assert_eq!(format!("{operation}"), NOT_OPERATION_NAME);
        let input = TestArray::vector(vec![1.0, 0.0, 1.0]);
        let outputs = operation.interpret(&mut (), &[input]).unwrap();
        assert_eq!(outputs[0].values(), &[0.0, 1.0, 0.0]);

        // The `!` operator implementation matches the interpretation.
        assert_eq!((!TestArray::vector(vec![1.0, 0.0, 1.0])).values(), &[0.0, 1.0, 0.0]);

        // Array type inference preserves the Boolean input type.
        let input_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, std::slice::from_ref(&input_type)),
            Ok(vec![input_type.clone()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &mut (), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, NotOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[3] .
                let %1:bool[3] = not %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
