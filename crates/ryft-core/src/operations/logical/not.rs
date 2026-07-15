use crate::macros::{define_elementwise_operation, define_tracer_operator};

/// Canonical operation name for [`NotOperation`].
pub const NOT_OPERATION_NAME: &str = "not";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise negation (i.e., `!input`) of one value while
    /// preserving its type metadata. This operation covers both logical (i.e., Boolean) and bitwise negation: the
    /// two semantics coincide on Boolean element types, and StableHLO's
    /// [`not`](https://openxla.org/stablehlo/spec#not) operation likewise serves both.
    NotOperation, NOT_OPERATION_NAME, Not, not,
    /// Value-level elementwise negation capability. [`Not`] is the fallible Ryft counterpart to [`std::ops::Not`]
    /// that [`NotOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something
    /// goes wrong (e.g., when a value's data type does not support negation), instead of panicking. Value types
    /// additionally provide [`std::ops::Not`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
);

define_tracer_operator!(@unary std::ops::Not, not, NotOperation, "`not` operation failed");

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::regions::EmptyRegionDriver;
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_not() {
        let operation = NotOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), NOT_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NotOperation");
        assert_eq!(format!("{operation}"), NOT_OPERATION_NAME);
        let input = TestArray::vector(vec![1.0, 0.0, 1.0]);
        let outputs = operation.interpret(&EagerContext::<TestArray>::new(), &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs[0].values(), &[0.0, 1.0, 0.0]);

        // The `!` operator implementation matches the interpretation.
        assert_eq!((!TestArray::vector(vec![1.0, 0.0, 1.0])).values(), &[0.0, 1.0, 0.0]);

        // Array type inference preserves the Boolean input type.
        let input_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, std::slice::from_ref(&input_type), &[]),
            Ok(vec![input_type.clone()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::<TestArray>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<TestArray, NotOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input], Vec::new()).unwrap()[0];
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
