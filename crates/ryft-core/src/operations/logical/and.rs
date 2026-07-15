use crate::macros::{define_elementwise_operation, define_tracer_operator};

/// Canonical operation name for [`AndOperation`].
pub const AND_OPERATION_NAME: &str = "and";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise conjunction (i.e., `left & right`) of two
    /// values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise conjunction: the two semantics coincide on Boolean element types, and StableHLO's
    /// [`and`](https://openxla.org/stablehlo/spec#and) operation likewise serves both.
    AndOperation, AND_OPERATION_NAME, And, and,
    /// Value-level elementwise conjunction capability. [`And`] is the fallible Ryft counterpart to
    /// [`std::ops::BitAnd`] that [`AndOperation`] interprets through, surfacing a
    /// [`ProgramError`](crate::ProgramError) when something goes wrong (e.g., when a value's data type does not
    /// support conjunction), instead of panicking. Value types additionally provide [`std::ops::BitAnd`] as
    /// ergonomic (albeit panicking) sugar layered on top of this capability.
);

define_tracer_operator!(@binary std::ops::BitAnd, bitand, AndOperation, "`and` operation failed");

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_and() {
        let operation = AndOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), AND_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "AndOperation");
        assert_eq!(format!("{operation}"), AND_OPERATION_NAME);
        let lhs = TestArray::vector(vec![1.0, 1.0, 0.0, 0.0]);
        let rhs = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        let outputs = operation.interpret(&EagerContext::<TestArray>::new(), &EmptyRegionDriver, &[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].values(), &[1.0, 0.0, 0.0, 0.0]);

        // The `&` operator implementation matches the interpretation, including scalar broadcasting.
        let lhs = TestArray::vector(vec![1.0, 1.0, 0.0, 0.0]);
        let rhs = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!((lhs & rhs).values(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!((TestArray::vector(vec![1.0, 0.0]) & TestArray::scalar(1.0)).values(), &[1.0, 0.0]);

        // Array type inference broadcasts the Boolean input types.
        let input_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[ArrayType::scalar(DataType::Boolean), input_type.clone()],
                &[],
            ),
            Ok(vec![input_type.clone()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, std::slice::from_ref(&input_type), &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<crate::EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::<TestArray>::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<TestArray, AndOperation>::new();
        let left = builder.add_input(input_type.clone());
        let right = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![program_output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[4], %1:bool[4] .
                let %2:bool[4] = and %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
