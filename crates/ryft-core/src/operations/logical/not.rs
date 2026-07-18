use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_non_differentiable_operation, impl_non_transposable_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NotOperation`].
pub const NOT_OPERATION_NAME: &str = "not";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise negation (i.e., `!input`) of one value while
    /// preserving its type metadata. This operation covers both logical (i.e., Boolean) and bitwise negation: the
    /// two semantics coincide on Boolean element types, and StableHLO's
    /// [`not`](https://openxla.org/stablehlo/spec#not) operation likewise serves both.
    NotOperation, NOT_OPERATION_NAME,
    Not, not,
);

impl_non_differentiable_operation!(NotOperation);
impl_non_transposable_operation!(NotOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise negation capability. [`Not`] is the fallible Ryft counterpart to [`std::ops::Not`]
    /// that [`NotOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something
    /// goes wrong (e.g., when a value's data type does not support negation), instead of panicking. Value types
    /// additionally provide [`std::ops::Not`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    Not, not, NotOperation,
);

define_tracer_operator!(@unary std::ops::Not, not, NotOperation, "`not` operation failed");

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    #[test]
    fn test_not() {
        let operation = NotOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), NOT_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NotOperation");
        assert_eq!(format!("{operation}"), NOT_OPERATION_NAME);
        let input = Array::vector(vec![true, false, true]);
        let outputs = operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs[0].values(), &[false, true, false]);

        // The `!` operator implementation matches the interpretation.
        assert_eq!((!Array::vector(vec![true, false, true])).values(), &[false, true, false]);

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
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Array, NotOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
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

    #[test]
    fn test_not_batching() {
        check_operation!(
            @batching @exact,
            operation = NotOperation,
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), Array::vector(vec![true, false]))],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]))],
                },
                {
                    inputs = [(@replicated, Array::scalar(true))],
                    outputs = [(@replicated, Array::scalar(false))],
                },
            ],
        );
    }

    #[test]
    fn test_not_partial_evaluation() {
        check_operation!(
            @partial_evaluation @fold_and_residualize,
            operation = NotOperation,
            inputs = [Scalar::from(true)],
            expected = Scalar::from(false),
        );
    }
}
