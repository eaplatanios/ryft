use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`XorOperation`].
pub const XOR_OPERATION_NAME: &str = "xor";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise exclusive disjunction (i.e., `left ^ right`) of
    /// two values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise exclusive disjunction: the two semantics coincide on Boolean element types, and
    /// StableHLO's [`xor`](https://openxla.org/stablehlo/spec#xor) operation likewise serves both.
    XorOperation, XOR_OPERATION_NAME,
    Xor, xor,
);

impl_differentiable_elementwise_operation!(@non_differentiable XorOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise exclusive-disjunction capability. [`Xor`] is the fallible Ryft counterpart to
    /// [`std::ops::BitXor`] that [`XorOperation`] interprets through, surfacing a
    /// [`ProgramError`](crate::ProgramError) when something goes wrong (e.g., when a value's data type does not
    /// support exclusive disjunction), instead of panicking. Value types additionally provide [`std::ops::BitXor`]
    /// as ergonomic (albeit panicking) sugar layered on top of this capability.
    Xor,
    /// Computes [`XorOperation`] elementwise for this value and `right`.
    xor(right),
    XorOperation,
);

define_tracer_operator!(@binary std::ops::BitXor, bitxor, XorOperation, "`xor` operation failed");

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    #[test]
    fn test_xor() {
        let operation = XorOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), XOR_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "XorOperation");
        assert_eq!(format!("{operation}"), XOR_OPERATION_NAME);
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        let outputs = operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[left, right]).unwrap();
        assert_eq!(outputs[0].values(), &[false, true, true, false]);

        // The `^` operator implementation matches the interpretation, including scalar broadcasting.
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!((left ^ right).values(), &[false, true, true, false]);
        assert_eq!((Array::vector(vec![true, false]) ^ Array::scalar(true)).values(), &[false, true]);

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
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Array, XorOperation>::new();
        let left = builder.add_input(input_type.clone());
        let right = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Array, Array), Array>(vec![program_output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[4], %1:bool[4] .
                let %2:bool[4] = xor %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_xor_batching() {
        check_operation_batching!(
            @exact,
            operation = XorOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false])),
                    (@replicated, Array::scalar(true)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]))],
            }],
        );
    }

    #[test]
    fn test_xor_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = XorOperation,
            inputs = [Scalar::from(true), Scalar::from(false)],
            expected = Scalar::from(true),
        );
    }
}
