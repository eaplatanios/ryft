use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_non_differentiable_operation, impl_non_transposable_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AndOperation`].
pub const AND_OPERATION_NAME: &str = "and";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise conjunction (i.e., `left & right`) of two
    /// values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise conjunction: the two semantics coincide on Boolean element types, and StableHLO's
    /// [`and`](https://openxla.org/stablehlo/spec#and) operation likewise serves both.
    AndOperation, AND_OPERATION_NAME,
    And, and,
);

impl_non_differentiable_operation!(AndOperation);
impl_non_transposable_operation!(AndOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise conjunction capability. [`And`] is the fallible Ryft counterpart to
    /// [`std::ops::BitAnd`] that [`AndOperation`] interprets through, surfacing a
    /// [`ProgramError`](crate::ProgramError) when something goes wrong (e.g., when a value's data type does not
    /// support conjunction), instead of panicking. Value types additionally provide [`std::ops::BitAnd`] as
    /// ergonomic (albeit panicking) sugar layered on top of this capability.
    And, and, AndOperation,
);

define_tracer_operator!(@binary std::ops::BitAnd, bitand, AndOperation, "`and` operation failed");

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::differentiation::forward::ForwardModeDifferentiate;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::control_flow::Select;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    /// `f(x) = select((x > 0) & (x > 1), 2x, 3x)` expressed over JVP duals of the eager [`Array`] context.
    fn masked_select(
        x: DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>,
    ) -> Result<DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
        let positive = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan)?;
        let above_one = x.compare(&x.one_like(), ComparisonDirection::GreaterThan)?;
        let mask = positive & above_one;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x))
    }

    #[test]
    fn test_and() {
        let operation = AndOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<ArrayType>::name(&operation), AND_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "AndOperation");
        assert_eq!(format!("{operation}"), AND_OPERATION_NAME);
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        let outputs = operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[left, right]).unwrap();
        assert_eq!(outputs[0].values(), &[true, false, false, false]);

        // The `&` operator implementation matches the interpretation, including scalar broadcasting.
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!((left & right).values(), &[true, false, false, false]);
        assert_eq!((Array::vector(vec![true, false]) & Array::scalar(true)).values(), &[true, false]);

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
        let mut builder = ProgramBuilder::<Array, AndOperation>::new();
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
                let %2:bool[4] = and %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_and_batching() {
        check_operation!(
            @batching @exact,
            operation = AndOperation,
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![true, false])),
                        (@replicated, Array::scalar(true)),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]))],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(false)),
                        (@mapped(axis = 0), Array::vector(vec![true, false])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, false]))],
                },
            ],
        );
    }

    #[test]
    fn test_and_differentiation() {
        // The logical conjunction of two Boolean comparisons drives the select, so the derivative is 2 when both
        // predicates hold (x > 1) and 3 otherwise.
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(masked_select, Array::scalar(2.0), Array::scalar(1.0))
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![2.0]);

        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(masked_select, Array::scalar(0.5), Array::scalar(1.0))
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![1.5]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_and_partial_evaluation() {
        check_operation!(
            @partial_evaluation @fold_and_residualize,
            operation = AndOperation,
            inputs = [Scalar::from(true), Scalar::from(false)],
            expected = Scalar::from(false),
        );
    }
}
