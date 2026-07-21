use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

/// Canonical operation name for [`NotOperation`].
pub const NOT_OPERATION_NAME: &str = "not";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise negation (i.e., `!input`) of one value while
    /// preserving its type metadata. This operation covers both logical (i.e., Boolean) and bitwise negation. The two
    /// semantics coincide on Boolean element types, and StableHLO's [`not`](https://openxla.org/stablehlo/spec#not)
    /// operation likewise serves both.
    NotOperation,
    NOT_OPERATION_NAME,
    Not,
    not,
);

impl_differentiable_elementwise_operation!(@non_differentiable NotOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise negation capability. [`Not`] is the fallible Ryft counterpart to [`std::ops::Not`]
    /// that [`NotOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something
    /// goes wrong (e.g., when a value's data type does not support negation), instead of panicking. Value types
    /// additionally provide [`std::ops::Not`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    Not,
    /// Computes [`NotOperation`] elementwise for this value.
    not,
    NotOperation,
);

define_tracer_operator!(@unary std::ops::Not, not, NotOperation, "`not` operation failed");

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_not() {
        // Check the operation-specific eager value semantics.
        assert_eq!((!Array::vector(vec![true, false, true])).values(), &[false, true, false]);

        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @unary,
            operation = NotOperation,
            cases = [{
                input_data_types = [DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );

        // Check mapped and replicated batching behavior.
        check_operation_batching!(
            @exact,
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

        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = NotOperation,
            inputs = [Scalar::from(true)],
            expected = Scalar::from(false),
        );
    }
}
