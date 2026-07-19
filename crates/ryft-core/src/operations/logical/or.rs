use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`OrOperation`].
pub const OR_OPERATION_NAME: &str = "or";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise disjunction (i.e., `left | right`) of two
    /// values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise disjunction: the two semantics coincide on Boolean element types, and StableHLO's
    /// [`or`](https://openxla.org/stablehlo/spec#or) operation likewise serves both.
    OrOperation, OR_OPERATION_NAME,
    Or, or,
);

impl_differentiable_elementwise_operation!(@non_differentiable OrOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise disjunction capability. [`Or`] is the fallible Ryft counterpart to
    /// [`std::ops::BitOr`] that [`OrOperation`] interprets through, surfacing a
    /// [`ProgramError`](crate::ProgramError) when something goes wrong (e.g., when a value's data type does not
    /// support disjunction), instead of panicking. Value types additionally provide [`std::ops::BitOr`] as
    /// ergonomic (albeit panicking) sugar layered on top of this capability.
    Or,
    /// Computes [`OrOperation`] elementwise for this value and `right`.
    or(right),
    OrOperation,
);

define_tracer_operator!(@binary std::ops::BitOr, bitor, OrOperation, "`or` operation failed");

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_or() {
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!((left | right).values(), &[true, true, true, false]);
        assert_eq!((Array::vector(vec![true, false]) | Array::scalar(false)).values(), &[true, false]);
    }

    #[test]
    fn test_or_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = OrOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );
    }

    #[test]
    fn test_or_batching() {
        check_operation_batching!(
            @exact,
            operation = OrOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false])),
                    (@replicated, Array::scalar(false)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]))],
            }],
        );
    }

    #[test]
    fn test_or_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = OrOperation,
            inputs = [Scalar::from(true), Scalar::from(false)],
            expected = Scalar::from(true),
        );
    }
}
