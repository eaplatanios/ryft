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
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_xor() {
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!((left ^ right).values(), &[false, true, true, false]);
        assert_eq!((Array::vector(vec![true, false]) ^ Array::scalar(true)).values(), &[false, true]);
    }

    #[test]
    fn test_xor_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = XorOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
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
