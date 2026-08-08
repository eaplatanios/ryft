use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

/// Canonical operation name for [`OrOperation`].
pub const OR_OPERATION_NAME: &str = "or";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise disjunction (i.e., `left | right`) of two values
    /// and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e., Boolean) and
    /// bitwise disjunction. The two semantics coincide on Boolean element types, and StableHLO's
    /// [`or`](https://openxla.org/stablehlo/spec#or) operation likewise serves both.
    OrOperation,
    OR_OPERATION_NAME,
    Or,
    or,
);

impl_differentiable_elementwise_operation!(@non_differentiable OrOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise disjunction capability. [`Or`] is the fallible Ryft counterpart to [`std::ops::BitOr`]
    /// that [`OrOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something goes
    /// wrong (e.g., when a value's data type does not support disjunction), instead of panicking. Value types
    /// additionally provide [`std::ops::BitOr`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    Or,
    /// Computes [`OrOperation`] elementwise for this value and `rhs`.
    or(rhs),
    OrOperation,
);

define_tracer_operator!(@binary std::ops::BitOr, bitor, capability = Or, method = or);

/// Implements [`Or`] for one host primitive type as logical or for `bool` and bitwise or for integers, matching
/// the reference backends and StableHLO.
macro_rules! impl_capability_for_primitive {
    // The `|` operator is logical for `bool` and bitwise for integer primitives, and cannot fail for either.
    ($type:ty) => {
        impl Or for $type {
            fn or(&self, rhs: &Self) -> Result<Self, ProgramError> {
                Ok(*self | *rhs)
            }
        }
    };
}

impl_capability_for_primitive!(bool);
impl_capability_for_primitive!(i8);
impl_capability_for_primitive!(i16);
impl_capability_for_primitive!(i32);
impl_capability_for_primitive!(i64);
impl_capability_for_primitive!(i128);
impl_capability_for_primitive!(isize);
impl_capability_for_primitive!(u8);
impl_capability_for_primitive!(u16);
impl_capability_for_primitive!(u32);
impl_capability_for_primitive!(u64);
impl_capability_for_primitive!(u128);
impl_capability_for_primitive!(usize);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, DataType};
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};

    use super::*;

    #[test]
    fn test_or() {
        // Check elementwise and scalar-broadcast eager value semantics.
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!((left | right).elements::<bool>(), Ok(vec![true, true, true, false]));
        assert_eq!((Array::vector(vec![true, false]) | Array::scalar(false)).elements::<bool>(), Ok(vec![true, false]));

        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @binary,
            operation = OrOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );

        // Check mixed mapped/replicated batching.
        check_operation_batching!(
            @exact,
            operation = OrOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false])),
                    (@replicated, Array::scalar(false)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]))],
            }],
        );

        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = OrOperation::new(),
            inputs = [Array::scalar(true), Array::scalar(false)],
            expected = Array::scalar(true),
        );
    }

    #[test]
    fn test_or_for_primitives() {
        assert_eq!(Or::or(&true, &false), Ok(true));
        assert_eq!(Or::or(&0b1100_u8, &0b1010), Ok(0b1110));
    }
}
