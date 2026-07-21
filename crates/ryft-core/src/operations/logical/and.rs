use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

/// Canonical operation name for [`AndOperation`].
pub const AND_OPERATION_NAME: &str = "and";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise conjunction (i.e., `left & right`) of two
    /// values and typically supports broadcasting semantics for arrays. This operation covers both logical (i.e.,
    /// Boolean) and bitwise conjunction. The two semantics coincide on Boolean element types, and StableHLO's
    /// [`and`](https://openxla.org/stablehlo/spec#and) operation likewise serves both.
    AndOperation,
    AND_OPERATION_NAME,
    And,
    and,
);

impl_differentiable_elementwise_operation!(@non_differentiable AndOperation);

define_elementwise_capability!(
    @binary
    /// Value-level elementwise conjunction capability. [`And`] is the fallible Ryft counterpart to [`std::ops::BitAnd`]
    /// that [`AndOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something goes
    /// wrong (e.g., when a value's data type does not support conjunction), instead of panicking. Value types
    /// additionally provide [`std::ops::BitAnd`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    And,
    /// Computes [`AndOperation`] elementwise for this value and `rhs`.
    and(rhs),
    AndOperation,
);

define_tracer_operator!(@binary std::ops::BitAnd, bitand, AndOperation, "`and` operation failed");

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::{DifferentiationTracer, jvp};
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::control_flow::Select;
    use crate::programs::ProgramError;
    use crate::types::DataType;

    use super::*;

    /// Computes `f(x) = select((x > 0) & (x > 1), 2x, 3x)`.
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
        // Check elementwise and scalar-broadcast eager value semantics.
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!((left & right).values(), &[true, false, false, false]);
        assert_eq!((Array::vector(vec![true, false]) & Array::scalar(true)).values(), &[true, false]);

        // Check the shared elementwise type-inference contract in both type universes.
        check_operation_type_inference!(
            @elementwise @binary,
            operation = AndOperation,
            cases = [{
                input_data_types = [DataType::Boolean, DataType::Boolean],
                output_data_types = [DataType::Boolean],
            }],
        );

        // Check both mixed mapped/replicated operand orderings.
        check_operation_batching!(
            @exact,
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

        // The logical conjunction of two Boolean comparisons drives the select, so the derivative is 2 when both
        // predicates hold (x > 1) and 3 otherwise.
        let (primal, tangent) = jvp(masked_select, Array::scalar(2.0), Array::scalar(1.0)).unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![2.0]);

        let (primal, tangent) = jvp(masked_select, Array::scalar(0.5), Array::scalar(1.0)).unwrap();
        assert_eq!(primal.to_f64s(), vec![1.5]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);

        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = AndOperation,
            inputs = [Scalar::from(true), Scalar::from(false)],
            expected = Scalar::from(false),
        );
    }
}
