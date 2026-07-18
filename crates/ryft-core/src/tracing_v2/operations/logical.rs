use crate::macros::{impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};

impl_non_differentiable_operation!(NotOperation);

impl_non_transposable_operation!(NotOperation);

impl_non_differentiable_operation!(AndOperation);

impl_non_transposable_operation!(AndOperation);

impl_non_differentiable_operation!(OrOperation);

impl_non_transposable_operation!(OrOperation);

impl_non_differentiable_operation!(XorOperation);

impl_non_transposable_operation!(XorOperation);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::control_flow::Select;
    use crate::programs::ProgramError;
    use crate::tests::TestArray;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};

    /// `f(x) = select((x > 0) & (x > 1), 2x, 3x)` expressed over JVP duals of the eager [`TestArray`] context.
    fn masked_select(
        x: DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
    ) -> Result<DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
        let positive = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan)?;
        let above_one = x.compare(&x.one_like(), ComparisonDirection::GreaterThan)?;
        let mask = positive & above_one;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x))
    }

    #[test]
    fn test_logical_jvp_emits_zero_tangents_and_piecewise_select_derivatives() {
        // The logical conjunction of two Boolean comparisons drives the select, so the derivative is 2 when both
        // predicates hold (x > 1) and 3 otherwise.
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(masked_select, TestArray::scalar(2.0), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(primal.values, vec![4.0]);
        assert_eq!(tangent.values, vec![2.0]);

        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(masked_select, TestArray::scalar(0.5), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(primal.values, vec![1.5]);
        assert_eq!(tangent.values, vec![3.0]);
    }
}
