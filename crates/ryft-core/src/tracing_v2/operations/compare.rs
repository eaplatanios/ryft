use crate::operations::compare::CompareOperation;

crate::impl_non_differentiable_operation!(CompareOperation);

crate::impl_non_transposable_operation!(CompareOperation);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::programs::ProgramError;
    use crate::tests::TestArray;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over JVP duals of the eager [`TestArray`] context.
    fn piecewise_select(
        x: DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
    ) -> Result<DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan)?;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x))
    }

    #[test]
    fn test_compare_jvp_emits_zero_tangents_and_piecewise_select_derivatives() {
        // `f(x) = select(x > 0, 2x, 3x)`: the comparison output is Boolean, so its tangent is symbolically zero and
        // the derivative comes entirely from the selected branch (2 for x > 0 and 3 for x <= 0).
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(piecewise_select, TestArray::scalar(2.0), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(primal.values, vec![4.0]);
        assert_eq!(tangent.values, vec![2.0]);

        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(piecewise_select, TestArray::scalar(-2.0), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(primal.values, vec![-6.0]);
        assert_eq!(tangent.values, vec![3.0]);
    }
}
