use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError, TransposableOperation,
    TranspositionDriver,
};
use crate::operations::compare::CompareOperation;
use crate::partial::PartialValue;
use crate::programs::operations::Operation;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

/// Forward-mode rule for [`CompareOperation`]: comparisons map into a discrete (Boolean) codomain, so the primal
/// operation is replayed on the input primals and each Boolean output is paired with a canonical typed zero tangent.
impl<C: Context> DifferentiableOperation<C> for CompareOperation
where
    C::Operation: From<CompareOperation>,
    CompareOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Transpose rule for [`CompareOperation`]: comparisons map into a discrete (Boolean) codomain and are not linear
/// maps, so a tangent program never contains a primal `compare` on a linear operand (its forward pairs the replayed
/// primal with a zero tangent) and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for CompareOperation
where
    CompareOperation: Operation<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no partition-aware transpose rule", self.name()),
        }
        .into())
    }
}

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
