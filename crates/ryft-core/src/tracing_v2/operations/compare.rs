use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::operations::Operation;
use crate::operations::compare::CompareOperation;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiableOperation, JvpTracer, replay_zero_tangent};

/// Forward-mode rule for [`CompareOperation`]: comparisons map into a discrete (Boolean) codomain, so the primal
/// operation is replayed on the input primals and each Boolean output is paired with a canonical typed zero tangent.
impl<C: Context> DifferentiableOperation<C> for CompareOperation
where
    C::Operation: Clone + From<CompareOperation>,
    CompareOperation: Operation<C::Type>,
{
    fn jvp(&self, context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        replay_zero_tangent(context, self.clone(), inputs)
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
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no partition-aware transpose rule", self.name()),
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::DifferentiationContext;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over staged tracers of any context with [`TestArray`] semantics.
    fn piecewise_select<C>(x: crate::tracing::Tracer<C>) -> crate::tracing::Tracer<C>
    where
        C: crate::contexts::StagingContext<
                Type = crate::types::ArrayType,
                Constant = TestArray,
                Operation = crate::tracing_v2::ArrayOperation<TestArray>,
            >,
    {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan).unwrap();
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x)).unwrap()
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
