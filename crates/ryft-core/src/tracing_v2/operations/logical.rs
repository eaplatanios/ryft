use crate::contexts::Context;
use crate::differentiation::TransposableOperation;
use crate::operations::Operation;
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiableOperation, DifferentiationDual, replay_zero_tangent};
use crate::types::ArrayType;

/// Implements the erroring [`TransposableOperation`] rule for Boolean-codomain logical operations: they are not
/// linear maps, so a tangent program never contains them on a linear operand (their forwards pair the replayed
/// primal with a zero tangent) and each rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
macro_rules! logical_unsupported_transpose {
    ($operation:ty) => {
        impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>> TransposableOperation<V, O> for $operation {
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
    };
}

logical_unsupported_transpose!(NotOperation);
logical_unsupported_transpose!(AndOperation);
logical_unsupported_transpose!(OrOperation);
logical_unsupported_transpose!(XorOperation);

/// Forward-mode rule for [`NotOperation`]: a Boolean output has no tangent, so the primal operation is replayed
/// on the input primals and paired with a canonical typed zero tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for NotOperation
where
    C::Operation: Clone + From<NotOperation>,
    NotOperation: Operation<ArrayType>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        replay_zero_tangent(context, self.clone(), inputs)
    }
}

/// Forward-mode rule for [`AndOperation`]: a Boolean output has no tangent, so the primal operation is replayed
/// on the input primals and paired with a canonical typed zero tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for AndOperation
where
    C::Operation: Clone + From<AndOperation>,
    AndOperation: Operation<ArrayType>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        replay_zero_tangent(context, self.clone(), inputs)
    }
}

/// Forward-mode rule for [`OrOperation`]: a Boolean output has no tangent, so the primal operation is replayed on
/// the input primals and paired with a canonical typed zero tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for OrOperation
where
    C::Operation: Clone + From<OrOperation>,
    OrOperation: Operation<ArrayType>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        replay_zero_tangent(context, self.clone(), inputs)
    }
}

/// Forward-mode rule for [`XorOperation`]: a Boolean output has no tangent, so the primal operation is replayed
/// on the input primals and paired with a canonical typed zero tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for XorOperation
where
    C::Operation: Clone + From<XorOperation>,
    XorOperation: Operation<ArrayType>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        replay_zero_tangent(context, self.clone(), inputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::control_flow::Select;
    use crate::programs::ProgramError;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::Differentiate;
    use crate::tracing_v2::differentiation::DifferentiationTracer;

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
