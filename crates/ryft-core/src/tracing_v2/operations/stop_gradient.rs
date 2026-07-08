use crate::contexts::Context;
use crate::differentiation::DifferentiationDual;
use crate::differentiation::TransposableOperation;
use crate::operations::Operation;
use crate::operations::stop_gradient::StopGradientOperation;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::differentiation::DifferentiableOperation;

/// Forward-mode rule for [`StopGradientOperation`]: the operation is the identity on the primal but severs the
/// tangent, so the primal is replayed (re-tagging the stop-gradient boundary) and paired with a typed zero tangent.
impl<C: Context> DifferentiableOperation<C> for StopGradientOperation
where
    C::Operation: Clone + From<StopGradientOperation>,
    StopGradientOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Transpose rule for [`StopGradientOperation`]: the operation severs the tangent, so a tangent program never
/// contains a primal `stop_gradient` on a linear operand (its forward pairs the replayed primal with a zero tangent)
/// and the rule reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for StopGradientOperation
where
    StopGradientOperation: Operation<V::Type>,
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
    use crate::contexts::EagerContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::stop_gradient::StopGradient;
    use crate::scalars::Scalar;
    use crate::tracing_v2::{Differentiate, value_and_gradient};

    #[test]
    fn test_stop_gradient_jvp_severs_the_tangent() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain.jvp(|x| Ok(x.stop_gradient()), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_eq!(primal, 2.0);
        assert_eq!(tangent, 0.0);
    }

    #[test]
    fn test_stop_gradient_composes_with_batch() {
        use crate::batching::Batch;
        use crate::batching::BatchAxis;
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| Ok(x.clone() * x.stop_gradient()),
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_stop_gradient_treats_the_marked_value_as_a_constant() {
        // The JAX documentation example: `f(x) = x * stop_gradient(x)` differentiates like
        // `x * c` with `c` frozen at the primal value, so `f'(x) = stop_gradient(x)`.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (value, gradient) =
            value_and_gradient(&domain, |x| x.clone() * x.stop_gradient(), Scalar::from(3.0)).unwrap();
        assert_eq!(value, 9.0);
        assert_eq!(gradient, 3.0);
    }
}
