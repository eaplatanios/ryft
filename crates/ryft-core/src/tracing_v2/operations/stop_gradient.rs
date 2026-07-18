use crate::contexts::Context;
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::impl_non_transposable_operation;
use crate::operations::stop_gradient::StopGradientOperation;
use crate::programs::operations::Operation;

/// Forward-mode rule for [`StopGradientOperation`]: the operation is the identity on the primal but severs the
/// tangent, so the primal is replayed (re-tagging the stop-gradient boundary) and paired with a typed zero tangent.
impl<C: Context> DifferentiableOperation<C> for StopGradientOperation
where
    C::Type: crate::differentiation::DifferentiableType,
    C::Operation: From<StopGradientOperation>,
    StopGradientOperation: Operation<C::Type>,
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

impl_non_transposable_operation!(StopGradientOperation);

#[cfg(test)]
mod tests {
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::operations::stop_gradient::StopGradient;
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};

    #[test]
    fn test_stop_gradient_jvp_severs_the_tangent() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain.jvp(|x| Ok(x.stop_gradient()), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_eq!(primal, 2.0);
        assert_eq!(tangent, 0.0);
    }

    #[test]
    fn test_stop_gradient_composes_with_batch() {
        use crate::batching::{Batch, BatchAxis};
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
            domain.value_and_gradient(|x| x.clone() * x.stop_gradient(), Scalar::from(3.0)).unwrap();
        assert_eq!(value, 9.0);
        assert_eq!(gradient, 3.0);
    }
}
