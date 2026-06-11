use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::InterpretableOperation;
use crate::operations::control_flow::{Select, SelectOperation};
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::differentiation::{JvpTracer, ResidualFactor, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, LinearOperationOf};
use crate::types::{ArrayType, Type, Typed};

/// Trait that represents linear [`Operation`](crate::operations::Operation) types that support/include a
/// captured-condition select. The captured-condition select is the linear-program counterpart of
/// [`SelectOperation`]: the Boolean condition is a primal value captured at linearization time, the two inputs are
/// the tangents (or cotangents) of the `on_true` and `on_false` branches, and the map
/// `(t, f) ↦ select(condition, t, f)` is linear in `(t, f)`. Linear operation enums implement this trait so that
/// the JVP rule of [`SelectOperation`] can stage the captured-condition select without knowing which linear
/// operation type is in use.
pub trait SupportsLinearSelect<T: Type, F> {
    /// Constructs the linear-operation representation of the captured-condition select.
    fn linear_select_operation(condition: F) -> Self;
}

impl<
    V: Value<ArrayType>
        + crate::operations::manipulation::Broadcast<Output = V>
        + crate::operations::manipulation::Transpose,
    C,
> crate::tracing_v2::batching::BatchableOperation<V, C> for SelectOperation
where
    SelectOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        crate::tracing_v2::batching::apply_elementwise_batch(self, inputs)
    }
}

/// JVP rule for [`SelectOperation`], mirroring JAX's rule for
/// [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html): the primal output is
/// `select(condition, on_true, on_false)` over the input primals, and the tangent is
/// `select(condition, on_true_tangent, on_false_tangent)` with the same primal condition. The condition is Boolean,
/// so its own tangent is identically zero and is ignored. When both branch tangents are symbolic
/// [`Tangent::Zero`]s, the output tangent is a symbolic zero of the output type and no linear operation is staged;
/// otherwise the rule captures the condition as a residual factor and stages the captured-condition select provided
/// by [`SupportsLinearSelect`].
impl<D> DifferentiableOperation<D> for SelectOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Select<Condition = D::Value>,
    LinearOperationOf<D>: SupportsLinearSelect<ArrayType, ResidualFactor<ArrayType, D::Value>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 3, ProgramError);
        let condition = &inputs[0];
        let on_true = &inputs[1];
        let on_false = &inputs[2];
        let primal = D::Value::select(condition.primal().clone(), on_true.primal().clone(), on_false.primal().clone())?;
        if on_true.tangent().is_zero() && on_false.tangent().is_zero() {
            let tangent_type = primal.r#type().into_owned();
            return Ok(vec![JvpTracer::from_zero_tangent(primal, tangent_type)]);
        }
        let condition_factor = condition.factor(context);
        let on_true_tangent = context.materialize_tangent(on_true.tangent().clone())?;
        let on_false_tangent = context.materialize_tangent(on_false.tangent().clone())?;
        let mut outputs = context.stage_operation(
            LinearOperationOf::<D>::linear_select_operation(condition_factor),
            &[on_true_tangent, on_false_tangent],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{DifferentiableDomainExtension, jacrev};

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over linearization tracers.
    fn piecewise_select<'domain>(
        x: crate::tracing_v2::LinearizationTracer<'domain, TestArrayDomain>,
    ) -> crate::tracing_v2::LinearizationTracer<'domain, TestArrayDomain> {
        let mask = x.clone().compare(x.zero_like(), ComparisonDirection::GreaterThan);
        Select::select(mask, x.clone() + x.clone(), x.clone() + x.clone() + x).unwrap()
    }

    #[test]
    fn test_select_jacrev_computes_piecewise_derivative() {
        // Reverse mode through `f(x) = select(x > 0, 2x, 3x)` exercises the captured-condition select transpose:
        // the on_true cotangent is `select(condition, cotangent, 0)` and the on_false cotangent is
        // `select(condition, 0, cotangent)`.
        let jacobian = jacrev(&TestArrayDomain, |x| Ok(piecewise_select(x)), TestArray::scalar(2.0)).unwrap();
        assert_close(jacobian.rows().partials().values()[0], 2.0);

        let jacobian = jacrev(&TestArrayDomain, |x| Ok(piecewise_select(x)), TestArray::scalar(-2.0)).unwrap();
        assert_close(jacobian.rows().partials().values()[0], 3.0);
    }

    #[test]
    fn test_select_jacfwd_computes_piecewise_derivative() {
        // Forward mode through the same function exercises the captured-condition select under batched basis
        // tangents (the direct batched JVP path).
        let jacobian = TestArrayDomain.jacfwd(|x| Ok(piecewise_select(x)), TestArray::scalar(2.0)).unwrap();
        assert_close(jacobian.rows().partials().values()[0], 2.0);

        let jacobian = TestArrayDomain.jacfwd(|x| Ok(piecewise_select(x)), TestArray::scalar(-2.0)).unwrap();
        assert_close(jacobian.rows().partials().values()[0], 3.0);
    }

    #[test]
    fn test_select_jacrev_over_vector_masks_per_element() {
        // Per-element masking over a vector input: the Jacobian of `select(x > 0, 2x, 3x)` is diagonal with entries
        // 2 where x > 0 and 3 elsewhere.
        let jacobian =
            jacrev(&TestArrayDomain, |x| Ok(piecewise_select(x)), TestArray::vector(vec![1.0, -1.0])).unwrap();
        let block = jacobian.rows().partials();
        assert_eq!(block.output_shape(), &[2]);
        assert_eq!(block.input_shape(), &[2]);
        assert_close(block.values()[0], 2.0);
        assert_close(block.values()[1], 0.0);
        assert_close(block.values()[2], 0.0);
        assert_close(block.values()[3], 3.0);
    }
}
