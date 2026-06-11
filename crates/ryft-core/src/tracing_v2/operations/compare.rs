use crate::macros::check_count;
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ZeroTangentOperation};
use crate::types::ArrayType;

impl<
    V: Value<ArrayType>
        + crate::operations::manipulation::Broadcast<Output = V>
        + crate::operations::manipulation::Transpose,
    C,
> crate::tracing_v2::batching::BatchableOperation<V, C> for CompareOperation
where
    CompareOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        crate::tracing_v2::batching::apply_elementwise_batch(self, inputs)
    }
}

/// Comparison outputs are Boolean, so [`CompareOperation`] uses the zero-tangent forward-mode rule. The rule is
/// generic over the context's metadata type and applies to every context whose values can be compared and
/// interpreted, covering both array ([`ArrayType`]) and scalar ([`DataType`](crate::types::DataType)) programs.
impl<D: DifferentiationContext> ZeroTangentOperation<D> for CompareOperation
where
    D::Value: Compare<Output = D::Value>,
    Self: InterpretableOperation<D::Type, D::Value>,
{
}

/// JVP rule for [`CompareOperation`]: the Boolean primal output is computed from the input primals and paired with a
/// symbolic [`Tangent::Zero`](crate::differentiation::Tangent::Zero). Refer to the documentation of
/// [`ZeroTangentOperation`] for why this is sound.
impl<D: DifferentiationContext> DifferentiableOperation<D> for CompareOperation
where
    D::Value: Compare<Output = D::Value>,
    Self: Operation<D::Type> + InterpretableOperation<D::Type, D::Value>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        self.zero_tangent_jvp(context, inputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::DifferentiationContext;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over linearization tracers.
    fn piecewise_select<'domain>(
        x: crate::tracing_v2::LinearizationTracer<'domain, TestArrayDomain>,
    ) -> crate::tracing_v2::LinearizationTracer<'domain, TestArrayDomain> {
        let mask = x.clone().compare(x.zero_like(), ComparisonDirection::GreaterThan);
        Select::select(mask, x.clone() + x.clone(), x.clone() + x.clone() + x).unwrap()
    }

    #[test]
    fn test_compare_jvp_emits_zero_tangents_and_piecewise_select_derivatives() {
        // `f(x) = select(x > 0, 2x, 3x)`: the comparison output is Boolean, so its tangent is symbolically zero and
        // the derivative comes entirely from the selected branch (2 for x > 0 and 3 for x <= 0).
        let (primal, tangent) =
            TestArrayDomain.jvp(piecewise_select, TestArray::scalar(2.0), TestArray::scalar(1.0)).unwrap();
        assert_eq!(primal.values, vec![4.0]);
        assert_eq!(tangent.values, vec![2.0]);

        let (primal, tangent) =
            TestArrayDomain.jvp(piecewise_select, TestArray::scalar(-2.0), TestArray::scalar(1.0)).unwrap();
        assert_eq!(primal.values, vec![-6.0]);
        assert_eq!(tangent.values, vec![3.0]);
    }
}
