use crate::operations::logical::{LogicalBinary, LogicalNot, LogicalOperation};
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ZeroTangentOperation};
use crate::types::ArrayType;

/// Logical inputs and outputs are Boolean, so [`LogicalOperation`] uses the zero-tangent forward-mode rule.
impl<D> ZeroTangentOperation<D> for LogicalOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: LogicalBinary + LogicalNot,
{
}

/// JVP rule for [`LogicalOperation`]: the Boolean primal output is computed from the input primals and paired with a
/// symbolic [`Tangent::Zero`](crate::differentiation::Tangent::Zero). Refer to the documentation of
/// [`ZeroTangentOperation`] for why this is sound.
impl<D> DifferentiableOperation<D> for LogicalOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: LogicalBinary + LogicalNot,
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
    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::control_flow::Select;
    use crate::operations::logical::{LogicalBinary, LogicalKind};
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::DifferentiationContext;

    /// `f(x) = select((x > 0) & (x > 1), 2x, 3x)` expressed over linearization tracers.
    fn masked_select<'domain>(
        x: crate::tracing_v2::LinearizationTracer<'domain, TestArrayDomain>,
    ) -> crate::tracing_v2::LinearizationTracer<'domain, TestArrayDomain> {
        let positive = x.clone().compare(x.zero_like(), ComparisonDirection::GreaterThan);
        let above_one = x.clone().compare(x.one_like(), ComparisonDirection::GreaterThan);
        let mask = positive.logical_binary(above_one, LogicalKind::And);
        Select::select(mask, x.clone() + x.clone(), x.clone() + x.clone() + x).unwrap()
    }

    #[test]
    fn test_logical_jvp_emits_zero_tangents_and_piecewise_select_derivatives() {
        // The logical conjunction of two Boolean comparisons drives the select, so the derivative is 2 when both
        // predicates hold (x > 1) and 3 otherwise.
        let (primal, tangent) =
            TestArrayDomain.jvp(masked_select, TestArray::scalar(2.0), TestArray::scalar(1.0)).unwrap();
        assert_eq!(primal.values, vec![4.0]);
        assert_eq!(tangent.values, vec![2.0]);

        let (primal, tangent) =
            TestArrayDomain.jvp(masked_select, TestArray::scalar(0.5), TestArray::scalar(1.0)).unwrap();
        assert_eq!(primal.values, vec![1.5]);
        assert_eq!(tangent.values, vec![3.0]);
    }
}
