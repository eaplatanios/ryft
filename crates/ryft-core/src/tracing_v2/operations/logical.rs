use std::ops::{BitAnd, BitOr, BitXor, Not};

use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ZeroTangentOperation};
use crate::types::ArrayType;

/// Logical inputs and outputs are Boolean, so [`NotOperation`] uses the zero-tangent forward-mode rule.
impl<D> ZeroTangentOperation<D> for NotOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Not<Output = D::Value>,
{
}

/// JVP rule for [`NotOperation`]: the Boolean primal output is computed from the input primals and paired with a
/// symbolic [`Tangent::Zero`](crate::differentiation::Tangent::Zero). Refer to the documentation of
/// [`ZeroTangentOperation`] for why this is sound.
impl<D> DifferentiableOperation<D> for NotOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Not<Output = D::Value>,
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

/// Logical inputs and outputs are Boolean, so [`AndOperation`] uses the zero-tangent forward-mode rule.
impl<D> ZeroTangentOperation<D> for AndOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: BitAnd<Output = D::Value>,
{
}

/// JVP rule for [`AndOperation`]: the Boolean primal output is computed from the input primals and paired with a
/// symbolic [`Tangent::Zero`](crate::differentiation::Tangent::Zero). Refer to the documentation of
/// [`ZeroTangentOperation`] for why this is sound.
impl<D> DifferentiableOperation<D> for AndOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: BitAnd<Output = D::Value>,
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

/// Logical inputs and outputs are Boolean, so [`OrOperation`] uses the zero-tangent forward-mode rule.
impl<D> ZeroTangentOperation<D> for OrOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: BitOr<Output = D::Value>,
{
}

/// JVP rule for [`OrOperation`]: the Boolean primal output is computed from the input primals and paired with a
/// symbolic [`Tangent::Zero`](crate::differentiation::Tangent::Zero). Refer to the documentation of
/// [`ZeroTangentOperation`] for why this is sound.
impl<D> DifferentiableOperation<D> for OrOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: BitOr<Output = D::Value>,
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

/// Logical inputs and outputs are Boolean, so [`XorOperation`] uses the zero-tangent forward-mode rule.
impl<D> ZeroTangentOperation<D> for XorOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: BitXor<Output = D::Value>,
{
}

/// JVP rule for [`XorOperation`]: the Boolean primal output is computed from the input primals and paired
/// with a symbolic [`Tangent::Zero`](crate::differentiation::Tangent::Zero). Refer to the documentation of
/// [`ZeroTangentOperation`] for why this is sound.
impl<D> DifferentiableOperation<D> for XorOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: BitXor<Output = D::Value>,
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
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::DifferentiationContext;

    /// `f(x) = select((x > 0) & (x > 1), 2x, 3x)` expressed over staged tracers of any context with [`TestArray`]
    /// semantics.
    fn masked_select<C>(x: crate::tracing::Tracer<C>) -> crate::tracing::Tracer<C>
    where
        C: crate::contexts::StagingContext<
                Type = crate::types::ArrayType,
                Constant = TestArray,
                Operation = crate::tracing_v2::ArrayOperation<crate::types::ArrayType, TestArray>,
            >,
    {
        let positive = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan);
        let above_one = x.compare(&x.one_like(), ComparisonDirection::GreaterThan);
        let mask = positive & above_one;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x)).unwrap()
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
