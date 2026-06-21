use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{MaybeZeroOperation, ZeroOperation};
use crate::operations::control_flow::{SELECT_OPERATION_NAME, Select, SelectCondition, SelectOperation};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::operations::primitive::transpose_captured_condition_select;
use crate::tracing_v2::{CapturedFactor, DifferentiableOperation, DifferentiationContext, LinearOperationOf};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Captured-condition select operation used in linear tangent and cotangent programs.
///
/// The ordinary [`SelectOperation`] is not the linear primitive because its operand list is
/// `(condition, on_true, on_false)`, and the Boolean condition is a primal value with no tangent space. The linear map
/// produced by the JVP of `select(condition, on_true, on_false)` instead fixes that primal condition as a captured
/// factor and acts only on the two branch tangents (or cotangents):
///
/// ```text
/// (on_true_tangent, on_false_tangent) -> select(condition, on_true_tangent, on_false_tangent)
/// ```
///
/// This operation stores that captured `condition` factor in the operation payload, which lets residualized
/// pushforwards remap or instantiate it like other captured factors. Its transpose routes the output cotangent into
/// the selected branch: `select(condition, cotangent, 0)` for the `on_true` input and
/// `select(condition, 0, cotangent)` for the `on_false` input.
#[derive(Clone, Debug)]
pub struct LinearSelectOperation<F> {
    /// Captured Boolean condition that drives the selection.
    condition: F,

    /// [`PhantomData`] marker tying the captured condition to the [`DataType`] it is interpreted against, mirroring the
    /// marker carried by [`ScaleOperation`](crate::operations::arithmetic::ScaleOperation). The `fn() -> DataType` form
    /// indexes by [`DataType`] without owning one, so this operation's `Send` and `Sync` depend only on `F`.
    marker: PhantomData<fn() -> DataType>,
}

impl<F> LinearSelectOperation<F> {
    /// Creates a new [`LinearSelectOperation`] capturing the provided Boolean condition.
    #[inline]
    pub fn new(condition: F) -> Self {
        Self { condition, marker: PhantomData }
    }

    /// Returns the captured Boolean condition that drives the selection.
    #[inline]
    pub fn condition(&self) -> &F {
        &self.condition
    }
}

impl<F: Display> Display for LinearSelectOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Display> Operation<DataType> for LinearSelectOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        // The captured-condition select is linear in its two branch tangents, which share a type that is the output
        // type. The Boolean primal condition is captured as a factor (the primal trace already typed it), so it is not
        // revalidated here; only the branch tangents are checked.
        check_count!("input", input_types, 2, TypeError);
        if input_types[0] != input_types[1] {
            return Err(TypeError {
                message: format!(
                    "select on_true data type {} differs from on_false data type {}",
                    input_types[0], input_types[1],
                ),
            });
        }
        Ok(vec![input_types[0]])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("condition", &self.condition))
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for LinearSelectOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        // The two branch tangents are the operation's inputs; the captured Boolean condition is prepended as the
        // selection operand so that the underlying `select` shape inference (which expects the condition first) runs.
        check_count!("input", input_types, 2, TypeError);
        SelectOperation.infer_output_types(&[
            self.condition.r#type().into_owned(),
            input_types[0].clone(),
            input_types[1].clone(),
        ])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("condition", &self.condition))
    }
}

/// Transpose rule for the captured-condition select, shared by the scalar
/// [`LinearScalarOperation::Select`](crate::tracing_v2::LinearScalarOperation) and array
/// [`LinearArrayOperation::Select`](crate::tracing_v2::LinearArrayOperation) variants. The forward linear map
/// `(t, f) ↦ select(condition, t, f)` routes the output cotangent into the branch the captured condition selected:
/// the `on_true` cotangent is `select(condition, cotangent, 0)` and the `on_false` cotangent is
/// `select(condition, 0, cotangent)`. The transposed select reuses the same captured condition, reconstructed from
/// `self` and staged via [`transpose_captured_condition_select`]. The impl is generic over the primary type `T` and
/// applies wherever `LinearSelectOperation<F>` implements [`Operation`] for `T` (i.e., [`DataType`] and
/// [`ArrayType`]).
impl<T: Type, V: Value<T>, O: Operation<T>, F: Clone> TransposableOperation<T, V, O> for LinearSelectOperation<F>
where
    Self: Operation<T>,
    O: From<ZeroOperation<T>> + From<LinearSelectOperation<F>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, T, V, O>,
        input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        transpose_captured_condition_select(
            || O::from(LinearSelectOperation::new(self.condition().clone())),
            context,
            input_types,
            output_cotangents,
        )
    }
}

impl<V: Value<ArrayType> + crate::operations::manipulation::Broadcast + crate::operations::manipulation::Transpose>
    crate::tracing_v2::batching::BatchableOperation<V, V::InterpretationContext> for SelectOperation
where
    SelectOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        crate::tracing_v2::batching::apply_elementwise_batch(context, self, inputs)
    }
}

/// JVP rule for [`SelectOperation`], mirroring JAX's rule for
/// [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html): the primal output is
/// `select(condition, on_true, on_false)` over the input primals, and the tangent is
/// `select(condition, on_true_tangent, on_false_tangent)` with the same primal condition. The condition is Boolean,
/// so its own tangent is identically zero and is ignored. When both branch tangents are canonical staged zeros, the
/// output tangent is a canonical staged zero of the output type and no linear operation is staged;
/// otherwise the rule captures the condition as a residual factor and stages the captured-condition select provided
/// by [`LinearSelectOperation`].
impl<D> DifferentiableOperation<D> for SelectOperation
where
    D: DifferentiationContext,
    SelectOperation: Operation<D::Type>,
    D::Value: SelectCondition + Select<Condition = <D::Value as SelectCondition>::Condition>,
    LinearOperationOf<D>: From<LinearSelectOperation<CapturedFactor<D::Type, D::Value>>> + From<ZeroOperation<D::Type>>,
    LinearOperationOf<D>: MaybeZeroOperation<D::Type>,
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
        let primal = D::Value::select(&condition.primal().select_condition()?, on_true.primal(), on_false.primal())?;
        if context.is_zero(on_true.tangent())? && context.is_zero(on_false.tangent())? {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        let condition_factor = condition.factor(context);
        let mut outputs = context
            .stage_operation(LinearSelectOperation::new(condition_factor), &[on_true.tangent(), on_false.tangent()])?;
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

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over staged tracers of any context with [`TestArray`] semantics.
    fn piecewise_select<C>(x: crate::tracing::Tracer<C>) -> crate::tracing::Tracer<C>
    where
        C: crate::contexts::StagingContext<
                Type = crate::types::ArrayType,
                Constant = TestArray,
                Operation = crate::tracing_v2::ArrayOperation<crate::types::ArrayType, TestArray>,
            >,
    {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan);
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x)).unwrap()
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

    #[test]
    fn test_scalar_select_jvp_and_gradient_flow_to_selected_branch() {
        // Differentiating `f(x, y) = select(x > y, 2x, 3y)` over `ScalarDomain` exercises the scalar captured-condition
        // select staged through `LinearScalarOperation::Select`: forward mode routes each branch tangent through the
        // selected branch, and reverse mode routes the cotangent there, so the derivative reaches only the selected
        // branch's input.
        use crate::scalars::ScalarDomain;
        use crate::tracing_v2::{DifferentiationContext, value_and_grad};

        fn piecewise<C>(x: crate::tracing::Tracer<C>, y: crate::tracing::Tracer<C>) -> crate::tracing::Tracer<C>
        where
            C: crate::contexts::StagingContext<
                    Type = crate::types::DataType,
                    Constant = f64,
                    Operation = crate::operations::scalars::ScalarOperation<f64>,
                >,
        {
            let mask = x.compare(&y, ComparisonDirection::GreaterThan);
            Select::select(&mask, &(x.clone() + x.clone()), &(y.clone() + y.clone() + y.clone())).unwrap()
        }

        let domain = ScalarDomain::<f64>::new();

        // `x > y`: the output is `2x`, so forward mode passes the `x` tangent through (scaled by 2) and zeroes the `y`
        // tangent, while the gradient is `(2, 0)`.
        let (primal, tangent) = domain.jvp(|(x, y)| piecewise(x, y), (3.0, 2.0), (1.0, 0.0)).unwrap();
        assert_close(primal, 6.0);
        assert_close(tangent, 2.0);
        let (_, tangent) = domain.jvp(|(x, y)| piecewise(x, y), (3.0, 2.0), (0.0, 1.0)).unwrap();
        assert_close(tangent, 0.0);
        let (value, gradient) = value_and_grad(&domain, |(x, y)| piecewise(x, y), (3.0, 2.0)).unwrap();
        assert_close(value, 6.0);
        assert_close(gradient.0, 2.0);
        assert_close(gradient.1, 0.0);

        // `x <= y`: the output is `3y`, so the roles flip; the gradient is `(0, 3)`.
        let (primal, tangent) = domain.jvp(|(x, y)| piecewise(x, y), (1.0, 2.0), (1.0, 0.0)).unwrap();
        assert_close(primal, 6.0);
        assert_close(tangent, 0.0);
        let (_, tangent) = domain.jvp(|(x, y)| piecewise(x, y), (1.0, 2.0), (0.0, 1.0)).unwrap();
        assert_close(tangent, 3.0);
        let (value, gradient) = value_and_grad(&domain, |(x, y)| piecewise(x, y), (1.0, 2.0)).unwrap();
        assert_close(value, 6.0);
        assert_close(gradient.0, 0.0);
        assert_close(gradient.1, 3.0);
    }
}
