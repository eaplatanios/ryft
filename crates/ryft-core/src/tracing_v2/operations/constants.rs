use std::fmt::{Debug, Display};

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::constants::{
    ConstantLike, ConstantLikeOperation, OneLike, OneLikeOperation, OneOperation, SupportsZeroLike, ZeroLike,
    ZeroLikeOperation, ZeroOperation,
};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::Parameter;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_elementwise_batch};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{Differentiable, DifferentiableOperation};
use crate::types::{ArrayType, Type};

impl<V, RuleContext> BatchableOperation<V, RuleContext> for ZeroLikeOperation
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    ZeroLikeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &RuleContext, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        apply_elementwise_batch(self, inputs)
    }
}

impl<V, RuleContext> BatchableOperation<V, RuleContext> for OneLikeOperation
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    OneLikeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &RuleContext, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        apply_elementwise_batch(self, inputs)
    }
}

impl<V, F: Clone + Debug + Display, RuleContext> BatchableOperation<V, RuleContext>
    for ConstantLikeOperation<ArrayType, F>
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    ConstantLikeOperation<ArrayType, F>: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &RuleContext, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        apply_elementwise_batch(self, inputs)
    }
}

/// [`ZeroOperation`] takes no inputs and produces a constant of its captured type. The same
/// constant is the right value for every batch lane, so the rule interprets the operation once
/// and wraps each output as a lane-uniform [`ArrayBatch`] (`batch_axis = None`). Downstream
/// elementwise consumers that need the constant materialized at the batched physical shape will
/// broadcast it through the internal elementwise batching rule.
impl<V, RuleContext> BatchableOperation<V, RuleContext> for ZeroOperation<ArrayType>
where
    V: Traceable<ArrayType>,
    ZeroOperation<ArrayType>: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &RuleContext, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`OneOperation`] is lane-uniform by the
/// same argument.
impl<V, RuleContext> BatchableOperation<V, RuleContext> for OneOperation<ArrayType>
where
    V: Traceable<ArrayType>,
    OneOperation<ArrayType>: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &RuleContext, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Operation<T>>
    LinearOperation<T, V, LinearOperationCarrier> for ZeroOperation<T>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearOperationCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearOperationCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearOperationCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for ZeroOperation<D::Type>
where
    D: Differentiable,
    ZeroOperation<D::Type>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, TracingError);
        Ok(vec![JvpTracer::from_zero_tangent(
            context.differentiable().zero_primal(self.r#type())?,
            self.r#type().clone(),
        )])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Operation<T>>
    LinearOperation<T, V, LinearOperationCarrier> for OneOperation<T>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearOperationCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearOperationCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearOperationCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for OneOperation<D::Type>
where
    D: Differentiable,
    OneOperation<D::Type>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, TracingError);
        Ok(vec![JvpTracer::from_zero_tangent(
            context.differentiable().one_primal(self.r#type())?,
            self.r#type().clone(),
        )])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Operation<T>>
    LinearOperation<T, V, LinearOperationCarrier> for ZeroLikeOperation
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearOperationCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearOperationCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearOperationCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D> DifferentiableOperation<D> for ZeroLikeOperation
where
    D: Differentiable,
    ZeroLikeOperation: Operation<D::Type>,
    D::Value: ZeroLike,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer::new(inputs[0].primal().zero_like(), inputs[0].tangent().zero_like())])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Operation<T>>
    LinearOperation<T, V, LinearOperationCarrier> for OneLikeOperation
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearOperationCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearOperationCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearOperationCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D> DifferentiableOperation<D> for OneLikeOperation
where
    D: Differentiable,
    OneLikeOperation: Operation<D::Type>,
    D::Value: OneLike,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer::new(inputs[0].primal().one_like(), inputs[0].tangent().zero_like())])
    }
}

impl<T, V, LinearOperationCarrier, F> LinearOperation<T, V, LinearOperationCarrier> for ConstantLikeOperation<T, F>
where
    T: Parameter + Type,
    V: Traceable<T>,
    LinearOperationCarrier: Operation<T>,
    F: Debug + Display,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearOperationCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearOperationCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearOperationCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D, F> DifferentiableOperation<D> for ConstantLikeOperation<D::Type, F>
where
    D: Differentiable,
    ConstantLikeOperation<D::Type, F>: Operation<D::Type>,
    D::Value: ConstantLike<F>,
    F: Clone,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer::new(
            inputs[0].primal().constant_like(self.value().clone()),
            inputs[0].tangent().zero_like(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::tracing::Program;
    use crate::tracing::domains::{ScalarDomain, TracingDomain};
    use crate::types::DataType;

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(angle), angle.sin());
        assert_eq!(Cos::cos(angle), angle.cos());

        let domain = ScalarDomain::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            domain.interpret_and_trace(|x| Ok(x.sin()), 2.0f64).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
