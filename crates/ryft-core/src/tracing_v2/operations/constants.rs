use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{
    ConstantOperation, FillOperation, OneLike, OneLikeOperation, OneOperation, ZeroLike, ZeroLikeOperation,
    ZeroOperation,
};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_elementwise_batch};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Type, Typed};

impl<V: Value<ArrayType> + crate::operations::manipulation::Broadcast + crate::operations::manipulation::Transpose>
    BatchableOperation<V, V::InterpretationContext> for ZeroLikeOperation
where
    ZeroLikeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        apply_elementwise_batch(context, self, inputs)
    }
}

impl<V: Value<ArrayType> + crate::operations::manipulation::Broadcast + crate::operations::manipulation::Transpose>
    BatchableOperation<V, V::InterpretationContext> for OneLikeOperation
where
    OneLikeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        apply_elementwise_batch(context, self, inputs)
    }
}

/// [`ZeroOperation`] takes no inputs and produces a constant of its captured type. The same
/// constant is the right value for every batch lane, so the rule interprets the operation once
/// and wraps each output as a lane-uniform [`ArrayBatch`] (`batch_axis = None`). Downstream
/// elementwise consumers that need the constant materialized at the batched physical shape will
/// broadcast it through the internal elementwise batching rule.
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for ZeroOperation<ArrayType>
where
    ZeroOperation<ArrayType>: InterpretableOperation<ArrayType, V>,
    <V as Value<ArrayType>>::InterpretationContext: Default,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(
            self,
            &<V as Value<ArrayType>>::InterpretationContext::default(),
            &[],
        )?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`OneOperation`] is lane-uniform by the
/// same argument.
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for OneOperation<ArrayType>
where
    OneOperation<ArrayType>: InterpretableOperation<ArrayType, V>,
    <V as Value<ArrayType>>::InterpretationContext: Default,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(
            self,
            &<V as Value<ArrayType>>::InterpretationContext::default(),
            &[],
        )?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`ConstantOperation`] is also lane-uniform because it has no
/// data inputs.
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for ConstantOperation<ArrayType, V>
where
    ConstantOperation<ArrayType, V>: InterpretableOperation<ArrayType, V>,
    <V as Value<ArrayType>>::InterpretationContext: Default,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(
            self,
            &<V as Value<ArrayType>>::InterpretationContext::default(),
            &[],
        )?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`FillOperation`] is also lane-uniform because it has no
/// data inputs.
impl<V: Value<ArrayType>, F: Clone + Display, C> BatchableOperation<V, C> for FillOperation<ArrayType, F>
where
    FillOperation<ArrayType, F>: InterpretableOperation<ArrayType, V>,
    <V as Value<ArrayType>>::InterpretationContext: Default,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(
            self,
            &<V as Value<ArrayType>>::InterpretationContext::default(),
            &[],
        )?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for ZeroOperation<T> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for ZeroOperation<D::Type>
where
    D: DifferentiationContext,
    D::Operation: From<ZeroOperation<D::Type>>,
    LinearOperationOf<D>: From<ZeroOperation<D::Type>>,
    ZeroOperation<D::Type>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, ProgramError);
        let operation = D::Operation::from(ZeroOperation::new(self.r#type().clone()));
        let mut primals = context.bind_primal(operation, &[])?;
        check_count!("output", primals, 1, ProgramError);
        let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(self.r#type().clone()))?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(primals.pop().unwrap(), tangent_outputs.remove(0))])
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for OneOperation<T> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for OneOperation<D::Type>
where
    D: DifferentiationContext,
    D::Operation: From<OneOperation<D::Type>>,
    LinearOperationOf<D>: From<ZeroOperation<D::Type>>,
    OneOperation<D::Type>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, ProgramError);
        let operation = D::Operation::from(OneOperation::new(self.r#type().clone()));
        let mut primals = context.bind_primal(operation, &[])?;
        check_count!("output", primals, 1, ProgramError);
        let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(self.r#type().clone()))?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(primals.pop().unwrap(), tangent_outputs.remove(0))])
    }
}

impl<T, V, O, F, Mode> TransposableOperation<T, V, O> for ConstantOperation<T, F, Mode>
where
    T: Type,
    V: Value<T>,
    O: Operation<T>,
    F: Clone + Display + Typed<T>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for ConstantOperation<D::Type, D::Constant>
where
    D: DifferentiationContext,
    D::Constant: Clone + Typed<D::Type>,
    LinearOperationOf<D>: From<ZeroOperation<D::Type>>,
    ConstantOperation<D::Type, D::Constant>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, ProgramError);
        let output_type = self.value().r#type().into_owned();
        let primal = context.differentiable().lift(self.value().clone())?;
        let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(output_type))?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))])
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for ZeroLikeOperation {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D> DifferentiableOperation<D> for ZeroLikeOperation
where
    D: DifferentiationContext,
    ZeroLikeOperation: Operation<D::Type>,
    D::Value: ZeroLike,
    LinearOperationOf<D>: From<ZeroOperation<D::Type>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().zero_like();
        let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(primal.r#type().into_owned()))?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        let tangent = tangent_outputs.remove(0);
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for OneLikeOperation {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D> DifferentiableOperation<D> for OneLikeOperation
where
    D: DifferentiationContext,
    OneLikeOperation: Operation<D::Type>,
    D::Value: OneLike,
    LinearOperationOf<D>: From<ZeroOperation<D::Type>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().one_like();
        let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(primal.r#type().into_owned()))?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        let tangent = tangent_outputs.remove(0);
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

impl<T, V, O, F> TransposableOperation<T, V, O> for FillOperation<T, F>
where
    T: Type,
    V: Value<T>,
    O: Operation<T>,
    F: Display,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for FillOperation<D::Type, f64>
where
    D: DifferentiationContext,
    D::Operation: From<FillOperation<D::Type, f64>>,
    LinearOperationOf<D>: From<ZeroOperation<D::Type>>,
    FillOperation<D::Type, f64>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, ProgramError);
        let operation = D::Operation::from(FillOperation::new(self.r#type().clone(), *self.value()));
        let mut primals = context.bind_primal(operation, &[])?;
        check_count!("output", primals, 1, ProgramError);
        let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(self.r#type().clone()))?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(primals.pop().unwrap(), tangent_outputs.remove(0))])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::programs::Program;
    use crate::scalars::ScalarDomain;
    use crate::tracing::TracingContext;
    use crate::types::DataType;

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(&angle), angle.sin());
        assert_eq!(Cos::cos(&angle), angle.cos());

        let domain = ScalarDomain::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            TracingContext::interpret_and_trace(&domain, |x| Ok(x.sin()), 2.0f64).unwrap();

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
