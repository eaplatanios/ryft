use std::fmt::Display;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{
    ConstantOperation, FillOperation, OneLike, OneLikeOperation, OneOperation, SupportsFill, SupportsOne, SupportsZero,
    SupportsZeroLike, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_elementwise_batch};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Type, Typed};

impl<
    V: Value<ArrayType>
        + crate::operations::manipulation::Broadcast
        + crate::operations::manipulation::Transpose,
    C,
> BatchableOperation<V, C> for ZeroLikeOperation
where
    ZeroLikeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        apply_elementwise_batch(self, inputs)
    }
}

impl<
    V: Value<ArrayType>
        + crate::operations::manipulation::Broadcast
        + crate::operations::manipulation::Transpose,
    C,
> BatchableOperation<V, C> for OneLikeOperation
where
    OneLikeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        apply_elementwise_batch(self, inputs)
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
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`OneOperation`] is lane-uniform by the
/// same argument.
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for OneOperation<ArrayType>
where
    OneOperation<ArrayType>: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`ConstantOperation`] is also lane-uniform because it has no
/// data inputs.
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for ConstantOperation<ArrayType, V>
where
    ConstantOperation<ArrayType, V>: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`FillOperation`] is also lane-uniform because it has no
/// data inputs.
impl<V: Value<ArrayType>, F: Clone + Display, C> BatchableOperation<V, C> for FillOperation<ArrayType, F>
where
    FillOperation<ArrayType, F>: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::unbatched).collect())
    }
}

impl<T: Parameter + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for ZeroOperation<T> {
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
    D::Operation: SupportsZero<D::Type>,
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
        let operation = <D::Operation as SupportsZero<D::Type>>::zero_operation(self.r#type().clone());
        let mut primals = context.bind_primal(operation, &[])?;
        check_count!("output", primals, 1, ProgramError);
        Ok(vec![JvpTracer::from_zero_tangent(primals.pop().unwrap(), self.r#type().clone())])
    }
}

impl<T: Parameter + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for OneOperation<T> {
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
    D::Operation: SupportsOne<D::Type>,
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
        let operation = <D::Operation as SupportsOne<D::Type>>::one_operation(self.r#type().clone());
        let mut primals = context.bind_primal(operation, &[])?;
        check_count!("output", primals, 1, ProgramError);
        Ok(vec![JvpTracer::from_zero_tangent(primals.pop().unwrap(), self.r#type().clone())])
    }
}

impl<T, V, O, F> TransposableOperation<T, V, O> for ConstantOperation<T, F>
where
    T: Parameter + Type,
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
        Ok(vec![JvpTracer::from_zero_tangent(primal, output_type)])
    }
}

impl<T: Parameter + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for ZeroLikeOperation {
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
    LinearOperationOf<D>: SupportsZeroLike<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(inputs[0].primal().zero_like(), inputs[0].tangent().zero_like())])
    }
}

impl<T: Parameter + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for OneLikeOperation {
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
    LinearOperationOf<D>: SupportsZeroLike<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(inputs[0].primal().one_like(), inputs[0].tangent().zero_like())])
    }
}

impl<T, V, O, F> TransposableOperation<T, V, O> for FillOperation<T, F>
where
    T: Parameter + Type,
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
    D::Operation: SupportsFill<D::Type, f64>,
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
        let operation =
            <D::Operation as SupportsFill<D::Type, f64>>::fill_operation(self.r#type().clone(), *self.value());
        let mut primals = context.bind_primal(operation, &[])?;
        check_count!("output", primals, 1, ProgramError);
        Ok(vec![JvpTracer::from_zero_tangent(primals.pop().unwrap(), self.r#type().clone())])
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
