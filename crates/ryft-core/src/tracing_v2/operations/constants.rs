use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{
    OneLike, OneLikeOperation, OneOperation, SupportsZeroLike, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::parameters::Parameter;
use crate::tracing::domains::Tracer;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
use crate::types::Type;

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Clone + Operation<T>>
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
    D: DifferentiableDomain,
    ZeroOperation<D::Type>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, TracingError);
        Ok(vec![JvpTracer::from_zero_tangent(context.domain.zero(&self.r#type)?, self.r#type.clone())])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Clone + Operation<T>>
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
    D: DifferentiableDomain,
    OneOperation<D::Type>: Operation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, TracingError);
        Ok(vec![JvpTracer::from_zero_tangent(context.domain.one(&self.r#type)?, self.r#type.clone())])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Clone + Operation<T>>
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
    D: DifferentiableDomain,
    ZeroLikeOperation: Operation<D::Type>,
    D::Value: ZeroLike,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.zero_like(), tangent: inputs[0].tangent.zero_like() }])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearOperationCarrier: Clone + Operation<T>>
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
    D: DifferentiableDomain,
    OneLikeOperation: Operation<D::Type>,
    D::Value: OneLike,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, D::Type, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.one_like(), tangent: inputs[0].tangent.zero_like() }])
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
