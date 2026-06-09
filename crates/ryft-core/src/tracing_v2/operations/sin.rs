use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, SupportsScale};
use crate::operations::trigonometric::{Cos, Sin, SinOperation};
use crate::programs::ProgramError;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, ResidualFactor, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};

impl<D> DifferentiableOperation<D> for SinOperation
where
    D: DifferentiationContext,
    SinOperation: Operation<D::Type>,
    D::Value: Sin + Cos,
    LinearOperationOf<D>: SupportsScale<D::Type, ResidualFactor<D::Type, D::Value>>,
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
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let tangent = match input.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(tangent.scale(context.factor(input.primal().clone().cos()))),
        };
        Ok(vec![JvpTracer::new(input.primal().clone().sin(), tangent)])
    }
}
