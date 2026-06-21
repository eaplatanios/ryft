use std::ops::Mul;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{AddOperation, MulOperation, Scale, ScaleOperation};
use crate::operations::constants::{HasZeroOperation, ZeroOperation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{CapturedFactor, DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Typed};

impl<D> DifferentiableOperation<D> for MulOperation
where
    D: DifferentiationContext,
    MulOperation: Operation<D::Type>,
    D::Value: Mul<Output = D::Value>,
    LinearOperationOf<D>: From<AddOperation>
        + From<ScaleOperation<D::Type, CapturedFactor<D::Type, D::Value>>>
        + From<ZeroOperation<D::Type>>,
    LinearOperationOf<D>: HasZeroOperation<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal().clone() * right.primal().clone();
        let left_term = if context.is_zero(left.tangent())? {
            None
        } else {
            Some(left.tangent().clone().scale(right.factor(context)))
        };
        let right_term = if context.is_zero(right.tangent())? {
            None
        } else {
            Some(right.tangent().clone().scale(left.factor(context)))
        };
        let tangent = match (left_term, right_term) {
            (Some(left_term), Some(right_term)) => left_term + right_term,
            (Some(term), None) | (None, Some(term)) => term,
            (None, None) => {
                let mut tangent_outputs =
                    context.stage_nullary_operation(ZeroOperation::new(primal.r#type().into_owned()))?;
                check_count!("output", tangent_outputs, 1, ProgramError);
                tangent_outputs.remove(0)
            }
        };
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Transpose rule for the linear `Mul` (the `Mul` variant of
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation)). A bilinear product is not a linear map in both
/// operands jointly, so a linear program never contains one: the JVP of [`MulOperation`] always lowers each tangent
/// term to a captured-factor [`ScaleOperation`]. This rule therefore rejects transposition with guidance to rewrite
/// to `Scale` first.
impl<V: Value<ArrayType>, O: Operation<ArrayType>> TransposableOperation<ArrayType, V, O> for MulOperation {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "linear `Mul` transpose is not supported (rewrite to `Scale` before transposition)".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::trigonometric::Sin;
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::DifferentiationContext;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) = domain.jvp(|(left, right)| left * right, (2.0f64, 5.0f64), (3.0f64, -1.0f64)).unwrap();

        approx_eq(primal, 10.0);
        approx_eq(tangent, 13.0);

        let (_, pushforward) = domain
            .linearize(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))
            .unwrap();
        let pushforward = pushforward.instantiate_program().unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=3] %0
                    %3:f64 = scale [factor=2] %1
                    %4:f64 = add %2 %3
                    %5:f64 = scale [factor=-0.4161468365471424] %0
                    %6:f64 = add %4 %5
                in (%6)
            "}
            .trim_end(),
        );
    }
}
