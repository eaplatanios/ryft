use std::ops::Mul;

#[cfg(test)]
use indoc::indoc;

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, ScaleOperation, SupportsScale};
use crate::parameters::Parameter;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::DifferentiableOperation;
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer, LinearOperationOf};
use crate::types::Type;

impl<T: Parameter + Type, V: Traceable<T>, O: Operation<T> + SupportsScale<T, V>> LinearOperation<T, V, O>
    for ScaleOperation<T, V>
where
    ScaleOperation<T, V>: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().scale(self.factor().clone()))]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<T: Parameter + Type, D> DifferentiableOperation<D> for ScaleOperation<T, D::CapturedValue>
where
    D: Differentiable<Type = T>,
    D::Value: Mul<Output = D::Value>,
    LinearOperationOf<D>: SupportsScale<T, D::Tangent, D::Value>,
    ScaleOperation<T, D::CapturedValue>: Operation<T>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor = context.differentiable().lift_captured_primal(self.factor().clone())?;
        Ok(vec![JvpTracer::new(factor.clone() * input.primal().clone(), input.tangent().clone().scale(factor))])
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::Context;
    use crate::parameters::Placeholder;
    use crate::tracing::domains::ProgramTracingDomain;
    use crate::tracing::{ProgramBuilder, ProgramTracingContext};
    use crate::tracing_v2::LinearArrayOperation;
    use crate::tracing_v2::test_util::TestArray;
    use crate::types::{ArrayType, DataType};
    use pretty_assertions::assert_eq;

    fn test_transposition_context<'transpose>(
        domain: &'transpose ProgramTracingDomain<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>>>,
    ) -> ProgramTracingContext<'transpose, ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>> {
        ProgramTracingContext::new(domain, builder)
    }

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_scale_transpose_scales_output_cotangents() {
        let transpose_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>::new(),
            ));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let domain = ProgramTracingDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = ScaleOperation::new(TestArray::scalar(3.0))
            .transpose(&mut context, &[Cotangent::Staged(output_cotangent)])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
        drop(context);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder
            .build::<TestArray, TestArray>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
        approx_eq(transpose_program.interpret(TestArray::scalar(2.0)).unwrap().values()[0], 6.0);
        assert_eq!(
            transpose_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale [factor=[3.0]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
