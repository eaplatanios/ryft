use std::ops::Mul;

#[cfg(test)]
use indoc::indoc;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, ScaleOperation, SupportsScale};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::DifferentiableOperation;
use crate::tracing_v2::differentiation::{
    DifferentiationContext, JvpTracer, LinearOperationOf, ResidualFactor, TangentContext,
};
use crate::types::Type;

impl<T: Parameter + Type, V: Value<T>, O: Operation<T> + SupportsScale<T, V>> TransposableOperation<T, V, O>
    for ScaleOperation<T, V>
where
    ScaleOperation<T, V>: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(cotangent.clone().scale(self.factor().clone()))]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<T: Parameter + Type, D> DifferentiableOperation<D> for ScaleOperation<T, D::Constant>
where
    D: DifferentiationContext<Type = T>,
    D::Value: Mul<Output = D::Value>,
    LinearOperationOf<D>: SupportsScale<T, ResidualFactor<T, D::Value>>,
    ScaleOperation<T, D::Constant>: Operation<T>,
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
        let factor = context.differentiable().lift(self.factor().clone())?;
        Ok(vec![JvpTracer::new(
            factor.clone() * input.primal().clone(),
            input.tangent().clone().scale(ResidualFactor::Constant(factor)),
        )])
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::contexts::StagingContext;
    use crate::domains::AbstractDomain;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tracing::AbstractTracingContext;
    use crate::tracing_v2::LinearArrayOperation;
    use crate::tracing_v2::test_util::TestArray;
    use crate::types::{ArrayType, DataType};
    use pretty_assertions::assert_eq;

    fn test_transposition_context<'transpose>(
        domain: &'transpose AbstractDomain<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>>>,
    ) -> AbstractTracingContext<'transpose, ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>> {
        AbstractTracingContext::new(domain, builder)
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
        let domain = AbstractDomain::new();
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
