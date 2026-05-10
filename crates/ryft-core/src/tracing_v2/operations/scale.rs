use std::ops::Mul;

#[cfg(test)]
use indoc::indoc;

use crate::differentiation::LinearOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, ScaleOperation, SupportsAdd, SupportsMul, SupportsScale};
use crate::parameters::Parameter;
use crate::tracing::domains::{ProgramTracer, RuntimeDomain, Tracer, TracingContext};
use crate::tracing::{ProgramTracingContext, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{
    DifferentiableDomain, DifferentiableOperation, DifferentiableTracingDomain, DifferentiableTracingOperationCarrier,
};
use crate::types::{ArrayType, DataType, Type};

impl<T: Parameter + Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsScale<T, V>> LinearOperation<T, V, O>
    for ScaleOperation<T, V>
where
    ScaleOperation<T, V>: Operation<T>,
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Option<ProgramTracer<'transpose, T, V, O>>],
    ) -> Result<Vec<Option<ProgramTracer<'transpose, T, V, O>>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Some(cotangent) => Ok(vec![Some(cotangent.clone().scale(self.factor.clone()))]),
            None => Ok(vec![None]),
        }
    }
}

impl<T: Parameter + Type, V, D> DifferentiableOperation<D> for ScaleOperation<T, V>
where
    V: Differentiable<T> + Scale<Output = V>,
    D: DifferentiableDomain<Type = T, Value = V>,
    D::LinearOperationCarrier: SupportsScale<T, D::Tangent, V>,
    ScaleOperation<T, V>: Operation<T>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().scale(self.factor.clone()),
            tangent: input.tangent.clone().scale(self.factor.clone()),
        }])
    }
}

impl<'domain, D, V, O> DifferentiableOperation<TracingContext<'domain, D>> for ScaleOperation<DataType, V>
where
    D: DifferentiableTracingDomain<Type = DataType, Value = V, OperationCarrier = O> + RuntimeDomain + 'domain,
    V: Value<DataType> + Differentiable<DataType>,
    O: DifferentiableTracingOperationCarrier<D> + SupportsAdd<DataType, V> + SupportsMul<DataType, V> + 'domain,
    Tracer<'domain, D>: Mul<Output = Tracer<'domain, D>>,
    <TracingContext<'domain, D> as DifferentiableDomain>::LinearOperationCarrier:
        SupportsScale<DataType, Tracer<'domain, D>>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, D>>,
        inputs: &[JvpTracer<Tracer<'domain, D>, Tracer<'jvp, TracingContext<'domain, D>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'domain, D>, Tracer<'jvp, TracingContext<'domain, D>>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor_tracer = context.domain.constant(self.factor.clone());
        let primal = factor_tracer.clone() * input.primal.clone();
        let tangent = input.tangent.clone().scale(factor_tracer);
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

impl<'domain, D, V, O> DifferentiableOperation<TracingContext<'domain, D>> for ScaleOperation<ArrayType, V>
where
    D: DifferentiableTracingDomain<Type = ArrayType, Value = V, OperationCarrier = O> + RuntimeDomain + 'domain,
    V: Value<ArrayType> + Differentiable<ArrayType>,
    O: DifferentiableTracingOperationCarrier<D> + SupportsAdd<ArrayType, V> + SupportsMul<ArrayType, V> + 'domain,
    Tracer<'domain, D>: Mul<Output = Tracer<'domain, D>>,
    <TracingContext<'domain, D> as DifferentiableDomain>::LinearOperationCarrier:
        SupportsScale<ArrayType, Tracer<'domain, D>>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, D>>,
        inputs: &[JvpTracer<Tracer<'domain, D>, Tracer<'jvp, TracingContext<'domain, D>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'domain, D>, Tracer<'jvp, TracingContext<'domain, D>>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor_tracer = context.domain.constant(self.factor.clone());
        let primal = factor_tracer.clone() * input.primal.clone();
        let tangent = input.tangent.clone().scale(factor_tracer);
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::tracing::domains::ProgramTracingDomain;
    use crate::tracing::{ProgramBuilder, ProgramTracingContext};
    use crate::tracing_v2::LinearArrayOperation;
    use crate::tracing_v2::test_util::TestArray;

    use super::*;

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
        let contribution_atom = ScaleOperation::new(TestArray::scalar(3.0))
            .transpose(&mut context, &[Some(output_cotangent)])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution")
            .expect("transpose should produce one cotangent contribution")
            .atom_id()
            .unwrap();
        drop(context);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder
            .build::<TestArray, TestArray>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
        approx_eq(transpose_program.interpret(TestArray::scalar(2.0)).unwrap().values[0], 6.0);
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
