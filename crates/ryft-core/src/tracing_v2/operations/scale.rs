use std::ops::Mul;

#[cfg(test)]
use indoc::indoc;

use crate::TracingContext;
use crate::differentiation::{LinearOperation, TranspositionContext};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scale, ScaleOperation, SupportsAdd, SupportsScale};
use crate::tracing::engines::Tracer;
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine};
use crate::types::{ArrayType, DataType, Type};

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T> + SupportsScale<T, V>> LinearOperation<T, V, O>
    for ScaleOperation<T, V>
where
    ScaleOperation<T, V>: Operation<T>,
{
    #[inline]
    fn transpose(
        &self,
        context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let cotangent_outputs = context.stage(O::scale_operation(self.factor.clone()), &[atom])?;
                check_count!("output", cotangent_outputs, 1, TracingError);
                Ok(vec![Some(cotangent_outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

impl<T: Type, V, E> DifferentiableOperation<E> for ScaleOperation<T, V>
where
    V: Differentiable<T> + Scale<Output = V>,
    E: DifferentiableEngine<Type = T, Value = V>,
    E::LinearOperationCarrier: SupportsScale<T, E::Tangent, V>,
    ScaleOperation<T, V>: Operation<T>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>],
    ) -> Result<Vec<JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().scale(self.factor.clone()),
            tangent: input.tangent.clone().scale(self.factor.clone()),
        }])
    }
}

impl<'engine, V, EInner> DifferentiableOperation<TracingContext<'engine, EInner>> for ScaleOperation<DataType, V>
where
    V: Value<DataType> + Differentiable<DataType>,
    EInner: DifferentiableTracingEngine<Type = DataType, Value = V>,
    EInner::OperationCarrier: SupportsAdd<DataType, V>,
    Tracer<'engine, EInner>: Mul<Output = Tracer<'engine, EInner>>,
    EInner::LinearOperationCarrier<'engine>: SupportsScale<DataType, Tracer<'engine, EInner>>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, Tracer<'jvp, TracingContext<'engine, EInner>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, Tracer<'jvp, TracingContext<'engine, EInner>>>>, TracingError>
    {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor_tracer = context.engine.constant(self.factor.clone());
        let primal = factor_tracer.clone() * input.primal.clone();
        let tangent = input.tangent.clone().scale(factor_tracer);
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

impl<'engine, V, EInner> DifferentiableOperation<TracingContext<'engine, EInner>> for ScaleOperation<ArrayType, V>
where
    V: Value<ArrayType> + Differentiable<ArrayType>,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V>,
    EInner::OperationCarrier: SupportsAdd<ArrayType, V>,
    Tracer<'engine, EInner>: Mul<Output = Tracer<'engine, EInner>>,
    EInner::LinearOperationCarrier<'engine>: SupportsScale<ArrayType, Tracer<'engine, EInner>>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, Tracer<'jvp, TracingContext<'engine, EInner>>>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, Tracer<'jvp, TracingContext<'engine, EInner>>>>, TracingError>
    {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor_tracer = context.engine.constant(self.factor.clone());
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

    use crate::differentiation::TranspositionContext;
    use crate::parameters::Placeholder;
    use crate::tracing::ProgramBuilder;
    use crate::tracing_v2::LinearArrayOperation;
    use crate::tracing_v2::test_util::TestArray;

    use super::*;

    fn test_transposition_context(
        builder: Rc<RefCell<ProgramBuilder<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>>>>,
    ) -> TranspositionContext<ArrayType, TestArray, LinearArrayOperation<TestArray, ArrayType>> {
        TranspositionContext::new(builder)
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
        let mut context = test_transposition_context(transpose_builder.clone());
        let contribution_atom = ScaleOperation::new(TestArray::scalar(3.0))
            .transpose(&mut context, &[Some(output_cotangent_atom)])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution")
            .expect("transpose should produce one cotangent contribution");
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
