use std::ops::Mul;

#[cfg(test)]
use indoc::indoc;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::{Scalable, Scale, ScaleOperation};
use crate::payloads::Input;
use crate::programs::{ProgramError, Value};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::differentiation::{DifferentiationContext, JvpTracer, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, ValueOrCapture};
use crate::types::Type;

impl<C, V> Scale<C::Type, Tracer<C>, ValueOrCapture<C::Type, V>, Input> for C
where
    C: StagingContext,
    V: Value<C::Type>,
    C::Operation: From<ScaleOperation<C::Type, ValueOrCapture<C::Type, V>, Input>>,
{
    #[inline]
    fn scale(&self, input: &Tracer<C>, factor: ValueOrCapture<C::Type, V>) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_operation(ScaleOperation::<C::Type, _, Input>::new(factor), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C, V> Scalable<ValueOrCapture<C::Type, V>, Input> for Tracer<C>
where
    C: StagingContext,
    V: Value<C::Type>,
    C::Operation: From<ScaleOperation<C::Type, ValueOrCapture<C::Type, V>, Input>>,
{
    #[inline]
    fn scale(&self, factor: ValueOrCapture<C::Type, V>) -> Result<Self, ProgramError> {
        let mut outputs = self.context().stage_operation(ScaleOperation::<C::Type, _, Input>::new(factor), &[self])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Transpose rule for [`ScaleOperation`]: scaling by a captured factor is self-adjoint, so the input cotangent is the
/// output cotangent scaled by the same factor. The captured factor type `F` is independent of the cotangent value type
/// `V` (they coincide only at the top level; inside a linear scan body the factor lives in the scan-local
/// `ValueOrCapture` namespace while the cotangent does not), so this impl is generic over both and stages the adjoint
/// scale into `O` using the same scale interpretation mode as `self`.
impl<T: Type, V: Value<T>, F: Value<T>, O: Operation<T> + From<ScaleOperation<T, F, Payload>>, Payload>
    TransposableOperation<T, V, O> for ScaleOperation<T, F, Payload>
{
    #[inline]
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(
                cotangent.unary(ScaleOperation::<T, F, Payload>::new(self.factor().clone())),
            )]),
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<T: Type, D> DifferentiableOperation<D> for ScaleOperation<T, D::Constant>
where
    D: DifferentiationContext<Type = T>,
    D::Value: Mul<Output = D::Value>,
    D::LinearOperation<D::Tangent, ValueOrCapture<D::Type, D::Value>>:
        From<ScaleOperation<T, ValueOrCapture<T, D::Value>, Input>>,
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
        let mut tangent_outputs = context.stage_operation(
            ScaleOperation::<T, ValueOrCapture<T, D::Value>, Input>::new(ValueOrCapture::Value(factor.clone())),
            &[input.tangent().clone()],
        )?;
        check_count!("output", tangent_outputs, 1, ProgramError);
        Ok(vec![JvpTracer::new(factor.clone() * input.primal().clone(), tangent_outputs.remove(0))])
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
    use crate::tests::TestArray;
    use crate::tracing::AbstractTracingContext;
    use crate::tracing_v2::{ArrayOperation, LinearArrayOperation};
    use crate::types::{ArrayType, DataType};
    use pretty_assertions::assert_eq;

    fn test_transposition_context<'transpose>(
        domain: &'transpose AbstractDomain<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>,
        >,
        builder: Rc<
            RefCell<
                ProgramBuilder<
                    ArrayType,
                    TestArray,
                    LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>,
                >,
            >,
        >,
    ) -> AbstractTracingContext<
        'transpose,
        ArrayType,
        TestArray,
        LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>,
    > {
        AbstractTracingContext::new(domain, builder)
    }

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_scale_transpose_scales_output_cotangents() {
        let transpose_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>,
        >::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = ScaleOperation::<ArrayType, TestArray, Input>::new(TestArray::scalar(3.0))
            .transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Staged(output_cotangent)])
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
