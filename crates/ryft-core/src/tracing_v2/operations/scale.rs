use std::fmt::{Debug, Display};
use std::ops::Mul;

#[cfg(test)]
use indoc::indoc;

use crate::macros::check_count;
use crate::operations::constants::ZeroLike;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::engines::Tracer;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{
    DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine, LinearArrayOperation,
};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use crate::operations::arithmetic::SupportsAdd;

/// Trait that represents [`Operation`] carrier types that support/include [`ScaleOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`ScaleOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsScale<T: Type, V: Traceable<T>, F: Traceable<T> = V> {
    /// Constructs the carrier-specific representation of the scaling [`Operation`].
    fn scale_operation(factor: F) -> Self;
}

/// Unary linear operation that multiplies its input by a captured factor.
///
/// In ordinary programs this represents "multiply by a closed-over constant." In linear programs
/// the same semantic idea is reused to scale tangent and cotangent terms.
#[derive(Clone, Debug)]
pub struct ScaleOperation<T: Type, V: Typed<T>> {
    /// Captured factor applied to every input of this unary linear op.
    pub factor: V,

    /// Phantom marker tying the captured factor to the abstract type it is interpreted against.
    pub marker: std::marker::PhantomData<T>,
}

impl<T: Type, V: Typed<T>> ScaleOperation<T, V> {
    /// Creates a new scale operation capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor, marker: std::marker::PhantomData }
    }
}

impl<T: Type, V: Typed<T>> ScaleOperation<T, V> {
    /// Validates abstract inputs without needing a concrete instance.
    ///
    /// This is mainly used by carrier-level wrappers that want to construct or validate a scale op
    /// from type information before they have committed to a concrete `ScaleOperation` value.
    pub fn abstract_eval_static(inputs: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", inputs, 1, TypeError);
        Ok(vec![inputs[0].clone()])
    }
}

impl<T: Type, V: Debug + Display + Typed<T>> Display for ScaleOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: Type, V: Debug + Display + Typed<T>> Operation<T> for ScaleOperation<T, V> {
    #[inline]
    fn name(&self) -> &'static str {
        "scale"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Self::abstract_eval_static(input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", &self.factor))
    }
}

impl<T: Type, V: Clone + Debug + Display + Typed<T> + Mul<Output = V>> InterpretableOperation<T, V>
    for ScaleOperation<T, V>
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![self.factor.clone() * inputs[0].clone()])
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + crate::parameters::Parameter + Mul<Output = V> + ZeroLike>
    LinearOperation<T, V, LinearArrayOperation<V, T>> for ScaleOperation<T, V>
where
    LinearArrayOperation<V, T>: Operation<T>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<T, V, LinearArrayOperation<V, T>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let cotangent_outputs =
                    context.stage(LinearArrayOperation::<V, T>::Scale { factor: self.factor.clone() }, &[atom])?;
                check_count!("output", cotangent_outputs, 1, TracingError);
                Ok(vec![Some(cotangent_outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

impl<V, E> DifferentiableOperation<E> for ScaleOperation<ArrayType, V>
where
    V: Differentiable<ArrayType> + Mul<Output = V>,
    E: DifferentiableEngine<Type = ArrayType, Value = V>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier:
        SupportsScale<ArrayType, E::Tangent, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsScale<
                ArrayType,
                E::Tangent,
                V,
            >>::scale_operation(self.factor.clone()),
            &[input.tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: self.factor.clone() * input.primal.clone(), tangent: tangent_outputs[0] }])
    }
}

impl<V, E> DifferentiableOperation<E> for ScaleOperation<DataType, V>
where
    V: Differentiable<DataType> + Mul<Output = V>,
    E: DifferentiableEngine<Type = DataType, Value = V>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier:
        SupportsScale<DataType, E::Tangent, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsScale<
                DataType,
                E::Tangent,
                V,
            >>::scale_operation(self.factor.clone()),
            &[input.tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: self.factor.clone() * input.primal.clone(), tangent: tangent_outputs[0] }])
    }
}

/// JVP rule for `ScaleOperation` under the
/// [`TracingContext`](crate::tracing::engines::TracingContext) wrapper.
///
/// The operation's captured factor is `V_inner` (the underlying engine's value type), but the
/// wrapper engine's [`Value`](crate::tracing::engines::Engine::Value) is
/// [`Tracer`](crate::tracing::engines::Tracer). The rule lifts the captured
/// factor into a `Tracer` constant in the outer trace and then stages both the primal product
/// and the tangent scale on traced primals.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for ScaleOperation<ArrayType, V>
where
    V: Value<ArrayType> + Differentiable<ArrayType>,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V>,
    EInner::OperationCarrier: SupportsAdd<ArrayType, V>,
    Tracer<'engine, EInner>: Mul<Output = Tracer<'engine, EInner>>,
    EInner::LinearOperationCarrier<'engine>: SupportsScale<ArrayType, Tracer<'engine, EInner>>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor_tracer = context.engine.constant(self.factor.clone());
        let tangent_outputs = context.stage(
            <EInner::LinearOperationCarrier<'engine> as SupportsScale<
                ArrayType,
                Tracer<'engine, EInner>,
            >>::scale_operation(factor_tracer.clone()),
            &[input.tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: factor_tracer * input.primal.clone(), tangent: tangent_outputs[0] }])
    }
}

impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for ScaleOperation<DataType, V>
where
    V: Value<DataType> + Differentiable<DataType>,
    EInner: DifferentiableTracingEngine<Type = DataType, Value = V>,
    EInner::OperationCarrier: SupportsAdd<DataType, V>,
    Tracer<'engine, EInner>: Mul<Output = Tracer<'engine, EInner>>,
    EInner::LinearOperationCarrier<'engine>: SupportsScale<DataType, Tracer<'engine, EInner>>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let factor_tracer = context.engine.constant(self.factor.clone());
        let tangent_outputs = context.stage(
            <EInner::LinearOperationCarrier<'engine> as SupportsScale<
                DataType,
                Tracer<'engine, EInner>,
            >>::scale_operation(factor_tracer.clone()),
            &[input.tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: factor_tracer * input.primal.clone(), tangent: tangent_outputs[0] }])
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::tracing::ProgramBuilder;
    use crate::tracing::transposition::TranspositionContext;
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
