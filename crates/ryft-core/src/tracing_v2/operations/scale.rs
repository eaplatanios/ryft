use std::{
    fmt::{Debug, Display},
    ops::Mul,
};

#[cfg(test)]
use indoc::indoc;

use crate::macros::check_input_count;
use crate::tracing::{OperationFormatter, Traceable, TracingError, Value};
use crate::tracing_v2::{
    LinearPrimitiveOperation,
    engines::{DifferentiableEngine, Engine},
    forward::{Differentiable, EngineTangent, JvpTracer, TangentSpace},
    jit::Tracer,
    operations::constants::ZeroLike,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, SupportsAdd, SupportsNeg,
    lift_jit_constant, mul::SupportsMul, unary_abstract,
};

/// Hidden carrier capability for staging the scaling primitive.
///
/// Ordinary tracing carriers and linear-program carriers can both implement this trait when they
/// support representing a captured-factor scale operation in their own operation universe.
#[doc(hidden)]
pub trait SupportsScale<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the scaling primitive with a captured factor.
    fn scale_operation(factor: V) -> Self;
}

/// Unary linear operation that multiplies its input by a captured factor.
///
/// In ordinary programs this represents "multiply by a closed-over constant." In linear programs
/// the same semantic idea is reused to scale tangent and cotangent terms.
#[derive(Clone)]
pub struct ScaleOperation<T: Type, V: Typed<T>> {
    /// Captured factor applied to every input of this unary linear op.
    factor: V,

    /// Phantom marker tying the captured factor to the abstract type it is interpreted against.
    _marker: std::marker::PhantomData<T>,
}

impl<T: Type, V: Typed<T>> ScaleOperation<T, V> {
    /// Creates a new scale operation capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor, _marker: std::marker::PhantomData }
    }

    /// Returns the captured scale factor.
    #[inline]
    pub fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V: Typed<ArrayType>> ScaleOperation<ArrayType, V> {
    /// Validates abstract inputs without needing a concrete instance.
    ///
    /// This is mainly used by carrier-level wrappers that want to construct or validate a scale op
    /// from type information before they have committed to a concrete `ScaleOperation` value.
    pub fn abstract_eval_static(inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<T: Type, V: Typed<T>> Debug for ScaleOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scale")
    }
}

impl<T: Type, V: Typed<T>> Display for ScaleOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scale")
    }
}

impl<V: Typed<ArrayType> + Display> Operation<ArrayType> for ScaleOperation<ArrayType, V> {
    fn name(&self) -> &'static str {
        "scale"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Self::abstract_eval_static(input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", self.factor()))
    }
}

impl<V: Typed<ArrayType> + Display + Clone + Mul<Output = V>> InterpretableOperation<ArrayType, V>
    for ScaleOperation<ArrayType, V>
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V> + ZeroLike> LinearOperation<ArrayType, V>
    for ScaleOperation<ArrayType, V>
{
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<
            '_,
            ArrayType,
            V,
            LinearPrimitiveOperation<V>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .apply_operation(&[atom], LinearPrimitiveOperation::Scale { factor: self.factor().clone() }, 1)?
                    .into_iter()
                    .next()
                    .expect("scale transpose should produce one cotangent contribution"),
            )]),
            None => Ok(vec![None]),
        }
    }
}

impl<
    'engine,
    V: Value<ArrayType> + ZeroLike + Mul<Output = V>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    InnerLinearOperation: Clone
        + Operation<ArrayType>
        + SupportsAdd<ArrayType, Tracer<'engine, E>>
        + SupportsNeg<ArrayType, Tracer<'engine, E>>
        + SupportsScale<ArrayType, Tracer<'engine, E>>,
> InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>
    for ScaleOperation<ArrayType, V>
where
    E::TracingOperation: SupportsMul<ArrayType, V> + SupportsScale<ArrayType, V> + 'static,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>, TracingError>
    {
        check_input_count!(inputs, 1);
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        Ok(vec![JvpTracer {
            primal: factor.clone() * inputs[0].primal.clone(),
            tangent: inputs[0].tangent.clone().scale(factor),
        }])
    }
}

impl<V, E> DifferentiableOperation<E> for ScaleOperation<ArrayType, V>
where
    V: Differentiable<ArrayType> + Mul<Output = V>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + ?Sized,
    E::LinearOperation: SupportsAdd<ArrayType, V> + SupportsNeg<ArrayType, V> + SupportsScale<ArrayType, V>,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<V, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<V, EngineTangent<E>>>, TracingError> {
        check_input_count!(inputs, 1);
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: self.factor().clone() * input.primal.clone(),
            tangent: EngineTangent::<E>::scale(self.factor().clone(), input.tangent.clone()),
        }])
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use pretty_assertions::assert_eq;

    use crate::{
        parameters::Placeholder,
        tracing::ProgramBuilder,
        tracing_v2::{LinearPrimitiveOperation, operations::TranspositionContext},
    };

    use super::*;

    fn test_transposition_context(
        builder: Rc<RefCell<ProgramBuilder<ArrayType, f64, LinearPrimitiveOperation<f64>>>>,
    ) -> TranspositionContext<'static, ArrayType, f64, LinearPrimitiveOperation<f64>> {
        TranspositionContext::new(builder)
    }

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_scale_transpose_scales_output_cotangents() {
        let transpose_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
        let mut context = test_transposition_context(transpose_builder.clone());
        let contribution_atom = ScaleOperation::new(3.0f64)
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
        let transpose_program =
            transpose_builder.build::<f64, f64>(vec![contribution_atom], Placeholder, Placeholder).unwrap();
        approx_eq(transpose_program.interpret(2.0f64).unwrap(), 6.0);
        assert_eq!(
            transpose_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale [factor=3] %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
