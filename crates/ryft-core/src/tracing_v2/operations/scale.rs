//! Scaling primitive for [`crate::tracing_v2`].
//!
//! `ScaleOp` is the main example of a primitive with captured constant state. Unlike bare
//! multiplication, the scale factor is part of the op object itself, which makes this module a good
//! reference for how traced constants move through replay, linearization, and higher-order traced
//! execution.

use std::{
    fmt::{Debug, Display},
    ops::Mul,
};

#[cfg(test)]
use indoc::indoc;

use crate::tracing_v2::{
    AtomId, Traceable, TracingError, Value, ZeroLike,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::Tracer,
    linear::LinearTerm,
};
use crate::types::{ArrayType, Type, Typed};

use super::{
    DifferentiableOp, InterpretableOp, LinearAddOperation, LinearNegOperation, LinearOperation, Op, VectorizableOp,
    expect_input_count, lift_jit_constant, mul::MulTracingOperation, unary_abstract,
};

/// Hidden staging trait for the scaling primitive.
#[doc(hidden)]
pub trait ScaleTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the scaling primitive with a captured factor.
    fn scale_op(factor: V) -> Self;
}

/// Hidden staging trait for the scaling primitive in linear programs.
#[doc(hidden)]
pub trait LinearScaleOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear scaling primitive with a captured factor.
    fn linear_scale_op(factor: V) -> Self;
}

/// Unary linear operation that multiplies its input by a captured factor.
///
/// In ordinary programs this represents "multiply by a closed-over constant." In linear programs
/// the same semantic idea is reused to scale tangent and cotangent terms.
#[derive(Clone)]
pub struct ScaleOp<T: Type, V: Typed<T>> {
    /// Captured factor applied to every input of this unary linear op.
    factor: V,

    /// Phantom marker tying the captured factor to the abstract type it is interpreted against.
    _marker: std::marker::PhantomData<T>,
}

impl<T: Type, V: Traceable<T>> ScaleOp<T, V> {
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

impl<V: Traceable<ArrayType>> ScaleOp<ArrayType, V> {
    /// Validates abstract inputs without needing a concrete instance.
    ///
    /// This is mainly used by carrier-level wrappers that want to construct or validate a scale op
    /// from type information before they have committed to a concrete `ScaleOp` value.
    pub fn abstract_eval_static(inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<T: Type, V: Traceable<T>> Debug for ScaleOp<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scale")
    }
}

impl<T: Type, V: Traceable<T>> Display for ScaleOp<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scale")
    }
}

impl<V: Traceable<ArrayType>> Op for ScaleOp<ArrayType, V> {
    fn name(&self) -> &'static str {
        "scale"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        Self::abstract_eval_static(inputs)
    }

    fn try_simplify(
        &self,
        inputs: &[AtomId],
        _is_zero_constant: &dyn Fn(AtomId) -> bool,
        _is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        if crate::tracing_v2::is_identity_one(&self.factor) { Some(inputs.to_vec()) } else { None }
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> InterpretableOp<ArrayType, V> for ScaleOp<ArrayType, V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V> + ZeroLike> LinearOperation<ArrayType, V> for ScaleOp<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0].clone().scale(self.factor().clone()))])
    }
}

impl<
    'engine,
    V: Value<ArrayType> + ZeroLike + Mul<Output = V>,
    O: MulTracingOperation<ArrayType, V> + ScaleTracingOperation<ArrayType, V> + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
    InnerLinearOperation: Clone
        + LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + LinearScaleOperation<ArrayType, Tracer<'engine, E>>,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>
    for ScaleOp<ArrayType, V>
where
    O: Op<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>, InnerLinearOperation>>, TracingError>
    {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        Ok(vec![JvpTracer {
            primal: factor.clone() * inputs[0].primal.clone(),
            tangent: inputs[0].tangent.clone().scale(factor),
        }])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOp<ArrayType, V, T, O, L> for ScaleOp<ArrayType, V>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: self.factor().clone() * input.primal.clone(),
            tangent: T::scale(self.factor().clone(), input.tangent.clone()),
        }])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> VectorizableOp<ArrayType, V> for ScaleOp<ArrayType, V> {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| self.factor().clone() * lane).collect())])
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use pretty_assertions::assert_eq;

    use crate::{parameters::Placeholder, tracing_v2::LinearProgramBuilder};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_scale_transpose_scales_output_cotangents() {
        let transpose_builder = Rc::new(RefCell::new(LinearProgramBuilder::<f64>::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(&1.0f64);
        let output_cotangent = LinearTerm::from_staged_parts(output_cotangent_atom, transpose_builder.clone());
        let contribution = ScaleOp::new(3.0f64)
            .transpose(&[output_cotangent])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution")
            .expect("transpose should produce one cotangent contribution");
        let contribution_atom = contribution.atom();
        drop(contribution);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder.build::<f64, f64>(vec![contribution_atom], Placeholder, Placeholder);
        approx_eq(transpose_program.call(2.0f64).unwrap(), 6.0);
        assert_eq!(
            transpose_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
