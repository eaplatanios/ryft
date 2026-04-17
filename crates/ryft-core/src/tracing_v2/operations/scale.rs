//! Scaling primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    ops::Mul,
};

#[cfg(test)]
use indoc::indoc;

use crate::tracing_v2::{
    TraceError, Traceable, ZeroLike,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOp, Op, OpSet, SupportsMul, SupportsScale, VectorizableOp},
};
use crate::types::{ArrayType, Type, Typed};

use super::{expect_input_count, lift_jit_constant, unary_abstract};

/// Unary linear operation that multiplies its input by a captured factor.
#[derive(Clone)]
pub struct ScaleOp<T: Type, V: Typed<T>> {
    factor: V,
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
    pub fn abstract_eval_static(inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
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

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Self::abstract_eval_static(inputs)
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        _is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        if crate::tracing_v2::graph::is_identity_one(&self.factor) { Some(inputs.to_vec()) } else { None }
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> InterpretableOp<ArrayType, V> for ScaleOp<ArrayType, V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V> + ZeroLike> LinearOp<ArrayType, V> for ScaleOp<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0].clone().scale(self.factor().clone()))])
    }
}

impl<
    V: Traceable<ArrayType> + ZeroLike + Mul<Output = V>,
    S: OpSet<ArrayType, V> + SupportsMul<ArrayType, V> + SupportsScale<ArrayType, V>,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>
    for ScaleOp<ArrayType, V>
where
    S::JitOp: Op<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        Ok(vec![JvpTracer {
            primal: factor.clone() * inputs[0].primal.clone(),
            tangent: inputs[0].tangent.clone().scale(factor),
        }])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>, T: TangentSpace<ArrayType, V>> DifferentiableOp<ArrayType, V, T>
    for ScaleOp<ArrayType, V>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: self.factor().clone() * input.primal.clone(),
            tangent: T::scale(self.factor().clone(), input.tangent.clone()),
        }])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> VectorizableOp<ArrayType, V> for ScaleOp<ArrayType, V> {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
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
        let transpose_graph = transpose_builder.build::<f64, f64>(vec![contribution_atom], Placeholder, Placeholder);
        approx_eq(transpose_graph.call(2.0f64).unwrap(), 6.0);
        assert_eq!(
            transpose_graph.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
