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
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOp, Op, VectorizableOp},
};
use crate::types::ArrayType;

use super::{expect_input_count, lift_jit_constant, unary_abstract};

/// Unary linear operation that multiplies its input by a captured factor.
#[derive(Clone)]
pub struct ScaleOp<V: Traceable> {
    factor: V,
}

impl<V: Traceable> ScaleOp<V> {
    /// Creates a new scale operation capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }

    /// Returns the captured scale factor.
    #[inline]
    pub fn factor(&self) -> &V {
        &self.factor
    }

    /// Validates abstract inputs without needing a concrete instance.
    pub fn abstract_eval_static(inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: Traceable> Debug for ScaleOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scale")
    }
}

impl<V: Traceable> Display for ScaleOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scale")
    }
}

impl<V: Traceable> Op for ScaleOp<V> {
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

impl<V: Traceable + Mul<Output = V>> InterpretableOp<V> for ScaleOp<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }
}

impl<V: Traceable + Mul<Output = V> + ZeroLike> LinearOp<V> for ScaleOp<V> {
    fn transpose(&self, output_cotangents: &[LinearTerm<V>]) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0].clone().scale(self.factor().clone()))])
    }
}

impl<V: Traceable + ZeroLike + Mul<Output = V>> InterpretableOp<crate::tracing_v2::linear::Linearized<JitTracer<V>>>
    for ScaleOp<V>
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<V>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<V>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        Ok(vec![JvpTracer {
            primal: factor.clone() * inputs[0].primal.clone(),
            tangent: inputs[0].tangent.clone().scale(factor),
        }])
    }
}

impl<V: Traceable + Mul<Output = V>, T: TangentSpace<V>> DifferentiableOp<V, T> for ScaleOp<V> {
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: self.factor().clone() * input.primal.clone(),
            tangent: T::scale(self.factor().clone(), input.tangent.clone()),
        }])
    }
}

impl<V: Traceable + Mul<Output = V>> VectorizableOp<V> for ScaleOp<V> {
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
