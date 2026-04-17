//! Multiplication primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    ops::Mul,
};

use crate::tracing_v2::{
    TraceError, Traceable,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    ops::{DifferentiableOp, InterpretableOp, Op, OpSet, VectorizableOp},
};
use crate::types::ArrayType;

use super::{binary_same_abstract, expect_batch_sizes_match, expect_input_count};

/// Elementwise multiplication primitive.
#[derive(Clone, Default)]
pub struct MulOp;

impl Debug for MulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Mul")
    }
}

impl Display for MulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "mul")
    }
}

impl Op for MulOp {
    fn name(&self) -> &'static str {
        "mul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![binary_same_abstract("mul", inputs)?])
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        if inputs.len() == 2 {
            if is_one_constant(inputs[0]) {
                Some(vec![inputs[1]])
            } else if is_one_constant(inputs[1]) {
                Some(vec![inputs[0]])
            } else if is_zero_constant(inputs[0]) {
                Some(vec![inputs[0]])
            } else if is_zero_constant(inputs[1]) {
                Some(vec![inputs[1]])
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> InterpretableOp<ArrayType, V> for MulOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>, T: TangentSpace<ArrayType, V>, S: OpSet<ArrayType, V>>
    DifferentiableOp<ArrayType, V, T, S> for MulOp
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, OpSet = S>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        let left = &inputs[0];
        let right = &inputs[1];
        Ok(vec![JvpTracer {
            primal: left.primal.clone() * right.primal.clone(),
            tangent: T::add(
                T::scale(right.primal.clone(), left.tangent.clone()),
                T::scale(left.primal.clone(), right.tangent.clone()),
            ),
        }])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> VectorizableOp<ArrayType, V> for MulOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![Batch::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left * right)
                .collect(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::{engine::ArrayScalarEngine, test_support};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let engine = ArrayScalarEngine::<f64>::new();
        let output = DifferentiableOp::<ArrayType, f64, f64, crate::tracing_v2::CoreOpSet>::jvp(
            &MulOp,
            &engine,
            &[JvpTracer { primal: 2.0f64, tangent: 3.0f64 }, JvpTracer { primal: 5.0f64, tangent: -1.0f64 }],
        )
        .unwrap()
        .pop()
        .unwrap();

        approx_eq(output.primal, 10.0);
        approx_eq(output.tangent, 13.0);
        test_support::assert_bilinear_pushforward_rendering();
    }
}
