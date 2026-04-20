//! Multiplication primitive for [`crate::tracing_v2`].
//!
//! `MulOperation` is the bilinear counterpart to [`super::AddOperation`]. It is used directly in user programs
//! and indirectly inside derivative rules for many other primitives, so its docs are a good place
//! to understand how one primitive threads through staging, replay, batching, and JVP evaluation.

use std::{
    fmt::{Debug, Display},
    ops::Mul,
};

use crate::broadcasting::Broadcastable;
use crate::tracing_v2::{
    AtomId, Traceable, TracingError,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
};
use crate::types::{ArrayType, Type};

use super::{
    DifferentiableOperation, InterpretableOperation, Operation, VectorizableOperation, expect_batch_sizes_match,
    expect_input_count,
};

/// Hidden staging trait for the multiplication primitive.
#[doc(hidden)]
pub trait MulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the multiplication primitive.
    fn mul_op() -> Self;
}

/// Elementwise multiplication primitive.
///
/// Multiplication is both a user-visible primitive and a building block for derivative rules such
/// as the JVP of [`super::SinOperation`] and the replay of captured scale factors.
#[derive(Clone, Default)]
pub struct MulOperation;

impl Debug for MulOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Mul")
    }
}

impl Display for MulOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "mul")
    }
}

impl Operation for MulOperation {
    fn name(&self) -> &'static str {
        "mul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        expect_input_count(inputs.len(), 2)?;
        inputs[0]
            .broadcast(&inputs[1])
            .map(|output| vec![output])
            .map_err(|_| TracingError::IncompatibleAbstractValues { op: "mul" })
    }

    fn try_simplify(
        &self,
        inputs: &[AtomId],
        is_zero_constant: &dyn Fn(AtomId) -> bool,
        is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
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

impl<V: Traceable<ArrayType> + Mul<Output = V>> InterpretableOperation<ArrayType, V> for MulOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOperation<ArrayType, V, T, O, L> for MulOperation
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
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

impl<V: Traceable<ArrayType> + Mul<Output = V>> VectorizableOperation<ArrayType, V> for MulOperation {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
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
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::{engine::ArrayScalarEngine, test_support};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let engine = ArrayScalarEngine::<f64>::new();
        let output = DifferentiableOperation::<
            ArrayType,
            f64,
            f64,
            crate::tracing_v2::PrimitiveOperation<ArrayType, f64>,
            crate::tracing_v2::LinearPrimitiveOperation<ArrayType, f64>,
        >::jvp(
            &MulOperation,
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

    #[test]
    fn test_mul_abstract_eval_broadcasts_and_promotes_inputs() {
        let output = <MulOperation as Operation>::abstract_eval(
            &MulOperation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(3)]), None, None).unwrap(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(
            output,
            vec![
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None,).unwrap()
            ]
        );
    }

    #[test]
    fn test_mul_abstract_eval_rejects_non_broadcastable_inputs() {
        let error = <MulOperation as Operation>::abstract_eval(
            &MulOperation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap_err();

        assert_eq!(error, TracingError::IncompatibleAbstractValues { op: "mul" });
    }
}
