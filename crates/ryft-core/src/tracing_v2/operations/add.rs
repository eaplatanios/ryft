//! Addition primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    ops::Add,
};

use crate::tracing_v2::{
    TraceError, TraceValue, Zero,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOp, Op, VectorizableOp},
};
use crate::types::ArrayType;

use super::{binary_same_abstract, expect_batch_sizes_match, expect_input_count};

/// Elementwise addition primitive.
#[derive(Clone, Default)]
pub struct AddOp;

impl Debug for AddOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Add")
    }
}

impl Display for AddOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "add")
    }
}

impl Op for AddOp {
    fn name(&self) -> &'static str {
        "add"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![binary_same_abstract("add", inputs)?])
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        if inputs.len() == 2 {
            if is_zero_constant(inputs[0]) {
                Some(vec![inputs[1]])
            } else if is_zero_constant(inputs[1]) {
                Some(vec![inputs[0]])
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<V: TraceValue + Add<Output = V>> InterpretableOp<V> for AddOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V: TraceValue + Add<Output = V> + Zero> LinearOp<V> for AddOp {
    fn transpose(&self, output_cotangents: &[LinearTerm<V>]) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0].clone()), Some(output_cotangents[0].clone())])
    }
}

impl<V: TraceValue + Add<Output = V>, T: TangentSpace<V>> DifferentiableOp<V, T> for AddOp {
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.clone() + inputs[1].primal.clone(),
            tangent: T::add(inputs[0].tangent.clone(), inputs[1].tangent.clone()),
        }])
    }
}

impl<V: TraceValue + Add<Output = V>> VectorizableOp<V> for AddOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![Batch::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left + right)
                .collect(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{
        tracing_v2::test_support,
        types::{DataType, Layout, Shape, StridedLayout},
    };

    use super::*;

    #[test]
    fn test_add_abstract_eval_rejects_incompatible_inputs() {
        let error =
            <AddOp as Op>::abstract_eval(&AddOp, &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)])
                .unwrap_err();

        assert_eq!(error, TraceError::IncompatibleAbstractValues { op: "add" });
        test_support::assert_reference_graph_rendering();
    }

    #[test]
    fn test_add_abstract_eval_drops_layout_when_inputs_disagree() {
        let output = <AddOp as Op>::abstract_eval(
            &AddOp,
            &[
                ArrayType::new(DataType::F32, Shape::scalar(), Some(Layout::Strided(StridedLayout::new(vec![]))), None)
                    .unwrap(),
                ArrayType::scalar(DataType::F32),
            ],
        )
        .unwrap();

        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);
    }

    #[test]
    fn test_add_batch_requires_matching_lane_counts() {
        let error = AddOp.batch(&[Batch::new(vec![1.0f64, 2.0f64]), Batch::new(vec![3.0f64])]).unwrap_err();

        assert_eq!(error, TraceError::MismatchedBatchSize);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
