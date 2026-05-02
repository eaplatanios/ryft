use std::collections::BTreeSet;
use std::fmt::{Debug, Display};
use std::ops::Mul;

use crate::broadcasting::Broadcastable;
use crate::macros::check_input_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::{SupportsAdd, SupportsScale};

/// Hidden carrier capability for staging the multiplication primitive.
#[doc(hidden)]
pub trait SupportsMul<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the multiplication primitive.
    fn mul_operation() -> Self;
}

impl<'engine, E: TracingEngine + ?Sized> Mul for Tracer<'engine, E>
where
    E::Operation: SupportsMul<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::mul_operation())
    }
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

impl Operation<ArrayType> for MulOperation {
    fn name(&self) -> &'static str {
        "mul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 2 {
            return Err(TypeError { message: format!("mul expected 2 input types but got {}", input_types.len()) });
        }
        match input_types[0].broadcast(&input_types[1]) {
            Ok(output) => Ok(vec![output]),
            Err(_) => {
                // As with `add`, keep generic `ArrayType` broadcasting strict and apply the
                // JAX-like implicit `pvary` behavior only at the primitive boundary. We retry
                // after clearing VMA metadata, then conservatively mark the result as varying over
                // every manual axis that either input may vary across.
                let original_varying_manual_axes = match (&input_types[0].sharding, &input_types[1].sharding) {
                    (None, None) => BTreeSet::new(),
                    (Some(left), None) => left.varying_manual_axes.clone(),
                    (None, Some(right)) => right.varying_manual_axes.clone(),
                    (Some(left), Some(right)) => {
                        left.varying_manual_axes.union(&right.varying_manual_axes).cloned().collect::<BTreeSet<_>>()
                    }
                };
                let mut left = input_types[0].clone();
                let mut right = input_types[1].clone();
                if let Some(sharding) = &mut left.sharding {
                    sharding.varying_manual_axes.clear();
                }
                if let Some(sharding) = &mut right.sharding {
                    sharding.varying_manual_axes.clear();
                }
                let mut output = left
                    .broadcast(&right)
                    .map_err(|_| TypeError { message: "mul input types are not broadcast-compatible".to_string() })?;
                if let Some(sharding) = &mut output.sharding {
                    sharding.varying_manual_axes = original_varying_manual_axes;
                }
                Ok(vec![output])
            }
        }
    }
}

impl Operation<DataType> for MulOperation {
    fn name(&self) -> &'static str {
        "mul"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        if input_types.len() != 2 {
            return Err(TypeError { message: format!("mul expected 2 input types but got {}", input_types.len()) });
        }
        input_types[0]
            .broadcast(&input_types[1])
            .map(|output| vec![output])
            .map_err(|_| TypeError { message: "mul input types are not broadcast-compatible".to_string() })
    }
}

impl<V: Typed<ArrayType> + Clone + Mul<Output = V>> InterpretableOperation<ArrayType, V> for MulOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<V: Typed<DataType> + Clone + Mul<Output = V>> InterpretableOperation<DataType, V> for MulOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<E> DifferentiableOperation<E> for MulOperation
where
    E: DifferentiableEngine + ?Sized,
    MulOperation: Operation<E::Type>,
    E::Value: Mul<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsAdd<E::Type, E::Value> + SupportsScale<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let left_term = context
            .apply_operation(
                &[left.tangent],
                <E::LinearOperation as SupportsScale<E::Type, E::Value>>::scale_operation(right.primal.clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("mul jvp scale should produce one tangent");
        let right_term = context
            .apply_operation(
                &[right.tangent],
                <E::LinearOperation as SupportsScale<E::Type, E::Value>>::scale_operation(left.primal.clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("mul jvp scale should produce one tangent");
        let tangent = context
            .apply_operation(
                &[left_term, right_term],
                <E::LinearOperation as SupportsAdd<E::Type, E::Value>>::add_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("mul jvp add should produce one tangent");
        Ok(vec![JvpTracer { primal: left.primal.clone() * right.primal.clone(), tangent }])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::Program;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::{LinearScalarOperation, Sin, jvp, jvp_program};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let engine = ScalarEngine::<f64>::new();
        let (primal, tangent) =
            jvp(&engine, |(left, right)| left * right, (2.0f64, 5.0f64), (3.0f64, -1.0f64)).unwrap();

        approx_eq(primal, 10.0);
        approx_eq(tangent, 13.0);

        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, (f64, f64), f64>) =
            jvp_program(&engine, |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64)).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=3] %0
                    %3:f64 = scale [factor=2] %1
                    %4:f64 = add %2 %3
                    %5:f64 = scale [factor=-0.4161468365471424] %0
                    %6:f64 = add %4 %5
                in (%6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_mul_abstract_eval_broadcasts_and_promotes_inputs() {
        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(
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
        let error = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &MulOperation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap_err();

        assert_eq!(error, TypeError { message: "mul input types are not broadcast-compatible".to_string() });
    }

    #[test]
    fn test_mul_abstract_eval_merges_varying_manual_axes_for_compatible_inputs() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let left = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let right = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(&MulOperation, &[left, right]).unwrap();

        assert_eq!(
            output[0].sharding.as_ref().unwrap().varying_manual_axes,
            BTreeSet::from(["x".to_string(), "y".to_string()])
        );
    }
}
