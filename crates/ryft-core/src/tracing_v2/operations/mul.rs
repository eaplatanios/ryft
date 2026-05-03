use std::fmt::Display;
use std::ops::Mul;

use crate::broadcasting::Broadcastable;
use crate::macros::check_input_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::{ElementwiseArrayOperation, InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearizableEngine};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::SupportsScale;

/// Trait that represents [`Operation`] carrier types that support/include [`MulOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`MulOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsMul<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the multiplication [`Operation`].
    fn mul_operation() -> Self;
}

impl<'engine, E: TracingEngine + ?Sized> Mul for Tracer<'engine, E>
where
    E::OperationCarrier: SupportsMul<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::OperationCarrier::mul_operation())
    }
}

/// Elementwise multiplication primitive.
///
/// Multiplication is both a user-visible primitive and a building block for derivative rules such
/// as the JVP of [`super::SinOperation`] and the replay of captured scale factors.
#[derive(Clone, Debug, Default)]
pub struct MulOperation;

impl Display for MulOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl ElementwiseArrayOperation for MulOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "mul"
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

impl Operation<DataType> for MulOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "mul"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_input_count!(input_types, 2, TypeError);
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
    E: LinearizableEngine + ?Sized,
    MulOperation: Operation<E::Type>,
    E::Value: Mul<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsAdd<E::Type, E::Value> + SupportsScale<E::Type, E::Value>,
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
            .stage(
                <E::LinearOperationCarrier as SupportsScale<E::Type, E::Value>>::scale_operation(right.primal.clone()),
                &[left.tangent],
            )?
            .into_iter()
            .next()
            .expect("mul jvp scale should produce one tangent");
        let right_term = context
            .stage(
                <E::LinearOperationCarrier as SupportsScale<E::Type, E::Value>>::scale_operation(left.primal.clone()),
                &[right.tangent],
            )?
            .into_iter()
            .next()
            .expect("mul jvp scale should produce one tangent");
        let tangent = context
            .stage(
                <E::LinearOperationCarrier as SupportsAdd<E::Type, E::Value>>::add_operation(),
                &[left_term, right_term],
            )?
            .into_iter()
            .next()
            .expect("mul jvp add should produce one tangent");
        Ok(vec![JvpTracer { primal: left.primal.clone() * right.primal.clone(), tangent }])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::Program;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::{LinearScalarOperation, Sin, jvp, linearize};
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
            linearize(&engine, |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64)).unwrap();

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
