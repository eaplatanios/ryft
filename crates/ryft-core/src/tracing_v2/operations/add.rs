use std::collections::BTreeSet;
use std::fmt::{Debug, Display};
use std::ops::Add;

use crate::broadcasting::Broadcastable;
use crate::macros::check_input_count;
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::operations::constants::ZeroLike;
use crate::tracing_v2::{DifferentiableEngine, LinearArrayOperation};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::{DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Hidden carrier capability for staging the addition primitive.
///
/// Backend-owned closed op carriers (such as [`ArrayOperation`](super::ArrayOperation) and the XLA backend's
/// `XlaOperation`) implement this trait so that generic transform code can stage `AddOperation` without
/// knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsAdd<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the addition primitive.
    fn add_operation() -> Self;
}

impl<'engine, E: TracingEngine + ?Sized> Add for Tracer<'engine, E>
where
    E::Operation: SupportsAdd<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::add_operation())
    }
}

/// Elementwise addition primitive.
///
/// In the larger architecture, [`AddOperation`] is the canonical "fully supported" primitive: nearly
/// every transform depends on addition being available in its staged carrier.
#[derive(Clone, Default)]
pub struct AddOperation;

impl Debug for AddOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Add")
    }
}

impl Display for AddOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "add")
    }
}

impl Operation<ArrayType> for AddOperation {
    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 2 {
            return Err(TypeError { message: format!("add expected 2 input types but got {}", input_types.len()) });
        }
        match input_types[0].broadcast(&input_types[1]) {
            Ok(output) => Ok(vec![output]),
            Err(_) => {
                // JAX keeps generic shape/type broadcasting conservative and instead makes binary
                // primitives tolerate differing VMA annotations by implicitly inserting `pvary`
                // where needed. We model that more narrowly here: retry abstract evaluation after
                // erasing only the varying-manual-axis metadata, then restore the union on the
                // result instead of weakening generic `ArrayType` broadcasting everywhere.
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
                    .map_err(|_| TypeError { message: "add input types are not broadcast-compatible".to_string() })?;
                if let Some(sharding) = &mut output.sharding {
                    sharding.varying_manual_axes = original_varying_manual_axes;
                }
                Ok(vec![output])
            }
        }
    }
}

impl Operation<DataType> for AddOperation {
    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        if input_types.len() != 2 {
            return Err(TypeError { message: format!("add expected 2 input types but got {}", input_types.len()) });
        }
        input_types[0]
            .broadcast(&input_types[1])
            .map(|output| vec![output])
            .map_err(|_| TypeError { message: "add input types are not broadcast-compatible".to_string() })
    }
}

impl<V: Typed<ArrayType> + Clone + Add<Output = V>> InterpretableOperation<ArrayType, V> for AddOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V: Typed<DataType> + Clone + Add<Output = V>> InterpretableOperation<DataType, V> for AddOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V> + ZeroLike> LinearOperation<ArrayType, V, LinearArrayOperation<V>>
    for AddOperation
{
    fn transpose(
        &self,
        _context: &mut crate::tracing_v2::operations::TranspositionContext<'_, ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<V: Traceable<DataType> + crate::parameters::Parameter + Add<Output = V> + ZeroLike>
    LinearOperation<DataType, V, LinearArrayOperation<V, DataType>> for AddOperation
{
    fn transpose(
        &self,
        _context: &mut crate::tracing_v2::operations::TranspositionContext<
            '_,
            DataType,
            V,
            LinearArrayOperation<V, DataType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![output_cotangents[0], output_cotangents[0]])
    }
}

impl<E> DifferentiableOperation<E> for AddOperation
where
    E: DifferentiableEngine + ?Sized,
    AddOperation: Operation<E::Type>,
    E::Value: Add<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsAdd<E::Type, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent, inputs[1].tangent],
                <E::LinearOperation as SupportsAdd<E::Type, E::Value>>::add_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("add jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone() + inputs[1].primal.clone(), tangent }])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::test_support;
    use crate::types::{DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_add_abstract_eval_broadcasts_and_promotes_inputs() {
        let output = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &AddOperation,
            &[
                ArrayType::scalar(DataType::F32),
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
    fn test_add_abstract_eval_rejects_non_broadcastable_inputs() {
        let error = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &AddOperation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap_err();

        assert_eq!(error, TypeError { message: "add input types are not broadcast-compatible".to_string() });
        test_support::assert_reference_program_rendering();
    }

    #[test]
    fn test_add_abstract_eval_drops_layout_when_inputs_disagree() {
        let output = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &AddOperation,
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
    fn test_add_abstract_eval_merges_varying_manual_axes_for_compatible_inputs() {
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

        let output = <AddOperation as Operation<ArrayType>>::infer_output_types(&AddOperation, &[left, right]).unwrap();

        assert_eq!(
            output[0].sharding.as_ref().unwrap().varying_manual_axes,
            BTreeSet::from(["x".to_string(), "y".to_string()])
        );
    }
}
