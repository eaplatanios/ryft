use std::collections::BTreeSet;
use std::fmt::Display;
use std::ops::Add;

use crate::broadcasting::Broadcastable;
use crate::macros::check_input_count;
use crate::operations::constants::ZeroLike;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearArrayOperation, LinearizableEngine};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Trait that represents [`Operation`] carrier types that support/include [`AddOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`AddOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsAdd<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the addition [`Operation`].
    fn add_operation() -> Self;
}

impl<'engine, E: TracingEngine<OperationCarrier: SupportsAdd<E::Type, E::Value>> + ?Sized> Add for Tracer<'engine, E> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::OperationCarrier::add_operation())
    }
}

/// Elementwise addition operation. Note that nearly every `ryft-core` transform depends on its [`Operation`] carrier
/// type implementing [`SupportsAdd`] and thus supporting this operation type.
#[derive(Clone, Debug, Default)]
pub struct AddOperation;

impl Display for AddOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl Operation<ArrayType> for AddOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_input_count!(input_types, 2, TypeError);
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
    #[inline]
    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_input_count!(input_types, 2, TypeError);
        input_types[0]
            .broadcast(&input_types[1])
            .map(|output| vec![output])
            .map_err(|_| TypeError { message: "add input types are not broadcast-compatible".to_string() })
    }
}

impl<V: Typed<DataType> + Clone + Add<Output = V>> InterpretableOperation<DataType, V> for AddOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V: Typed<ArrayType> + Clone + Add<Output = V>> InterpretableOperation<ArrayType, V> for AddOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V> + ZeroLike>
    LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>> for AddOperation
{
    fn transpose(
        &self,
        _context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            V,
            LinearArrayOperation<V, ArrayType>,
        >,
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
        _context: &mut crate::tracing::transposition::TranspositionContext<
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
    E: LinearizableEngine + ?Sized,
    AddOperation: Operation<E::Type>,
    E::Value: Add<Output = E::Value> + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsAdd<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent, inputs[1].tangent],
                <E::LinearOperationCarrier as SupportsAdd<E::Type, E::Value>>::add_operation(),
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::ProgramBuilder;
    use crate::tracing_v2::ScalarOperation;
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

        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let x = builder.add_input(<f64 as Typed<DataType>>::r#type(&1.0f64).into_owned());
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_instruction(ScalarOperation::Add, vec![x, three]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder).unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
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
