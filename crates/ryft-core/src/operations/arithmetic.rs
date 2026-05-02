use std::fmt::Display;
use std::ops::Add;

use crate::broadcasting::Broadcastable;
use crate::macros::check_input_count;
use crate::operations::{ElementwiseArrayOperation, InterpretableOperation, Operation};
use crate::tracing::{Traceable, Tracer, TracingEngine, TracingError};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// [`Operation`] that consumes two values and produces their broadcasted elementwise sum.
#[derive(Clone, Debug, Default)]
pub struct AddOperation;

impl Display for AddOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<DataType>>::name(self))
    }
}

impl Operation<DataType> for AddOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "add"
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_input_count!(input_types, 2, TypeError);
        input_types[0]
            .broadcast(&input_types[1])
            .map(|output| vec![output])
            .map_err(|_| TypeError { message: "add input types are not broadcast-compatible".to_string() })
    }
}

impl ElementwiseArrayOperation for AddOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "add"
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

impl<V: Typed<DataType> + Clone + Add<Output = V>> InterpretableOperation<DataType, V> for AddOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V: Typed<ArrayType> + Clone + Add<Output = V>> InterpretableOperation<ArrayType, V> for AddOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`AddOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`AddOperation`] without
/// knowing which carrier is in use.
pub trait SupportsAdd<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`AddOperation`].
    fn add_operation() -> Self;
}

impl<'engine, E: TracingEngine<OperationCarrier: SupportsAdd<E::Type, E::Value>> + ?Sized> Add for Tracer<'engine, E> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::OperationCarrier::add_operation())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_add_operation() {
        let operation = AddOperation;

        assert_eq!(Operation::<DataType>::name(&operation), "add");
        assert_eq!(format!("{operation:?}"), "AddOperation");
        assert_eq!(format!("{operation}"), "add");
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.0, 3.5]), Ok(vec![5.5]));
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[2.0, 3.5]), Ok(vec![5.5]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F64)]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.0]),
            Err(TracingError::InvalidInputCount { expected: 2, got: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[2.0]),
            Err(TracingError::InvalidInputCount { expected: 2, got: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32]),
            Err(TypeError { message: "add input types are not broadcast-compatible".to_string() }),
        );

        let mut builder = ProgramBuilder::<DataType, f64, AddOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![left, right]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![output], (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_add_array_operation_broadcasts_and_promotes_inputs() {
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
    fn test_add_array_operation_rejects_non_broadcastable_inputs() {
        let error = <AddOperation as Operation<ArrayType>>::infer_output_types(
            &AddOperation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap_err();

        assert_eq!(error, TypeError { message: "add input types are not broadcast-compatible".to_string() });
    }

    #[test]
    fn test_add_array_operation_drops_layout_when_inputs_disagree() {
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
    fn test_add_array_operation_merges_varying_manual_axes_for_compatible_inputs() {
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
