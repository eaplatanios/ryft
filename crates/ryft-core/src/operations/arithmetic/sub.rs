use std::fmt::Display;
use std::ops::{Neg, Sub};

use crate::broadcasting::Broadcastable;
use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`SubOperation`].
pub const SUB_OPERATION_NAME: &'static str = "sub";

/// [`Operation`] that subtracts two values and typically supports broadcasting semantics for arrays.
#[derive(Clone, Debug, Default)]
pub struct SubOperation;

impl Display for SubOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(SUB_OPERATION_NAME)
    }
}

impl Operation<DataType> for SubOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SUB_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        input_types[0].broadcast(&input_types[1]).map(|output| vec![output]).map_err(|_| TypeError {
            message: format!("{SUB_OPERATION_NAME} input types are not broadcast-compatible"),
        })
    }
}

impl ElementwiseOperation for SubOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SUB_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

// TODO(eaplatanios): The following two implementations can be unified.
impl<V: Clone + Typed<DataType> + Sub<Output = V>> InterpretableOperation<DataType, V> for SubOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone() - inputs[1].clone()])
    }
}

impl<V: Clone + Typed<ArrayType> + Sub<Output = V>> InterpretableOperation<ArrayType, V> for SubOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone() - inputs[1].clone()])
    }
}

impl<C: StagingContext<Operation: From<SubOperation>>> Sub for Tracer<C> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.binary(&rhs, SubOperation)
    }
}

impl<T: Type, V: Value<T> + Sub<Output = V> + Neg<Output = V>> Sub for Tangent<T, V> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Zero(r#type), Self::Zero(_)) => Self::Zero(r#type),
            (other, Self::Zero(_)) => other,
            (Self::Zero(_), Self::Value(right)) => Self::Value(-right),
            (Self::Value(left), Self::Value(right)) => Self::Value(left - right),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_sub() {
        let operation = SubOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SUB_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "SubOperation");
        assert_eq!(format!("{operation}"), SUB_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.0, 3.5]), Ok(vec![-1.5]));
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[2.0, 3.5]), Ok(vec![-1.5]));

        // Array type inference broadcasts shapes and promotes data types.
        let output = <SubOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            ],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))]);

        // Array type inference drops layout metadata when inputs disagree.
        let output = <SubOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::scalar()).with_layout(Layout::Strided(StridedLayout::new(vec![]))),
                ArrayType::scalar(DataType::F32),
            ],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);

        // Array type inference tolerates compatible inputs that only disagree on varying manual axes.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        let right = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            )
            .unwrap();
        let output = <SubOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right]).unwrap();
        assert_eq!(
            output[0].sharding().as_ref().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
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
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[2.0]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32]),
            Err(TypeError { message: format!("{SUB_OPERATION_NAME} input types are not broadcast-compatible") }),
        );
        let error = <SubOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("{SUB_OPERATION_NAME} input types are not broadcast-compatible") }
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, f64, SubOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![left, right]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![output], (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = sub %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
