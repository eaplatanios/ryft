use std::fmt::Display;
use std::ops::Div;

use crate::broadcasting::Broadcastable;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::tracing::{Context, Traceable, Tracer, TracingError};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`DivOperation`].
pub const DIV_OPERATION_NAME: &'static str = "div";

/// [`Operation`] that divides two values and typically supports broadcasting semantics for arrays.
#[derive(Clone, Debug, Default)]
pub struct DivOperation;

impl Display for DivOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(DIV_OPERATION_NAME)
    }
}

impl Operation<DataType> for DivOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DIV_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        input_types[0].broadcast(&input_types[1]).map(|output| vec![output]).map_err(|_| TypeError {
            message: format!("{DIV_OPERATION_NAME} input types are not broadcast-compatible"),
        })
    }
}

impl ElementwiseOperation for DivOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DIV_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

impl<V: Clone + Typed<DataType> + Div<Output = V>> InterpretableOperation<DataType, V> for DivOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() / inputs[1].clone()])
    }
}

impl<V: Clone + Typed<ArrayType> + Div<Output = V>> InterpretableOperation<ArrayType, V> for DivOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        Ok(vec![inputs[0].clone() / inputs[1].clone()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`DivOperation`]. Backend-owned closed [`Operation`]
/// types implement this trait so that generic transform code can stage [`DivOperation`] without knowing which type is
/// in use.
pub trait SupportsDiv<T: Type, V: Traceable<T>> {
    /// Constructs an instance of [`DivOperation`] for this [`Operation`] type.
    fn div_operation() -> Self;
}

impl<C: Context<Operation: SupportsDiv<C::Type, C::Value>>> Div for Tracer<C> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.binary(rhs, C::Operation::div_operation())
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
    fn test_div() {
        let operation = DivOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), DIV_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "DivOperation");
        assert_eq!(format!("{operation}"), DIV_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[7.0, 2.0]), Ok(vec![3.5]));
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[7.0, 2.0]), Ok(vec![3.5]));

        // Array type inference broadcasts shapes and promotes data types.
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
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

        // Array type inference drops layout metadata when inputs disagree.
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::scalar(), Some(Layout::Strided(StridedLayout::new(vec![]))), None)
                    .unwrap(),
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
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right]).unwrap();
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
            Err(TracingError::InvalidInputCount { expected: 2, got: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[2.0]),
            Err(TracingError::InvalidInputCount { expected: 2, got: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32]),
            Err(TypeError { message: format!("{DIV_OPERATION_NAME} input types are not broadcast-compatible") }),
        );
        let error = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("{DIV_OPERATION_NAME} input types are not broadcast-compatible") }
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, f64, DivOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![left, right]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![output], (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = div %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
