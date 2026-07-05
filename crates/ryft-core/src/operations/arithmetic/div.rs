use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::Context;
use crate::contexts::StagingContext;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

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

impl Operation<ArrayType> for DivOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DIV_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for DivOperation {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

impl<T: Type, V: Clone + Value<T> + Div, C> InterpretableOperation<T, V, C> for DivOperation
where
    Self: Operation<T>,
{
    #[inline]
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].div(&inputs[1])?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for DivOperation where C::Operation: From<DivOperation> {}

/// Value-level elementwise division capability. [`Div`] is the fallible Ryft counterpart to [`std::ops::Div`]
/// that [`DivOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
/// panicking. Value types additionally provide [`std::ops::Div`] as ergonomic (albeit panicking) sugar layered on top
/// of this capability.
pub trait Div: Sized {
    /// Divides `self` by `rhs`, returning a [`ProgramError`] if something goes wrong.
    fn div(&self, rhs: &Self) -> Result<Self, ProgramError>;
}

impl<C: StagingContext<Operation: From<DivOperation>>> Div for Tracer<C, C::Meta> {
    #[inline]
    fn div(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.binary(rhs, DivOperation))
    }
}

impl<C: StagingContext<Operation: From<DivOperation>>> std::ops::Div for Tracer<C, C::Meta> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.binary(&rhs, DivOperation)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::scalars::Scalar;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout};

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
        assert_eq!(
            InterpretableOperation::<DataType, Scalar, EagerContext<DataType, Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(7.0), Scalar::from(2.0)],
            ),
            Ok(vec![Scalar::from(3.5)]),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray, EagerContext<ArrayType, TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::scalar(7.0), TestArray::scalar(2.0)],
            ),
            Ok(vec![TestArray::scalar(3.5)]),
        );

        // Array type inference broadcasts shapes and promotes data types.
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            ],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))]);

        // Array type inference drops layout metadata when inputs disagree.
        let output = <DivOperation as Operation<ArrayType>>::infer_output_types(
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
            InterpretableOperation::<DataType, Scalar, EagerContext<DataType, Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(2.0)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray, EagerContext<ArrayType, TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32]),
            Err(TypeError { message: format!("{DIV_OPERATION_NAME} input types are not broadcast-compatible") }),
        );
        let error = <DivOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("{DIV_OPERATION_NAME} input types are not broadcast-compatible") }
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, Scalar, DivOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
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
