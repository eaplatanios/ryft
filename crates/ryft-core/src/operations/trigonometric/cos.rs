use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Canonical operation name for [`CosOperation`].
pub const COS_OPERATION_NAME: &'static str = "cos";

/// [`Operation`] that computes the elementwise cosine of one value while preserving its type metadata.
#[derive(Clone, Debug, Default)]
pub struct CosOperation;

impl Display for CosOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(COS_OPERATION_NAME)
    }
}

impl Operation<DataType> for CosOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COS_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl Operation<ArrayType> for CosOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COS_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for CosOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<T: Type, V: Clone + Value<T> + Cos> InterpretableOperation<T, V> for CosOperation
where
    Self: Operation<T>,
{
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].cos()?])
    }
}

impl<T: Type, V: Value<T>, O> PartiallyEvaluatableOperation<T, V, O> for CosOperation {}

/// Value-level elementwise cosine capability. [`Cos`] fills the same role for [`CosOperation`] that
/// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic [`Operation`]s.
pub trait Cos: Sized {
    /// Computes the elementwise cosine of this value, returning a [`ProgramError`] if something goes wrong.
    fn cos(&self) -> Result<Self, ProgramError>;
}

impl<C: StagingContext<Operation: From<CosOperation>>> Cos for Tracer<C, C::Meta> {
    #[inline]
    fn cos(&self) -> Result<Self, ProgramError> {
        Ok(self.unary(CosOperation))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
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
    fn test_cos() {
        assert_eq!(Scalar::from(0.5f32).cos().unwrap(), 0.5f32.cos());
        assert_eq!(Scalar::from(0.5f64).cos().unwrap(), 0.5f64.cos());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).cos().unwrap(), bf16::from_f32(0.5f32.cos()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).cos().unwrap(), f16::from_f32(0.5f32.cos()));

        let operation = CosOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), COS_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "CosOperation");
        assert_eq!(format!("{operation}"), COS_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(
            InterpretableOperation::<DataType, Scalar>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(0.5)],
            ),
            Ok(vec![Scalar::from(0.5f64.cos())]),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::scalar(0.5)],
            ),
            Ok(vec![TestArray::scalar(0.5f64.cos())]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            <CosOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input)),
            Ok(vec![input]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, Scalar>::interpret(&operation, &EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, Scalar, CosOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = cos %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
