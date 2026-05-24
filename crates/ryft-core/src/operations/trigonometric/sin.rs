use std::fmt::Display;

use half::{bf16, f16};
use crate::{AddOperation, CosOperation};
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::tracing::{Context, Traceable, Tracer, TracingError};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`SinOperation`].
pub const SIN_OPERATION_NAME: &'static str = "sin";

/// [`Operation`] that computes the elementwise sine of one value while preserving its type metadata.
#[derive(Clone, Debug, Default)]
pub struct SinOperation;

impl Display for SinOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(SIN_OPERATION_NAME)
    }
}

impl Operation<DataType> for SinOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SIN_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl ElementwiseOperation for SinOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SIN_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Clone + Typed<DataType> + Sin> InterpretableOperation<DataType, V> for SinOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V: Clone + Typed<ArrayType> + Sin> InterpretableOperation<ArrayType, V> for SinOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().sin()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`SinOperation`]. Backend-owned closed [`Operation`]
/// types implement this trait so that generic transform code can stage [`SinOperation`] without knowing which type is
/// in use.
pub trait SupportsSin<T: Type, V: Traceable<T>> {
    /// Constructs an instance of [`SinOperation`] for this [`Operation`] type.
    fn sin_operation() -> Self;
}

/// Value-level elementwise sine capability. [`Sin`] fills the same role for [`SinOperation`] that
/// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic [`Operation`]s.
pub trait Sin: Sized {
    /// Computes the elementwise sine of this value.
    fn sin(self) -> Self;
}

impl Sin for f32 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
}

impl Sin for f64 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
}

impl Sin for bf16 {
    #[inline]
    fn sin(self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }
}

impl Sin for f16 {
    #[inline]
    fn sin(self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }
}

impl<C: Context<Operation: SupportsSin<C::Type, C::Value>>> Sin for Tracer<C> {
    #[inline]
    fn sin(self) -> Self {
        self.unary(C::Operation::sin_operation())
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_sin() {
        assert_eq!(Sin::sin(0.5f32), 0.5f32.sin());
        assert_eq!(Sin::sin(0.5f64), 0.5f64.sin());
        assert_eq!(Sin::sin(bf16::from_f32(0.5)), bf16::from_f32(0.5f32.sin()));
        assert_eq!(Sin::sin(f16::from_f32(0.5)), f16::from_f32(0.5f32.sin()));

        let operation = SinOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SIN_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "SinOperation");
        assert_eq!(format!("{operation}"), SIN_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[0.5]), Ok(vec![0.5f64.sin()]));
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[0.5]), Ok(vec![0.5f64.sin()]));

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Static(2), Size::Static(3)]),
            Some(Layout::Strided(StridedLayout::new(vec![3, 1]))),
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        assert_eq!(
            <SinOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input)),
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
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, f64, SinOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
