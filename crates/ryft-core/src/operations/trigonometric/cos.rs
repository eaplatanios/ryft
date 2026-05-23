use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::tracing::{Context, Traceable, Tracer, TracingError};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

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

impl ElementwiseOperation for CosOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COS_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Clone + Typed<DataType> + Cos> InterpretableOperation<DataType, V> for CosOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<V: Clone + Typed<ArrayType> + Cos> InterpretableOperation<ArrayType, V> for CosOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().cos()])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`CosOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`CosOperation`] without
/// knowing which carrier is in use.
pub trait SupportsCos<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`CosOperation`].
    fn cos_operation() -> Self;
}

/// Value-level elementwise cosine capability. [`Cos`] fills the same role for [`CosOperation`] that
/// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic [`Operation`]s.
pub trait Cos: Sized {
    /// Computes the elementwise cosine of this value.
    fn cos(self) -> Self;
}

impl Cos for f32 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Cos for f64 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Cos for bf16 {
    #[inline]
    fn cos(self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }
}

impl Cos for f16 {
    #[inline]
    fn cos(self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }
}

impl<C: Context<Operation: SupportsCos<C::Type, C::Value>>> Cos for Tracer<C> {
    #[inline]
    fn cos(self) -> Self {
        self.unary(C::Operation::cos_operation())
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
    fn test_cos() {
        assert_eq!(Cos::cos(0.5f32), 0.5f32.cos());
        assert_eq!(Cos::cos(0.5f64), 0.5f64.cos());
        assert_eq!(Cos::cos(bf16::from_f32(0.5)), bf16::from_f32(0.5f32.cos()));
        assert_eq!(Cos::cos(f16::from_f32(0.5)), f16::from_f32(0.5f32.cos()));

        let operation = CosOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), COS_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "CosOperation");
        assert_eq!(format!("{operation}"), COS_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[0.5]), Ok(vec![0.5f64.cos()]));
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[0.5]), Ok(vec![0.5f64.cos()]));

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
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, f64, CosOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
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
