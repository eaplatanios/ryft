use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`SinOperation`].
pub const SIN_OPERATION_NAME: &'static str = "sin";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise sine of one value while preserving its type
    /// metadata.
    SinOperation, SIN_OPERATION_NAME, Sin, sin,
    /// Value-level elementwise sine capability. [`Sin`] fills the same role for [`SinOperation`] that
    /// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic
    /// [`Operation`](crate::Operation)s.
);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout, TypeError};

    use super::*;

    #[test]
    fn test_sin() {
        assert_eq!(Scalar::from(0.5f32).sin().unwrap(), 0.5f32.sin());
        assert_eq!(Scalar::from(0.5f64).sin().unwrap(), 0.5f64.sin());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).sin().unwrap(), bf16::from_f32(0.5f32.sin()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).sin().unwrap(), f16::from_f32(0.5f32.sin()));

        let operation = SinOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SIN_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "SinOperation");
        assert_eq!(format!("{operation}"), SIN_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(
            InterpretableOperation::<Scalar, EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(0.5)],
            ),
            Ok(vec![Scalar::from(0.5f64.sin())]),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::scalar(0.5)]
            ),
            Ok(vec![TestArray::scalar(0.5f64.sin())]),
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
            InterpretableOperation::<Scalar, EagerContext<Scalar>>::interpret(&operation, &EagerContext::new(), &[],),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, SinOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
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
