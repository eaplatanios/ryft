use crate::macros::{define_elementwise_operation, define_tracer_operator};

/// Canonical operation name for [`SubOperation`].
pub const SUB_OPERATION_NAME: &str = "sub";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that subtracts two values and typically supports broadcasting semantics for arrays.
    SubOperation, SUB_OPERATION_NAME, Sub, sub,
    /// Value-level elementwise subtraction capability. [`Sub`] is the fallible Ryft counterpart to [`std::ops::Sub`]
    /// that [`SubOperation`] interprets through, surfacing a [`ProgramError`](crate::ProgramError) when something
    /// goes wrong, instead of panicking. Value types additionally provide [`std::ops::Sub`] as ergonomic (albeit
    /// panicking) sugar layered on top of this capability.
);

define_tracer_operator!(@binary std::ops::Sub, sub, SubOperation, "`sub` operation failed");

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout, TypeError};

    use super::*;

    #[test]
    fn test_sub() {
        let operation = SubOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SUB_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "SubOperation");
        assert_eq!(format!("{operation}"), SUB_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0), Scalar::from(3.5)],
            ),
            Ok(vec![Scalar::from(-1.5)])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(2.0), TestArray::scalar(3.5)],
            ),
            Ok(vec![TestArray::scalar(-1.5)]),
        );

        // Array type inference broadcasts shapes and promotes data types.
        let output = <SubOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            ],
            &[],
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
            &[],
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
        let output =
            <SubOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right], &[]).unwrap();
        assert_eq!(
            output[0].sharding().as_ref().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F64)], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError { message: format!("'{SUB_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );
        let error = <SubOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ],
            &[],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("'{SUB_OPERATION_NAME}' input types are not broadcast-compatible") },
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, SubOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![left, right], Vec::new()).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
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
