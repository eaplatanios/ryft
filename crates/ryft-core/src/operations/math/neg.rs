use crate::macros::{
    check_types, define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};
use crate::programs::types::TypeError;
use crate::types::DataType;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NegOperation`].
pub const NEG_OPERATION_NAME: &str = "neg";

/// Infers the output data types for numeric negation.
fn infer_neg_output_data_types(input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
    check_types!(@numeric, NEG_OPERATION_NAME, input_types);
    let input_type = input_types[0];
    if input_type == DataType::F8E8M0FNU {
        return Err(TypeError { message: "'neg' does not support input data type f8e8m0fnu".to_string() });
    }
    Ok(vec![input_type])
}

define_elementwise_operation!(
    @unary
    /// [`Operation`] that negates one integer, floating-point, or complex value while preserving its array metadata
    /// and reduction state. Boolean, token, structural-zero, and the unsigned-only `f8e8m0fnu` data types are rejected.
    NegOperation, NEG_OPERATION_NAME,
    Neg, neg,
    infer_data_types = infer_neg_output_data_types,
);

impl_differentiable_elementwise_operation! {
    @linear
    NegOperation,
    rule = [@negative]
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise negation capability. [`Neg`] is the fallible Ryft counterpart to [`std::ops::Neg`]
    /// that [`NegOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
    /// panicking. Value types additionally provide [`std::ops::Neg`] as ergonomic (albeit panicking) sugar layered on
    /// top of this capability.
    Neg,
    /// Negates `self`, returning a [`ProgramError`] if something goes wrong.
    neg,
    NegOperation,
);

define_tracer_operator!(@unary std::ops::Neg, neg, NegOperation, "`neg` operation failed");

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_neg() {
        let operation = NegOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), NEG_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NegOperation");
        assert_eq!(format!("{operation}"), NEG_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Ok(vec![DataType::F32]),
        );
        let output_types = Operation::<DataType>::infer_output_types(&operation, &[DataType::U8], &[]);
        assert_eq!(output_types, Ok(vec![DataType::U8]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Ok(vec![Scalar::from(-2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(1u8)],
            ),
            Ok(vec![Scalar::from(u8::MAX)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)],
            ),
            Ok(vec![Array::scalar(-2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, -2.0))],
            ),
            Ok(vec![Scalar::from(Complex::new(-1.0f64, 2.0))]),
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
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            <NegOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
            Ok(vec![input]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, NegOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_neg_type_inference() {
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::F8E8M0FNU] {
            let expected =
                TypeError { message: format!("'{NEG_OPERATION_NAME}' does not support input data type {input_type}") };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&NegOperation, &[input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(&NegOperation, &[ArrayType::scalar(input_type)], &[]),
                Err(expected),
            );
        }

        // Negation is linear, so partial-sum and reduced markers pass through unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&NegOperation, std::slice::from_ref(&unreduced), &[]),
            Ok(vec![unreduced]),
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&NegOperation, std::slice::from_ref(&reduced), &[]),
            Ok(vec![reduced]),
        );
    }

    #[test]
    fn test_neg_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = NegOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-1.0, 2.0]))],
            }],
        );
    }

    #[test]
    fn test_neg_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = NegOperation,
            cases = [{
                primals = [Array::scalar(2.0)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(-2.0)],
                tangent_outputs = [Array::scalar(-3.0)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = neg %0
                        %3:f64[] = neg %1
                    in (%2, %3)
                "},
            }],
        );
        check_gradient!(@scalar, |x| -x, at = 0.7, step = 1e-6, tolerance = 1e-6);
        assert_eq!(
            gradient_holomorphic(|input| -input, Scalar::from(Complex::new(0.7f64, -0.3))),
            Ok(Scalar::from(Complex::new(-1.0, 0.0))),
        );

        // Second-order differentiation recovers d²(-x)/dx² = 0.
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| -x, x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            0.0,
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_neg_partial_evaluation() {
        check_operation_partial_evaluation!(operation = NegOperation, inputs = [2.0], expected = -2.0,);
    }

    #[test]
    fn test_neg_transposition() {
        check_operation_transposition!(
            @exact,
            operation = NegOperation,
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0)],
                input_cotangents = [Array::scalar(-3.0)],
                pullback = indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = neg %0
                    in (%1)
                "},
            }],
        );
    }
}
