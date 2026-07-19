use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SubOperation`].
pub const SUB_OPERATION_NAME: &str = "sub";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that subtracts two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that carry partial sums must both be unreduced over exactly the same
    /// mesh axes (subtraction is linear, so the difference of two partial sums over the same axes is another valid
    /// partial sum); mixing an unreduced operand with an already reduced operand would duplicate the reduced
    /// contribution when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    SubOperation, SUB_OPERATION_NAME,
    Sub, sub,
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @linear
    SubOperation,
    rule = [@positive, @negative]
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise subtraction capability. [`Sub`] is the fallible Ryft counterpart to [`std::ops::Sub`]
    /// that [`SubOperation`] interprets through, surfacing a [`ProgramError`] when something
    /// goes wrong, instead of panicking. Value types additionally provide [`std::ops::Sub`] as ergonomic (albeit
    /// panicking) sugar layered on top of this capability.
    Sub,
    /// Subtracts `right` from this value, returning a [`ProgramError`] if something goes wrong.
    sub(right),
    SubOperation,
);

define_tracer_operator!(@binary std::ops::Sub, sub, SubOperation, "`sub` operation failed");

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

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
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

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
                &[Scalar::from(2.0f32), Scalar::from(3.5f64)],
            ),
            Ok(vec![Scalar::from(-1.5f64)])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0), Array::scalar(3.5)],
            ),
            Ok(vec![Array::scalar(-1.5)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, 2.0)), Scalar::from(Complex::new(0.5f64, -1.0))],
            ),
            Ok(vec![Scalar::from(Complex::new(0.5f64, 3.0))]),
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
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let right = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["y"])
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
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, SubOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
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

    #[test]
    fn test_sub_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&SubOperation, &[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError { message: format!("'{SUB_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean] {
            let expected =
                TypeError { message: format!("'{SUB_OPERATION_NAME}' does not support input data type {input_type}") };
            assert_eq!(
                Operation::<DataType>::infer_output_types(&SubOperation, &[input_type, input_type], &[]),
                Err(expected.clone()),
            );
            assert_eq!(
                Operation::<ArrayType>::infer_output_types(
                    &SubOperation,
                    &[ArrayType::scalar(input_type), ArrayType::scalar(input_type)],
                    &[],
                ),
                Err(expected),
            );
        }
        assert_eq!(
            <SubOperation as Operation<ArrayType>>::infer_output_types(
                &SubOperation,
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError { message: format!("'{SUB_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = || {
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let unreduced = || {
            plain()
                .with_sharding(plain().sharding().unwrap().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };

        let output =
            Operation::<ArrayType>::infer_output_types(&SubOperation, &[unreduced(), unreduced()], &[]).unwrap();
        assert_eq!(output[0].unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&SubOperation, &[unreduced(), plain()], &[]),
            Err(TypeError { message: "'sub' operands must be unreduced over the same axes".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&SubOperation, &[plain(), unreduced()], &[]),
            Err(TypeError { message: "'sub' operands must be unreduced over the same axes".to_string() }),
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = SubOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sub_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SubOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    (@replicated, Array::scalar(3.0)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-2.0, -5.0]))],
            }],
        );
    }

    #[test]
    fn test_sub_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SubOperation,
            cases = [{
                primals = [Array::scalar(5.0), Array::scalar(2.0)],
                tangents = [Array::scalar(3.0), Array::scalar(1.0)],
                primal_outputs = [Array::scalar(3.0)],
                tangent_outputs = [Array::scalar(2.0)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = sub %0 %1
                        %5:f64[] = sub %2 %3
                    in (%4, %5)
                "},
            }],
        );
        fn subtract_negated<V>(input: V) -> V
        where
            V: Clone + std::ops::Neg<Output = V> + std::ops::Sub<Output = V>,
        {
            input.clone() - -input
        }
        check_gradient!(@scalar, subtract_negated, at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = Complex::new(0.7f64, -0.3);
        assert_eq!(
            gradient_holomorphic(subtract_negated, Scalar::from(input)),
            Ok(Scalar::from(Complex::new(2.0, 0.0))),
        );

        // Second-order differentiation recovers d²(x - (-x))/dx² = 0.
        assert_abs_diff_eq!(
            gradient(|x| gradient(subtract_negated, x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            0.0,
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_sub_partial_evaluation() {
        check_operation_partial_evaluation!(operation = SubOperation, inputs = [2.0, 3.5], expected = -1.5,);
    }

    #[test]
    fn test_sub_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = SubOperation,
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0)],
                    input_cotangents = [Array::scalar(3.0), Array::scalar(-3.0)],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        let %1:f64[] = neg %0
                        in (%0, %1)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = vector_type.clone())),
                    ],
                    output_cotangents = [Array::from_f64s(vector_type.clone(), vec![2.0, 3.0, 4.0])],
                    input_cotangents = [
                        Array::scalar(9.0),
                        Array::from_f64s(vector_type, vec![-2.0, -3.0, -4.0]),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                            %2:f64[3] = neg %0
                        in (%1, %2)
                    "},
                },
            ],
        );
    }
}
