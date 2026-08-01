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

define_tracer_operator!(@binary std::ops::Sub, sub, capability = Sub, method = sub);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Dimension, Shape};

    use super::*;

    #[test]
    fn test_sub() {
        let operation = SubOperation;

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
    }

    #[test]
    fn test_sub_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = SubOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("'{SUB_OPERATION_NAME}' input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = || {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let unreduced = || {
            plain()
                .with_sharding(plain().sharding().unwrap().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };

        check_operation_type_inference!(
            operation = SubOperation,
            cases = [
                {
                    input_types = [unreduced(), unreduced()],
                    output_types = [unreduced()],
                },
                {
                    input_types = [unreduced(), plain()],
                    error = "'sub' operands must be unreduced over the same axes",
                },
                {
                    input_types = [plain(), unreduced()],
                    error = "'sub' operands must be unreduced over the same axes",
                },
            ],
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
    }

    #[test]
    fn test_sub_partial_evaluation() {
        check_operation_partial_evaluation!(operation = SubOperation, inputs = [2.0, 3.5], expected = -1.5,);
    }

    #[test]
    fn test_sub_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
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
