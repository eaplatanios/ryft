use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AddOperation`].
pub const ADD_OPERATION_NAME: &str = "add";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that adds two numeric values elementwise, promoting their element [`DataType`](crate::DataType)s
    /// and broadcasting their [`Shape`](crate::Shape)s. Array operands that carry partial sums must both be unreduced
    /// over exactly the same mesh axes. Mixing an unreduced operand with an already reduced operand would duplicate the
    /// reduced contribution when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    AddOperation,
    ADD_OPERATION_NAME,
    Add,
    add,
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @linear
    AddOperation,
    rule = [@positive, @positive],
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise addition capability. [`Add`] is the fallible Ryft counterpart to [`std::ops::Add`]
    /// that [`AddOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
    /// panicking. Value types additionally provide [`std::ops::Add`] as ergonomic (albeit panicking) sugar layered
    /// on top of this capability.
    Add,
    /// Adds `rhs` to this value.
    add(rhs),
    AddOperation,
);

define_tracer_operator!(@binary std::ops::Add, add, capability = Add, method = add);

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
    fn test_add() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &AddOperation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0f32), Scalar::from(3.5f64)],
            ),
            Ok(vec![Scalar::from(5.5f64)])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &AddOperation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0), Array::vector(vec![3.5, -1.0])],
            ),
            Ok(vec![Array::vector(vec![5.5, 1.0])]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &AddOperation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(Complex::new(1.0f64, 2.0)), Scalar::from(Complex::new(0.5f64, -1.0))],
            ),
            Ok(vec![Scalar::from(Complex::new(1.5f64, 1.0))]),
        );
    }

    #[test]
    fn test_add_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = AddOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("'{ADD_OPERATION_NAME}' input types are not broadcast-compatible"),
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
            operation = AddOperation,
            cases = [
                {
                    input_types = [unreduced(), unreduced()],
                    output_types = [unreduced()],
                },
                {
                    input_types = [unreduced(), plain()],
                    error = "'add' operands must be unreduced over the same axes",
                },
                {
                    input_types = [plain(), unreduced()],
                    error = "'add' operands must be unreduced over the same axes",
                },
            ],
        );

        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = AddOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_add_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AddOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    (@replicated, Array::scalar(3.0)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![4.0, 1.0]))],
            }],
        );
    }

    #[test]
    fn test_add_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AddOperation,
            cases = [{
                primals = [Array::scalar(2.0), Array::scalar(5.0)],
                tangents = [Array::scalar(3.0), Array::scalar(-1.0)],
                primal_outputs = [Array::scalar(7.0)],
                tangent_outputs = [Array::scalar(2.0)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = add %0 %1
                        %5:f64[] = add %2 %3
                    in (%4, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_add_partial_evaluation() {
        check_operation_partial_evaluation!(operation = AddOperation, inputs = [2.0, 3.5], expected = 5.5,);
    }

    #[test]
    fn test_add_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = AddOperation,
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0)],
                    input_cotangents = [Array::scalar(3.0), Array::scalar(3.0)],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        in (%0, %0)
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
                        Array::from_f64s(vector_type, vec![2.0, 3.0, 4.0]),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                        in (%1, %0)
                    "},
                },
            ],
        );
    }
}
