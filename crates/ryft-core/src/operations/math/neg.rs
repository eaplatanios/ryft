use crate::macros::{
    check_types, define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};
use crate::programs::types::TypeError;
use crate::types::DataType;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NegOperation`].
pub const NEG_OPERATION_NAME: &str = "neg";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that negates one integer, floating-point, or complex value while preserving its array metadata
    /// and reduction state. Boolean, token, structural-zero, and the unsigned-only `f8e8m0fnu` data types are rejected.
    NegOperation, NEG_OPERATION_NAME,
    Neg, neg,
    infer_data_types = |input_types: &[DataType]| {
        check_types!(@numeric, NEG_OPERATION_NAME, input_types);
        let input_type = input_types[0];
        if input_type == DataType::F8E8M0FNU {
            return Err(TypeError::invalid("'neg' does not support input data type f8e8m0fnu".to_string()));
        }
        Ok(vec![input_type])
    },
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
    use crate::types::{ArrayType, Shape, Size};

    use super::*;

    #[test]
    fn test_neg() {
        let operation = NegOperation;

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
    }

    #[test]
    fn test_neg_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = NegOperation,
            cases = [{
                input_data_types = [DataType::F64],
                output_data_types = [DataType::F64],
            }],
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::F8E8M0FNU] {
            let message = format!("'{NEG_OPERATION_NAME}' does not support input data type {input_type}");
            check_operation_type_inference!(
                @elementwise @unary,
                operation = NegOperation,
                cases = [{
                    input_data_types = [input_type],
                    error = message,
                }],
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
        check_operation_type_inference!(
            operation = NegOperation,
            cases = [{
                input_types = [unreduced.clone()],
                output_types = [unreduced],
            }],
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = NegOperation,
            cases = [{
                input_types = [reduced.clone()],
                output_types = [reduced],
            }],
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
