use std::ops::{Add as StandardAdd, Div as StandardDiv};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SqrtOperation`].
pub const SQRT_OPERATION_NAME: &str = "sqrt";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise square root of one value (i.e., `x ↦ √x`, the
    /// principal branch `√z` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    SqrtOperation, SQRT_OPERATION_NAME,
    Sqrt, sqrt,
    check_data_types = [@floating_or_complex],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    SqrtOperation,
    jvp<C> where C::Value: StandardAdd<Output = C::Value> + StandardDiv<Output = C::Value> {
        |(_, input_tangent) -> output| input_tangent / (output.clone() + output)
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise square-root capability. [`Sqrt`] fills the same role for
    /// [`SqrtOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Sqrt,
    /// Computes [`SqrtOperation`] elementwise for this value.
    sqrt,
    SqrtOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
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
    use crate::programs::types::{TypeError, Typed};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_sqrt() {
        assert_eq!(Scalar::from(0.25f32).sqrt().unwrap(), 0.5f32);
        assert_eq!(Scalar::from(0.25f64).sqrt().unwrap(), 0.5f64);
        assert_eq!(Scalar::from(bf16::from_f32(0.25)).sqrt().unwrap(), bf16::from_f32(0.5));
        assert_eq!(Scalar::from(f16::from_f32(0.25)).sqrt().unwrap(), f16::from_f32(0.5));
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Scalar::from(input).sqrt().unwrap(), Scalar::from(input.sqrt()), epsilon = 1e-12);
        // The principal branch maps the negative real axis to the positive imaginary axis.
        assert_abs_diff_eq!(
            Scalar::from(ComplexNumber::new(-4.0f64, 0.0)).sqrt().unwrap(),
            Scalar::from(ComplexNumber::new(0.0f64, 2.0)),
            epsilon = 1e-12,
        );

        let operation = SqrtOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SQRT_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "SqrtOperation");
        assert_eq!(format!("{operation}"), SQRT_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(4.0)],
            ),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(4.0)],
            ),
            Ok(vec![Array::scalar(2.0)]),
        );

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
            Operation::<ArrayType>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
            Ok(vec![input]),
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

        let mut builder = ProgramBuilder::<Scalar, SqrtOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sqrt %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sqrt_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&SqrtOperation, &[DataType::C64], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&SqrtOperation, &[DataType::I32], &[]),
            Err(TypeError { message: "'sqrt' does not support input data type i32".to_string() }),
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SqrtOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sqrt_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SqrtOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.sqrt(), 2.0f64.sqrt()]))],
            }],
        );
    }

    #[test]
    fn test_sqrt_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SqrtOperation,
            cases = [{
                primals = [Array::scalar(2.0)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(2.0f64.sqrt())],
                tangent_outputs = [Array::scalar(3.0 / (2.0 * 2.0f64.sqrt()))],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = sqrt %0
                        %3:f64[] = add %2 %2
                        %4:f64[] = div %1 %3
                    in (%2, %4)
                "},
            }],
        );
        check_gradient!(@scalar, |input| input.sqrt(), at = 2.0, step = 1e-6, tolerance = 1e-6);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.sqrt().unwrap(), Scalar::from(input)),
            Ok(Scalar::from(ComplexNumber::new(1.0, 0.0) / (input.sqrt() + input.sqrt()))),
        );

        // Second-order differentiation recovers d²(√x)/dx² = -1/(4 · x^(3/2)).
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.sqrt().unwrap(), x).unwrap(), Scalar::from(2.0f64)).unwrap(),
            -0.25 * 2.0f64.powf(-1.5),
            epsilon = 1e-9,
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = context.jvp(|input| input.sqrt(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.values()[0], 3.0 / (2.0 * 2.0f64.sqrt()), epsilon = 1e-6);

        // The widened staged tangent program recomputes the denominator in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SqrtOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = sqrt %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = sqrt %3
                    %5:f32[] = add %4 %4
                    %6:f32[] = div %1 %5
                in (%2, %6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sqrt_partial_evaluation() {
        check_operation_partial_evaluation!(operation = SqrtOperation, inputs = [4.0], expected = 2.0,);
    }

    #[test]
    fn test_sqrt_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = SqrtOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
