use std::ops::{Mul as StandardMul, Neg as StandardNeg};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};

use super::Sin;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`CosOperation`].
pub const COS_OPERATION_NAME: &str = "cos";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise cosine of a floating-point or complex value while
    /// preserving its array metadata. Array operands that still carry partial sums are rejected.
    CosOperation, COS_OPERATION_NAME,
    Cos, cos,
    check_data_types = [@floating_or_complex],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    CosOperation,
    jvp<C> where C::Value: Sin + StandardNeg<Output = C::Value> + StandardMul<Output = C::Value> {
        |(input, input_tangent)| -(input.sin()? * input_tangent)
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise cosine capability. [`Cos`] fills the same role for [`CosOperation`] that
    /// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic
    /// [`Operation`]s.
    Cos,
    /// Computes [`CosOperation`] elementwise for this value.
    cos,
    CosOperation,
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
    fn test_cos() {
        assert_eq!(Scalar::from(0.5f32).cos().unwrap(), 0.5f32.cos());
        assert_eq!(Scalar::from(0.5f64).cos().unwrap(), 0.5f64.cos());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).cos().unwrap(), bf16::from_f32(0.5f32.cos()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).cos().unwrap(), f16::from_f32(0.5f32.cos()));
        let Scalar::C128(extreme) = Scalar::from(ComplexNumber::new(0.0f64, 1000.0)).cos().unwrap() else {
            panic!("expected a c128 result")
        };
        assert!(extreme.re.is_infinite() && extreme.re.is_sign_positive());
        assert_eq!(extreme.im, 0.0);

        let operation = CosOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), COS_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "CosOperation");
        assert_eq!(format!("{operation}"), COS_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Ok(vec![DataType::F32]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.5)],
            ),
            Ok(vec![Scalar::from(0.5f64.cos())]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.5)],
            ),
            Ok(vec![Array::scalar(0.5f64.cos())]),
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
            <CosOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
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
        let mut builder = ProgramBuilder::<Scalar, CosOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
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

    #[test]
    fn test_cos_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&CosOperation, &[DataType::I32], &[]),
            Err(TypeError { message: "'cos' does not support input data type i32".to_string() }),
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = CosOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_cos_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = CosOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.cos(), (-1.0f64).cos()]))],
            }],
        );
    }

    #[test]
    fn test_cos_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = CosOperation,
            cases = [{
                primals = [Array::scalar(2.0)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(2.0f64.cos())],
                tangent_outputs = [Array::scalar(-3.0 * 2.0f64.sin())],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = cos %0
                        %3:f64[] = sin %0
                        %4:f64[] = mul %3 %1
                        %5:f64[] = neg %4
                    in (%2, %5)
                "},
            }],
        );
        check_gradient!(@scalar, |input| input.cos(), at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.cos().unwrap(), Scalar::from(input)),
            Ok(Scalar::from(-input.sin())),
        );

        // Second-order differentiation recovers d²(cos(x))/dx² = -cos(x).
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.cos().unwrap(), x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            -0.7f64.cos(),
            epsilon = 1e-9,
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![4.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = context.jvp(|input| input.cos(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.values()[0], -3.0 * 4.0f64.sin(), epsilon = 1e-6);

        // The widened staged tangent program computes the coefficient in the widened differential representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(CosOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = cos %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = sin %3
                    %5:f32[] = mul %4 %1
                    %6:f32[] = neg %5
                in (%2, %6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_cos_partial_evaluation() {
        check_operation_partial_evaluation!(operation = CosOperation, inputs = [0.5], expected = 0.5f64.cos(),);
    }

    #[test]
    fn test_cos_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = CosOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
