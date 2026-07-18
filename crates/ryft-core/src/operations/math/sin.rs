use std::ops::Mul as StandardMul;

use crate::contexts::Context;
use crate::differentiation::elementwise::{ElementwiseDerivativeAlignment, unary_elementwise_jvp};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::macros::{define_elementwise_capability, define_elementwise_operation, impl_non_transposable_operation};
use crate::programs::operations::Operation;

use super::Cos;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SinOperation`].
pub const SIN_OPERATION_NAME: &str = "sin";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise sine of a floating-point or complex value while
    /// preserving its array metadata. Array operands that still carry partial sums are rejected.
    SinOperation, SIN_OPERATION_NAME,
    Sin, sin,
    check_data_types = [@floating_or_complex],
    check_array_types = [@no_unreduced],
);

impl<C: Context> DifferentiableOperation<C> for SinOperation
where
    C::Type: DifferentiableType,
    C::Value: Sin + Cos + StandardMul<Output = C::Value> + ElementwiseDerivativeAlignment<C::Type>,
    SinOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(sin(x)) = cos(x) · dx.
        unary_elementwise_jvp(
            self,
            inputs,
            |input| input.sin(),
            |operands| Ok(operands.input_primal()?.cos()? * operands.input_tangent()?),
        )
    }
}

impl_non_transposable_operation!(SinOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise sine capability. [`Sin`] fills the same role for [`SinOperation`] that
    /// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic
    /// [`Operation`]s.
    Sin, sin, SinOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_gradient;
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
    fn test_sin() {
        assert_eq!(Scalar::from(0.5f32).sin().unwrap(), 0.5f32.sin());
        assert_eq!(Scalar::from(0.5f64).sin().unwrap(), 0.5f64.sin());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).sin().unwrap(), bf16::from_f32(0.5f32.sin()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).sin().unwrap(), f16::from_f32(0.5f32.sin()));
        let Scalar::C128(extreme) = Scalar::from(ComplexNumber::new(0.0f64, 1000.0)).sin().unwrap() else {
            panic!("expected a c128 result")
        };
        assert_eq!(extreme.re, 0.0);
        assert!(extreme.im.is_infinite() && extreme.im.is_sign_positive());

        let operation = SinOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SIN_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "SinOperation");
        assert_eq!(format!("{operation}"), SIN_OPERATION_NAME);
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
            Ok(vec![Scalar::from(0.5f64.sin())]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.5)]
            ),
            Ok(vec![Array::scalar(0.5f64.sin())]),
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
            <SinOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input), &[]),
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
        let mut builder = ProgramBuilder::<Scalar, SinOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
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

    #[test]
    fn test_sin_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&SinOperation, &[DataType::I32], &[]),
            Err(TypeError { message: "'sin' does not support input data type i32".to_string() }),
        );
        crate::operations::math::tests::assert_rejects_unreduced(SinOperation, SIN_OPERATION_NAME, 1);
    }

    #[test]
    fn test_sin_batching() {
        crate::operations::math::tests::assert_unary_batching(
            SinOperation,
            &[0.5, -1.0],
            &[0.5f64.sin(), (-1.0f64).sin()],
        );
    }

    #[test]
    fn test_sin_differentiation() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = context.jvp(|input| input.sin(), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_abs_diff_eq!(primal, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 3.0 * 2.0f64.cos(), epsilon = 1e-9);
        check_gradient!(@scalar, |input| input.sin(), at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.sin().unwrap(), Scalar::from(input)),
            Ok(Scalar::from(input.cos())),
        );

        // Second-order differentiation recovers d²(sin(x))/dx² = -sin(x).
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.sin().unwrap(), x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            -0.7f64.sin(),
            epsilon = 1e-9,
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = context.jvp(|input| input.sin(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.values()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-6);

        // The plain staged tangent program computes the coefficient directly on the input.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = sin %0
                    %3:f64[] = cos %0
                    %4:f64[] = mul %3 %1
                in (%2, %4)
            "}
            .trim_end(),
        );

        // The widened staged tangent program computes the coefficient in the widened differential representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = sin %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = cos %3
                    %5:f32[] = mul %4 %1
                in (%2, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sin_partial_evaluation() {
        crate::operations::math::tests::assert_partial_evaluation(SinOperation, &[0.5], 0.5f64.sin());
    }

    #[test]
    fn test_sin_transposition() {
        crate::operations::math::tests::assert_rejects_nonlinear_transposition(SinOperation, SIN_OPERATION_NAME, 1);
    }
}
