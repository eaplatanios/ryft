use std::ops::{Add as StandardAdd, Div as StandardDiv};

use crate::contexts::Context;
use crate::differentiation::elementwise::{ElementwiseDerivativeAlignment, unary_elementwise_jvp};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::macros::{define_elementwise_capability, define_elementwise_operation, impl_non_transposable_operation};
use crate::programs::operations::Operation;

/// Canonical operation name for [`SqrtOperation`].
pub const SQRT_OPERATION_NAME: &str = "sqrt";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise square root of one value (i.e., `x ↦ √x`, the
    /// principal branch `√z` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    SqrtOperation, SQRT_OPERATION_NAME,
    Sqrt, sqrt,
    validate = super::validate_floating_or_complex_input_types,
    validate_array = super::validate_no_unreduced_inputs,
);

impl<C: Context> DifferentiableOperation<C> for SqrtOperation
where
    C::Type: DifferentiableType,
    C::Value: Sqrt
        + StandardAdd<Output = C::Value>
        + StandardDiv<Output = C::Value>
        + ElementwiseDerivativeAlignment<C::Type>,
    SqrtOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(√x) = dx / (2 · √x), reusing the primal output as the denominator when no widening is required.
        unary_elementwise_jvp(
            self,
            inputs,
            |input| input.sqrt(),
            |operands| {
                let denominator = operands.output_primal_at_tangent_type()?;
                Ok(operands.input_tangent()? / (denominator.clone() + denominator))
            },
        )
    }
}

impl_non_transposable_operation!(SqrtOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise square-root capability. [`Sqrt`] fills the same role for
    /// [`SqrtOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Sqrt, sqrt, SqrtOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::{TypeError, Typed};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::{TestArray, check_gradient};
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};
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
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(4.0)],
            ),
            Ok(vec![TestArray::scalar(2.0)]),
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
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
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
        crate::operations::math::tests::assert_rejects_unreduced(SqrtOperation, SQRT_OPERATION_NAME, 1);
    }

    #[test]
    fn test_sqrt_batching() {
        crate::operations::math::tests::assert_unary_batching(
            SqrtOperation,
            &[0.5, 2.0],
            &[0.5f64.sqrt(), 2.0f64.sqrt()],
        );
    }

    #[test]
    fn test_sqrt_differentiation() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = context.jvp(|input| input.sqrt(), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_abs_diff_eq!(primal, 2.0f64.sqrt(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 3.0 / (2.0 * 2.0f64.sqrt()), epsilon = 1e-9);
        check_gradient!(|input| input.sqrt().unwrap(), 2.0, 1e-6, 1e-6);
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

        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primal = TestArray::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = context.jvp(|input| input.sqrt(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.values()[0], 3.0 / (2.0 * 2.0f64.sqrt()), epsilon = 1e-9);

        // The plain staged tangent program reuses the primal `sqrt` as the denominator instead of staging a
        // duplicate.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SqrtOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = sqrt %0
                    %3:f64[] = add %2 %2
                    %4:f64[] = div %1 %3
                in (%2, %4)
            "}
            .trim_end(),
        );

        // The widened staged tangent program recomputes the denominator in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SqrtOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        crate::operations::math::tests::assert_partial_evaluation(SqrtOperation, &[4.0], 2.0);
    }

    #[test]
    fn test_sqrt_transposition() {
        crate::operations::math::tests::assert_rejects_nonlinear_transposition(SqrtOperation, SQRT_OPERATION_NAME, 1);
    }
}
