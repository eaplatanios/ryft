use std::ops::Mul as StandardMul;

use crate::contexts::Context;
use crate::differentiation::elementwise::{ElementwiseDerivativeAlignment, unary_elementwise_jvp};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::macros::{define_elementwise_capability, define_elementwise_operation, impl_non_transposable_operation};
use crate::programs::operations::Operation;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ExpOperation`].
pub const EXP_OPERATION_NAME: &str = "exp";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise natural exponential of one value (i.e.,
    /// `x ↦ eˣ`, the analytic continuation `e^z` on complex operands) while preserving its array metadata. Only
    /// floating-point and complex operands are supported, and operands that still carry partial sums are rejected.
    ExpOperation, EXP_OPERATION_NAME,
    Exp, exp,
    check_data_types = [@floating_or_complex],
    check_array_types = [@no_unreduced],
);

impl<C: Context> DifferentiableOperation<C> for ExpOperation
where
    C::Type: DifferentiableType,
    C::Value: Exp + StandardMul<Output = C::Value> + ElementwiseDerivativeAlignment<C::Type>,
    ExpOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(eˣ) = eˣ · dx, reusing the primal output as the coefficient when no widening is required.
        unary_elementwise_jvp(
            self,
            inputs,
            |input| input.exp(),
            |operands| {
                let output_primal = operands.output_primal_at_tangent_type()?;
                Ok(output_primal * operands.input_tangent()?)
            },
        )
    }
}

impl_non_transposable_operation!(ExpOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise natural-exponential capability. [`Exp`] fills the same role for
    /// [`ExpOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Exp, exp, ExpOperation,
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
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::{TypeError, Typed};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_exp() {
        assert_eq!(Scalar::from(0.5f32).exp().unwrap(), 0.5f32.exp());
        assert_eq!(Scalar::from(0.5f64).exp().unwrap(), 0.5f64.exp());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).exp().unwrap(), bf16::from_f32(0.5f32.exp()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).exp().unwrap(), f16::from_f32(0.5f32.exp()));
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Scalar::from(input).exp().unwrap(), Scalar::from(input.exp()), epsilon = 1e-12);
        // Euler's identity: e^{iπ} = -1.
        assert_abs_diff_eq!(
            Scalar::from(ComplexNumber::new(0.0f64, std::f64::consts::PI)).exp().unwrap(),
            Scalar::from(ComplexNumber::new(-1.0f64, 0.0)),
            epsilon = 1e-12,
        );

        let operation = ExpOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), EXP_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ExpOperation");
        assert_eq!(format!("{operation}"), EXP_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.7)],
            ),
            Ok(vec![Scalar::from(0.7f64.exp())]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.7)],
            ),
            Ok(vec![Array::scalar(0.7f64.exp())]),
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

        let mut builder = ProgramBuilder::<Scalar, ExpOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = exp %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_exp_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&ExpOperation, &[DataType::C64], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&ExpOperation, &[DataType::I32], &[]),
            Err(TypeError { message: "'exp' does not support input data type i32".to_string() }),
        );
        crate::operations::math::tests::assert_rejects_unreduced(ExpOperation, EXP_OPERATION_NAME, 1);
    }

    #[test]
    fn test_exp_batching() {
        crate::operations::math::tests::assert_unary_batching(
            ExpOperation,
            &[0.5, -1.0],
            &[0.5f64.exp(), (-1.0f64).exp()],
        );
    }

    #[test]
    fn test_exp_differentiation() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = context.jvp(|input| input.exp(), Scalar::from(0.7), Scalar::from(3.0)).unwrap();
        assert_abs_diff_eq!(primal, 0.7f64.exp(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 3.0 * 0.7f64.exp(), epsilon = 1e-9);
        check_gradient!(@scalar, |input| input.exp(), at = 0.7, step = 1e-6, tolerance = 1e-6);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.exp().unwrap(), Scalar::from(input)),
            Ok(Scalar::from(input.exp())),
        );

        // Second-order differentiation recovers d²(eˣ)/dx² = eˣ.
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.exp().unwrap(), x).unwrap(), Scalar::from(0.7f64)).unwrap(),
            0.7f64.exp(),
            epsilon = 1e-9,
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (primal_output, tangent) = context.jvp(|input| input.exp(), primal, input_tangent).unwrap();
        // The primal output stays genuinely `f8e8m0fnu`-encoded (not an `f64` pun): `exp(2) ≈ 7.39` rounds to the
        // nearest representable power of two, `8 = 2^3`, whose biased-exponent encoding is `0x82`.
        assert_eq!(primal_output.r#type().as_ref(), &ArrayType::scalar(DataType::F8E8M0FNU));
        assert_eq!(
            primal_output.values(),
            &[Scalar::from_low_precision_float_bits(DataType::F8E8M0FNU, 0x82).unwrap()],
        );
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.values()[0], 3.0 * 2.0f64.exp(), epsilon = 1e-6);

        // The plain staged tangent program reuses the primal `exp` as the coefficient instead of staging a duplicate.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(ExpOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = exp %0
                    %3:f64[] = mul %2 %1
                in (%2, %3)
            "}
            .trim_end(),
        );

        // The widened staged tangent program recomputes the coefficient in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(ExpOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = exp %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = exp %3
                    %5:f32[] = mul %4 %1
                in (%2, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_exp_partial_evaluation() {
        crate::operations::math::tests::assert_partial_evaluation(ExpOperation, &[0.7], 0.7f64.exp());
    }

    #[test]
    fn test_exp_transposition() {
        crate::operations::math::tests::assert_rejects_nonlinear_transposition(ExpOperation, EXP_OPERATION_NAME, 1);
    }
}
