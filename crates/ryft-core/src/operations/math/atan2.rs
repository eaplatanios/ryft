use std::ops::{Add as StandardAdd, Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg};

use crate::contexts::Context;
use crate::differentiation::elementwise::ElementwiseDerivativeAlignment;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::macros::{check_count, define_elementwise_operation, impl_non_transposable_operation};
use crate::programs::operations::Operation;
use crate::programs::types::Typed;
use crate::programs::{MaybeZero, ProgramError, Value};

/// Canonical operation name for [`Atan2Operation`].
pub const ATAN2_OPERATION_NAME: &str = "atan2";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise two-argument arc tangent of its operands (i.e.,
    /// `(y, x) ↦ atan2(y, x)`, the angle of the point `(x, y)` in the correct quadrant for real operands), promoting
    /// their element types and broadcasting their shapes. For complex operands, the principal value is defined as
    /// `-i · log((x + i · y) / sqrt(x² + y²))`. Only floating-point and complex operands are supported, and array
    /// operands that still carry partial sums are rejected, with their reduced-axis markers required to agree.
    Atan2Operation, ATAN2_OPERATION_NAME,
    Atan2, atan2,
    validate_data_types = super::validate_floating_or_complex_input_types,
    validate_array_types = super::validate_binary_reduction_state,
);

impl<C: Context> DifferentiableOperation<C> for Atan2Operation
where
    C::Type: DifferentiableType,
    C::Value: Atan2
        + StandardNeg<Output = C::Value>
        + StandardAdd<Output = C::Value>
        + StandardMul<Output = C::Value>
        + StandardDiv<Output = C::Value>
        + ElementwiseDerivativeAlignment<C::Type>,
    Atan2Operation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // d(atan2(y, x)) = x / (x² + y²) · dy - y / (x² + y²) · dx. The shared denominator is computed once for both
        // terms, and each divided coefficient is formed independently, matching the primitive's numerical rule:
        // combining the terms into one numerator can produce `inf - inf` before division for large finite inputs even
        // when the two finite quotient terms cancel. This rule stays hand-written instead of delegating to
        // `binary_elementwise_jvp` because independent per-side term closures would recompute the shared denominator.
        check_count!("input", inputs, 2, ProgramError);
        let y = &inputs[0];
        let x = &inputs[1];
        let primal = y.primal().atan2(x.primal())?;
        let target = primal.r#type().tangent();
        let has_y_tangent = y.tangent().as_value().is_some();
        let has_x_tangent = x.tangent().as_value().is_some();
        if !has_y_tangent && !has_x_tangent {
            return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Zero(target))?]);
        }
        if target.is_zero_space() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'atan2' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        let x_primal = x.primal().align_tangent(&target)?;
        let y_primal = y.primal().align_tangent(&target)?;
        let denominator = x_primal.clone() * x_primal.clone() + y_primal.clone() * y_primal.clone();
        let y_term = y
            .tangent()
            .as_value()
            .map(|tangent| {
                Ok::<_, DifferentiationError>(
                    (x_primal.clone() / denominator.clone()) * tangent.align_tangent(&target)?,
                )
            })
            .transpose()?;
        let x_term = x
            .tangent()
            .as_value()
            .map(|tangent| {
                Ok::<_, DifferentiationError>(
                    -(y_primal.clone() / denominator.clone()) * tangent.align_tangent(&target)?,
                )
            })
            .transpose()?;
        let tangent = y_term
            .into_iter()
            .chain(x_term)
            .reduce(|y_term, x_term| y_term + x_term)
            .map_or_else(|| MaybeZero::Zero(target), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl_non_transposable_operation!(Atan2Operation);

/// Value-level elementwise two-argument arc-tangent capability, computing `atan2(self, x)`. [`Atan2`] fills the same
/// role for [`Atan2Operation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait Atan2: Sized {
    /// Computes the elementwise two-argument arc tangent `atan2(self, x)` (i.e., with this value as the `y`
    /// coordinate), promoting both operands to a common floating-point or complex element type and returning a
    /// [`ProgramError`] if something goes wrong.
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<Atan2Operation>>>> Atan2 for V {
    #[inline]
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(Atan2Operation, Vec::new(), &[self.clone(), x.clone()])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, jvp, value_and_gradient, value_and_gradient_holomorphic};
    use crate::interpretation::InterpretableOperation;
    use crate::operations::constants::OneLike;
    use crate::operations::manipulation::ConvertElementType;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::{TestArray, check_gradient};
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_atan2() {
        assert_eq!(Scalar::from(0.5f32).atan2(&Scalar::from(-0.25f32)).unwrap(), 0.5f32.atan2(-0.25f32));
        assert_eq!(Scalar::from(0.5f64).atan2(&Scalar::from(-0.25f64)).unwrap(), 0.5f64.atan2(-0.25f64));
        assert_eq!(
            Scalar::from(bf16::from_f32(0.5)).atan2(&Scalar::from(bf16::from_f32(-0.25))).unwrap(),
            bf16::from_f32(0.5f32.atan2(-0.25f32)),
        );
        assert_eq!(
            Scalar::from(f16::from_f32(0.5)).atan2(&Scalar::from(f16::from_f32(-0.25))).unwrap(),
            f16::from_f32(0.5f32.atan2(-0.25f32)),
        );
        let y = Complex::new(0.5f32, 0.25);
        let x = Complex::new(-0.75f32, 0.125);
        let imaginary_unit = Complex::new(0.0, 1.0);
        assert_abs_diff_eq!(
            Scalar::from(y).atan2(&Scalar::from(x)).unwrap(),
            Scalar::from(-imaginary_unit * ((x + imaginary_unit * y) / (x * x + y * y).sqrt()).ln()),
            epsilon = 1e-6,
        );
        let y = Complex::new(0.5f64, 0.0);
        let x = Complex::new(-0.75f64, 0.125);
        let imaginary_unit = Complex::new(0.0, 1.0);
        assert_abs_diff_eq!(
            Scalar::from(0.5f32).atan2(&Scalar::from(x)).unwrap(),
            Scalar::from(-imaginary_unit * ((x + imaginary_unit * y) / (x * x + y * y).sqrt()).ln()),
            epsilon = 1e-12,
        );

        let operation = Atan2Operation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), ATAN2_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "Atan2Operation");
        assert_eq!(format!("{operation}"), ATAN2_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F32], &[]),
            Ok(vec![DataType::F32]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.5), Scalar::from(-0.25)],
            ),
            Ok(vec![Scalar::from(0.5f64.atan2(-0.25f64))]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(0.5), TestArray::scalar(-0.25)],
            ),
            Ok(vec![TestArray::scalar(0.5f64.atan2(-0.25f64))]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its identical inputs.
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
            <Atan2Operation as Operation<ArrayType>>::infer_output_types(
                &operation,
                &[input.clone(), input.clone()],
                &[],
            ),
            Ok(vec![input]),
        );

        // Floating-point and complex operands promote to a common type.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::C64, DataType::C64], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::C128], &[]),
            Ok(vec![DataType::C128]),
        );
        let complex_type = ArrayType::new(DataType::C64, Shape::new(vec![Size::Static(2)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[complex_type.clone(), complex_type.clone()], &[]),
            Ok(vec![complex_type]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.5f32), Scalar::from(-0.25f64)],
            ),
            Ok(vec![Scalar::from(0.5f64.atan2(-0.25f64))]),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, Atan2Operation>::new();
        let y = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![y, x]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = atan2 %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_atan2_type_inference() {
        assert_eq!(
            Operation::<DataType>::infer_output_types(&Atan2Operation, &[DataType::I32, DataType::F32], &[]),
            Err(TypeError { message: "'atan2' does not support input data type i32".to_string() }),
        );
        crate::operations::math::tests::assert_rejects_unreduced(Atan2Operation, ATAN2_OPERATION_NAME, 2);
        crate::operations::math::tests::assert_rejects_mismatched_reduced(Atan2Operation, ATAN2_OPERATION_NAME);
    }

    #[test]
    fn test_atan2_batching() {
        crate::operations::math::tests::assert_binary_batching(
            Atan2Operation,
            &[0.5, -1.0],
            2.0,
            &[0.5f64.atan2(2.0), (-1.0f64).atan2(2.0)],
        );
    }

    #[test]
    fn test_atan2_differentiation() {
        let (y, x) = (0.7f64, -0.3f64);
        let (value, (y_gradient, x_gradient)) =
            value_and_gradient(|(y, x)| y.atan2(&x).unwrap(), (Scalar::from(y), Scalar::from(x))).unwrap();
        assert_abs_diff_eq!(value, y.atan2(x), epsilon = 1e-9);
        assert_abs_diff_eq!(y_gradient, x / (x * x + y * y), epsilon = 1e-9);
        assert_abs_diff_eq!(x_gradient, -y / (x * x + y * y), epsilon = 1e-9);
        check_gradient!(|y| y.atan2(&y.one_like()).unwrap(), 0.7, 1e-6, 1e-6);

        // Second-order differentiation recovers d²(atan2(y, 1))/dy² = -2y / (1 + y²)².
        assert_abs_diff_eq!(
            gradient(|y| gradient(|y| y.atan2(&y.one_like()).unwrap(), y).unwrap(), Scalar::from(0.7f64)).unwrap(),
            -2.0 * 0.7 / ((1.0 + 0.7f64 * 0.7) * (1.0 + 0.7f64 * 0.7)),
            epsilon = 1e-9,
        );

        let (_, tangent): (Scalar, Scalar) = jvp(
            |(y, x)| y.atan2(&x),
            (Scalar::from(1.0e308), Scalar::from(1.0e308)),
            (Scalar::from(1.0e308), Scalar::from(1.0e308)),
        )
        .unwrap();
        assert_eq!(tangent, Scalar::from(0.0));

        let y = Complex::new(0.7f64, -0.2);
        let x = Complex::new(-0.3f64, 0.4);
        let (value, (y_gradient, x_gradient)) =
            value_and_gradient_holomorphic(|(y, x)| y.atan2(&x).unwrap(), (Scalar::from(y), Scalar::from(x))).unwrap();
        let denominator = x * x + y * y;
        let imaginary_unit = Complex::new(0.0, 1.0);
        assert_abs_diff_eq!(
            value,
            Scalar::from(-imaginary_unit * ((x + imaginary_unit * y) / denominator.sqrt()).ln()),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(y_gradient, Scalar::from(x / denominator), epsilon = 1e-12);
        assert_abs_diff_eq!(x_gradient, Scalar::from(-y / denominator), epsilon = 1e-12);

        let y = Scalar::from(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let x = Scalar::from(4.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let (primal, tangent): (Scalar, Scalar) =
            jvp(|(y, x)| y.atan2(&x), (y, x), (Scalar::from(1.0f32), Scalar::from(1.0f32))).unwrap();
        assert!(matches!(primal, Scalar::F8E8M0FNU(_)));
        let Scalar::F32(tangent) = tangent else { panic!("expected an f32 tangent") };
        assert_abs_diff_eq!(tangent, 0.1f32, epsilon = 1e-6);

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let y = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(Atan2Operation, Vec::new(), vec![y, x]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = atan2 %0 %1
                    %5:f64[] = mul %1 %1
                    %6:f64[] = mul %0 %0
                    %7:f64[] = add %5 %6
                    %8:f64[] = div %1 %7
                    %9:f64[] = mul %8 %2
                    %10:f64[] = div %0 %7
                    %11:f64[] = neg %10
                    %12:f64[] = mul %11 %3
                    %13:f64[] = add %9 %12
                in (%4, %13)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_atan2_partial_evaluation() {
        crate::operations::math::tests::assert_partial_evaluation(Atan2Operation, &[0.5, -0.25], 0.5f64.atan2(-0.25));
    }

    #[test]
    fn test_atan2_transposition() {
        crate::operations::math::tests::assert_rejects_nonlinear_transposition(Atan2Operation, ATAN2_OPERATION_NAME, 2);
    }
}
