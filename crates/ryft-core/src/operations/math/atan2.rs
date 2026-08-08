use std::ops::{Add as StandardAdd, Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg};

use crate::differentiation::{
    DifferentiableType, DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment,
};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_differentiable_operation,
};
use crate::programs::{MaybeZero, ProgramError, Type, Typed};

// TODO(eaplatanios): Review this module.

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
    check_data_types = [@float],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_operation! {
    <T> Atan2Operation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Atan2
            + StandardNeg<Output = C::Value>
            + StandardAdd<Output = C::Value>
            + StandardMul<Output = C::Value>
            + StandardDiv<Output = C::Value>
            + ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, _context, _driver, inputs| {
            // d(atan2(y, x)) = x / (x² + y²) · dy - y / (x² + y²) · dx. The shared denominator is computed once for
            // both terms, and each divided coefficient is formed independently, matching the primitive's numerical
            // rule: combining the terms into one numerator can produce `inf - inf` before division for large finite
            // inputs even when the two finite quotient terms cancel. The custom form also computes the shared
            // denominator only once; independent per-side term expressions would recompute it.
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
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise two-argument arc-tangent capability, computing `atan2(self, x)`. [`Atan2`] fills the
    /// same role for [`Atan2Operation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Atan2,
    /// Computes the elementwise two-argument arc tangent `atan2(self, x)` (i.e., with this value as the `y`
    /// coordinate), promoting both operands to a common floating-point or complex element type and returning a
    /// [`ProgramError`] if something goes wrong.
    atan2(x),
    Atan2Operation,
);

/// Implements [`Atan2`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Atan2 for $type {
            fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
                Ok(<$type>::atan2(*self, *x))
            }
        }
    };
}

impl_capability_for_primitive!(f32);
impl_capability_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::differentiation::{jvp, value_and_gradient_holomorphic};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::manipulation::conversion::ConvertElementType;

    use super::*;

    #[test]
    fn test_atan2() {
        assert_eq!(
            Array::scalar(0.5f32).atan2(&Array::scalar(-0.25f32)).unwrap(),
            Array::scalar(0.5f32.atan2(-0.25f32)),
        );
        assert_eq!(
            Array::scalar(0.5f64).atan2(&Array::scalar(-0.25f64)).unwrap(),
            Array::scalar(0.5f64.atan2(-0.25f64)),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).atan2(&Array::scalar(bf16::from_f32(-0.25))).unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.atan2(-0.25f32))),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).atan2(&Array::scalar(f16::from_f32(-0.25))).unwrap(),
            Array::scalar(f16::from_f32(0.5f32.atan2(-0.25f32))),
        );
        let y = Complex::new(0.5f32, 0.25);
        let x = Complex::new(-0.75f32, 0.125);
        let imaginary_unit = Complex::new(0.0, 1.0);
        assert_abs_diff_eq!(
            Array::scalar(y).atan2(&Array::scalar(x)).unwrap(),
            Array::scalar(-imaginary_unit * ((x + imaginary_unit * y) / (x * x + y * y).sqrt()).ln()),
            epsilon = 1e-6,
        );
        let y = Complex::new(0.5f64, 0.0);
        let x = Complex::new(-0.75f64, 0.125);
        let imaginary_unit = Complex::new(0.0, 1.0);
        assert_abs_diff_eq!(
            Array::scalar(0.5f32).atan2(&Array::scalar(x)).unwrap(),
            Array::scalar(-imaginary_unit * ((x + imaginary_unit * y) / (x * x + y * y).sqrt()).ln()),
            epsilon = 1e-12,
        );

        assert_eq!(Array::scalar(0.5).atan2(&Array::scalar(-0.25)).unwrap(), Array::scalar(0.5f64.atan2(-0.25f64)),);
    }

    #[test]
    fn test_atan2_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = Atan2Operation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::F32, DataType::C128],
                    output_data_types = [DataType::C128],
                },
                {
                    input_data_types = [DataType::I32, DataType::F32],
                    error = "'atan2' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = Atan2Operation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = Atan2Operation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_atan2_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = Atan2Operation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![0.5, -1.0])),
                    (@replicated, Array::scalar(2.0)),
                ],
                outputs = [(@mapped(
                    axis = 0
                ), Array::vector(vec![0.5f64.atan2(2.0), (-1.0f64).atan2(2.0)]))],
            }],
        );
    }

    #[test]
    fn test_atan2_differentiation() {
        let (y, x) = (0.7f64, -0.3f64);
        let (y_tangent, x_tangent) = (0.4f64, -0.2f64);
        let tangent = (x * y_tangent - y * x_tangent) / (x * x + y * y);
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = Atan2Operation::new(),
            cases = [{
                primals = [Array::scalar(y), Array::scalar(x)],
                tangents = [Array::scalar(y_tangent), Array::scalar(x_tangent)],
                primal_outputs = [Array::scalar(y.atan2(x))],
                tangent_outputs = [Array::scalar(tangent)],
                jvp = indoc! {"
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
                "},
            }],
        );
    }

    #[test]
    fn test_atan2_differentiation_avoids_overflow() {
        let (_, tangent): (Array, Array) = jvp(
            |(y, x)| y.atan2(&x),
            (Array::scalar(1.0e308), Array::scalar(1.0e308)),
            (Array::scalar(1.0e308), Array::scalar(1.0e308)),
        )
        .unwrap();
        assert_eq!(tangent, Array::scalar(0.0));
    }

    #[test]
    fn test_atan2_complex_differentiation() {
        let y = Complex::new(0.7f64, -0.2);
        let x = Complex::new(-0.3f64, 0.4);
        let (value, (y_gradient, x_gradient)) =
            value_and_gradient_holomorphic(|(y, x)| y.atan2(&x).unwrap(), (Array::scalar(y), Array::scalar(x)))
                .unwrap();
        let denominator = x * x + y * y;
        let imaginary_unit = Complex::new(0.0, 1.0);
        assert_abs_diff_eq!(
            value,
            Array::scalar(-imaginary_unit * ((x + imaginary_unit * y) / denominator.sqrt()).ln()),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(y_gradient, Array::scalar(x / denominator), epsilon = 1e-12);
        assert_abs_diff_eq!(x_gradient, Array::scalar(-y / denominator), epsilon = 1e-12);
    }

    #[test]
    fn test_atan2_low_precision_differentiation_uses_widened_tangents() {
        let y = Array::scalar(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let x = Array::scalar(4.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let (primal, tangent): (Array, Array) =
            jvp(|(y, x)| y.atan2(&x), (y, x), (Array::scalar(1.0f32), Array::scalar(1.0f32))).unwrap();
        assert_eq!(primal.r#type().data_type(), DataType::F8E8M0FNU);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 0.1f32 as f64, epsilon = 1e-6);
    }

    #[test]
    fn test_atan2_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = Atan2Operation::new(),
            inputs = [Array::scalar(0.5), Array::scalar(-0.25)],
            expected = Array::scalar(0.5f64.atan2(-0.25)),
        );
    }

    #[test]
    fn test_atan2_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = Atan2Operation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_atan2_for_primitives() {
        assert_eq!(Atan2::atan2(&1.0_f64, &1.0), Ok(std::f64::consts::FRAC_PI_4));
    }
}
