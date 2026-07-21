use crate::differentiation::elementwise::ElementwiseDerivativeAlignment;
use crate::differentiation::forward::DifferentiationDual;
use crate::differentiation::types::DifferentiableType;
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::complex::{Complex, Conjugate, Imaginary, Real};
use crate::operations::constants::{OneLike, ZeroLike};
use crate::operations::control_flow::Select;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::types::{Type, TypeError, Typed};
use crate::types::DataType;

/// Canonical operation name for [`AbsOperation`].
pub const ABS_OPERATION_NAME: &str = "abs";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise absolute value of a value (i.e., `x ↦ |x|` and the magnitude `|z|`
    /// for complex operands with a real result) while preserving all other type metadata. Inputs that still represent
    /// partial sums over unreduced mesh axes are rejected because taking an absolute value does not preserve
    /// partial-sum semantics. Matching the operand constraints of
    /// [StableHLO's `abs`](https://openxla.org/stablehlo/spec#abs), signed-integer (including the sub-byte
    /// [`DataType::I2`] and [`DataType::I4`] types, with the minimum value wrapping to itself), floating-point,
    /// and complex inputs are supported, while unsigned-integer, Boolean, token, structural-zero, and single-bit
    /// [`DataType::I1`] inputs (whose only negative value `-1` has no representable absolute value) are rejected.
    AbsOperation,
    ABS_OPERATION_NAME,
    Abs,
    abs,
    infer_data_types = |input_types: &[DataType]| {
        let input_type = input_types[0];
        let output_type = if input_type == DataType::C64 {
            DataType::F32
        } else if input_type == DataType::C128 {
            DataType::F64
        } else if (input_type.is_signed() && input_type != DataType::I1) || input_type.is_floating_point() {
            input_type
        } else {
            return Err(TypeError {
                message: format!("cannot compute the absolute value of a value of data type {input_type}"),
            });
        };
        Ok(vec![output_type])
    },
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @custom
    AbsOperation,
    jvp<C>
    where
        C::Type: DifferentiableType,
        C::Value: Abs
            + Compare<Output = C::Value>
            + Complex
            + Conjugate
            + Imaginary
            + Real
            + Select
            + ZeroLike
            + OneLike
            + std::ops::Neg<Output = C::Value>
            + std::ops::Mul<Output = C::Value>
            + std::ops::Div<Output = C::Value>
            + ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, _context, _driver, inputs| {
            // Away from zero, the real derivative is `d|x| = sign(x) · dx`, while the complex magnitude is a ℂ → ℝ map
            // with `d|z| = Re(z̄ · dz) / |z|`. At the real origin, choose the right derivative and return `dx`. At the
            // complex origin, replace the zero denominator with one so the zero numerator yields zero. These
            // conventions keep the rule finite and stable under higher-order transforms. A structural zero tangent
            // stays symbolic, retyped to the real output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().abs()?;
            let primal_tangent_type = primal.r#type().tangent();
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal_tangent_type),
                MaybeZero::Value(_) if primal_tangent_type.is_zero_space() => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!("'abs' output type {} has no tangent space", primal.r#type()),
                    }
                    .into());
                }
                MaybeZero::Value(tangent) => {
                    if input.primal().r#type().is_complex() {
                        let denominator = primal.align_tangent(&primal_tangent_type)?;
                        let zero = denominator.zero_like();
                        let one = denominator.one_like();
                        let denominator_is_zero = denominator.compare(&zero, ComparisonDirection::Equal)?;
                        let denominator = C::Value::select(&denominator_is_zero, &one, &denominator)?;
                        // Normalize `conj(z) / |z|` before multiplying by `dz`. Computing `conj(z) * dz` first is
                        // algebraically equivalent but can overflow even when the final directional derivative is
                        // finite.
                        let conjugate = input.primal().conjugate()?;
                        let real = conjugate.real()? / denominator.clone();
                        let imaginary = conjugate.imaginary()? / denominator.clone();
                        let coefficient = real.complex(&imaginary)?;
                        let input_tangent_type = input.primal().r#type().tangent();
                        let tangent = tangent.align_tangent(&input_tangent_type)?;
                        MaybeZero::Value((tangent * coefficient).real()?.align_tangent(&primal_tangent_type)?)
                    } else {
                        let input = input.primal().align_tangent(&primal_tangent_type)?;
                        let tangent = tangent.align_tangent(&primal_tangent_type)?;
                        let zero = input.zero_like();
                        let non_negative = input.compare(&zero, ComparisonDirection::GreaterThanOrEqual)?;
                        MaybeZero::Value(C::Value::select(&non_negative, &tangent, &-tangent.clone())?)
                    }
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

// TODO(eaplatanios): Review from here onwards.

define_elementwise_capability!(
    @unary
    /// Value-level elementwise absolute-value capability. [`Abs`] fills the same role for [`AbsOperation`] that
    /// [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Abs,
    /// Computes the elementwise absolute value of this value (i.e., the magnitude for complex values, with a real
    /// result), returning a [`ProgramError`] if something goes wrong (e.g., when the value's data type carries no
    /// absolute value, such as a Boolean).
    abs,
    AbsOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::{gradient, jvp, value_and_gradient};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::math::{Reduce, ReductionKind};
    use crate::programs::regions::EmptyRegionDriver;
    use crate::types::ArrayType;

    use super::*;

    #[test]
    fn test_abs() {
        let operation = AbsOperation;

        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(-2.0)],
            ),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(ComplexNumber::new(3.0f64, -4.0f64))],
            ),
            Ok(vec![Scalar::from(5.0)]),
        );
    }

    #[test]
    fn test_abs_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = AbsOperation,
            cases = [
                {
                    input_data_types = [DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::I2],
                    output_data_types = [DataType::I2],
                },
                {
                    input_data_types = [DataType::I4],
                    output_data_types = [DataType::I4],
                },
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::F32],
                },
                {
                    input_data_types = [DataType::C128],
                    output_data_types = [DataType::F64],
                },
            ],
        );

        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::I1, DataType::U32] {
            let message = format!("cannot compute the absolute value of a value of data type {input_type}");
            check_operation_type_inference!(
                @elementwise @unary,
                operation = AbsOperation,
                cases = [{
                    input_data_types = [input_type],
                    error = message,
                }],
            );
        }

        check_operation_type_inference!(
            @reject @unreduced,
            operation = AbsOperation,
            input_types = [ArrayType::scalar(DataType::F32)],
        );
    }

    #[test]
    fn test_abs_partial_evaluation() {
        check_operation_partial_evaluation!(operation = AbsOperation, inputs = [-2.0], expected = 2.0,);
    }

    #[test]
    fn test_abs_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AbsOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
            }],
        );
    }

    #[test]
    fn test_abs_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AbsOperation,
            cases = [
                {
                    primals = [Array::scalar(0.7)],
                    tangents = [Array::scalar(3.0)],
                    primal_outputs = [Array::scalar(0.7)],
                    tangent_outputs = [Array::scalar(3.0)],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = abs %0
                            %3:f64[] = zero_like %0
                            %4:bool[] = compare [direction=GreaterThanOrEqual] %0 %3
                            %5:f64[] = neg %1
                            %6:f64[] = select %4 %1 %5
                        in (%2, %6)
                    "},
                },
                {
                    primals = [Array::scalar(-2.5)],
                    tangents = [Array::scalar(2.0)],
                    primal_outputs = [Array::scalar(2.5)],
                    tangent_outputs = [Array::scalar(-2.0)],
                },
            ],
        );
    }

    #[test]
    fn test_abs_differentiation_at_zero() {
        // The real rule chooses the right derivative at zero and remains constant under another derivative.
        assert_abs_diff_eq!(gradient(|x| x.abs().unwrap(), Scalar::from(0.0f64)).unwrap(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(
            gradient(|x| gradient(|x| x.abs().unwrap(), x).unwrap(), Scalar::from(0.0f64)).unwrap(),
            0.0,
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_abs_complex_differentiation() {
        // |z| is a ℂ → ℝ function and so it flows through the plain gradient entry point. With
        // d|z| = Re(z̄ · dz) / |z|, the bilinear-pairing gradient is z̄ / |z| (the unit-magnitude conjugate direction):
        // the reverse-mode counterpart of ∇|z|² = 2z̄ after the chain rule through the square root.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) = value_and_gradient(|z| z.abs().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z.norm()));
        let expected = z.conj() / z.norm();
        let Scalar::C128(actual) = gradient_value else { panic!("expected a c128 gradient") };
        assert!((actual - expected).norm() < 1e-12, "expected {expected} but got {actual}");

        // The array universe agrees: summing the elementwise magnitudes of a complex vector is again ℂⁿ → ℝ, and the
        // finite-difference oracle perturbs each element's real and imaginary parts independently.
        check_gradient!(
            @array,
            |z| z.abs().map(|magnitudes| magnitudes.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![ComplexNumber::new(0.7f64, -0.3), ComplexNumber::new(-1.2f64, 0.8)]),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // The complex rule replaces a zero magnitude denominator with one, so the zero numerator produces a finite
        // zero tangent and gradient at the origin.
        assert_eq!(
            jvp(
                |z| z.abs(),
                Scalar::from(ComplexNumber::new(0.0f64, 0.0f64)),
                Scalar::from(ComplexNumber::new(1.0f64, 2.0f64)),
            ),
            Ok((Scalar::from(0.0f64), Scalar::from(0.0f64))),
        );
        assert_eq!(
            gradient(|z| z.abs().unwrap(), Scalar::from(ComplexNumber::new(0.0f64, 0.0f64))),
            Ok(Scalar::from(ComplexNumber::new(0.0f64, 0.0f64))),
        );
    }

    #[test]
    fn test_abs_complex_differentiation_avoids_overflow() {
        // Normalizing the complex coefficient before applying the tangent avoids overflowing the otherwise finite
        // directional derivative `Re((conj(z) / |z|) * dz)`.
        assert_eq!(
            jvp(
                |z| z.abs(),
                Scalar::from(ComplexNumber::new(1e308f64, 0.0)),
                Scalar::from(ComplexNumber::new(2.0f64, 0.0)),
            ),
            Ok((Scalar::from(1e308f64), Scalar::from(2.0f64))),
        );
    }

    #[test]
    fn test_abs_low_precision_differentiation_uses_widened_tangents() {
        // The coefficient and tangent are computed in the widened differential representation.
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = jvp(|input| input.abs(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_abs_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = AbsOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
