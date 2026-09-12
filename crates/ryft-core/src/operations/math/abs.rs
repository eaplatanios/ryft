use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, DataType, NumericArrayElement};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, dispatch_on_array_element_type,
    impl_differentiable_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::complex::{Complex, Conjugate, Imaginary, Real};
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::programs::{MaybeZero, ProgramError, Type, TypeError, Typed};

// TODO(eaplatanios): Review this module.

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
            return Err(TypeError::invalid(format!("cannot compute the absolute value of a value of data type `{input_type}`")));
        };
        Ok(vec![output_type])
    },
    check_array_types = [@no_unreduced],
);

impl_differentiable_operation! {
    <T> AbsOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Abs
            + Compare<C::Value>
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
        |_operation, context, _driver, inputs| {
            // Away from zero, the real derivative is `d|x| = sign(x) · dx`, while the complex magnitude is a ℂ → ℝ map
            // with `d|z| = Re(z̄ · dz) / |z|`. At the real origin, choose the right derivative and return `dx`. At the
            // complex origin, replace the zero denominator with one so the zero numerator yields zero. These
            // conventions keep the rule finite and stable under higher-order transforms. A structural zero tangent
            // stays symbolic, retyped to the real output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().abs()?;
            let primal_tangent_type = primal.r#type().tangent()?;
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal_tangent_type),
                MaybeZero::Value(_) if primal_tangent_type.is_zero_space() => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!("`{}` output type {} has no tangent space", ABS_OPERATION_NAME, primal.r#type()),
                    }
                    .into());
                }
                MaybeZero::Value(tangent) => {
                    let primal = context.primal_to_tangent(primal.clone())?;
                    let input_primal = context.primal_to_tangent(input.primal().clone())?;
                    if input.primal().r#type().is_complex() {
                        let denominator = primal.align_tangent(&primal_tangent_type, &primal)?;
                        let zero = denominator.zero_like()?;
                        let one = denominator.one_like()?;
                        let denominator_is_zero = denominator.compare(&zero, ComparisonDirection::Equal)?;
                        let denominator = C::Value::select(&denominator_is_zero, &one, &denominator)?;
                        // Normalize `conj(z) / |z|` before multiplying by `dz`. Computing `conj(z) * dz` first is
                        // algebraically equivalent but can overflow even when the final directional derivative is
                        // finite.
                        let conjugate = input_primal.conjugate()?;
                        let real = conjugate.real()? / denominator.clone();
                        let imaginary = conjugate.imaginary()? / denominator.clone();
                        let coefficient = real.complex(&imaginary)?;
                        let input_tangent_type = input.primal().r#type().tangent()?;
                        let tangent = tangent.align_tangent(&input_tangent_type, &input_primal)?;
                        MaybeZero::Value((tangent * coefficient).real()?.align_tangent(&primal_tangent_type, &primal)?)
                    } else {
                        let input = input_primal.align_tangent(&primal_tangent_type, &primal)?;
                        let tangent = tangent.align_tangent(&primal_tangent_type, &primal)?;
                        let zero = input.zero_like()?;
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

/// Implements [`Abs`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Signed integer primitives use checked absolute values so that the `MIN` overflow reports an error instead of
    // wrapping like the XLA-mirroring reference backends do on devices.
    (@signed $type:ty) => {
        impl Abs for $type {
            fn abs(&self) -> Result<Self, ProgramError> {
                self.checked_abs().ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` result does not fit in {}", ABS_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Unsigned integer primitives are their own absolute values.
    (@unsigned $type:ty) => {
        impl Abs for $type {
            fn abs(&self) -> Result<Self, ProgramError> {
                Ok(*self)
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 absolute values, which cannot fail.
    (@float $type:ty) => {
        impl Abs for $type {
            fn abs(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::abs(*self))
            }
        }
    };
}

impl_capability_for_primitive!(@signed i8);
impl_capability_for_primitive!(@signed i16);
impl_capability_for_primitive!(@signed i32);
impl_capability_for_primitive!(@signed i64);
impl_capability_for_primitive!(@signed i128);
impl_capability_for_primitive!(@signed isize);
impl_capability_for_primitive!(@unsigned u8);
impl_capability_for_primitive!(@unsigned u16);
impl_capability_for_primitive!(@unsigned u32);
impl_capability_for_primitive!(@unsigned u64);
impl_capability_for_primitive!(@unsigned u128);
impl_capability_for_primitive!(@unsigned usize);
impl_capability_for_primitive!(@float f32);
impl_capability_for_primitive!(@float f64);

impl Abs for Array {
    fn abs(&self) -> Result<Self, ProgramError> {
        // The absolute value of a complex array is its elementwise magnitude, so the element data type maps to its
        // real part data type, mirroring the `AbsOperation` type-inference contract.
        let data_type = match self.r#type().data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other,
        };
        let output_type = self.r#type().into_owned().with_data_type(data_type);
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }
        let input_type = self.r#type().data_type();
        if !((input_type.is_signed() && input_type != DataType::I1)
            || input_type.is_floating_point()
            || input_type.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "cannot compute the absolute value of a scalar of data type `{input_type}`",
            ))
            .into());
        }
        dispatch_on_array_element_type!(@numeric input_type, |Element| {
            self.map_elements::<Element, <Element as NumericArrayElement>::Magnitude>(output_type, |value| {
                <Element as NumericArrayElement>::abs(value)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, f8e4m3fn, f8e8m0fnu, i4};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_abs() {
        assert_eq!(AbsOperation::<ArrayType>::new().to_string(), "abs");
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
            let message = format!("cannot compute the absolute value of a value of data type `{input_type}`");
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
            operation = AbsOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F32)],
        );
    }

    #[test]
    fn test_abs_interpretation() {
        let operation = AbsOperation::new();

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(-2.0).unwrap()],
            ),
            Ok(vec![Array::scalar(2.0).unwrap()]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(ComplexNumber::new(3.0f64, -4.0f64)).unwrap()],
            ),
            Ok(vec![Array::scalar(5.0).unwrap()]),
        );
    }

    #[test]
    fn test_abs_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = AbsOperation::new(),
            inputs = [Array::scalar(-2.0).unwrap()],
            expected = Array::scalar(2.0).unwrap(),
        );
    }

    #[test]
    fn test_abs_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AbsOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_abs_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AbsOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(0.7).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap()],
                    primal_outputs = [Array::scalar(0.7).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
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
                    primals = [Array::scalar(-2.5).unwrap()],
                    tangents = [Array::scalar(2.0).unwrap()],
                    primal_outputs = [Array::scalar(2.5).unwrap()],
                    tangent_outputs = [Array::scalar(-2.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_abs_differentiation_at_zero() {
        // The real rule chooses the right derivative at zero and remains constant under another derivative.
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(0.0f64).unwrap()).gradient(|x| x.abs().unwrap()).unwrap(),
            Array::scalar(1.0).unwrap(),
            epsilon = 1e-9,
        );
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(0.0f64).unwrap())
                .gradient(|x| { differentiate_at(x).gradient(|x| x.abs().unwrap()).unwrap() })
                .unwrap(),
            Array::scalar(0.0).unwrap(),
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_abs_complex_differentiation() {
        // |z| is a ℂ → ℝ function and so it flows through the plain gradient entry point. With
        // d|z| = Re(z̄ · dz) / |z|, the bilinear-pairing gradient is z̄ / |z| (the unit-magnitude conjugate direction):
        // the reverse-mode counterpart of ∇|z|² = 2z̄ after the chain rule through the square root.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) =
            differentiate_at(Array::scalar(z).unwrap()).value_and_gradient(|z| z.abs().unwrap()).unwrap();
        assert_eq!(value, Array::scalar(z.norm()).unwrap());
        let expected = z.conj() / z.norm();
        assert_abs_diff_eq!(gradient_value, Array::scalar(expected).unwrap(), epsilon = 1e-12);

        // The array universe agrees: summing the elementwise magnitudes of a complex vector is again ℂⁿ → ℝ, and the
        // finite-difference oracle perturbs each element's real and imaginary parts independently.
        check_gradient!(
            |z| z.abs().map(|magnitudes| magnitudes.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![ComplexNumber::new(0.7f64, -0.3), ComplexNumber::new(-1.2f64, 0.8)]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // The complex rule replaces a zero magnitude denominator with one, so the zero numerator produces a finite
        // zero tangent and gradient at the origin.
        assert_eq!(
            differentiate_at(Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap())
                .jvp(Array::scalar(ComplexNumber::new(1.0f64, 2.0f64)).unwrap(), |z| z.abs()),
            Ok((Array::scalar(0.0f64).unwrap(), Array::scalar(0.0f64).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap()).gradient(|z| z.abs().unwrap()),
            Ok(Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap()),
        );
    }

    #[test]
    fn test_abs_complex_differentiation_avoids_overflow() {
        // Normalizing the complex coefficient before applying the tangent avoids overflowing the otherwise finite
        // directional derivative `Re((conj(z) / |z|) * dz)`.
        assert_eq!(
            differentiate_at(Array::scalar(ComplexNumber::new(1e308f64, 0.0)).unwrap())
                .jvp(Array::scalar(ComplexNumber::new(2.0f64, 0.0)).unwrap(), |z| z.abs()),
            Ok((Array::scalar(1e308f64).unwrap(), Array::scalar(2.0f64).unwrap())),
        );
    }

    #[test]
    fn test_abs_low_precision_differentiation_uses_widened_tangents() {
        // The coefficient and tangent are computed in the widened differential representation.
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.abs()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_abs_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = AbsOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_abs_for_primitives() {
        assert_eq!(Abs::abs(&-5_i32), Ok(5));
        assert_eq!(
            Abs::abs(&i8::MIN),
            Err(ProgramError::InvalidArgument { message: "`abs` result does not fit in i8".to_string() }),
        );
        assert_eq!(Abs::abs(&5_usize), Ok(5));
        assert_eq!(Abs::abs(&-2.5_f64), Ok(2.5));
    }

    #[test]
    fn test_abs_for_array() {
        assert_eq!(Array::vector(vec![-1.5, 2.5]).unwrap().abs().unwrap(), Array::vector(vec![1.5, 2.5]).unwrap());
        // The absolute value of a complex array is its elementwise magnitude with a real element data type.
        let complex = Array::vector(vec![3.0]).unwrap().complex(&Array::vector(vec![4.0]).unwrap()).unwrap();
        let magnitude = complex.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), ArrayType::new_static(DataType::F64, [1]));
        assert_abs_diff_eq!(magnitude, Array::vector(vec![5.0]).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn test_abs_for_array_low_precision() {
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let negative = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[-1.0, -2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(negative.abs().unwrap(), left);
    }

    #[test]
    fn test_abs_for_array_complex() {
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        // The absolute value is the elementwise magnitude with a real element data type.
        let magnitude = left.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2]));
        assert_abs_diff_eq!(
            magnitude,
            Array::vector(vec![left_values[0].norm(), left_values[1].norm()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_abs_for_array_integers() {
        // Sub-byte arithmetic uses the declared bit width for every wrapping operation.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(narrow.abs().unwrap().elements::<i4>(), Ok(vec![i4::new(7).unwrap(), i4::MIN]));
    }
}
