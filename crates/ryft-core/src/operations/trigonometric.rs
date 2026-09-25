//! Operations that compute trigonometric and hyperbolic functions elementwise. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`SinOperation`]) together with a value capability trait (e.g.,
//! [`Sin`]) whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so the same code
//! executes immediately or records into a program depending on the value it runs on:
//!
//!   - [`Sin`] and [`Cos`] compute the sine and cosine of angles measured in radians.
//!   - [`Atan2`] computes the two-argument arc tangent (i.e., `(y, x) ↦ atan2(y, x)`, the angle of the point `(x, y)`
//!     in its correct quadrant), with the principal value `-i · log((x + i · y) / sqrt(x² + y²))` for complex values.
//!   - [`Tanh`] computes the hyperbolic tangent (i.e., `x ↦ tanh(x)`).
//!
//! Floating-point and complex inputs are supported, same as for StableHLO's
//! [`sine`](https://openxla.org/stablehlo/spec#sine), [`cosine`](https://openxla.org/stablehlo/spec#cosine),
//! [`atan2`](https://openxla.org/stablehlo/spec#atan2), and [`tanh`](https://openxla.org/stablehlo/spec#tanh).
//! Unary operations preserve the metadata of their input, and [`Atan2`] promotes the element types and broadcasts the
//! shapes of its inputs. Inputs that carry partial sums over unreduced mesh axes are rejected. The derivatives of sine,
//! cosine, and the hyperbolic tangent are the cosine, the negated sine, and `1 - tanh(x)²`, and the derivative of
//! `atan2(y, x)` is `(x · dy - y · dx) / (x² + y²)`. Every operation is nonlinear, so reverse-mode differentiation
//! transposes its linearization instead.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{Array, Cos, ProgramError, Sin};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::scalar(0.0f64)?;
//! assert_eq!(input.sin()?, Array::scalar(0.0)?);
//! assert_eq!(input.cos()?, Array::scalar(1.0)?);
//! # Ok(())
//! # }
//! ```

use crate::arrays::FloatingPointArrayElement;
use crate::differentiation::{
    DifferentiableType, DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment,
};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_array_elementwise_operation,
    impl_differentiable_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::arithmetic::{Add, Div, Mul, Neg, Sub};
use crate::operations::constants::one_like::OneLike;
use crate::programs::{MaybeZero, ProgramError, Type, Typed};

/// Canonical operation name for [`SinOperation`].
pub const SIN_OPERATION_NAME: &str = "sin";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise sine of a floating-point or complex value while
    /// preserving its array metadata. Array inputs that still carry partial sums are rejected.
    SinOperation,
    SIN_OPERATION_NAME,
    Sin,
    sin,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    SinOperation,
    jvp<C> where C::Value: Mul + Cos {
        |(input, input_tangent)| input.cos()?.mul(&input_tangent)?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise sines. Concrete arrays compute immediately while context-carrying
    /// values apply [`SinOperation`] through their context.
    Sin,
    /// Computes the sine of each floating-point or complex element; real angles are measured in radians.
    /// Returns an error if the input types or metadata are unsupported.
    sin,
    SinOperation,
);

impl_array_elementwise_operation!(
    @unary
    Sin,
    sin,
    operation = "sin",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::sin(input),
);

/// Implements [`Sin`] for one host primitive type.
macro_rules! impl_sin_for_primitive {
    ($type:ty) => {
        impl Sin for $type {
            #[inline]
            fn sin(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::sin(*self))
            }
        }
    };
}

impl_sin_for_primitive!(f32);
impl_sin_for_primitive!(f64);

/// Canonical operation name for [`CosOperation`].
pub const COS_OPERATION_NAME: &str = "cos";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise cosine of a floating-point or complex value while
    /// preserving its array metadata. Array inputs that still carry partial sums are rejected.
    CosOperation,
    COS_OPERATION_NAME,
    Cos,
    cos,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    CosOperation,
    jvp<C> where C::Value: Neg + Mul + Sin {
        |(input, input_tangent)| input.sin()?.mul(&input_tangent)?.neg()?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise cosines. Concrete arrays compute immediately while
    /// context-carrying values apply [`CosOperation`] through their context.
    Cos,
    /// Computes the cosine of each floating-point or complex element; real angles are measured in radians.
    /// Returns an error if the input types or metadata are unsupported.
    cos,
    CosOperation,
);

impl_array_elementwise_operation!(
    @unary
    Cos,
    cos,
    operation = "cos",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::cos(input),
);

/// Implements [`Cos`] for one host primitive type.
macro_rules! impl_cos_for_primitive {
    ($type:ty) => {
        impl Cos for $type {
            #[inline]
            fn cos(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::cos(*self))
            }
        }
    };
}

impl_cos_for_primitive!(f32);
impl_cos_for_primitive!(f64);

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`Atan2Operation`].
pub const ATAN2_OPERATION_NAME: &str = "atan2";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise two-argument arc tangent of its inputs (i.e.,
    /// `(y, x) ↦ atan2(y, x)`, the angle of the point `(x, y)` in the correct quadrant for real inputs), promoting
    /// their element types and broadcasting their shapes. For complex inputs, the principal value is defined as
    /// `-i · log((x + i · y) / sqrt(x² + y²))`. Only floating-point and complex inputs are supported, and array inputs
    /// that still carry partial sums are rejected, with their reduced-axis markers required to agree.
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
        C::Value: Neg
            + Add
            + Mul
            + Div
            + Atan2
            + ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, context, _driver, inputs| {
            // d(atan2(y, x)) = x / (x² + y²) · dy - y / (x² + y²) · dx. The shared denominator is computed once for
            // both terms, and each divided coefficient is formed independently, matching the primitive's numerical
            // rule: combining the terms into one numerator can produce `inf - inf` before division for large finite
            // inputs even when the two finite quotient terms cancel. The custom form also computes the shared
            // denominator only once; independent per-side term expressions would recompute it.
            check_count!("input", inputs, 2, ProgramError);
            let y = &inputs[0];
            let x = &inputs[1];
            let primal = y.primal().atan2(x.primal())?;
            let target = primal.r#type().tangent()?;
            let has_y_tangent = y.tangent().as_value().is_some();
            let has_x_tangent = x.tangent().as_value().is_some();
            if !has_y_tangent && !has_x_tangent {
                return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Zero(target))?]);
            }
            if target.is_zero_space() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` output type `{}` has no tangent space",
                        ATAN2_OPERATION_NAME,
                        primal.r#type(),
                    ),
                }
                .into());
            }
            let output_primal = primal;
            let primal = context.primal_to_tangent(output_primal.clone())?;
            let x_primal = context.primal_to_tangent(x.primal().clone())?.align_tangent(&target, &primal)?;
            let y_primal = context.primal_to_tangent(y.primal().clone())?.align_tangent(&target, &primal)?;
            let denominator = x_primal.mul(&x_primal)?.add(&y_primal.mul(&y_primal)?)?;
            let y_term = y
                .tangent()
                .as_value()
                .map(|tangent| {
                    Ok::<_, DifferentiationError>(
                        x_primal.div(&denominator)?.mul(&tangent.align_tangent(&target, &primal)?)?,
                    )
                })
                .transpose()?;
            let x_term = x
                .tangent()
                .as_value()
                .map(|tangent| {
                    Ok::<_, DifferentiationError>(
                        y_primal.div(&denominator)?.neg()?.mul(&tangent.align_tangent(&target, &primal)?)?,
                    )
                })
                .transpose()?;
            let tangent = match (y_term, x_term) {
                (Some(y_term), Some(x_term)) => MaybeZero::Value(y_term.add(&x_term)?),
                (Some(term), None) | (None, Some(term)) => MaybeZero::Value(term),
                (None, None) => MaybeZero::Zero(target),
            };
            Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to compute elementwise two-argument arc tangents. Concrete arrays compute immediately
    /// while context-carrying values apply [`Atan2Operation`] through their context. Refer to that operation for
    /// supported types and exceptional-value behavior.
    Atan2,
    /// Computes `atan2(self, x)`, with `self` as the vertical coordinate and `x` as the horizontal coordinate,
    /// promoting and broadcasting the inputs. Returns an error if the input types or metadata are unsupported.
    atan2(x),
    Atan2Operation,
);

impl_array_elementwise_operation!(
    @binary
    Atan2, atan2,
    operation = "atan2",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| FloatingPointArrayElement::atan2(lhs, rhs),
);

/// Implements [`Atan2`] for one host primitive type.
macro_rules! impl_atan2_for_primitive {
    // Implements the capability for one floating-point primitive.
    ($type:ty) => {
        impl Atan2 for $type {
            fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
                Ok(<$type>::atan2(*self, *x))
            }
        }
    };
}

impl_atan2_for_primitive!(f32);
impl_atan2_for_primitive!(f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`TanhOperation`].
pub const TANH_OPERATION_NAME: &str = "tanh";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise hyperbolic tangent of one value (i.e.,
    /// `x ↦ tanh(x)`, the analytic continuation `tanh(z)` on complex inputs) while preserving its array metadata. Only
    /// floating-point and complex inputs are supported, and inputs that still carry partial sums are rejected.
    TanhOperation, TANH_OPERATION_NAME,
    Tanh, tanh,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    TanhOperation,
    jvp<C>
    where
        C::Value: OneLike + Sub + Mul,
    {
        // `d(tanh(x)) = (1 - tanh(x)²) · dx`, reusing the primal output evaluated at the tangent type.
        |(_, input_tangent) -> output| output.one_like()?.sub(&output.mul(&output)?)?.mul(&input_tangent)?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise hyperbolic tangents. Concrete arrays compute immediately while
    /// context-carrying values apply [`TanhOperation`] through their context. Refer to that operation for supported
    /// types and exceptional-value behavior.
    Tanh,
    /// Computes the hyperbolic tangent of each floating-point or complex element. Returns an error if the input types
    /// or metadata are unsupported.
    tanh,
    TanhOperation,
);

impl_array_elementwise_operation!(
    @unary
    Tanh, tanh,
    operation = "tanh",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::tanh(input),
);

/// Implements [`Tanh`] for one host primitive type.
macro_rules! impl_tanh_for_primitive {
    // Implements the capability for one floating-point primitive.
    ($type:ty) => {
        impl Tanh for $type {
            fn tanh(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::tanh(*self))
            }
        }
    };
}

impl_tanh_for_primitive!(f32);
impl_tanh_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Layout, StridedLayout, f8e8m0fnu};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::manipulation::conversions::ConvertElementType;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, TypeError, Typed};

    use super::*;

    #[test]
    fn test_sin_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = SinOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`sin` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sin_interpretation() {
        assert_eq!(
            SinOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_sin_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SinOperation::new(),
            inputs = [Array::scalar(0.5).unwrap()],
            expected = Array::scalar(0.5f64.sin()).unwrap(),
        );
    }

    #[test]
    fn test_sin_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SinOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.sin(), (-1.0f64).sin()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_sin_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SinOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(2.0f64.sin()).unwrap()],
                tangent_outputs = [Array::scalar(3.0 * 2.0f64.cos()).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = sin %0
                        %3:f64[] = cos %0
                        %4:f64[] = mul %3 %1
                    in (%2, %4)
                "},
            }],
        );
    }

    #[test]
    fn test_sin_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            differentiate_at(Array::scalar(input).unwrap()).holomorphic().gradient(|input| input.sin().unwrap()),
            Ok(Array::scalar(input.cos()).unwrap()),
        );
    }

    #[test]
    fn test_sin_differentiation_low_precision_uses_widened_tangents() {
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.sin()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-6);

        // The widened staged tangent program computes the coefficient in the widened differential representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
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
    fn test_sin_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = SinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_sin() {
        assert_eq!(Array::scalar(0.5f32).unwrap().sin().unwrap(), Array::scalar(0.5f32.sin()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().sin().unwrap(), Array::scalar(0.5f64.sin()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().sin().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.sin())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().sin().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.sin())).unwrap(),
        );
        let extreme = Array::scalar(ComplexNumber::new(0.0f64, 1000.0))
            .unwrap()
            .sin()
            .unwrap()
            .elements::<ComplexNumber<f64>>()
            .unwrap()[0];
        assert_eq!(extreme.re, 0.0);
        assert!(extreme.im.is_infinite() && extreme.im.is_sign_positive());

        assert_eq!(Array::scalar(0.5).unwrap().sin().unwrap(), Array::scalar(0.5f64.sin()).unwrap(),);
    }

    #[test]
    fn test_array_sin_vector() {
        let vector = Array::vector(vec![0.0, 1.0]).unwrap();
        assert_abs_diff_eq!(vector.sin().unwrap(), Array::vector(vec![0.0, 1.0f64.sin()]).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn test_array_sin_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        assert_abs_diff_eq!(
            left.sin().unwrap(),
            Array::vector(vec![left_values[0].sin(), left_values[1].sin()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_sin_for_primitives() {
        assert_eq!(Sin::sin(&0.0f64), Ok(0.0));
    }

    #[test]
    fn test_cos_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = CosOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`cos` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = CosOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_cos_interpretation() {
        assert_eq!(
            CosOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(1.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_cos_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CosOperation::new(),
            inputs = [Array::scalar(0.5).unwrap()],
            expected = Array::scalar(0.5f64.cos()).unwrap(),
        );
    }

    #[test]
    fn test_cos_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = CosOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.cos(), (-1.0f64).cos()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_cos_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = CosOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(2.0f64.cos()).unwrap()],
                tangent_outputs = [Array::scalar(-3.0 * 2.0f64.sin()).unwrap()],
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
    }

    #[test]
    fn test_cos_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            differentiate_at(Array::scalar(input).unwrap()).holomorphic().gradient(|input| input.cos().unwrap()),
            Ok(Array::scalar(-input.sin()).unwrap()),
        );
    }

    #[test]
    fn test_cos_differentiation_low_precision_uses_widened_tangents() {
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[4.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.cos()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.to_f64s()[0], -3.0 * 4.0f64.sin(), epsilon = 1e-6);

        // The widened staged tangent program computes the coefficient in the widened differential representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(CosOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
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
    fn test_cos_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = CosOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_cos() {
        assert_eq!(Array::scalar(0.5f32).unwrap().cos().unwrap(), Array::scalar(0.5f32.cos()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().cos().unwrap(), Array::scalar(0.5f64.cos()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().cos().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.cos())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().cos().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.cos())).unwrap(),
        );
        let extreme = Array::scalar(ComplexNumber::new(0.0f64, 1000.0))
            .unwrap()
            .cos()
            .unwrap()
            .elements::<ComplexNumber<f64>>()
            .unwrap()[0];
        assert!(extreme.re.is_infinite() && extreme.re.is_sign_positive());
        assert_eq!(extreme.im, 0.0);

        assert_eq!(Array::scalar(0.5).unwrap().cos().unwrap(), Array::scalar(0.5f64.cos()).unwrap(),);
    }

    #[test]
    fn test_array_cos_vector() {
        let vector = Array::vector(vec![0.0, 1.0]).unwrap();
        assert_abs_diff_eq!(vector.cos().unwrap(), Array::vector(vec![1.0, 1.0f64.cos()]).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn test_array_cos_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        assert_abs_diff_eq!(
            left.cos().unwrap(),
            Array::vector(vec![left_values[0].cos(), left_values[1].cos()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_cos_for_primitives() {
        assert_eq!(Cos::cos(&0.0f64), Ok(1.0));
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
                    error = "`atan2` does not support input data type `i32`",
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
    fn test_atan2_interpretation() {
        assert_eq!(
            Atan2Operation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap(), Array::scalar(1.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_atan2_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = Atan2Operation::new(),
            inputs = [Array::scalar(0.5).unwrap(), Array::scalar(-0.25).unwrap()],
            expected = Array::scalar(0.5f64.atan2(-0.25)).unwrap(),
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
                    (@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap()),
                    (@replicated, Array::scalar(2.0).unwrap()),
                ],
                outputs = [(@mapped(
                    axis = 0
                ), Array::vector(vec![0.5f64.atan2(2.0), (-1.0f64).atan2(2.0)]).unwrap())],
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
                primals = [Array::scalar(y).unwrap(), Array::scalar(x).unwrap()],
                tangents = [Array::scalar(y_tangent).unwrap(), Array::scalar(x_tangent).unwrap()],
                primal_outputs = [Array::scalar(y.atan2(x)).unwrap()],
                tangent_outputs = [Array::scalar(tangent).unwrap()],
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
        let (_, tangent) = differentiate_at((Array::scalar(1.0e308).unwrap(), Array::scalar(1.0e308).unwrap()))
            .jvp((Array::scalar(1.0e308).unwrap(), Array::scalar(1.0e308).unwrap()), |(y, x)| y.atan2(&x))
            .unwrap();
        assert_eq!(tangent, Array::scalar(0.0).unwrap());
    }

    #[test]
    fn test_atan2_differentiation_complex() {
        let y = ComplexNumber::new(0.7f64, -0.2);
        let x = ComplexNumber::new(-0.3f64, 0.4);
        let (value, (y_gradient, x_gradient)) =
            differentiate_at((Array::scalar(y).unwrap(), Array::scalar(x).unwrap()))
                .holomorphic()
                .value_and_gradient(|(y, x)| y.atan2(&x).unwrap())
                .unwrap();
        let denominator = x * x + y * y;
        let imaginary_unit = ComplexNumber::new(0.0, 1.0);
        assert_abs_diff_eq!(
            value,
            Array::scalar(-imaginary_unit * ((x + imaginary_unit * y) / denominator.sqrt()).ln()).unwrap(),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(y_gradient, Array::scalar(x / denominator).unwrap(), epsilon = 1e-12);
        assert_abs_diff_eq!(x_gradient, Array::scalar(-y / denominator).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn test_atan2_differentiation_low_precision_uses_widened_tangents() {
        let y = Array::scalar(2.0f32).unwrap().convert_element_type(DataType::F8E8M0FNU).unwrap();
        let x = Array::scalar(4.0f32).unwrap().convert_element_type(DataType::F8E8M0FNU).unwrap();
        let (primal, tangent) = differentiate_at((y, x))
            .jvp((Array::scalar(1.0f32).unwrap(), Array::scalar(1.0f32).unwrap()), |(y, x)| y.atan2(&x))
            .unwrap();
        assert_eq!(primal.r#type().data_type(), DataType::F8E8M0FNU);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 0.1f32 as f64, epsilon = 1e-6);
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
    fn test_array_atan2() {
        assert_eq!(
            Array::scalar(0.5f32).unwrap().atan2(&Array::scalar(-0.25f32).unwrap()).unwrap(),
            Array::scalar(0.5f32.atan2(-0.25f32)).unwrap(),
        );
        assert_eq!(
            Array::scalar(0.5f64).unwrap().atan2(&Array::scalar(-0.25f64).unwrap()).unwrap(),
            Array::scalar(0.5f64.atan2(-0.25f64)).unwrap(),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5))
                .unwrap()
                .atan2(&Array::scalar(bf16::from_f32(-0.25)).unwrap())
                .unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.atan2(-0.25f32))).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5))
                .unwrap()
                .atan2(&Array::scalar(f16::from_f32(-0.25)).unwrap())
                .unwrap(),
            Array::scalar(f16::from_f32(0.5f32.atan2(-0.25f32))).unwrap(),
        );
        let y = ComplexNumber::new(0.5f32, 0.25);
        let x = ComplexNumber::new(-0.75f32, 0.125);
        let imaginary_unit = ComplexNumber::new(0.0, 1.0);
        assert_abs_diff_eq!(
            Array::scalar(y).unwrap().atan2(&Array::scalar(x).unwrap()).unwrap(),
            Array::scalar(-imaginary_unit * ((x + imaginary_unit * y) / (x * x + y * y).sqrt()).ln()).unwrap(),
            epsilon = 1e-6,
        );
        let y = ComplexNumber::new(0.5f64, 0.0);
        let x = ComplexNumber::new(-0.75f64, 0.125);
        let imaginary_unit = ComplexNumber::new(0.0, 1.0);
        assert_abs_diff_eq!(
            Array::scalar(0.5f32).unwrap().atan2(&Array::scalar(x).unwrap()).unwrap(),
            Array::scalar(-imaginary_unit * ((x + imaginary_unit * y) / (x * x + y * y).sqrt()).ln()).unwrap(),
            epsilon = 1e-12,
        );

        assert_eq!(
            Array::scalar(0.5).unwrap().atan2(&Array::scalar(-0.25).unwrap()).unwrap(),
            Array::scalar(0.5f64.atan2(-0.25f64)).unwrap(),
        );

        assert_abs_diff_eq!(
            Array::vector(vec![1.0]).unwrap().atan2(&Array::vector(vec![1.0]).unwrap()).unwrap(),
            Array::vector(vec![std::f64::consts::FRAC_PI_4]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_atan2_layout_and_promotion() {
        // Binary kernels perform complete broadcasting after promoting both physical inputs to their common type.
        let left_type =
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[0.0f32, 1.0]).unwrap();
        let right_type =
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[1.0f64, 1.0, -1.0]).unwrap();
        let angles = left.atan2(&right).unwrap();
        assert_eq!(angles.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3]));
        assert_abs_diff_eq!(
            angles,
            Array::matrix(
                2,
                3,
                vec![
                    0.0,
                    0.0,
                    std::f64::consts::PI,
                    std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_4,
                    3.0 * std::f64::consts::FRAC_PI_4,
                ],
            )
            .unwrap(),
            epsilon = 1e-12,
        );
        assert!(matches!(
            Array::scalar(1i32).unwrap().atan2(&Array::scalar(1.0f64).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`atan2` does not support input data type `i32`",
        ));
    }

    #[test]
    fn test_atan2_for_primitives() {
        assert_eq!(Atan2::atan2(&1.0f64, &1.0), Ok(std::f64::consts::FRAC_PI_4));
    }

    #[test]
    fn test_tanh_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = TanhOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`tanh` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = TanhOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_tanh_interpretation() {
        assert_eq!(
            TanhOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_tanh_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = TanhOperation::new(),
            inputs = [Array::scalar(0.7).unwrap()],
            expected = Array::scalar(0.7f64.tanh()).unwrap(),
        );
    }

    #[test]
    fn test_tanh_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = TanhOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.tanh(), (-1.0f64).tanh()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_tanh_differentiation() {
        let expected_tangent = 3.0 * (1.0 - 0.7f64.tanh() * 0.7f64.tanh());
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = TanhOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.7f64.tanh()).unwrap()],
                tangent_outputs = [Array::scalar(expected_tangent).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = tanh %0
                        %3:f64[] = one_like %2
                        %4:f64[] = mul %2 %2
                        %5:f64[] = sub %3 %4
                        %6:f64[] = mul %5 %1
                    in (%2, %6)
                "},
            }],
        );
    }

    #[test]
    fn test_tanh_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = {
            let tanh = input.tanh();
            ComplexNumber::new(1.0, 0.0) - tanh * tanh
        };
        assert_abs_diff_eq!(
            Array::scalar(expected).unwrap(),
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| { input.tanh().unwrap() })
                .unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_tanh_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = TanhOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_tanh() {
        assert_eq!(Array::scalar(0.5f32).unwrap().tanh().unwrap(), Array::scalar(0.5f32.tanh()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().tanh().unwrap(), Array::scalar(0.5f64.tanh()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().tanh().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.tanh())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().tanh().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.tanh())).unwrap(),
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().tanh().unwrap(),
            Array::scalar(input.tanh()).unwrap(),
            epsilon = 1e-12,
        );

        assert_eq!(Array::scalar(0.7).unwrap().tanh().unwrap(), Array::scalar(0.7f64.tanh()).unwrap(),);
    }

    #[test]
    fn test_tanh_for_primitives() {
        assert_eq!(Tanh::tanh(&0.0f64), Ok(0.0));
    }
}
