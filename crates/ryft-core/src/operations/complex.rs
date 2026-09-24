//! Construction, conjugation, and component extraction for complex values.
//!
//! This module provides the following:
//!
//!   - [`Complex`] and [`ComplexOperation`], which combine identically typed real and imaginary parts.
//!   - [`Conjugate`] and [`ConjugateOperation`], which negate the imaginary component.
//!   - [`Real`] and [`RealOperation`], which extract the real component.
//!   - [`Imaginary`] and [`ImaginaryOperation`], which extract the imaginary component.
//!
//! Parts have `f32` or `f64` elements, corresponding to `c64` or `c128` complex elements. Construction and extraction
//! preserve shape, sharding, and memory placement, but clear byte-strided layouts when element width changes.
//! Conjugation preserves the entire type. Partial evaluation folds known inputs, and batching applies each operation
//! elementwise. Differentiation treats these maps as real-linear; transposition uses Ryft's bilinear complex pairing,
//! including the negative imaginary cotangent for construction and imaginary-part extraction.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError};
//! # use ryft_core::operations::complex::{Complex, Conjugate, Imaginary, Real};
//! # fn main() -> Result<(), ProgramError> {
//! let real = Array::vector(vec![1.0f32, 2.0])?;
//! let imaginary = Array::vector(vec![3.0f32, -4.0])?;
//! let value = real.complex(&imaginary)?;
//! assert_eq!(value.real()?, real);
//! assert_eq!(value.imaginary()?, imaginary);
//! assert_eq!(value.conjugate()?.imaginary()?, Array::vector(vec![-3.0f32, 4.0])?);
//! # Ok(())
//! # }
//! ```

use num_complex::Complex as ComplexNumber;

use crate::arrays::{Array, ArrayType, DataType};
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::constants::zero::Zero;
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::manipulation::conversions::ElementType;
use crate::operations::math::neg::NegOperation;
use crate::programs::{MaybeZero, ProgramError, Type, TypeError, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ComplexOperation`].
pub const COMPLEX_OPERATION_NAME: &str = "complex";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that constructs a complex value from its real and imaginary parts (i.e., `(re, im) ↦ re + im·i`,
    /// with `(f32, f32) ↦ c64` and `(f64, f64) ↦ c128`). This is the analogue of
    /// [JAX's `lax.complex`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.complex.html) and the inverse of the
    /// [`RealOperation`]/[`ImaginaryOperation`] pair. The two parts must have identical types. Array shape, sharding,
    /// and memory placement are preserved; byte-strided layouts are cleared because complex elements are wider.
    ///
    /// As a map from the pair of real parts, the operation is linear, and its transpose is the
    /// `ȳ ↦ (real(ȳ), imaginary(-ȳ))` pair under the bilinear (i.e., conjugation-free) pairing that Ryft's
    /// transposition uses over complex types.
    ComplexOperation, COMPLEX_OPERATION_NAME,
    Complex, complex,
    infer_data_types = |input_types: &[DataType]| {
        if input_types[0] != input_types[1] {
            return Err(TypeError::invalid(format!(
                "`{COMPLEX_OPERATION_NAME}` requires identical part types but got `{}` and `{}`",
                input_types[0], input_types[1],
            )));
        }
        Ok(vec![part_to_complex_data_type(input_types[0], COMPLEX_OPERATION_NAME)?])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        if input_types[0] != input_types[1] {
            return Err(TypeError::invalid(format!(
                "`{COMPLEX_OPERATION_NAME}` requires identical part types but got `{}` and `{}`",
                input_types[0], input_types[1],
            )));
        }
        let data_type = part_to_complex_data_type(input_types[0].data_type(), COMPLEX_OPERATION_NAME)?;
        Ok(vec![input_types[0].with_element_type(data_type)])
    },
);

impl_differentiable_operation! {
    <T> ComplexOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C: Zero<C::Value>,
        C::Value: Complex,
    {
        |_operation, context, _driver, inputs| {
            check_count!("input", inputs, 2, ProgramError);
            let real = &inputs[0];
            let imaginary = &inputs[1];
            let primal = real.primal().complex(imaginary.primal())?;
            // Complex construction is linear in its two real parts: `d(complex(re, im)) = complex(dre, dim)`. When both
            // part tangents are structural zeros the output tangent stays a symbolic zero of the complex output type;
            // when only one is, the missing part is materialized as a real zero through the context so the staged
            // `complex` keeps its two-part arity.
            let tangent = match (real.tangent(), imaginary.tangent()) {
                (MaybeZero::Zero(_), MaybeZero::Zero(_)) => MaybeZero::Zero(primal.r#type().tangent()?),
                (real_tangent, imaginary_tangent) => MaybeZero::Value(
                    real_tangent
                        .clone()
                        .materialize(context.tangent())?
                        .complex(&imaginary_tangent.clone().materialize(context.tangent())?)?,
                ),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: Type,
        V::Type: DifferentiableType,
        O: From<NegOperation<V::Type>> + From<RealOperation<V::Type>> + From<ImaginaryOperation<V::Type>>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for the linear [`ComplexOperation`]. Under the bilinear (i.e., conjugation-free) pairing
            // used for complex transposition, the transpose of `(re, im) ↦ re + im·i` maps the output cotangent `ȳ`
            // to `(real(ȳ), imaginary(-ȳ))`: pairing `Re(ȳ · (re + im·i))` against
            // `(re, im)` picks out the real part of `ȳ` for `re` and the *negated* imaginary part for `im`. Like the
            // `Add` rule, a known part contributes an additive constant whose adjoint is dropped at the pullback
            // output boundary.
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(output_cotangent) => {
                    let contribution = MaybeZero::Value(output_cotangent.unary(RealOperation::new()));
                    accumulators[0].accumulate(context, contribution)?;
                    let contribution =
                        MaybeZero::Value(output_cotangent.unary(NegOperation::new()).unary(ImaginaryOperation::new()));
                    accumulators[1].accumulate(context, contribution)?;
                    Ok(())
                }
            }
        }
    },
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to construct complex values from real and imaginary parts. Concrete arrays compute
    /// immediately; context-carrying values apply [`ComplexOperation`] through their context. The parts must have
    /// identical types with `f32` or `f64` elements; construction neither promotes nor broadcasts them.
    Complex,
    /// Constructs `self + imaginary·i` elementwise, producing `c64` from `f32` parts or `c128` from `f64` parts.
    /// Returns an error if the parts have different types or unsupported element types. Refer to [`ComplexOperation`]
    /// for output metadata and differentiation semantics.
    ///
    /// # Parameters
    ///
    ///   - `imaginary`: Imaginary part, with the same type as this real part.
    complex(imaginary),
    ComplexOperation,
);

// TODO(eaplatanios): Review this.

impl Complex for Array {
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        // Construction requires identical part types and combines their values without promotion or broadcasting.
        if self.r#type() != imaginary.r#type() {
            return Err(TypeError::invalid(format!(
                "`{COMPLEX_OPERATION_NAME}` requires identical part types but got `{}` and `{}`",
                self.r#type(),
                imaginary.r#type(),
            ))
            .into());
        }
        let data_type = match self.r#type().data_type() {
            DataType::F32 => DataType::C64,
            DataType::F64 => DataType::C128,
            other => {
                return Err(TypeError::invalid(format!(
                    "cannot construct a complex value from parts of data type `{other}`",
                ))
                .into());
            }
        };
        let output_type = self.r#type().with_element_type(data_type);
        if data_type == DataType::C64 {
            self.map_element_pairs::<f32, ComplexNumber<f32>>(imaginary, output_type, |real, imaginary| {
                Ok(ComplexNumber::new(real, imaginary))
            })
        } else {
            self.map_element_pairs::<f64, ComplexNumber<f64>>(imaginary, output_type, |real, imaginary| {
                Ok(ComplexNumber::new(real, imaginary))
            })
        }
    }
}

/// Canonical operation name for [`ConjugateOperation`].
pub const CONJUGATE_OPERATION_NAME: &str = "conjugate";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise complex conjugate of one complex value (i.e., `z ↦ z̄`, negating
    /// the imaginary part) while preserving its type metadata. Only `c64` and `c128` inputs are supported.
    ///
    /// Conjugation is ℝ-linear but not ℂ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's
    /// transposition uses over complex types, it is self-adjoint: the transpose of `z ↦ z̄` is `ȳ ↦ ȳ̄`.
    ConjugateOperation, CONJUGATE_OPERATION_NAME,
    Conjugate, conjugate,
    infer_data_types = |input_types: &[DataType]| {
        complex_to_part_data_type(input_types[0], CONJUGATE_OPERATION_NAME)?;
        Ok(vec![input_types[0]])
    },
);

impl_differentiable_operation! {
    <T> ConjugateOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Conjugate,
    {
        |_operation, _context, _driver, inputs| {
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().conjugate()?;
            // Conjugation is ℝ-linear (but not ℂ-linear): `d(z̄) = d̄z`. A structural zero tangent stays
            // symbolic.
            let tangent = match input.tangent() {
                MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.conjugate()?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: Type,
        V::Type: DifferentiableType,
        O: From<ConjugateOperation<V::Type>>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for the ℝ-linear [`ConjugateOperation`]. Under the bilinear (i.e., conjugation-free)
            // pairing that Ryft's transposition uses over complex types, conjugation is self-adjoint: pairing
            // `Re(ȳ · z̄)` against `z` shows that the transpose of `z ↦ z̄` is `ȳ ↦ ȳ̄`.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(output_cotangent) => {
                    let contribution = MaybeZero::Value(output_cotangent.unary(ConjugateOperation::new()));
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
            }
        }
    },
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to conjugate complex values elementwise. Concrete arrays compute immediately;
    /// context-carrying values apply [`ConjugateOperation`] through their context. Conjugation preserves the input
    /// type and negates each imaginary component.
    Conjugate,
    /// Returns the elementwise complex conjugate, or an error if the input does not have `c64` or `c128` elements.
    /// Refer to [`ConjugateOperation`] for differentiation semantics.
    conjugate,
    ConjugateOperation,
);

impl Conjugate for Array {
    fn conjugate(&self) -> Result<Self, ProgramError> {
        match self.r#type().data_type() {
            DataType::C64 => self
                .map_elements::<ComplexNumber<f32>, ComplexNumber<f32>>(self.r#type().into_owned(), |value| {
                    Ok(value.conj())
                }),
            DataType::C128 => self
                .map_elements::<ComplexNumber<f64>, ComplexNumber<f64>>(self.r#type().into_owned(), |value| {
                    Ok(value.conj())
                }),
            other => Err(TypeError::invalid(format!("cannot conjugate a value of data type `{other}`")).into()),
        }
    }
}

/// Canonical operation name for [`RealOperation`].
pub const REAL_OPERATION_NAME: &str = "real";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that extracts the elementwise real part of one complex value (i.e., `z ↦ Re(z)`, with
    /// `c64 ↦ f32` and `c128 ↦ f64`) while preserving shape, sharding, and memory placement. Byte-strided layouts are
    /// cleared because the output elements are narrower. This is the analogue of
    /// [JAX's `lax.real`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.real.html).
    ///
    /// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
    /// over complex types, the transpose of `z ↦ Re(z)` is `ȳ ↦ complex(ȳ, 0)`.
    RealOperation, REAL_OPERATION_NAME,
    Real, real,
    infer_data_types = |input_types: &[DataType]| {
        Ok(vec![complex_to_part_data_type(input_types[0], REAL_OPERATION_NAME)?])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        let data_type = complex_to_part_data_type(input_types[0].data_type(), REAL_OPERATION_NAME)?;
        Ok(vec![input_types[0].with_element_type(data_type)])
    },
);

impl_differentiable_operation! {
    <T> RealOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Real,
    {
        |_operation, _context, _driver, inputs| {
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().real()?;
            // Real-part extraction is ℝ-linear: `d(Re(z)) = Re(dz)`. A structural zero tangent stays symbolic,
            // retyped to the real output type.
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.real()?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: Type,
        V::Type: DifferentiableType,
        O: From<ComplexOperation<V::Type>> + From<ZeroLikeOperation<V::Type>>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for the ℝ-linear [`RealOperation`]. Under the bilinear (i.e., conjugation-free) pairing
            // that Ryft's transposition uses over complex types, pairing `t · Re(z)` against `z` shows that the
            // transpose of `z ↦ Re(z)` is `t ↦ complex(t, 0)`, injecting the real cotangent with a zero imaginary part.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(output_cotangent) => {
                    let zero = output_cotangent.unary(ZeroLikeOperation::new());
                    let contribution = MaybeZero::Value(output_cotangent.binary(&zero, ComplexOperation::new()));
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
            }
        }
    },
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to extract the real component of complex values elementwise. Concrete arrays
    /// compute immediately; context-carrying values apply [`RealOperation`] through their context. The output has
    /// `f32` elements for `c64` inputs and `f64` elements for `c128` inputs.
    Real,
    /// Returns the elementwise real component, or an error if the input is not complex valued. Refer to
    /// [`RealOperation`] for output metadata and differentiation semantics.
    real,
    RealOperation,
);

impl Real for Array {
    fn real(&self) -> Result<Self, ProgramError> {
        // Extract components at their original precision; changing element width clears byte-strided layouts.
        match self.r#type().data_type() {
            DataType::C64 => self
                .map_elements::<ComplexNumber<f32>, f32>(self.r#type().with_element_type(DataType::F32), |value| {
                    Ok(value.re)
                }),
            DataType::C128 => self
                .map_elements::<ComplexNumber<f64>, f64>(self.r#type().with_element_type(DataType::F64), |value| {
                    Ok(value.re)
                }),
            other => {
                Err(TypeError::invalid(format!("cannot extract the real part of a value of data type `{other}`"))
                    .into())
            }
        }
    }
}

/// Canonical operation name for [`ImaginaryOperation`].
pub const IMAGINARY_OPERATION_NAME: &str = "imaginary";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that extracts the elementwise imaginary part of one complex value (i.e., `z ↦ Im(z)`, with
    /// `c64 ↦ f32` and `c128 ↦ f64`) while preserving shape, sharding, and memory placement. Byte-strided layouts are
    /// cleared because the output elements are narrower. This is the analogue of
    /// [JAX's `lax.imag`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.imag.html).
    ///
    /// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
    /// over complex types, the transpose of `z ↦ Im(z)` is `ȳ ↦ complex(0, -ȳ)`.
    ImaginaryOperation, IMAGINARY_OPERATION_NAME,
    Imaginary, imaginary,
    infer_data_types = |input_types: &[DataType]| {
        Ok(vec![complex_to_part_data_type(input_types[0], IMAGINARY_OPERATION_NAME)?])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        let data_type = complex_to_part_data_type(input_types[0].data_type(), IMAGINARY_OPERATION_NAME)?;
        Ok(vec![input_types[0].with_element_type(data_type)])
    },
);

impl_differentiable_operation! {
    <T> ImaginaryOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Imaginary,
    {
        |_operation, _context, _driver, inputs| {
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().imaginary()?;
            // Imaginary-part extraction is ℝ-linear: `d(Im(z)) = Im(dz)`. A structural zero tangent stays symbolic,
            // retyped to the real output type.
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.imaginary()?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: Type,
        V::Type: DifferentiableType,
        O: From<NegOperation<V::Type>> + From<ComplexOperation<V::Type>> + From<ZeroLikeOperation<V::Type>>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for the ℝ-linear [`ImaginaryOperation`]. Under the bilinear (i.e., conjugation-free)
            // pairing that Ryft's transposition uses over complex types, pairing `t · Im(z)` against `z` shows that the
            // transpose of `z ↦ Im(z)` is `t ↦ complex(0, -t)`, injecting the *negated* real cotangent as the imaginary
            // part.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(output_cotangent) => {
                    let zero = output_cotangent.unary(ZeroLikeOperation::new());
                    let negated = output_cotangent.unary(NegOperation::new());
                    let contribution = MaybeZero::Value(zero.binary(&negated, ComplexOperation::new()));
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
            }
        }
    },
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to extract the imaginary component of complex values elementwise. Concrete arrays
    /// compute immediately; context-carrying values apply [`ImaginaryOperation`] through their context. The output has
    /// `f32` elements for `c64` inputs and `f64` elements for `c128` inputs.
    Imaginary,
    /// Returns the elementwise imaginary component, or an error if the input is not complex valued. Refer to
    /// [`ImaginaryOperation`] for output metadata and differentiation semantics.
    imaginary,
    ImaginaryOperation,
);

impl Imaginary for Array {
    fn imaginary(&self) -> Result<Self, ProgramError> {
        // Extract components at their original precision; changing element width clears byte-strided layouts.
        match self.r#type().data_type() {
            DataType::C64 => self
                .map_elements::<ComplexNumber<f32>, f32>(self.r#type().with_element_type(DataType::F32), |value| {
                    Ok(value.im)
                }),
            DataType::C128 => self
                .map_elements::<ComplexNumber<f64>, f64>(self.r#type().with_element_type(DataType::F64), |value| {
                    Ok(value.im)
                }),
            other => {
                Err(TypeError::invalid(format!("cannot extract the imaginary part of a value of data type `{other}`"))
                    .into())
            }
        }
    }
}

/// Maps a real part element [`DataType`] to the complex [`DataType`] it constructs (i.e., `f32 → c64` and
/// `f64 → c128`), reporting a [`TypeError`] under `operation_name`'s name for any other part data type.
fn part_to_complex_data_type(part: DataType, operation_name: &'static str) -> Result<DataType, TypeError> {
    match part {
        DataType::F32 => Ok(DataType::C64),
        DataType::F64 => Ok(DataType::C128),
        other => Err(TypeError::invalid(format!("`{operation_name}` requires `f32` or `f64` parts but got `{other}`"))),
    }
}

/// Maps a complex element [`DataType`] to the [`DataType`] of its real and imaginary parts (i.e., `c64 → f32` and
/// `c128 → f64`), reporting a [`TypeError`] under `operation_name`'s name for non-complex input data types.
fn complex_to_part_data_type(complex: DataType, operation_name: &'static str) -> Result<DataType, TypeError> {
    match complex {
        DataType::C64 => Ok(DataType::F32),
        DataType::C128 => Ok(DataType::F64),
        other => Err(TypeError::invalid(format!("`{operation_name}` requires a complex input but got `{other}`"))),
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, Layout, StridedLayout};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::programs::{EmptyRegionDriver, Operation};

    use super::*;

    #[test]
    fn test_complex_type_inference() {
        check_operation_type_inference!(
            operation = ComplexOperation::<DataType>::new(),
            cases = [
                {
                    input_types = [DataType::F32, DataType::F32],
                    output_types = [DataType::C64],
                },
                {
                    input_types = [DataType::F64, DataType::F64],
                    output_types = [DataType::C128],
                },
                {
                    input_types = [DataType::F32, DataType::F64],
                    error = "`complex` requires identical part types but got `f32` and `f64`",
                },
                {
                    input_types = [DataType::I32, DataType::I32],
                    error = "`complex` requires `f32` or `f64` parts but got `i32`",
                },
            ],
        );
    }

    #[test]
    fn test_complex_type_inference_layout() {
        // Byte strides are cleared when the element width changes.
        let input_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(
            Operation::infer_output_types(
                &ComplexOperation::<ArrayType>::new(),
                &[input_type.clone(), input_type],
                &[],
            ),
            Ok(vec![ArrayType::new_static(DataType::C64, [2])]),
        );
    }

    #[test]
    fn test_complex_interpretation() {
        assert_eq!(
            ComplexOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.5f32).unwrap(), Array::scalar(-2.0f32).unwrap()],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(1.5f32, -2.0f32)).unwrap()]),
        );
    }

    #[test]
    fn test_complex_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ComplexOperation::new(),
            inputs = [Array::scalar(1.5f64).unwrap(), Array::scalar(-2.0f64).unwrap()],
            expected = Array::scalar(ComplexNumber::new(1.5f64, -2.0f64)).unwrap(),
        );
    }

    #[test]
    fn test_complex_batching() {
        check_operation_batching!(
            @exact,
            operation = ComplexOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.5f64, 0.5f64]).unwrap()),
                    (@mapped(axis = 0), Array::vector(vec![-2.0f64, 1.0f64]).unwrap()),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![
                        ComplexNumber::new(1.5f64, -2.0f64),
                        ComplexNumber::new(0.5f64, 1.0f64),
                    ]).unwrap()
                )],
            }],
        );
    }

    #[test]
    fn test_complex_differentiation() {
        // Construction: d(complex(re, im)) = complex(dre, dim), including the mixed case where one part tangent is a
        // structural zero that must be materialized to keep the staged `complex` arity.
        let (primal, tangent) = differentiate_at((Array::scalar(1.5f64).unwrap(), Array::scalar(-2.0f64).unwrap()))
            .jvp((Array::scalar(0.25f64).unwrap(), Array::scalar(4.0f64).unwrap()), |(real, imaginary)| {
                real.complex(&imaginary)
            })
            .unwrap();
        assert_eq!(primal, Array::scalar(ComplexNumber::new(1.5f64, -2.0f64)).unwrap());
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(0.25f64, 4.0f64)).unwrap());
        let (_, tangent) = differentiate_at((Array::scalar(1.5f64).unwrap(), Array::scalar(-2.0f64).unwrap()))
            .jvp((Array::scalar(0.25f64).unwrap(), Array::scalar(4.0f64).unwrap()), |(real, imaginary)| {
                let constant = imaginary.context().lift(Array::scalar(0.0f64).unwrap())?;
                real.complex(&constant)
            })
            .unwrap();
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(0.25f64, 0.0f64)).unwrap());
    }

    #[test]
    fn test_complex_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ComplexOperation::new(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, -4.0f64)).unwrap()],
                input_cotangents = [Array::scalar(3.0f64).unwrap(), Array::scalar(4.0f64).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_complex() {
        // Construction retains both components at each supported precision.
        assert_eq!(
            Array::vector(vec![1f32, 2.0]).unwrap().complex(&Array::vector(vec![3f32, -4.0]).unwrap()),
            Ok(Array::vector(vec![ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![1f64, 2.0]).unwrap().complex(&Array::vector(vec![3f64, -4.0]).unwrap()),
            Ok(Array::vector(vec![ComplexNumber::new(1f64, 3.0), ComplexNumber::new(2f64, -4.0)]).unwrap()),
        );

        // Part types must agree, and integer parts cannot represent a complex array.
        assert_eq!(
            Array::scalar(1f32).unwrap().complex(&Array::scalar(2f64).unwrap()),
            Err(TypeError::invalid("`complex` requires identical part types but got `f32[]` and `f64[]`").into()),
        );
        assert_eq!(
            Array::vector(vec![1f32]).unwrap().complex(&Array::vector(vec![2f32, 3.0]).unwrap()),
            Err(TypeError::invalid("`complex` requires identical part types but got `f32[1]` and `f32[2]`").into()),
        );
        assert_eq!(
            Array::scalar(1i32).unwrap().complex(&Array::scalar(2i32).unwrap()),
            Err(TypeError::invalid("cannot construct a complex value from parts of data type `i32`").into()),
        );
    }

    #[test]
    fn test_array_complex_layout() {
        // Byte strides are cleared when the output element width changes.
        let input_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let input = Array::from_elements(input_type, &[1f32, 2.0]).unwrap();
        assert_eq!(
            input.complex(&input),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::C64, [2]),
                &[ComplexNumber::new(1f32, 1.0), ComplexNumber::new(2f32, 2.0)],
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_conjugate_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = ConjugateOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::C128],
                    output_data_types = [DataType::C128],
                },
                {
                    input_data_types = [DataType::F64],
                    error = "`conjugate` requires a complex input but got `f64`",
                },
            ],
        );
    }

    #[test]
    fn test_conjugate_type_inference_layout() {
        // Byte strides are cleared when the element width changes.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        assert_eq!(
            Operation::infer_output_types(&ConjugateOperation::<ArrayType>::new(), &[input_type.clone()], &[]),
            Ok(vec![input_type]),
        );
    }

    #[test]
    fn test_conjugate_interpretation() {
        assert_eq!(
            ConjugateOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])
                    .unwrap()],
            ),
            Ok(vec![
                Array::vector(vec![ComplexNumber::new(1.5f64, 2.0f64), ComplexNumber::new(0.5f64, -1.0f64)]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_conjugate_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ConjugateOperation::new(),
            inputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0f64)).unwrap()],
            expected = Array::scalar(ComplexNumber::new(1.5f64, 2.0f64)).unwrap(),
        );
    }

    #[test]
    fn test_conjugate_batching() {
        check_operation_batching!(
            @exact,
            operation = ConjugateOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![
                        ComplexNumber::new(1.5f64, -2.0f64),
                        ComplexNumber::new(0.5f64, 1.0f64),
                    ]).unwrap()
                )],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![
                        ComplexNumber::new(1.5f64, 2.0f64),
                        ComplexNumber::new(0.5f64, -1.0f64),
                    ]).unwrap()
                )],
            }],
        );
    }

    #[test]
    fn test_conjugate_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);

        // Conjugation: d(z̄) = d̄z.
        let (primal, tangent) = differentiate_at(Array::scalar(input).unwrap())
            .jvp(Array::scalar(tangent_seed).unwrap(), |input| input.conjugate())
            .unwrap();
        assert_eq!(primal, Array::scalar(input.conj()).unwrap());
        assert_eq!(tangent, Array::scalar(tangent_seed.conj()).unwrap());
    }

    #[test]
    fn test_conjugate_differentiation_squared_magnitude() {
        // The squared magnitude is real-valued and non-holomorphic. The conjugation-free transposition pairing
        // contributes the conjugate input through each multiplication branch, giving twice the conjugate input.
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let gradient = differentiate_at(Array::scalar(input).unwrap())
            .gradient(|input| (input.clone() * input.conjugate().unwrap()).real().unwrap())
            .unwrap();
        assert_eq!(gradient, Array::scalar(input.conj() + input.conj()).unwrap());

        // Forward and reverse agree through the ℝ-linear rules: the jvp of f at tangent ż is 2·Re(z̄ · ż).
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);
        let (primal, tangent) = differentiate_at(Array::scalar(input).unwrap())
            .jvp(Array::scalar(tangent_seed).unwrap(), |input| (input.clone() * input.conjugate()?).real())
            .unwrap();
        assert_eq!(primal, Array::scalar(input.norm_sqr()).unwrap());
        assert_eq!(tangent, Array::scalar((tangent_seed * input.conj() + input * tangent_seed.conj()).re).unwrap());
    }

    #[test]
    fn test_conjugate_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ConjugateOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::C128)))],
                output_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, -4.0f64)).unwrap()],
                input_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, 4.0f64)).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_conjugate() {
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)])
                .unwrap()
                .conjugate(),
            Ok(Array::vector(vec![ComplexNumber::new(1f32, -3.0), ComplexNumber::new(2f32, 4.0)]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1f64, 3.0), ComplexNumber::new(2f64, -4.0)])
                .unwrap()
                .conjugate(),
            Ok(Array::vector(vec![ComplexNumber::new(1f64, -3.0), ComplexNumber::new(2f64, 4.0)]).unwrap()),
        );
        assert_eq!(
            Array::scalar(1f32).unwrap().conjugate(),
            Err(TypeError::invalid("cannot conjugate a value of data type `f32`").into()),
        );
    }

    #[test]
    fn test_array_conjugate_layout() {
        // Conjugation preserves byte strides because the element width is unchanged.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let input =
            Array::from_elements(input_type.clone(), &[ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)])
                .unwrap();
        assert_eq!(
            input.conjugate(),
            Ok(Array::from_elements(input_type, &[ComplexNumber::new(1f32, -3.0), ComplexNumber::new(2f32, 4.0)])
                .unwrap()),
        );
    }

    #[test]
    fn test_real_type_inference() {
        check_operation_type_inference!(
            operation = RealOperation::<DataType>::new(),
            cases = [
                {
                    input_types = [DataType::C64],
                    output_types = [DataType::F32],
                },
                {
                    input_types = [DataType::C128],
                    output_types = [DataType::F64],
                },
                {
                    input_types = [DataType::F32],
                    error = "`real` requires a complex input but got `f32`",
                },
            ],
        );
    }

    #[test]
    fn test_real_type_inference_layout() {
        // Byte strides are cleared when the element width changes.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        assert_eq!(
            Operation::infer_output_types(&RealOperation::<ArrayType>::new(), &[input_type], &[]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2])]),
        );
    }

    #[test]
    fn test_real_interpretation() {
        assert_eq!(
            RealOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])
                    .unwrap()],
            ),
            Ok(vec![Array::vector(vec![1.5f64, 0.5f64]).unwrap()]),
        );
    }

    #[test]
    fn test_real_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RealOperation::new(),
            inputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0f64)).unwrap()],
            expected = Array::scalar(1.5f64).unwrap(),
        );
    }

    #[test]
    fn test_real_batching() {
        check_operation_batching!(
            @exact,
            operation = RealOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![
                        ComplexNumber::new(1.5f64, -2.0f64),
                        ComplexNumber::new(0.5f64, 1.0f64),
                    ]).unwrap()
                )],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.5f64, 0.5f64]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_real_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);

        // Real-part extraction propagates the real component of the tangent.
        let (primal, tangent) = differentiate_at(Array::scalar(input).unwrap())
            .jvp(Array::scalar(tangent_seed).unwrap(), |input| input.real())
            .unwrap();
        assert_eq!(primal, Array::scalar(input.re).unwrap());
        assert_eq!(tangent, Array::scalar(tangent_seed.re).unwrap());
    }

    #[test]
    fn test_real_transposition() {
        check_operation_transposition!(
            @exact,
            operation = RealOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::C128)))],
                output_cotangents = [Array::scalar(3.0f64).unwrap()],
                input_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, 0.0f64)).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_real() {
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap().real(),
            Ok(Array::vector(vec![1f32, 2.0]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1f64, 3.0), ComplexNumber::new(2f64, -4.0)]).unwrap().real(),
            Ok(Array::vector(vec![1f64, 2.0]).unwrap()),
        );
        assert_eq!(
            Array::scalar(1f32).unwrap().real(),
            Err(TypeError::invalid("cannot extract the real part of a value of data type `f32`").into()),
        );
    }

    #[test]
    fn test_array_real_layout() {
        // Extraction clears byte strides because the output elements are narrower.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let input =
            Array::from_elements(input_type, &[ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap();
        assert_eq!(
            input.real(),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1f32, 2.0]).unwrap()),
        );
    }

    #[test]
    fn test_imaginary_type_inference() {
        check_operation_type_inference!(
            operation = ImaginaryOperation::<DataType>::new(),
            cases = [
                {
                    input_types = [DataType::C64],
                    output_types = [DataType::F32],
                },
                {
                    input_types = [DataType::C128],
                    output_types = [DataType::F64],
                },
                {
                    input_types = [DataType::Boolean],
                    error = "`imaginary` requires a complex input but got `bool`",
                },
            ],
        );
    }

    #[test]
    fn test_imaginary_type_inference_layout() {
        // Byte strides are cleared when the element width changes.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        assert_eq!(
            Operation::infer_output_types(&ImaginaryOperation::<ArrayType>::new(), &[input_type], &[]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2])]),
        );
    }

    #[test]
    fn test_imaginary_interpretation() {
        assert_eq!(
            ImaginaryOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])
                    .unwrap()],
            ),
            Ok(vec![Array::vector(vec![-2.0f64, 1.0f64]).unwrap()]),
        );
    }

    #[test]
    fn test_imaginary_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ImaginaryOperation::new(),
            inputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0f64)).unwrap()],
            expected = Array::scalar(-2.0f64).unwrap(),
        );
    }

    #[test]
    fn test_imaginary_batching() {
        check_operation_batching!(
            @exact,
            operation = ImaginaryOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![
                        ComplexNumber::new(1.5f64, -2.0f64),
                        ComplexNumber::new(0.5f64, 1.0f64),
                    ]).unwrap()
                )],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-2.0f64, 1.0f64]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_imaginary_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);

        let (primal, tangent) = differentiate_at(Array::scalar(input).unwrap())
            .jvp(Array::scalar(tangent_seed).unwrap(), |input| input.imaginary())
            .unwrap();
        assert_eq!(primal, Array::scalar(input.im).unwrap());
        assert_eq!(tangent, Array::scalar(tangent_seed.im).unwrap());
    }

    #[test]
    fn test_imaginary_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ImaginaryOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::C128)))],
                output_cotangents = [Array::scalar(3.0f64).unwrap()],
                input_cotangents = [Array::scalar(ComplexNumber::new(0.0f64, -3.0f64)).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_imaginary() {
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)])
                .unwrap()
                .imaginary(),
            Ok(Array::vector(vec![3f32, -4.0]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1f64, 3.0), ComplexNumber::new(2f64, -4.0)])
                .unwrap()
                .imaginary(),
            Ok(Array::vector(vec![3f64, -4.0]).unwrap()),
        );
        assert_eq!(
            Array::scalar(1f32).unwrap().imaginary(),
            Err(TypeError::invalid("cannot extract the imaginary part of a value of data type `f32`").into()),
        );
    }

    #[test]
    fn test_array_imaginary_layout() {
        // Extraction clears byte strides because the output elements are narrower.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let input =
            Array::from_elements(input_type, &[ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap();
        assert_eq!(
            input.imaginary(),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[3f32, -4.0]).unwrap()),
        );
    }
}
