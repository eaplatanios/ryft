//! Operations that construct complex values from their real and imaginary parts, conjugate them, and extract their
//! parts. Each operation is defined by an [`Operation`](crate::Operation) type (e.g., [`ComplexOperation`]) together
//! with a value capability trait (e.g., [`Complex`]) whose functions apply it to eager [`Array`]s and traced values
//! alike, so the same code executes immediately or records into a program depending on the value it runs on:
//!
//!   - [`Complex`] combines a real and an imaginary part into one complex value (i.e., `(re, im) ↦ re + im·i`). The
//!     parts must have identical types, so construction neither broadcasts nor promotes them.
//!   - [`Conjugate`] negates the imaginary part of a complex value (i.e., `z ↦ z̄`), preserving its type.
//!   - [`Real`] and [`Imaginary`] extract the real and imaginary parts of a complex value
//!     (i.e., `z ↦ Re(z)` and `z ↦ Im(z)`).
//!
//! Construction and extraction preserve shape, sharding, and memory placement, but clear byte-strided layouts because
//! they change the element width, whereas conjugation preserves the entire type. Batching maps each operation
//! elementwise over the batch axis.
//!
//! Every operation is ℝ-linear, so its tangent applies the same operation to the input tangents (e.g., the tangent
//! of `z̄` is the conjugate of the tangent of `z`). Transposition pairs complex values bilinearly (i.e., without
//! conjugation). Under that pairing, conjugation is self-adjoint, the transpose of construction maps an output
//! cotangent `ȳ` to `(Re(ȳ), -Im(ȳ))`, and the transposes of real-part and imaginary-part extraction map a cotangent
//! `t` to `complex(t, 0)` and `complex(0, -t)`, respectively.
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

/// Canonical operation name for [`ComplexOperation`].
pub const COMPLEX_OPERATION_NAME: &str = "complex";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that constructs a complex value from its real and imaginary parts (i.e., `(re, im) ↦ re + im·i`,
    /// with `(f32, f32) ↦ c64` and `(f64, f64) ↦ c128`). This operation is the Ryft analogue of JAX's
    /// [`lax.complex`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.complex.html) and the inverse of the
    /// [`RealOperation`] and [`ImaginaryOperation`] pair. The two parts must have identical types. Array shape,
    /// sharding, and memory placement are preserved. Note however that byte-strided layouts are cleared because
    /// complex elements are wider.
    ///
    /// As a map from the pair of real parts, the operation is linear, and its transpose is the `ȳ ↦ (real(ȳ),
    /// imaginary(-ȳ))` pair under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses over
    /// complex types.
    ComplexOperation,
    COMPLEX_OPERATION_NAME,
    Complex,
    complex,
    infer_data_types = |input_types: &[DataType]| {
        if input_types[0] != input_types[1] {
            return Err(TypeError::invalid(format!(
                "`{}` requires identical part types but got `{}` and `{}`",
                COMPLEX_OPERATION_NAME,
                input_types[0],
                input_types[1],
            )));
        }
        let data_type = match input_types[0] {
            DataType::F32 => DataType::C64,
            DataType::F64 => DataType::C128,
            other => {
                return Err(TypeError::invalid(format!(
                    "`{}` requires `f32` or `f64` parts but got `{}`",
                    COMPLEX_OPERATION_NAME,
                    other,
                )));
            }
        };
        Ok(vec![data_type])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        if input_types[0] != input_types[1] {
            return Err(TypeError::invalid(format!(
                "`{}` requires identical part types but got `{}` and `{}`",
                COMPLEX_OPERATION_NAME,
                input_types[0],
                input_types[1],
            )));
        }
        let data_type = match input_types[0].data_type() {
            DataType::F32 => DataType::C64,
            DataType::F64 => DataType::C128,
            other => {
                return Err(TypeError::invalid(format!(
                    "`{}` requires `f32` or `f64` parts but got `{}`",
                    COMPLEX_OPERATION_NAME,
                    other,
                )));
            }
        };
        Ok(vec![input_types[0].with_element_type(data_type)])
    },
);

impl_differentiable_operation! {
    <T> ComplexOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Complex,
        C: Zero<C::Value>,
    {
        |_operation, context, _driver, inputs| {
            // Complex construction is linear in its two real parts: `d(complex(re, im)) = complex(dre, dim)`. When both
            // part tangents are structural zeros the output tangent stays a symbolic zero of the complex output type.
            // When only one is, the missing part is materialized as a real zero through the context so the staged
            // `complex` keeps its two-part arity.
            check_count!("input", inputs, 2, ProgramError);
            let real = &inputs[0];
            let imaginary = &inputs[1];
            let primal = real.primal().complex(imaginary.primal())?;
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
            // Under the bilinear (i.e., conjugation-free) pairing used for complex transposition, the
            // transpose of `(re, im) ↦ re + im·i` maps the output cotangent `ȳ` to `(real(ȳ), imaginary(-ȳ))`:
            // pairing `Re(ȳ · (re + im·i))` against `(re, im)` picks out the real part of `ȳ` for `re` and the
            // _negated_ imaginary part for `im`. Like the `Add` rule, a known part contributes an additive constant
            // whose adjoint is dropped at the pullback output boundary.
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(output_cotangent) => {
                    let contribution = MaybeZero::Value(output_cotangent.unary(RealOperation::new()));
                    accumulators[0].accumulate(context, contribution)?;
                    let contribution = MaybeZero::Value(
                        output_cotangent.unary(NegOperation::new()).unary(ImaginaryOperation::new()),
                    );
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
    /// immediately while context-carrying values apply [`ComplexOperation`] through their context. The parts must have
    /// identical types with `f32` or `f64` elements as construction neither promotes nor broadcasts them.
    Complex,
    /// Constructs `self + imaginary·i` elementwise, producing `c64` from `f32` parts or `c128` from `f64` parts.
    /// Returns an error if the parts have different types or unsupported element types.
    ///
    /// # Parameters
    ///
    ///   - `imaginary`: Imaginary part, with the same type as this real part.
    complex(imaginary),
    ComplexOperation,
);

impl Complex for Array {
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        // Construction requires identical part types and combines their values without promotion or broadcasting.
        if self.r#type() != imaginary.r#type() {
            return Err(TypeError::invalid(format!(
                "`{}` requires identical part types but got `{}` and `{}`",
                COMPLEX_OPERATION_NAME,
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
    /// transposition uses over complex types, it is self-adjoint (i.e., the transpose of `z ↦ z̄` is `ȳ ↦ ȳ̄`).
    ConjugateOperation,
    CONJUGATE_OPERATION_NAME,
    Conjugate,
    conjugate,
    infer_data_types = |input_types: &[DataType]| {
        match input_types[0] {
            DataType::C64 | DataType::C128 => Ok(vec![input_types[0]]),
            other => Err(TypeError::invalid(format!(
                "`{}` requires a complex input but got `{}`",
                CONJUGATE_OPERATION_NAME,
                other,
            ))),
        }
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

            // Conjugation is ℝ-linear (but not ℂ-linear): `d(z̄) = d̄z`. A structural zero tangent stays symbolic.
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
            // Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses over complex
            // types, conjugation is self-adjoint: pairing `Re(ȳ · z̄)` against `z` shows that the transpose of
            // `z ↦ z̄` is `ȳ ↦ ȳ̄`.
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
    /// Represents the ability to conjugate complex values elementwise. Concrete arrays compute immediately while
    /// context-carrying values apply [`ConjugateOperation`] through their context. Conjugation preserves the input
    /// type and negates each imaginary component.
    Conjugate,
    /// Returns the elementwise complex conjugate, or an error if the input does not have `c64` or `c128` elements.
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
    /// [`Operation`] that extracts the elementwise real part of one complex value (i.e., `z ↦ Re(z)`, with `c64 ↦ f32`
    /// and `c128 ↦ f64`) while preserving shape, sharding, and memory placement. Note however that byte-strided layouts
    /// are cleared because the output elements are narrower. This operation is the Ryft analogue of JAX's
    /// [`lax.real`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.real.html).
    ///
    /// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
    /// over complex types, the transpose of `z ↦ Re(z)` is `ȳ ↦ complex(ȳ, 0)`.
    RealOperation,
    REAL_OPERATION_NAME,
    Real,
    real,
    infer_data_types = |input_types: &[DataType]| {
        let data_type = match input_types[0] {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => {
                return Err(TypeError::invalid(format!(
                    "`{}` requires a complex input but got `{}`",
                    REAL_OPERATION_NAME,
                    other,
                )));
            }
        };
        Ok(vec![data_type])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        let data_type = match input_types[0].data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => {
                return Err(TypeError::invalid(format!(
                    "`{}` requires a complex input but got `{}`",
                    REAL_OPERATION_NAME,
                    other,
                )));
            }
        };
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
            // but is retyped to the real output type.
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
            // Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses over complex types,
            // pairing `t · Re(z)` against `z` shows that the transpose of `z ↦ Re(z)` is `t ↦ complex(t, 0)`, injecting
            // the real cotangent with a zero imaginary part.
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
    /// Represents the ability to extract the real component of complex values elementwise. Concrete arrays compute
    /// immediately while context-carrying values apply [`RealOperation`] through their context. The output has `f32`
    /// elements for `c64` inputs and `f64` elements for `c128` inputs.
    Real,
    /// Returns the elementwise real component, or an error if the input is not complex valued.
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
    /// `c64 ↦ f32` and `c128 ↦ f64`) while preserving shape, sharding, and memory placement. Note however that
    /// byte-strided layouts are cleared because the output elements are narrower. This operation is the Ryft analogue
    /// of JAX's [`lax.imag`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.imag.html).
    ///
    /// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
    /// over complex types, the transpose of `z ↦ Im(z)` is `ȳ ↦ complex(0, -ȳ)`.
    ImaginaryOperation,
    IMAGINARY_OPERATION_NAME,
    Imaginary,
    imaginary,
    infer_data_types = |input_types: &[DataType]| {
        let data_type = match input_types[0] {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => {
                return Err(TypeError::invalid(format!(
                    "`{}` requires a complex input but got `{}`",
                    IMAGINARY_OPERATION_NAME,
                    other,
                )));
            }
        };
        Ok(vec![data_type])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        let data_type = match input_types[0].data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => {
                return Err(TypeError::invalid(format!(
                    "`{}` requires a complex input but got `{}`",
                    IMAGINARY_OPERATION_NAME,
                    other,
                )));
            }
        };
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
            // but is retyped to the real output type.
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
            // Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses over complex types,
            // pairing `t · Im(z)` against `z` shows that the transpose of `z ↦ Im(z)` is `t ↦ complex(0, -t)`,
            // injecting the _negated_ real cotangent as the imaginary part.
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
    /// Represents the ability to extract the imaginary component of complex values elementwise. Concrete arrays compute
    /// immediately while context-carrying values apply [`ImaginaryOperation`] through their context. The output has
    /// `f32` elements for `c64` inputs and `f64` elements for `c128` inputs.
    Imaginary,
    /// Returns the elementwise imaginary component, or an error if the input is not complex valued.
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

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, Layout, Memory, StridedLayout};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DifferentiableOperation, DifferentiationContext, differentiate_at};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::{EmptyRegionDriver, Operation};

    use super::*;

    #[test]
    fn test_complex() {
        assert_eq!(ComplexOperation::<ArrayType>::new().to_string(), "complex");
    }

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

        // Array parts must have identical types, so construction neither broadcasts nor promotes them.
        check_operation_type_inference!(
            operation = ComplexOperation::<ArrayType>::new(),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [1]), ArrayType::new_static(DataType::F32, [2])],
                error = "`complex` requires identical part types but got `f32[1]` and `f32[2]`",
            }],
        );
    }

    #[test]
    fn test_complex_type_inference_layout() {
        // Construction preserves memory placement but clears byte strides because complex elements are wider.
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            Operation::infer_output_types(
                &ComplexOperation::<ArrayType>::new(),
                &[input_type.clone(), input_type],
                &[],
            ),
            Ok(vec![ArrayType::new_static(DataType::C64, [2]).with_memory(Memory::Host { pinned: true })]),
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
            Ok(vec![Array::scalar(ComplexNumber::new(1.5f32, -2.0)).unwrap()]),
        );
    }

    #[test]
    fn test_complex_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ComplexOperation::new(),
            inputs = [Array::scalar(1.5f64).unwrap(), Array::scalar(-2.0f64).unwrap()],
            expected = Array::scalar(ComplexNumber::new(1.5f64, -2.0)).unwrap(),
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
                    (@mapped(axis = 0), Array::vector(vec![1.5f64, 0.5]).unwrap()),
                    (@mapped(axis = 0), Array::vector(vec![-2.0f64, 1.0]).unwrap()),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_complex_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ComplexOperation::new(),
            cases = [{
                primals = [Array::scalar(1.5f64).unwrap(), Array::scalar(-2.0f64).unwrap()],
                tangents = [Array::scalar(0.25f64).unwrap(), Array::scalar(4.0f64).unwrap()],
                primal_outputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0)).unwrap()],
                tangent_outputs = [Array::scalar(ComplexNumber::new(0.25f64, 4.0)).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:c128[] = complex %0 %1
                        %5:c128[] = complex %2 %3
                    in (%4, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_complex_differentiation_zero_tangents() {
        // A structural zero part tangent is materialized as a real zero so that the tangent keeps both parts.
        assert_eq!(
            differentiate_at((Array::scalar(1.5f64).unwrap(), Array::scalar(-2.0f64).unwrap()))
                .jvp((Array::scalar(0.25f64).unwrap(), Array::scalar(4.0f64).unwrap()), |(real, imaginary)| real
                    .complex(&imaginary.context().lift(Array::scalar(0.0f64).unwrap())?),),
            Ok((
                Array::scalar(ComplexNumber::new(1.5f64, 0.0)).unwrap(),
                Array::scalar(ComplexNumber::new(0.25f64, 0.0)).unwrap(),
            )),
        );

        // When both part tangents are structural zeros, the output tangent stays a symbolic zero of the complex type.
        let outputs = ComplexOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(1.5f64).unwrap()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(-2.0f64).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(ComplexNumber::new(1.5f64, -2.0)).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::C128))
        );
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
                output_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, -4.0)).unwrap()],
                input_cotangents = [Array::scalar(3.0f64).unwrap(), Array::scalar(4.0f64).unwrap()],
                pullback = indoc! {"
                    lambda %0:c128[] .
                    let %1:f64[] = real %0
                        %2:c128[] = neg %0
                        %3:f64[] = imaginary %2
                    in (%1, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_complex_for_array() {
        // Construction retains both parts at each supported precision.
        assert_eq!(
            Array::vector(vec![1f32, 2.0]).unwrap().complex(&Array::vector(vec![3f32, -4.0]).unwrap()),
            Ok(Array::vector(vec![ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![1f64, 2.0]).unwrap().complex(&Array::vector(vec![3f64, -4.0]).unwrap()),
            Ok(Array::vector(vec![ComplexNumber::new(1f64, 3.0), ComplexNumber::new(2f64, -4.0)]).unwrap()),
        );

        // Part types must agree exactly, and integer parts cannot represent a complex array.
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
    fn test_complex_for_array_layout() {
        // Construction traverses strided parts and produces a densely laid out complex array.
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
    fn test_conjugate() {
        assert_eq!(ConjugateOperation::<ArrayType>::new().to_string(), "conjugate");
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
        // Conjugation preserves byte strides because the element width is unchanged.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![16])));
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
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![ComplexNumber::new(1.5f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap()]),
        );
    }

    #[test]
    fn test_conjugate_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ConjugateOperation::new(),
            inputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0)).unwrap()],
            expected = Array::scalar(ComplexNumber::new(1.5f64, 2.0)).unwrap(),
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
                    Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap(),
                )],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![ComplexNumber::new(1.5f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_conjugate_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ConjugateOperation::new(),
            cases = [{
                primals = [Array::scalar(ComplexNumber::new(0.7f64, -0.3)).unwrap()],
                tangents = [Array::scalar(ComplexNumber::new(0.5f64, 2.0)).unwrap()],
                primal_outputs = [Array::scalar(ComplexNumber::new(0.7f64, 0.3)).unwrap()],
                tangent_outputs = [Array::scalar(ComplexNumber::new(0.5f64, -2.0)).unwrap()],
                jvp = indoc! {"
                    lambda %0:c128[], %1:c128[] .
                    let %2:c128[] = conjugate %0
                        %3:c128[] = conjugate %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_conjugate_differentiation_squared_magnitude() {
        // The squared magnitude is real valued and non-holomorphic. The conjugation-free transposition pairing
        // contributes the conjugate input through each multiplication branch, giving twice the conjugate input.
        let input = ComplexNumber::new(0.7f64, -0.3);
        assert_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .gradient(|input| (input.clone() * input.conjugate().unwrap()).real().unwrap()),
            Ok(Array::scalar(input.conj() + input.conj()).unwrap()),
        );

        // Forward and reverse mode agree through the ℝ-linear rules: the tangent at `ż` is `2·Re(z̄·ż)`.
        let tangent = ComplexNumber::new(0.5f64, 2.0);
        assert_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .jvp(Array::scalar(tangent).unwrap(), |input| (input.clone() * input.conjugate()?).real()),
            Ok((
                Array::scalar(input.norm_sqr()).unwrap(),
                Array::scalar((tangent * input.conj() + input * tangent.conj()).re).unwrap(),
            )),
        );
    }

    #[test]
    fn test_conjugate_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ConjugateOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::C128)))],
                output_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, -4.0)).unwrap()],
                input_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, 4.0)).unwrap()],
                pullback = indoc! {"
                    lambda %0:c128[] .
                    let %1:c128[] = conjugate %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_conjugate_for_array() {
        // Conjugation negates the imaginary part at each supported precision and rejects real inputs.
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
    fn test_conjugate_for_array_layout() {
        // Conjugation preserves byte strides because the element width is unchanged.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![16])));
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
    fn test_real() {
        assert_eq!(RealOperation::<ArrayType>::new().to_string(), "real");
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
        // Extraction preserves memory placement but clears byte strides because real parts are narrower.
        let input_type = ArrayType::new_static(DataType::C64, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            Operation::infer_output_types(&RealOperation::<ArrayType>::new(), &[input_type], &[]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true })]),
        );
    }

    #[test]
    fn test_real_interpretation() {
        assert_eq!(
            RealOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![1.5f64, 0.5]).unwrap()]),
        );
    }

    #[test]
    fn test_real_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RealOperation::new(),
            inputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0)).unwrap()],
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
                    Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap(),
                )],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.5f64, 0.5]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_real_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RealOperation::new(),
            cases = [{
                primals = [Array::scalar(ComplexNumber::new(0.7f64, -0.3)).unwrap()],
                tangents = [Array::scalar(ComplexNumber::new(0.5f64, 2.0)).unwrap()],
                primal_outputs = [Array::scalar(0.7f64).unwrap()],
                tangent_outputs = [Array::scalar(0.5f64).unwrap()],
                jvp = indoc! {"
                    lambda %0:c128[], %1:c128[] .
                    let %2:f64[] = real %0
                        %3:f64[] = real %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_real_differentiation_zero_tangent() {
        // A structural zero tangent stays symbolic and is retyped to the real output type.
        let outputs = RealOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(Array::scalar(ComplexNumber::new(0.7f64, -0.3)).unwrap())
                    .unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(0.7f64).unwrap());
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_real_transposition() {
        check_operation_transposition!(
            @exact,
            operation = RealOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::C128)))],
                output_cotangents = [Array::scalar(3.0f64).unwrap()],
                input_cotangents = [Array::scalar(ComplexNumber::new(3.0f64, 0.0)).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = zero_like %0
                        %2:c128[] = complex %0 %1
                    in (%2)
                "},
            }],
        );
    }

    #[test]
    fn test_real_for_array() {
        // Extraction keeps each supported precision and rejects real inputs.
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
    fn test_real_for_array_layout() {
        // Extraction traverses strided inputs and produces a densely laid out real array.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![16])));
        let input =
            Array::from_elements(input_type, &[ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap();
        assert_eq!(
            input.real(),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1f32, 2.0]).unwrap()),
        );
    }

    #[test]
    fn test_imaginary() {
        assert_eq!(ImaginaryOperation::<ArrayType>::new().to_string(), "imaginary");
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
        // Extraction preserves memory placement but clears byte strides because imaginary parts are narrower.
        let input_type = ArrayType::new_static(DataType::C64, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            Operation::infer_output_types(&ImaginaryOperation::<ArrayType>::new(), &[input_type], &[]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true })]),
        );
    }

    #[test]
    fn test_imaginary_interpretation() {
        assert_eq!(
            ImaginaryOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![-2.0f64, 1.0]).unwrap()]),
        );
    }

    #[test]
    fn test_imaginary_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ImaginaryOperation::new(),
            inputs = [Array::scalar(ComplexNumber::new(1.5f64, -2.0)).unwrap()],
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
                    Array::vector(vec![ComplexNumber::new(1.5f64, -2.0), ComplexNumber::new(0.5f64, 1.0)]).unwrap(),
                )],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-2.0f64, 1.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_imaginary_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ImaginaryOperation::new(),
            cases = [{
                primals = [Array::scalar(ComplexNumber::new(0.7f64, -0.3)).unwrap()],
                tangents = [Array::scalar(ComplexNumber::new(0.5f64, 2.0)).unwrap()],
                primal_outputs = [Array::scalar(-0.3f64).unwrap()],
                tangent_outputs = [Array::scalar(2.0f64).unwrap()],
                jvp = indoc! {"
                    lambda %0:c128[], %1:c128[] .
                    let %2:f64[] = imaginary %0
                        %3:f64[] = imaginary %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_imaginary_differentiation_zero_tangent() {
        // A structural zero tangent stays symbolic and is retyped to the real output type.
        let outputs = ImaginaryOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(Array::scalar(ComplexNumber::new(0.7f64, -0.3)).unwrap())
                    .unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(-0.3f64).unwrap());
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_imaginary_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ImaginaryOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::C128)))],
                output_cotangents = [Array::scalar(3.0f64).unwrap()],
                input_cotangents = [Array::scalar(ComplexNumber::new(0.0f64, -3.0)).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = zero_like %0
                        %2:f64[] = neg %0
                        %3:c128[] = complex %1 %2
                    in (%3)
                "},
            }],
        );
    }

    #[test]
    fn test_imaginary_for_array() {
        // Extraction keeps each supported precision and rejects real inputs.
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
    fn test_imaginary_for_array_layout() {
        // Extraction traverses strided inputs and produces a densely laid out real array.
        let input_type =
            ArrayType::new_static(DataType::C64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![16])));
        let input =
            Array::from_elements(input_type, &[ComplexNumber::new(1f32, 3.0), ComplexNumber::new(2f32, -4.0)]).unwrap();
        assert_eq!(
            input.imaginary(),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[3f32, -4.0]).unwrap()),
        );
    }
}
