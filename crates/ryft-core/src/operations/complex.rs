use crate::arrays::{ArrayType, DataType};
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::constants::zero::Zero;
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::math::neg::NegOperation;
use crate::programs::{MaybeZero, Type, TypeError, Typed};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ComplexOperation`].
pub const COMPLEX_OPERATION_NAME: &str = "complex";

/// Canonical operation name for [`ConjugateOperation`].
pub const CONJUGATE_OPERATION_NAME: &str = "conjugate";

/// Canonical operation name for [`RealOperation`].
pub const REAL_OPERATION_NAME: &str = "real";

/// Canonical operation name for [`ImaginaryOperation`].
pub const IMAGINARY_OPERATION_NAME: &str = "imaginary";

/// Maps a real part element [`DataType`] to the complex [`DataType`] it constructs (i.e., `f32 → c64` and
/// `f64 → c128`), reporting a [`TypeError`] under `op`'s name for any other part data type.
fn part_to_complex_data_type(part: DataType, op: &'static str) -> Result<DataType, TypeError> {
    match part {
        DataType::F32 => Ok(DataType::C64),
        DataType::F64 => Ok(DataType::C128),
        other => Err(TypeError::invalid(format!("'{op}' requires f32 or f64 parts but got {other}"))),
    }
}

/// Maps a complex element [`DataType`] to the [`DataType`] of its real and imaginary parts (i.e., `c64 → f32` and
/// `c128 → f64`), reporting a [`TypeError`] under `op`'s name for non-complex operand data types.
fn complex_to_part_data_type(complex: DataType, op: &'static str) -> Result<DataType, TypeError> {
    match complex {
        DataType::C64 => Ok(DataType::F32),
        DataType::C128 => Ok(DataType::F64),
        other => Err(TypeError::invalid(format!("'{op}' requires a complex operand but got {other}"))),
    }
}

define_elementwise_operation!(
    @binary
    /// [`Operation`] that constructs a complex value from its real and imaginary parts (i.e., `(re, im) ↦ re + im·i`,
    /// with `(f32, f32) ↦ c64` and `(f64, f64) ↦ c128`). This is the analogue of
    /// [JAX's `lax.complex`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.complex.html) and the inverse of the
    /// [`RealOperation`]/[`ImaginaryOperation`] pair. The two parts must have identical types.
    ///
    /// As a map from the pair of real parts, the operation is linear, and its transpose is the
    /// `ȳ ↦ (real(ȳ), imaginary(-ȳ))` pair under the bilinear (i.e., conjugation-free) pairing that Ryft's
    /// transposition uses over complex types.
    ComplexOperation, COMPLEX_OPERATION_NAME,
    Complex, complex,
    infer_data_types = |input_types: &[DataType]| {
        if input_types[0] != input_types[1] {
            return Err(TypeError::invalid(format!(
                    "'{}' requires identical part types but got {} and {}",
                    COMPLEX_OPERATION_NAME,
                    input_types[0],
                input_types[1],
                )));
        }
        Ok(vec![part_to_complex_data_type(input_types[0], COMPLEX_OPERATION_NAME)?])
    },
    infer_array_types = |input_types: &[ArrayType]| {
        if input_types[0] != input_types[1] {
            return Err(TypeError::invalid(format!(
                    "'{}' requires identical part types but got {} and {}",
                    COMPLEX_OPERATION_NAME,
                    input_types[0],
                    input_types[1],
                )));
        }
        let data_type = part_to_complex_data_type(input_types[0].data_type(), COMPLEX_OPERATION_NAME)?;
        Ok(vec![ArrayType { data_type, ..input_types[0].clone() }])
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
                (MaybeZero::Zero(_), MaybeZero::Zero(_)) => MaybeZero::Zero(primal.r#type().tangent()),
                (real_tangent, imaginary_tangent) => MaybeZero::Value(
                    real_tangent
                        .clone()
                        .materialize(context)?
                        .complex(&imaginary_tangent.clone().materialize(context)?)?,
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
        |_operation, _context, _driver, inputs, outputs| {
            // Transpose rule for the linear [`ComplexOperation`]. Under the bilinear (i.e., conjugation-free) pairing
            // that Ryft's transposition uses over complex types, the transpose of `(re, im) ↦ re + im·i` maps the output
            // cotangent `ȳ` to the part cotangents `(real(ȳ), imaginary(-ȳ))`: pairing `Re(ȳ · (re + im·i))` against
            // `(re, im)` picks out the real part of `ȳ` for `re` and the *negated* imaginary part for `im`. Like the
            // `Add` rule, known-ness is ignored — a known part contributes an additive constant whose adjoint is dropped
            // at the pullback output boundary.
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            Ok(match &outputs[0] {
                MaybeZero::Zero(_) =>
                    inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect(),
                MaybeZero::Value(output_cotangent) => vec![
                    MaybeZero::Value(output_cotangent.unary(RealOperation::new())),
                    MaybeZero::Value(output_cotangent.unary(NegOperation::new()).unary(ImaginaryOperation::new())),
                ],
            })
        }
    },
}

define_elementwise_capability!(
    @binary
    /// Value-level capability that constructs a complex value from this value as the real part and `imaginary` as the
    /// imaginary part. [`Complex`] fills the same role for [`ComplexOperation`] that [`Sin`](crate::Sin) fills for
    /// [`SinOperation`](crate::SinOperation).
    Complex,
    /// Constructs the complex value `self + imaginary·i`, returning a [`ProgramError`] if something goes wrong (e.g.,
    /// when the parts are not both `f32` or both `f64` valued).
    complex(imaginary),
    ComplexOperation,
);

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise complex conjugate of one complex value (i.e., `z ↦ z̄`, negating
    /// the imaginary part) while preserving its type metadata. Real operands are rejected: the conjugate of a real
    /// value is the identity, so real call sites should simply not conjugate.
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
        |_operation, _context, _driver, inputs, outputs| {
            // Transpose rule for the ℝ-linear [`ConjugateOperation`]. Under the bilinear (i.e., conjugation-free)
            // pairing that Ryft's transposition uses over complex types, conjugation is self-adjoint: pairing
            // `Re(ȳ · z̄)` against `z` shows that the transpose of `z ↦ z̄` is `ȳ ↦ ȳ̄`.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            Ok(match &outputs[0] {
                MaybeZero::Zero(_) => vec![MaybeZero::Zero(inputs[0].r#type().cotangent())],
                MaybeZero::Value(output_cotangent) =>
                    vec![MaybeZero::Value(output_cotangent.unary(ConjugateOperation::new()))],
            })
        }
    },
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise complex-conjugation capability. [`Conjugate`] fills the same role for
    /// [`ConjugateOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Conjugate,
    /// Computes the elementwise complex conjugate of this value, returning a [`ProgramError`] if something goes wrong
    /// (e.g., when the value is not complex valued).
    conjugate,
    ConjugateOperation,
);

define_elementwise_operation!(
    @unary
    /// [`Operation`] that extracts the elementwise real part of one complex value (i.e., `z ↦ Re(z)`, with
    /// `c64 ↦ f32` and `c128 ↦ f64`) while preserving all other type metadata. This is the analogue of
    /// [JAX's `lax.real`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.real.html).
    ///
    /// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
    /// over complex types, the transpose of `z ↦ Re(z)` is `ȳ ↦ complex(ȳ, 0)`.
    RealOperation, REAL_OPERATION_NAME,
    Real, real,
    infer_data_types = |input_types: &[DataType]| {
        Ok(vec![complex_to_part_data_type(input_types[0], REAL_OPERATION_NAME)?])
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
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
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
        |_operation, _context, _driver, inputs, outputs| {
            // Transpose rule for the ℝ-linear [`RealOperation`]. Under the bilinear (i.e., conjugation-free) pairing
            // that Ryft's transposition uses over complex types, pairing `t · Re(z)` against `z` shows that the
            // transpose of `z ↦ Re(z)` is `t ↦ complex(t, 0)`, injecting the real cotangent with a zero imaginary part.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            Ok(match &outputs[0] {
                MaybeZero::Zero(_) => vec![MaybeZero::Zero(inputs[0].r#type().cotangent())],
                MaybeZero::Value(output_cotangent) => {
                    let zero = output_cotangent.unary(ZeroLikeOperation::new());
                    vec![MaybeZero::Value(output_cotangent.binary(&zero, ComplexOperation::new()))]
                }
            })
        }
    },
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise real-part extraction capability. [`Real`] fills the same role for [`RealOperation`] that
    /// [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Real,
    /// Extracts the elementwise real part of this value, returning a [`ProgramError`] if something goes wrong (e.g.,
    /// when the value is not complex valued).
    real,
    RealOperation,
);

define_elementwise_operation!(
    @unary
    /// [`Operation`] that extracts the elementwise imaginary part of one complex value (i.e., `z ↦ Im(z)`, with
    /// `c64 ↦ f32` and `c128 ↦ f64`) while preserving all other type metadata. This is the analogue of
    /// [JAX's `lax.imag`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.imag.html).
    ///
    /// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
    /// over complex types, the transpose of `z ↦ Im(z)` is `ȳ ↦ complex(0, -ȳ)`.
    ImaginaryOperation, IMAGINARY_OPERATION_NAME,
    Imaginary, imaginary,
    infer_data_types = |input_types: &[DataType]| {
        Ok(vec![complex_to_part_data_type(input_types[0], IMAGINARY_OPERATION_NAME)?])
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
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
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
        |_operation, _context, _driver, inputs, outputs| {
            // Transpose rule for the ℝ-linear [`ImaginaryOperation`]. Under the bilinear (i.e., conjugation-free)
            // pairing that Ryft's transposition uses over complex types, pairing `t · Im(z)` against `z` shows that the
            // transpose of `z ↦ Im(z)` is `t ↦ complex(0, -t)`, injecting the *negated* real cotangent as the imaginary
            // part.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            Ok(match &outputs[0] {
                MaybeZero::Zero(_) => vec![MaybeZero::Zero(inputs[0].r#type().cotangent())],
                MaybeZero::Value(output_cotangent) => {
                    let zero = output_cotangent.unary(ZeroLikeOperation::new());
                    let negated = output_cotangent.unary(NegOperation::new());
                    vec![MaybeZero::Value(zero.binary(&negated, ComplexOperation::new()))]
                }
            })
        }
    },
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise imaginary-part extraction capability. [`Imaginary`] fills the same role for
    /// [`ImaginaryOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Imaginary,
    /// Extracts the elementwise imaginary part of this value, returning a [`ProgramError`] if something goes wrong
    /// (e.g., when the value is not complex valued).
    imaginary,
    ImaginaryOperation,
);

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::Array;
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_type_inference;
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_complex() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &ComplexOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.5f32), Array::scalar(-2.0f32)],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(1.5f32, -2.0f32))]),
        );
    }

    #[test]
    fn test_complex_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = ComplexOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F32],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::F64, DataType::F64],
                    output_data_types = [DataType::C128],
                },
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    error = "'complex' requires identical part types but got f32 and f64",
                },
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    error = "'complex' requires f32 or f64 parts but got i32",
                },
            ],
        );
    }

    #[test]
    fn test_conjugate() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &ConjugateOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])],
            ),
            Ok(vec![Array::vector(vec![ComplexNumber::new(1.5f64, 2.0f64), ComplexNumber::new(0.5f64, -1.0f64)])]),
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
                    input_data_types = [DataType::F64],
                    error = "'conjugate' requires a complex operand but got f64",
                },
            ],
        );
    }

    #[test]
    fn test_real() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &RealOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])],
            ),
            Ok(vec![Array::vector(vec![1.5f64, 0.5f64])]),
        );
    }

    #[test]
    fn test_real_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = RealOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::F32],
                },
                {
                    input_data_types = [DataType::C128],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F32],
                    error = "'real' requires a complex operand but got f32",
                },
            ],
        );
    }

    #[test]
    fn test_imaginary() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &ImaginaryOperation::<ArrayType>::new(),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])],
            ),
            Ok(vec![Array::vector(vec![-2.0f64, 1.0f64])]),
        );
    }

    #[test]
    fn test_imaginary_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = ImaginaryOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::F32],
                },
                {
                    input_data_types = [DataType::Boolean],
                    error = "'imaginary' requires a complex operand but got bool",
                },
            ],
        );
    }

    #[test]
    fn test_complex_differentiation() {
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);

        // Conjugation: d(z̄) = d̄z.
        let (primal, tangent) =
            differentiate_at(Array::scalar(z)).jvp(Array::scalar(tangent_seed), |x| x.conjugate()).unwrap();
        assert_eq!(primal, Array::scalar(z.conj()));
        assert_eq!(tangent, Array::scalar(tangent_seed.conj()));

        // Part extraction: d(Re(z)) = Re(dz) and d(Im(z)) = Im(dz).
        let (primal, tangent) =
            differentiate_at(Array::scalar(z)).jvp(Array::scalar(tangent_seed), |x| x.real()).unwrap();
        assert_eq!(primal, Array::scalar(z.re));
        assert_eq!(tangent, Array::scalar(tangent_seed.re));
        let (primal, tangent) =
            differentiate_at(Array::scalar(z)).jvp(Array::scalar(tangent_seed), |x| x.imaginary()).unwrap();
        assert_eq!(primal, Array::scalar(z.im));
        assert_eq!(tangent, Array::scalar(tangent_seed.im));

        // Construction: d(complex(re, im)) = complex(dre, dim), including the mixed case where one part tangent is a
        // structural zero that must be materialized to keep the staged `complex` arity.
        let (primal, tangent) = differentiate_at((Array::scalar(1.5f64), Array::scalar(-2.0f64)))
            .jvp((Array::scalar(0.25f64), Array::scalar(4.0f64)), |(real, imaginary)| real.complex(&imaginary))
            .unwrap();
        assert_eq!(primal, Array::scalar(ComplexNumber::new(1.5f64, -2.0f64)));
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(0.25f64, 4.0f64)));
        let (_, tangent) = differentiate_at((Array::scalar(1.5f64), Array::scalar(-2.0f64)))
            .jvp((Array::scalar(0.25f64), Array::scalar(4.0f64)), |(real, imaginary)| {
                let constant = imaginary.context().lift(Array::scalar(0.0f64))?;
                let _ = imaginary;
                real.complex(&constant)
            })
            .unwrap();
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(0.25f64, 0.0f64)));
    }

    #[test]
    fn test_complex_gradient_of_squared_magnitude_is_twice_the_conjugate() {
        // The canonical non-holomorphic example: f(z) = Re(z · z̄) = |z|² is a ℂ → ℝ function, so it flows through the
        // *plain* gradient entry point (the output is real; no holomorphy promise is involved). Under the bilinear
        // (i.e., conjugation-free) transposition pairing, the pullback of the real unit seed accumulates
        // z̄ (from the `z` factor) plus conjugate(z) (from the transposed conjugation branch), so the gradient is
        // 2·z̄ — the same value JAX's `grad` returns for real-valued functions of complex inputs.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let gradient = differentiate_at(Array::scalar(z))
            .gradient(|x| (x.clone() * x.conjugate().unwrap()).real().unwrap())
            .unwrap();
        assert_eq!(gradient, Array::scalar(z.conj() + z.conj()));

        // Forward and reverse agree through the ℝ-linear rules: the jvp of f at tangent ż is 2·Re(z̄ · ż).
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);
        let (primal, tangent) = differentiate_at(Array::scalar(z))
            .jvp(Array::scalar(tangent_seed), |x| Ok((x.clone() * x.conjugate()?).real()?))
            .unwrap();
        assert_eq!(primal, Array::scalar(z.norm_sqr()));
        assert_eq!(tangent, Array::scalar((tangent_seed * z.conj() + z * tangent_seed.conj()).re));
    }
}
