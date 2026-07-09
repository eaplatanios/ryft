use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, DifferentiationDual, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::NegOperation;
use crate::operations::complex::{
    Complex, ComplexOperation, Conjugate, ConjugateOperation, Imaginary, ImaginaryOperation, Real, RealOperation,
};
use crate::operations::constants::{Zero, ZeroLikeOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::Typed;

impl<C: Context> DifferentiableOperation<C> for ComplexOperation
where
    C: Zero<C::Value>,
    C::Operation: Clone,
    C::Value: Complex,
    ComplexOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let real = &inputs[0];
        let imaginary = &inputs[1];
        let primal = real.primal().complex(imaginary.primal())?;
        // Complex construction is linear in its two real parts: `d(complex(re, im)) = complex(dre, dim)`. When both
        // part tangents are structural zeros the output tangent stays a symbolic zero of the complex output type;
        // when only one is, the missing part is materialized as a real zero through the context so the staged
        // `complex` keeps its two-part arity.
        let tangent = match (real.tangent(), imaginary.tangent()) {
            (MaybeZero::Zero(_), MaybeZero::Zero(_)) => MaybeZero::Zero(primal.r#type().into_owned()),
            (real_tangent, imaginary_tangent) => MaybeZero::Value(
                real_tangent
                    .clone()
                    .materialize(context)?
                    .complex(&imaginary_tangent.clone().materialize(context)?)?,
            ),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for the linear [`ComplexOperation`]. Under the bilinear (i.e., conjugation-free) pairing that
/// Ryft's transposition uses over complex types, the transpose of `(re, im) ↦ re + im·i` maps the output cotangent
/// `ȳ` to the part cotangents `(real(ȳ), imaginary(-ȳ))`: pairing `Re(ȳ · (re + im·i))` against `(re, im)` picks out
/// the real part of `ȳ` for `re` and the *negated* imaginary part for `im`. Like the `Add` rule, known-ness is
/// ignored — a known part contributes an additive constant whose adjoint is dropped at the pullback output boundary.
impl<V: Value, O> TransposableOperation<V, O> for ComplexOperation
where
    O: Operation<V::Type> + From<NegOperation> + From<RealOperation> + From<ImaginaryOperation>,
    ComplexOperation: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(match &outputs[0] {
            MaybeZero::Zero(_) => inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect(),
            MaybeZero::Value(output_cotangent) => vec![
                MaybeZero::Value(output_cotangent.unary(RealOperation)),
                MaybeZero::Value(output_cotangent.unary(NegOperation).unary(ImaginaryOperation)),
            ],
        })
    }
}

impl<C: Context> DifferentiableOperation<C> for ConjugateOperation
where
    C::Operation: Clone,
    C::Value: Conjugate,
    ConjugateOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().conjugate()?;
        // Conjugation is ℝ-linear (but not ℂ-linear): `d(z̄) = d̄z`. A structural zero tangent stays symbolic.
        let tangent = match input.tangent() {
            MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.conjugate()?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for the ℝ-linear [`ConjugateOperation`]. Under the bilinear (i.e., conjugation-free) pairing that
/// Ryft's transposition uses over complex types, conjugation is self-adjoint: pairing `Re(ȳ · z̄)` against `z` shows
/// that the transpose of `z ↦ z̄` is `ȳ ↦ ȳ̄`.
impl<V: Value, O: Operation<V::Type> + From<ConjugateOperation>> TransposableOperation<V, O> for ConjugateOperation
where
    ConjugateOperation: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(match &outputs[0] {
            MaybeZero::Zero(_) => vec![MaybeZero::Zero(inputs[0].r#type().into_owned())],
            MaybeZero::Value(output_cotangent) => vec![MaybeZero::Value(output_cotangent.unary(ConjugateOperation))],
        })
    }
}

impl<C: Context> DifferentiableOperation<C> for RealOperation
where
    C::Operation: Clone,
    C::Value: Real,
    RealOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().real()?;
        // Real-part extraction is ℝ-linear: `d(Re(z)) = Re(dz)`. A structural zero tangent stays symbolic, retyped
        // to the real output type.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.real()?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for the ℝ-linear [`RealOperation`]. Under the bilinear (i.e., conjugation-free) pairing that Ryft's
/// transposition uses over complex types, pairing `t · Re(z)` against `z` shows that the transpose of `z ↦ Re(z)` is
/// `t ↦ complex(t, 0)`, injecting the real cotangent with a zero imaginary part.
impl<V: Value, O> TransposableOperation<V, O> for RealOperation
where
    O: Operation<V::Type> + From<ComplexOperation> + From<ZeroLikeOperation>,
    RealOperation: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(match &outputs[0] {
            MaybeZero::Zero(_) => vec![MaybeZero::Zero(inputs[0].r#type().into_owned())],
            MaybeZero::Value(output_cotangent) => {
                let zero = output_cotangent.unary(ZeroLikeOperation);
                vec![MaybeZero::Value(output_cotangent.binary(&zero, ComplexOperation))]
            }
        })
    }
}

impl<C: Context> DifferentiableOperation<C> for ImaginaryOperation
where
    C::Operation: Clone,
    C::Value: Imaginary,
    ImaginaryOperation: Operation<C::Type>,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let primal = input.primal().imaginary()?;
        // Imaginary-part extraction is ℝ-linear: `d(Im(z)) = Im(dz)`. A structural zero tangent stays symbolic,
        // retyped to the real output type.
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.imaginary()?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for the ℝ-linear [`ImaginaryOperation`]. Under the bilinear (i.e., conjugation-free) pairing that
/// Ryft's transposition uses over complex types, pairing `t · Im(z)` against `z` shows that the transpose of
/// `z ↦ Im(z)` is `t ↦ complex(0, -t)`, injecting the *negated* real cotangent as the imaginary part.
impl<V: Value, O> TransposableOperation<V, O> for ImaginaryOperation
where
    O: Operation<V::Type> + From<NegOperation> + From<ComplexOperation> + From<ZeroLikeOperation>,
    ImaginaryOperation: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(match &outputs[0] {
            MaybeZero::Zero(_) => vec![MaybeZero::Zero(inputs[0].r#type().into_owned())],
            MaybeZero::Value(output_cotangent) => {
                let zero = output_cotangent.unary(ZeroLikeOperation);
                let negated = output_cotangent.unary(NegOperation);
                vec![MaybeZero::Value(zero.binary(&negated, ComplexOperation))]
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::contexts::{Context, EagerContext};
    use crate::operations::complex::{Complex, Conjugate, Imaginary, Real};
    use crate::operations::scalars::ScalarOperation;
    use crate::scalars::Scalar;
    use crate::tracing_v2::ForwardModeDifferentiate;

    #[test]
    fn test_complex_part_jvps_are_real_linear() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);

        // Conjugation: d(z̄) = d̄z.
        let (primal, tangent) = domain.jvp(|x| x.conjugate(), Scalar::from(z), Scalar::from(tangent_seed)).unwrap();
        assert_eq!(primal, Scalar::from(z.conj()));
        assert_eq!(tangent, Scalar::from(tangent_seed.conj()));

        // Part extraction: d(Re(z)) = Re(dz) and d(Im(z)) = Im(dz).
        let (primal, tangent) = domain.jvp(|x| x.real(), Scalar::from(z), Scalar::from(tangent_seed)).unwrap();
        assert_eq!(primal, Scalar::from(z.re));
        assert_eq!(tangent, Scalar::from(tangent_seed.re));
        let (primal, tangent) = domain.jvp(|x| x.imaginary(), Scalar::from(z), Scalar::from(tangent_seed)).unwrap();
        assert_eq!(primal, Scalar::from(z.im));
        assert_eq!(tangent, Scalar::from(tangent_seed.im));

        // Construction: d(complex(re, im)) = complex(dre, dim), including the mixed case where one part tangent is a
        // structural zero that must be materialized to keep the staged `complex` arity.
        let (primal, tangent) = domain
            .jvp(
                |(real, imaginary)| real.complex(&imaginary),
                (Scalar::from(1.5f64), Scalar::from(-2.0f64)),
                (Scalar::from(0.25f64), Scalar::from(4.0f64)),
            )
            .unwrap();
        assert_eq!(primal, Scalar::from(ComplexNumber::new(1.5f64, -2.0f64)));
        assert_eq!(tangent, Scalar::from(ComplexNumber::new(0.25f64, 4.0f64)));
        let (_, tangent) = domain
            .jvp(
                |(real, imaginary)| {
                    let constant = imaginary.context().lift(Scalar::from(0.0f64))?;
                    let _ = imaginary;
                    real.complex(&constant)
                },
                (Scalar::from(1.5f64), Scalar::from(-2.0f64)),
                (Scalar::from(0.25f64), Scalar::from(4.0f64)),
            )
            .unwrap();
        assert_eq!(tangent, Scalar::from(ComplexNumber::new(0.25f64, 0.0f64)));
    }

    #[test]
    fn test_gradient_of_squared_magnitude_is_twice_the_conjugate() {
        // The canonical non-holomorphic example: f(z) = Re(z · z̄) = |z|² is a ℂ → ℝ function, so it flows through the
        // *plain* gradient entry point (the output is real; no holomorphy promise is involved). Under the bilinear
        // (i.e., conjugation-free) transposition pairing, the pullback of the real unit seed accumulates
        // z̄ (from the `z` factor) plus conjugate(z) (from the transposed conjugation branch), so the gradient is
        // 2·z̄ — the same value JAX's `grad` returns for real-valued functions of complex inputs.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let gradient =
            crate::tracing_v2::gradient(|x| (x.clone() * x.conjugate().unwrap()).real().unwrap(), Scalar::from(z))
                .unwrap();
        assert_eq!(gradient, Scalar::from(z.conj() + z.conj()));

        // Forward and reverse agree through the ℝ-linear rules: the jvp of f at tangent ż is 2·Re(z̄ · ż).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let tangent_seed = ComplexNumber::new(0.5f64, 2.0f64);
        let (primal, tangent) = domain
            .jvp(|x| Ok((x.clone() * x.conjugate()?).real()?), Scalar::from(z), Scalar::from(tangent_seed))
            .unwrap();
        assert_eq!(primal, Scalar::from(z.norm_sqr()));
        assert_eq!(tangent, Scalar::from((tangent_seed * z.conj() + z * tangent_seed.conj()).re));
    }
}
