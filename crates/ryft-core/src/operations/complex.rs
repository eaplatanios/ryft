use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::programs::values::Value;
use crate::types::{ArrayType, DataType};

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
        other => Err(TypeError { message: format!("'{op}' requires f32 or f64 parts but got {other}") }),
    }
}

/// Maps a complex element [`DataType`] to the [`DataType`] of its real and imaginary parts (i.e., `c64 → f32` and
/// `c128 → f64`), reporting a [`TypeError`] under `op`'s name for non-complex operand data types.
fn complex_to_part_data_type(complex: DataType, op: &'static str) -> Result<DataType, TypeError> {
    match complex {
        DataType::C64 => Ok(DataType::F32),
        DataType::C128 => Ok(DataType::F64),
        other => Err(TypeError { message: format!("'{op}' requires a complex operand but got {other}") }),
    }
}

/// [`Operation`] that constructs a complex value from its real and imaginary parts (i.e., `(re, im) ↦ re + im·i`,
/// with `(f32, f32) ↦ c64` and `(f64, f64) ↦ c128`). This is the analogue of
/// [JAX's `lax.complex`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.complex.html) and the inverse of the
/// [`RealOperation`]/[`ImaginaryOperation`] pair. The two parts must have identical types.
///
/// As a map from the pair of real parts, the operation is linear, and its transpose is the
/// `ȳ ↦ (real(ȳ), imaginary(-ȳ))` pair under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition
/// uses over complex types.
#[derive(Clone, Debug, Default)]
pub struct ComplexOperation;

impl Display for ComplexOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(COMPLEX_OPERATION_NAME)
    }
}

impl Operation<DataType> for ComplexOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COMPLEX_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        if input_types[0] != input_types[1] {
            return Err(TypeError {
                message: format!(
                    "'{COMPLEX_OPERATION_NAME}' requires identical part types but got {} and {}",
                    input_types[0], input_types[1],
                ),
            });
        }
        Ok(vec![part_to_complex_data_type(input_types[0].clone(), COMPLEX_OPERATION_NAME)?])
    }
}

impl Operation<ArrayType> for ComplexOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COMPLEX_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        if input_types[0] != input_types[1] {
            return Err(TypeError {
                message: format!(
                    "'{COMPLEX_OPERATION_NAME}' requires identical part types but got {} and {}",
                    input_types[0], input_types[1],
                ),
            });
        }
        let data_type = part_to_complex_data_type(input_types[0].data_type(), COMPLEX_OPERATION_NAME)?;
        Ok(vec![ArrayType { data_type, ..input_types[0].clone() }])
    }
}

impl ElementwiseOperation for ComplexOperation {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain<Value: Complex>> InterpretableOperation<C> for ComplexOperation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].complex(&inputs[1])?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ComplexOperation where C::Operation: From<ComplexOperation> {}

/// Value-level capability that constructs a complex value from this value as the real part and `imaginary` as the
/// imaginary part. [`Complex`] fills the same role for [`ComplexOperation`] that [`Sin`](crate::Sin) fills for
/// [`SinOperation`](crate::SinOperation).
pub trait Complex: Sized {
    /// Constructs the complex value `self + imaginary·i`, returning a [`ProgramError`] if something goes wrong (e.g.,
    /// when the parts are not both `f32` or both `f64` valued).
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<ComplexOperation>>>> Complex for V {
    #[inline]
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ComplexOperation, Vec::new(), &[self.clone(), imaginary.clone()])?
            .remove(0))
    }
}

/// [`Operation`] that computes the elementwise complex conjugate of one complex value (i.e., `z ↦ z̄`, negating the
/// imaginary part) while preserving its type metadata. Real operands are rejected: the conjugate of a real value is
/// the identity, so real call sites should simply not conjugate.
///
/// Conjugation is ℝ-linear but not ℂ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's
/// transposition uses over complex types, it is self-adjoint: the transpose of `z ↦ z̄` is `ȳ ↦ ȳ̄`.
#[derive(Clone, Debug, Default)]
pub struct ConjugateOperation;

impl Display for ConjugateOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(CONJUGATE_OPERATION_NAME)
    }
}

impl Operation<DataType> for ConjugateOperation {
    #[inline]
    fn name(&self) -> &'static str {
        CONJUGATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        complex_to_part_data_type(input_types[0].clone(), CONJUGATE_OPERATION_NAME)?;
        Ok(vec![input_types[0].clone()])
    }
}

impl Operation<ArrayType> for ConjugateOperation {
    #[inline]
    fn name(&self) -> &'static str {
        CONJUGATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        complex_to_part_data_type(input_types[0].data_type(), CONJUGATE_OPERATION_NAME)?;
        Ok(vec![input_types[0].clone()])
    }
}

impl ElementwiseOperation for ConjugateOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain<Value: Conjugate>> InterpretableOperation<C> for ConjugateOperation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].conjugate()?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ConjugateOperation where C::Operation: From<ConjugateOperation> {}

/// Value-level elementwise complex-conjugation capability. [`Conjugate`] fills the same role for
/// [`ConjugateOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait Conjugate: Sized {
    /// Computes the elementwise complex conjugate of this value, returning a [`ProgramError`] if something goes wrong
    /// (e.g., when the value is not complex valued).
    fn conjugate(&self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<ConjugateOperation>>>> Conjugate for V {
    #[inline]
    fn conjugate(&self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(ConjugateOperation, Vec::new(), &[self.clone()])?.remove(0))
    }
}

/// [`Operation`] that extracts the elementwise real part of one complex value (i.e., `z ↦ Re(z)`, with `c64 ↦ f32`
/// and `c128 ↦ f64`) while preserving all other type metadata. This is the analogue of
/// [JAX's `lax.real`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.real.html).
///
/// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
/// over complex types, the transpose of `z ↦ Re(z)` is `ȳ ↦ complex(ȳ, 0)`.
#[derive(Clone, Debug, Default)]
pub struct RealOperation;

impl Display for RealOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REAL_OPERATION_NAME)
    }
}

impl Operation<DataType> for RealOperation {
    #[inline]
    fn name(&self) -> &'static str {
        REAL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![complex_to_part_data_type(input_types[0].clone(), REAL_OPERATION_NAME)?])
    }
}

impl Operation<ArrayType> for RealOperation {
    #[inline]
    fn name(&self) -> &'static str {
        REAL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let data_type = complex_to_part_data_type(input_types[0].data_type(), REAL_OPERATION_NAME)?;
        Ok(vec![ArrayType { data_type, ..input_types[0].clone() }])
    }
}

impl ElementwiseOperation for RealOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain<Value: Real>> InterpretableOperation<C> for RealOperation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].real()?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for RealOperation where C::Operation: From<RealOperation> {}

/// Value-level elementwise real-part extraction capability. [`Real`] fills the same role for [`RealOperation`] that
/// [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait Real: Sized {
    /// Extracts the elementwise real part of this value, returning a [`ProgramError`] if something goes wrong (e.g.,
    /// when the value is not complex valued).
    fn real(&self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<RealOperation>>>> Real for V {
    #[inline]
    fn real(&self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(RealOperation, Vec::new(), &[self.clone()])?.remove(0))
    }
}

/// [`Operation`] that extracts the elementwise imaginary part of one complex value (i.e., `z ↦ Im(z)`, with
/// `c64 ↦ f32` and `c128 ↦ f64`) while preserving all other type metadata. This is the analogue of
/// [JAX's `lax.imag`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.imag.html).
///
/// The extraction is ℝ-linear. Under the bilinear (i.e., conjugation-free) pairing that Ryft's transposition uses
/// over complex types, the transpose of `z ↦ Im(z)` is `ȳ ↦ complex(0, -ȳ)`.
#[derive(Clone, Debug, Default)]
pub struct ImaginaryOperation;

impl Display for ImaginaryOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(IMAGINARY_OPERATION_NAME)
    }
}

impl Operation<DataType> for ImaginaryOperation {
    #[inline]
    fn name(&self) -> &'static str {
        IMAGINARY_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![complex_to_part_data_type(input_types[0].clone(), IMAGINARY_OPERATION_NAME)?])
    }
}

impl Operation<ArrayType> for ImaginaryOperation {
    #[inline]
    fn name(&self) -> &'static str {
        IMAGINARY_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let data_type = complex_to_part_data_type(input_types[0].data_type(), IMAGINARY_OPERATION_NAME)?;
        Ok(vec![ArrayType { data_type, ..input_types[0].clone() }])
    }
}

impl ElementwiseOperation for ImaginaryOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain<Value: Imaginary>> InterpretableOperation<C> for ImaginaryOperation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].imaginary()?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ImaginaryOperation where C::Operation: From<ImaginaryOperation> {}

/// Value-level elementwise imaginary-part extraction capability. [`Imaginary`] fills the same role for
/// [`ImaginaryOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait Imaginary: Sized {
    /// Extracts the elementwise imaginary part of this value, returning a [`ProgramError`] if something goes wrong
    /// (e.g., when the value is not complex valued).
    fn imaginary(&self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<ImaginaryOperation>>>> Imaginary for V {
    #[inline]
    fn imaginary(&self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(ImaginaryOperation, Vec::new(), &[self.clone()])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::types::{Shape, Size};

    use super::*;

    #[test]
    fn test_complex() {
        let operation = ComplexOperation;

        // Operation identity and type inference: identical `f32`/`f64` parts construct the matching complex type, and
        // mismatched or non-float parts are rejected.
        assert_eq!(Operation::<DataType>::name(&operation), COMPLEX_OPERATION_NAME);
        assert_eq!(format!("{operation}"), COMPLEX_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F32], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64, DataType::F64], &[]),
            Ok(vec![DataType::C128]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Err(TypeError { message: "'complex' requires identical part types but got f32 and f64".to_string() }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::I32, DataType::I32], &[]),
            Err(TypeError { message: "'complex' requires f32 or f64 parts but got i32".to_string() }),
        );

        // Array type inference swaps the element data type and keeps the shape.
        let part = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[part.clone(), part], &[]),
            Ok(vec![ArrayType::new(DataType::C64, Shape::new(vec![Size::Static(2)]))]),
        );

        // Concrete interpretation constructs the complex value in both the scalar and the array universes.
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(1.5f32), Scalar::from(-2.0f32)],
            ),
            Ok(vec![Scalar::from(ComplexNumber::new(1.5f32, -2.0f32))]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.5), Array::scalar(-2.0)],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(1.5f64, -2.0f64))]),
        );
    }

    #[test]
    fn test_conjugate() {
        let operation = ConjugateOperation;

        // Type inference preserves the complex operand type and rejects real operands.
        assert_eq!(Operation::<DataType>::name(&operation), CONJUGATE_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::C64], &[]),
            Ok(vec![DataType::C64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "'conjugate' requires a complex operand but got f64".to_string() }),
        );

        // Concrete interpretation negates the imaginary part, in both the scalar and the array universes.
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(ComplexNumber::new(1.5f64, -2.0f64))],
            ),
            Ok(vec![Scalar::from(ComplexNumber::new(1.5f64, 2.0f64))]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])],
            ),
            Ok(vec![Array::vector(vec![ComplexNumber::new(1.5f64, 2.0f64), ComplexNumber::new(0.5f64, -1.0f64)])]),
        );
    }

    #[test]
    fn test_real() {
        let operation = RealOperation;

        // Type inference maps the complex operand to its part data type and rejects real operands.
        assert_eq!(Operation::<DataType>::name(&operation), REAL_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::C64], &[]),
            Ok(vec![DataType::F32])
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::C128], &[]),
            Ok(vec![DataType::F64])
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Err(TypeError { message: "'real' requires a complex operand but got f32".to_string() }),
        );
        let complex = ArrayType::new(DataType::C128, Shape::new(vec![Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[complex], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))]),
        );

        // Concrete interpretation extracts the real part, in both the scalar and the array universes (where the
        // output element data type maps to the part data type).
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(ComplexNumber::new(1.5f64, -2.0f64))],
            ),
            Ok(vec![Scalar::from(1.5f64)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])],
            ),
            Ok(vec![Array::vector(vec![1.5f64, 0.5f64])]),
        );
    }

    #[test]
    fn test_imaginary() {
        let operation = ImaginaryOperation;

        // Type inference maps the complex operand to its part data type and rejects real operands.
        assert_eq!(Operation::<DataType>::name(&operation), IMAGINARY_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::C64], &[]),
            Ok(vec![DataType::F32])
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::Boolean], &[]),
            Err(TypeError { message: "'imaginary' requires a complex operand but got bool".to_string() }),
        );

        // Concrete interpretation extracts the imaginary part, in both the scalar and the array universes (where the
        // output element data type maps to the part data type).
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(ComplexNumber::new(1.5f64, -2.0f64))],
            ),
            Ok(vec![Scalar::from(-2.0f64)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![ComplexNumber::new(1.5f64, -2.0f64), ComplexNumber::new(0.5f64, 1.0f64)])],
            ),
            Ok(vec![Array::vector(vec![-2.0f64, 1.0f64])]),
        );
    }
}
