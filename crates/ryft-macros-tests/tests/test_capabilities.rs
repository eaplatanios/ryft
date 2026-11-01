extern crate ryft_core as core_alias;

use std::fmt::Debug;

use pretty_assertions::assert_eq;

use ryft::{
    Array, ArrayIrType, ArrayIrValue, ArrayType, Capability, DimensionValue, ProgramError, TypeError, capability,
};

/// Doubles values, implemented for array IR values by projecting onto their array members.
#[capability(projection(ArrayIrType => ArrayType))]
trait Double<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns twice this value.
    fn double(&self) -> Result<Self, ProgramError>;

    /// Returns four times this value, composing [`Double::double`].
    fn quadruple(&self) -> Result<Self, ProgramError> {
        self.double()?.double()
    }
}

impl Double<ArrayType> for Array {
    fn double(&self) -> Result<Self, ProgramError> {
        Ok(self.clone() + self.clone())
    }
}

/// Splits values into a head and a tail, generated against a renamed `ryft-core` import.
#[capability(crate = "::core_alias", projection(core_alias::ArrayIrType => core_alias::ArrayType))]
trait Split<T = <Self as core_alias::Capability>::Universe>: core_alias::Capability + Sized {
    /// Returns this value as the head and `others` as the tail.
    fn split(&self, others: &[Self]) -> Result<(Self, Vec<Self>), core_alias::ProgramError>;
}

impl Split<ArrayType> for Array {
    fn split(&self, others: &[Self]) -> Result<(Self, Vec<Self>), ProgramError> {
        Ok((self.clone(), others.to_vec()))
    }
}

/// Combines values, with a `Self`-defaulted right input, an optional input, inputs whose names match the local
/// variables of the generated functions, and additional supertraits and `where` predicates.
#[capability(projection(ArrayIrType => ArrayType))]
trait Combine<Rhs = Self, T = <Self as Capability>::Universe>: Capability + Sized + Clone
where
    Self: Debug,
{
    /// Returns this value plus `receiver`, plus `mask` when it is present, scaled by `output`.
    fn combine(&self, receiver: &Rhs, mask: Option<&Self>, output: usize) -> Result<Self, ProgramError>;
}

impl Combine<Array, ArrayType> for Array {
    fn combine(&self, receiver: &Array, mask: Option<&Array>, output: usize) -> Result<Self, ProgramError> {
        let sum = mask.into_iter().fold(self.clone() + receiver.clone(), |sum, mask| sum + mask.clone());
        Ok((1..output).fold(sum.clone(), |total, _| total + sum.clone()))
    }
}

#[test]
fn test_capability() {
    let vector = |values: Vec<f32>| ArrayIrValue::Array(Array::vector(values).unwrap());

    // Required functions are projected, and provided functions keep their default bodies.
    assert_eq!(vector(vec![1.0, 2.0]).double(), Ok(vector(vec![2.0, 4.0])));
    assert_eq!(vector(vec![1.0, 2.0]).quadruple(), Ok(vector(vec![4.0, 8.0])));

    // Members of other kinds are rejected by the projection.
    let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
    assert_eq!(
        dimension.double(),
        Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
    );
}

#[test]
fn test_capability_crate_path() {
    let vector = |values: Vec<f32>| ArrayIrValue::Array(Array::vector(values).unwrap());
    assert_eq!(
        vector(vec![1.0]).split(&[vector(vec![2.0]), vector(vec![3.0])]),
        Ok((vector(vec![1.0]), vec![vector(vec![2.0]), vector(vec![3.0])])),
    );
}

#[test]
fn test_capability_projected_inputs() {
    // `Self`-defaulted inputs and optional inputs are projected, while other inputs are passed through unchanged,
    // even when they are named like the local variables of the generated function.
    let vector = |values: Vec<f32>| ArrayIrValue::Array(Array::vector(values).unwrap());
    assert_eq!(vector(vec![1.0]).combine(&vector(vec![2.0]), None, 1), Ok(vector(vec![3.0])));
    assert_eq!(vector(vec![1.0]).combine(&vector(vec![2.0]), Some(&vector(vec![3.0])), 2), Ok(vector(vec![12.0])));
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/capabilities/error_arguments.rs");
    test_cases.compile_fail("tests/capabilities/error_traits.rs");
}
