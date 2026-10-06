//! Capability attribute integration through the default `ryft` path and through a renamed core-only import.

// TODO(eaplatanios): Review this module.

extern crate ryft_core as core_alias;

use pretty_assertions::assert_eq;

use ryft::{
    Array, ArrayIrType, ArrayIrValue, ArrayType, Capability, DimensionValue, ProgramError, TypeError, capability,
};

/// Doubles values, implemented for composite values by projecting onto their array members.
#[capability(projection(ArrayIrType => ArrayType))]
trait Double<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns twice this value.
    fn double(&self) -> Result<Self, ProgramError>;
}

impl Double<ArrayType> for Array {
    fn double(&self) -> Result<Self, ProgramError> {
        Ok(self.clone() + self.clone())
    }
}

/// Splits values into a head and a tail, generated against a renamed core-only import.
#[ryft::macros::capability(crate = "::core_alias", projection(core_alias::ArrayIrType => core_alias::ArrayType))]
trait Split<T = <Self as core_alias::Capability>::Universe>: core_alias::Capability + Sized {
    /// Returns this value as the head and `others` as the tail.
    fn split(&self, others: &[Self]) -> Result<(Self, Vec<Self>), core_alias::ProgramError>;
}

impl Split<ArrayType> for Array {
    fn split(&self, others: &[Self]) -> Result<(Self, Vec<Self>), ProgramError> {
        Ok((self.clone(), others.to_vec()))
    }
}

#[test]
fn test_capability() {
    let vector = |values: Vec<f32>| ArrayIrValue::Array(Array::vector(values).unwrap());
    assert_eq!(vector(vec![1.0, 2.0]).double(), Ok(vector(vec![2.0, 4.0])));
    assert_eq!(
        vector(vec![1.0]).split(&[vector(vec![2.0]), vector(vec![3.0])]),
        Ok((vector(vec![1.0]), vec![vector(vec![2.0]), vector(vec![3.0])])),
    );

    // Members of other kinds are rejected by the projection.
    let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
    assert_eq!(
        dimension.double(),
        Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
    );
}
