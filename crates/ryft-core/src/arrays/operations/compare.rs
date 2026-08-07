//! Array IR instantiations of the comparison operation family contracts.
//!
//! Comparing two first-class runtime dimensions is host integer work whose result is ordinary rank-zero Boolean array
//! data, so the array universe answers the comparison contract at both the dimension member level and the composite
//! level.

use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::DimensionType;
use crate::backends::Array;
use crate::operations::{Compare, ComparisonDirection};
use crate::programs::{ProgramError, Value, ValueProjection};

impl Compare<Array> for DimensionValue {
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Array, ProgramError> {
        let result = match direction {
            ComparisonDirection::Equal => self.extent() == rhs.extent(),
            ComparisonDirection::NotEqual => self.extent() != rhs.extent(),
            ComparisonDirection::LessThan => self.extent() < rhs.extent(),
            ComparisonDirection::LessThanOrEqual => self.extent() <= rhs.extent(),
            ComparisonDirection::GreaterThan => self.extent() > rhs.extent(),
            ComparisonDirection::GreaterThanOrEqual => self.extent() >= rhs.extent(),
        };
        Ok(Array::scalar(result))
    }
}

impl<A: Value<Type = ArrayType>> Compare for ArrayIrValue<A>
where
    DimensionValue: Compare<A>,
{
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        let left = <Self as ValueProjection<DimensionType>>::projected(self)?;
        let right = <Self as ValueProjection<DimensionType>>::projected(rhs)?;
        Ok(Self::Array(left.compare(right, direction)?))
    }
}
