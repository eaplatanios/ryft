//! Array IR instantiations of the comparison operation family contracts.
//!
//! Comparing two first-class runtime dimensions is host integer work whose result is ordinary rank-zero Boolean array
//! data, so the array universe answers the comparison contract at both the dimension member level and the composite
//! level.

use std::cmp::Ordering;
use std::sync::Arc;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::DimensionType;
use crate::operations::{Compare, ComparisonDirection, ElementType};
use crate::programs::{ProgramError, TypeError, Typed, Value, ValueProjection};

// TODO(eaplatanios): Review this.

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

impl Array {
    /// Compares two same-type arrays elementwise using their typed value semantics rather than their physical byte
    /// patterns. The equality directions apply to every element type, while the ordered directions apply to the
    /// partially ordered element types and are rejected with an error for the unordered complex ones.
    fn compare_elements(
        &self,
        rhs: &Self,
        output_type: ArrayType,
        direction: ComparisonDirection,
    ) -> Result<Self, ProgramError> {
        let data_type = self.r#type().data_type();
        if data_type.is_complex() {
            // The unordered complex element types define only the equality comparison directions. The compare
            // operation's type inference already rejects ordered complex comparisons, but the direct `Array`
            // comparison API reaches this kernel without it.
            if !matches!(direction, ComparisonDirection::Equal | ComparisonDirection::NotEqual) {
                return Err(TypeError::invalid(format!(
                    "cannot apply an ordered comparison to unordered complex scalars of data type {data_type}",
                ))
                .into());
            }
            let equal = matches!(direction, ComparisonDirection::Equal);
            return dispatch_on_array_element_type!(@complex data_type, |Element| {
                self.binary_elements::<Element, bool>(rhs, output_type, |left, right| {
                    Ok(if equal { left == right } else { left != right })
                })
            });
        }
        dispatch_on_array_element_type!(@ordered data_type, |Element| {
            self.binary_elements::<Element, bool>(rhs, output_type, |left, right| {
                // An unordered pair (a comparison involving a floating-point NaN) satisfies only `NotEqual`.
                let ordering = left.partial_cmp(&right);
                Ok(match direction {
                    ComparisonDirection::Equal => ordering == Some(Ordering::Equal),
                    ComparisonDirection::NotEqual => ordering != Some(Ordering::Equal),
                    ComparisonDirection::LessThan => ordering == Some(Ordering::Less),
                    ComparisonDirection::LessThanOrEqual => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
                    ComparisonDirection::GreaterThan => ordering == Some(Ordering::Greater),
                    ComparisonDirection::GreaterThanOrEqual => {
                        matches!(ordering, Some(Ordering::Greater | Ordering::Equal))
                    }
                })
            })
        })
    }
}

impl Compare for Array {
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        // Broadcast the operand types together (including element-type promotion) so mixed-precision comparisons
        // mirror the `CompareOperation` type-inference contract, then compare the promoted elements pairwise. The
        // output type is the Boolean-typed counterpart of the broadcast type.
        let (broadcast_type, operands) = Self::broadcast_promoted(&[self, rhs])?;
        let target = broadcast_type.data_type();
        let output_type = broadcast_type.with_element_type(DataType::Boolean);
        // Empty comparisons inspect no elements, so they succeed vacuously even for payload-free data types.
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }
        if target == DataType::Token {
            return Err(TypeError::invalid("cannot compare token scalars".to_string()).into());
        }
        if target == DataType::Zero {
            return Err(TypeError::invalid("cannot compare scalars of data types zero and zero".to_string()).into());
        }

        // `broadcast_promoted` converts only mismatched inputs, so equal-typed inputs retain their exact physical
        // storage and are decoded one addressed element at a time by the shared binary loop.
        let [left, right] = <[_; 2]>::try_from(operands).unwrap();
        left.compare_elements(&right, output_type, direction)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::arrays::encoding::{f8e5m2, i2};
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_compare() {
        let left = Array::vector(vec![1.0, 2.0, 3.0]);
        let right = Array::vector(vec![2.0, 2.0, 2.0]);
        let less_than = left.compare(&right, ComparisonDirection::LessThan).unwrap();
        assert_eq!(less_than.r#type().into_owned(), array_type(DataType::Boolean, &[3]));
        assert_eq!(less_than, Array::vector(vec![true, false, false]));
        // Operands broadcast and promote before comparing.
        let mixed = Array::vector(vec![1.0f32, 3.0]).compare(&Array::scalar(2.0f64), ComparisonDirection::GreaterThan);
        assert_eq!(mixed.unwrap(), Array::vector(vec![false, true]));

        // Sealed sub-byte elements use their signed value ordering and participate in full NumPy-style broadcasting.
        let left = Array::matrix(2, 1, vec![i2::new(-1).unwrap(), i2::new(1).unwrap()]);
        let right = Array::matrix(1, 3, vec![i2::new(-2).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]);
        assert_eq!(
            left.compare(&right, ComparisonDirection::LessThan).unwrap().elements::<bool>(),
            Ok(vec![false, true, true, false, false, false]),
        );

        // Addressed input and output layouts remain physical contracts; comparison writes only the Boolean element
        // ranges and leaves output holes zero.
        let strided_type = array_type(DataType::U16, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let left = Array::from_elements(strided_type.clone(), &[1u16, 3]).unwrap();
        let right = Array::from_elements(strided_type, &[2u16, 2]).unwrap();
        let compared = left.compare(&right, ComparisonDirection::LessThan).unwrap();
        assert_eq!(compared.elements::<bool>(), Ok(vec![true, false]));
        assert_eq!(compared.storage_bytes(), [1, 0, 0, 0, 0]);

        // Floating-point NaNs are unordered, while complex arrays expose only equality comparisons.
        let nan = Array::vector(vec![f8e5m2::NAN]);
        assert_eq!(nan.compare(&nan, ComparisonDirection::NotEqual).unwrap().elements::<bool>(), Ok(vec![true]));
        let complex = Array::vector(vec![ComplexNumber::new(1.0f32, 2.0)]);
        assert_eq!(complex.compare(&complex, ComparisonDirection::Equal).unwrap().elements::<bool>(), Ok(vec![true]),);
        assert!(matches!(
            complex.compare(&complex, ComparisonDirection::LessThan),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot apply an ordered comparison to unordered complex scalars of data type c64",
        ));

        // Empty payload-free comparisons are vacuous because they evaluate no unsupported element comparison; a
        // nonempty token array retains the established scalar-backend error.
        let empty_token = Array::from_logical_bytes(array_type(DataType::Token, &[0]), &[]).unwrap();
        assert_eq!(
            empty_token.compare(&empty_token, ComparisonDirection::Equal).unwrap().elements::<bool>(),
            Ok(vec![]),
        );
        let token = Array::from_logical_bytes(array_type(DataType::Token, &[1]), &[]).unwrap();
        assert!(matches!(
            token.compare(&token, ComparisonDirection::Equal),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == "cannot compare token scalars",
        ));
    }
}
