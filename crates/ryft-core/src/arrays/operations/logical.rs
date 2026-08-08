//! Reference [`Array`] kernels for the logical operation family contracts.
//!
//! Boolean and integer logic acts independently on every bit, so these kernels combine validated element encodings
//! byte by byte without decoding them. Negation masks the declared bit width so sub-byte encodings stay valid.

use std::sync::Arc;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::types::data::DataType;
use crate::operations::{And, Not, Or, Xor};
use crate::programs::{ProgramError, TypeError};

// TODO(eaplatanios): Review this.

impl Array {
    /// Applies a binary logical or bitwise operation directly to validated Boolean or integer element bytes. Since
    /// bitwise operations act independently on every bit, their result is independent of integer signedness and host
    /// endianness. Logical Boolean encodings use the same `0` and `1` bitwise truth tables.
    fn binary_logical(
        &self,
        rhs: &Self,
        operation: &str,
        function: impl Fn(u8, u8) -> u8,
    ) -> Result<Self, ProgramError> {
        let left_data_type = self.r#type.data_type();
        let right_data_type = rhs.r#type.data_type();
        let output_type = Broadcastable::broadcast(&self.r#type, &rhs.r#type)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        if left_data_type != right_data_type || !(left_data_type.is_boolean() || left_data_type.is_integer()) {
            return Err(TypeError::invalid(format!(
                "cannot apply `{operation}` to arrays of element data types {left_data_type} and {right_data_type}",
            ))
            .into());
        }

        let output_shape = output_type.static_shape().unwrap();
        let left_shape = self.r#type.static_shape().unwrap();
        let right_shape = rhs.r#type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let left_strides = left_shape.row_major_strides();
        let right_strides = right_shape.row_major_strides();
        let left_addressing = ArrayAddressing::new(self.r#type.clone())?;
        let right_addressing = ArrayAddressing::new(rhs.r#type.clone())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let left_bytes = self.storage_bytes();
        let right_bytes = rhs.storage_bytes();
        let element_byte_width = output_addressing.element_byte_width();
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let left_range = left_addressing.byte_range_for_flat_index(Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &left_shape,
                &left_strides,
            ));
            let right_range = right_addressing.byte_range_for_flat_index(Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &right_shape,
                &right_strides,
            ));
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            for byte in 0..element_byte_width {
                output_bytes[output_range.start + byte] =
                    function(left_bytes[left_range.start + byte], right_bytes[right_range.start + byte]);
            }
        }
        // Valid inputs, bitwise closure outputs, and zero-initialized unoccupied storage preserve every `Array`
        // encoding invariant without a second validation traversal.
        Ok(Self { r#type: output_type, bytes: Arc::new(output_bytes) })
    }
}

impl Not for Array {
    fn not(&self) -> Result<Self, ProgramError> {
        let mask = match self.r#type.data_type() {
            DataType::Boolean | DataType::I1 | DataType::U1 => 0b1,
            DataType::I2 | DataType::U2 => 0b11,
            DataType::I4 | DataType::U4 => 0b1111,
            data_type if data_type.is_integer() => u8::MAX,
            data_type => {
                return Err(TypeError::invalid(format!(
                    "cannot apply `not` to an array of element data type {data_type}"
                ))
                .into());
            }
        };
        let addressing = ArrayAddressing::new(self.r#type.clone())?;
        let input_bytes = self.storage_bytes();
        let mut bytes = vec![0; addressing.storage_byte_len()];
        for element in 0..addressing.element_count() {
            for byte in addressing.byte_range_for_flat_index(element) {
                bytes[byte] = !input_bytes[byte] & mask;
            }
        }
        // Masking retains valid Boolean and sub-byte encodings; full-width integers admit every bit pattern, and
        // zero-initialization preserves all layout holes and padding.
        Ok(Self { r#type: self.r#type.clone(), bytes: Arc::new(bytes) })
    }
}

impl And for Array {
    fn and(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, "and", |left, right| left & right)
    }
}

impl Or for Array {
    fn or(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, "or", |left, right| left | right)
    }
}

impl Xor for Array {
    fn xor(&self, rhs: &Self) -> Result<Self, ProgramError> {
        self.binary_logical(rhs, "xor", |left, right| left ^ right)
    }
}

impl std::ops::Not for Array {
    type Output = Self;

    fn not(self) -> Self::Output {
        Not::not(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::BitAnd for Array {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        And::and(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::BitOr for Array {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Or::or(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::BitXor for Array {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Xor::xor(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::arrays::encoding::{i2, u4};
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_logical_operations() {
        let left = Array::vector(vec![true, true, false, false]);
        let right = Array::vector(vec![true, false, true, false]);
        assert_eq!(left.and(&right).unwrap(), Array::vector(vec![true, false, false, false]));
        assert_eq!(left.or(&right).unwrap(), Array::vector(vec![true, true, true, false]));
        assert_eq!(left.xor(&right).unwrap(), Array::vector(vec![false, true, true, false]));
        assert_eq!(left.not().unwrap(), Array::vector(vec![false, false, true, true]));
        // General NumPy-style broadcasting maps each input coordinate into the common output shape.
        assert_eq!(
            Array::matrix(2, 1, vec![true, false]).and(&Array::matrix(1, 3, vec![true, false, true])).unwrap(),
            Array::matrix(2, 3, vec![true, false, true, false, false, false]),
        );
        // Same-data-type integers combine bitwise directly over all bytes of each encoding.
        let bits = Array::vector(vec![0b1100u8]).and(&Array::vector(vec![0b1010u8])).unwrap();
        assert_eq!(bits.elements::<u8>(), Ok(vec![0b1000]));
        assert_eq!(Array::vector(vec![0x00ff_i16, -1]).not().unwrap().elements::<i16>(), Ok(vec![-256, 0]));
        // Sub-byte negation complements only the declared low bits, retaining a valid sign-extended encoding.
        let signed_sub_byte = Array::vector(vec![i2::MIN, i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::MAX]);
        assert_eq!(
            signed_sub_byte.not().unwrap().elements::<i2>(),
            Ok(vec![i2::MAX, i2::new(0).unwrap(), i2::new(-1).unwrap(), i2::MIN]),
        );
        assert_eq!(
            Array::scalar(u4::new(0b1100).unwrap())
                .xor(&Array::scalar(u4::new(0b1010).unwrap()))
                .unwrap()
                .elements::<u4>(),
            Ok(vec![u4::new(0b0110).unwrap()]),
        );
        // Physical layouts are traversed through addressing, so holes stay zero rather than being complemented.
        let strided_type =
            array_type(DataType::Boolean, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let strided = Array::new(strided_type.clone(), vec![1, 0, 0]).unwrap().not().unwrap();
        assert_eq!(strided.r#type().as_ref(), &strided_type);
        assert_eq!(strided.storage_bytes(), [0, 0, 1]);
        assert_eq!(strided.elements::<bool>(), Ok(vec![false, true]));
        // Real floating-point operands are rejected, matching the scalar reference backend.
        assert!(matches!(
            Array::vector(vec![1.0]).and(&Array::vector(vec![0.0])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot apply `and` to arrays of element data types f64 and f64",
        ));
        // The `std::ops` sugar delegates to the fallible capabilities.
        assert_eq!(left.clone() & right.clone(), Array::vector(vec![true, false, false, false]));
        assert_eq!(!left.clone(), Array::vector(vec![false, false, true, true]));
    }
}
