//! Logical and bitwise operations on Boolean and integer values.
//!
//! Eager kernels combine validated encodings byte by byte. Their shared traversal handles broadcasting and physical
//! layouts without requiring any particular logical operation.

use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, Broadcastable};
use crate::programs::{ProgramError, TypeError, Typed};

pub mod and;
pub mod not;
pub mod or;
pub mod xor;

pub use and::{AND_OPERATION_NAME, And, AndOperation};
pub use not::{NOT_OPERATION_NAME, Not, NotOperation};
pub use or::{OR_OPERATION_NAME, Or, OrOperation};
pub use xor::{XOR_OPERATION_NAME, Xor, XorOperation};

// TODO(eaplatanios): Review this.

impl Array {
    /// Applies a binary logical or bitwise operation directly to validated Boolean or integer element bytes. Since
    /// bitwise operations act independently on every bit, their result is independent of integer signedness and host
    /// endianness. Logical Boolean encodings use the same `0` and `1` bitwise truth tables.
    pub(crate) fn binary_logical(
        &self,
        rhs: &Self,
        operation: &str,
        function: impl Fn(u8, u8) -> u8,
    ) -> Result<Self, ProgramError> {
        let left_data_type = self.r#type().data_type();
        let right_data_type = rhs.r#type().data_type();
        let output_type = Broadcastable::broadcast(self.r#type().as_ref(), rhs.r#type().as_ref())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        if left_data_type != right_data_type || !(left_data_type.is_boolean() || left_data_type.is_integer()) {
            return Err(TypeError::invalid(format!(
                "cannot apply `{operation}` to arrays of element data types `{left_data_type}` and `{right_data_type}`",
            ))
            .into());
        }

        let output_shape = output_type.static_shape().unwrap();
        let left_shape = self.r#type().static_shape().unwrap();
        let right_shape = rhs.r#type().static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let left_strides = left_shape.row_major_strides();
        let right_strides = right_shape.row_major_strides();
        let left_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let right_addressing = ArrayAddressing::new(rhs.r#type().into_owned())?;
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
        Ok(Self::new_unchecked(output_type, Arc::new(output_bytes)))
    }
}
