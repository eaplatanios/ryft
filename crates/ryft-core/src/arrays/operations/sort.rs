//! Reference [`Array`] kernels for the sorting operation family contracts.
//!
//! Sorting ranks keys through a total order computed directly from each element's encoding, then applies the
//! resulting permutation by moving whole element encodings. Non-key operands therefore sort without being decoded,
//! including element data types that have no scalar representation.

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::encoding::{ArrayElement, i1, i2, i4, u1, u2, u4};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::operations::sort::{
    ArgMax, ArgMin, Sort, SortDirection, TopK, extremal_index_from_index_passenger, sort_permutation,
    top_k_from_index_passenger, top_k_via_squeezed_view,
};
use crate::programs::{ProgramError, TypeError, Typed};

// TODO(eaplatanios): Review this.

impl Array {
    /// Returns the stable-sort rank of one ordered element, or `None` for complex and payload-free element data
    /// types. Signed integers use sign-biased two's complement and floating-point values use IEEE total ordering.
    fn element_total_order_rank(data_type: DataType, bytes: &[u8]) -> Option<u64> {
        /// Maps an `f64` to its IEEE 754 total-order rank.
        fn floating_point_rank(value: f64) -> u64 {
            let bits = value.to_bits();
            if bits >> 63 == 1 { !bits } else { bits | (1 << 63) }
        }

        Some(match data_type {
            DataType::Boolean => u64::from(bool::decode(bytes)),
            DataType::I1 => (i1::decode(bytes).value() as u64) ^ (1 << 63),
            DataType::I2 => (i2::decode(bytes).value() as u64) ^ (1 << 63),
            DataType::I4 => (i4::decode(bytes).value() as u64) ^ (1 << 63),
            DataType::I8 => (i8::decode(bytes) as u64) ^ (1 << 63),
            DataType::I16 => (i16::decode(bytes) as u64) ^ (1 << 63),
            DataType::I32 => (i32::decode(bytes) as u64) ^ (1 << 63),
            DataType::I64 => (i64::decode(bytes) as u64) ^ (1 << 63),
            DataType::U1 => u64::from(u1::decode(bytes).value()),
            DataType::U2 => u64::from(u2::decode(bytes).value()),
            DataType::U4 => u64::from(u4::decode(bytes).value()),
            DataType::U8 => u64::from(u8::decode(bytes)),
            DataType::U16 => u64::from(u16::decode(bytes)),
            DataType::U32 => u64::from(u32::decode(bytes)),
            DataType::U64 => u64::decode(bytes),
            data_type if data_type.is_floating_point() => {
                floating_point_rank(Self::element_as_f64(data_type, bytes).unwrap())
            }
            DataType::C64 | DataType::C128 | DataType::Token | DataType::Zero => return None,
            _ => unreachable!(),
        })
    }
}

impl Sort for Array {
    fn sort_with_key_count(
        operands: &[Self],
        axis: usize,
        direction: SortDirection,
        key_count: usize,
    ) -> Result<Vec<Self>, ProgramError> {
        let Some(key) = operands.first() else {
            return Err(ProgramError::UnsupportedOperation { message: "'sort' needs at least one input".to_string() });
        };
        if key_count == 0 {
            return Err(ProgramError::UnsupportedOperation {
                message: "'sort' key_count must be at least 1".to_string(),
            });
        }
        if key_count > operands.len() {
            return Err(TypeError::invalid(format!(
                "'sort' key_count {} exceeds operand count {}",
                key_count,
                operands.len(),
            ))
            .into());
        }
        let shape = key.r#type().static_shape().unwrap();
        if axis >= shape.rank() {
            return Err(
                TypeError::invalid(format!("'sort' axis {axis} is out of bounds for rank {}", shape.rank())).into()
            );
        }
        for operand in operands {
            if operand.r#type().shape() != key.r#type().shape() {
                return Err(TypeError::invalid(format!(
                    "'sort' operands must agree on shape but got {} and {}",
                    key.r#type().shape(),
                    operand.r#type().shape(),
                ))
                .into());
            }
        }
        let key_ranks = operands[..key_count]
            .iter()
            .map(|key| {
                let data_type = key.r#type().data_type();
                let unsupported = || {
                    ProgramError::from(TypeError::invalid(format!("'sort' does not support key data type {data_type}")))
                };
                let addressing = ArrayAddressing::new(key.r#type().into_owned())?;
                (0..addressing.element_count())
                    .map(|index| {
                        Self::element_total_order_rank(
                            data_type,
                            &key.storage_bytes()[addressing.byte_range_for_flat_index(index)],
                        )
                        .ok_or_else(unsupported)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let key_rank_slices = key_ranks.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let gather = sort_permutation(key_rank_slices.as_slice(), shape.dimensions(), axis, direction);
        // Applying the gather map moves whole element encodings, so non-key operands of any element data type
        // (including the sub-byte ones without a scalar representation) sort without being decoded.
        operands
            .iter()
            .map(|operand| operand.gather_elements(operand.r#type().into_owned(), |index| gather[index]))
            .collect()
    }
}

/// Materializes the `i32` index passenger that rides a ranking sort for the concrete eager [`Array`] backend
/// (the transform tracers stage it as an [`IotaOperation`](crate::operations::constants::IotaOperation) instead),
/// returning it together with the operand's static dimensions.
fn eager_index_passenger(value: &Array, axis: usize) -> Result<(Array, Vec<usize>), ProgramError> {
    let shape = value.r#type().static_shape().unwrap();
    let dimensions = shape.dimensions().to_vec();
    if axis >= dimensions.len() {
        return Err(
            TypeError::invalid(format!("'sort' axis {} is out of bounds for rank {}", axis, dimensions.len())).into()
        );
    }
    let inner_stride: usize = dimensions[axis + 1..].iter().product();
    let axis_size = dimensions[axis];
    let indices = Array::from_fn_elements(ArrayType::new(DataType::I32, value.r#type().shape().clone()), |index| {
        Ok(((index / inner_stride) % axis_size) as i32)
    })?;
    Ok((indices, dimensions))
}

impl TopK for Array {
    fn top_k(&self, k: usize, axis: usize) -> Result<(Self, Self), ProgramError> {
        if let Some(outputs) = top_k_via_squeezed_view(self, k, axis)? {
            return Ok(outputs);
        }
        let (indices, dimensions) = eager_index_passenger(self, axis)?;
        top_k_from_index_passenger(self, indices, dimensions.as_slice(), k, axis)
    }
}

impl ArgMax for Array {
    fn argmax(&self, axis: usize) -> Result<Self, ProgramError> {
        let (indices, dimensions) = eager_index_passenger(self, axis)?;
        extremal_index_from_index_passenger(self, indices, dimensions.as_slice(), axis, SortDirection::Descending)
    }
}

impl ArgMin for Array {
    fn argmin(&self, axis: usize) -> Result<Self, ProgramError> {
        let (indices, dimensions) = eager_index_passenger(self, axis)?;
        extremal_index_from_index_passenger(self, indices, dimensions.as_slice(), axis, SortDirection::Ascending)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::arrays::types::layouts::{Layout, StridedLayout};

    use super::*;

    #[test]
    fn test_array_sort() {
        // Non-key operands sort by moving whole element encodings, so sub-byte operands (which have no scalar
        // representation) ride an f32 key without being decoded.
        let key = Array::vector(vec![3.0f32, 1.0, 2.0]);
        let passenger = Array::from_elements(
            array_type(DataType::I4, &[3]),
            &[i4::new(-8).unwrap(), i4::new(0).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let outputs = Array::sort_with_key_count(&[key, passenger.clone()], 0, SortDirection::Ascending, 1).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 2.0, 3.0]);
        assert_eq!(outputs[1].storage_bytes(), [0x00, 0x07, 0x08]);

        // Sub-byte keys decode directly through arbitrary physical layouts and therefore sort like every other
        // ordered element type.
        let key_type = array_type(DataType::I4, &[3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let key =
            Array::from_elements(key_type, &[i4::new(3).unwrap(), i4::new(-2).unwrap(), i4::new(1).unwrap()]).unwrap();
        let output = Array::sort_with_key_count(&[key], 0, SortDirection::Ascending, 1).unwrap().remove(0);
        assert_eq!(output.elements::<i4>(), Ok(vec![i4::new(-2).unwrap(), i4::new(1).unwrap(), i4::new(3).unwrap()]),);
    }
}
