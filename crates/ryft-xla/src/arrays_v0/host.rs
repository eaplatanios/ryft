use ryft_core::types::{ArrayType, StaticShape};

use crate::arrays::ArrayTypeExtension;
use crate::{Array, Error as XlaError, ToPjrt};

use super::*;

/// Returns row-major element strides for `global_shape`.
///
/// The returned vector has the same rank as `global_shape`, with `strides[i]` giving the number of
/// logical elements skipped when incrementing dimension `i` by one in a dense major-to-minor
/// layout.
fn row_major_element_strides(global_shape: &[usize], element_type: DataType) -> Result<Vec<usize>, ArrayError> {
    let mut strides = vec![1usize; global_shape.len()];
    let mut stride = 1usize;
    for dimension in (0..global_shape.len()).rev() {
        strides[dimension] = stride;
        stride = stride.checked_mul(global_shape[dimension]).ok_or_else(|| XlaError::SizeLimitExceeded {
            message: format!(
                "row-major stride for array with shape {global_shape:?} and element type {element_type} exceeds the \
                 maximum allowed size of {}",
                usize::MAX,
            ),
        })?;
    }
    Ok(strides)
}

/// Extracts the dense row-major bytes corresponding to `shard_slices` from `host_data`.
pub(crate) fn extract_dense_shard_bytes(
    host_data: &[u8],
    global_shape: &[usize],
    shard_slices: &[Range<usize>],
    element_type: DataType,
) -> Result<Vec<u8>, ArrayError> {
    if shard_slices.is_empty() {
        return Ok(host_data.to_vec());
    }

    let strides = row_major_element_strides(global_shape, element_type)?;
    let shard_shape = shard_slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
    let shard_byte_count = ArrayType::new(element_type, StaticShape::new(shard_shape).into(), None, None)
        .map_err(XlaError::from)?
        .size_in_bytes()?;
    let element_size_in_bytes = element_type.to_pjrt().element_size_in_bytes().map_err(XlaError::from)?;
    let mut shard_bytes = Vec::with_capacity(shard_byte_count);
    append_dense_shard_bytes(
        host_data,
        shard_slices,
        strides.as_slice(),
        0,
        0,
        element_size_in_bytes,
        &mut shard_bytes,
    );
    Ok(shard_bytes)
}

/// Appends the row-major bytes for the shard slice at `dimension` to `shard_bytes`.
fn append_dense_shard_bytes(
    host_data: &[u8],
    shard_slices: &[Range<usize>],
    strides: &[usize],
    dimension: usize,
    base_element_offset: usize,
    element_size_in_bytes: usize,
    shard_bytes: &mut Vec<u8>,
) {
    let slice = &shard_slices[dimension];
    if dimension + 1 == shard_slices.len() {
        let start_element_offset =
            base_element_offset + slice.start.checked_mul(strides[dimension]).expect("validated shard offsets fit");
        let end_element_offset =
            base_element_offset + slice.end.checked_mul(strides[dimension]).expect("validated shard offsets fit");
        let start_byte_offset =
            start_element_offset.checked_mul(element_size_in_bytes).expect("validated shard byte offsets fit");
        let end_byte_offset =
            end_element_offset.checked_mul(element_size_in_bytes).expect("validated shard byte offsets fit");
        shard_bytes.extend_from_slice(&host_data[start_byte_offset..end_byte_offset]);
        return;
    }

    for index in slice.start..slice.end {
        let element_offset =
            base_element_offset + index.checked_mul(strides[dimension]).expect("validated shard offsets fit");
        append_dense_shard_bytes(
            host_data,
            shard_slices,
            strides,
            dimension + 1,
            element_offset,
            element_size_in_bytes,
            shard_bytes,
        );
    }
}

pub(crate) fn materialize_dense_array_bytes(array: &Array<'_>) -> Result<Vec<u8>, ArrayError> {
    let global_shape = array.shape();
    let element_type = array.data_type();
    let total_byte_count = array.r#type.size_in_bytes()?;
    let mut global_bytes = vec![0u8; total_byte_count];
    let mut written = vec![false; total_byte_count];

    for shard in array.shards() {
        let device = shard.device();
        let shard_index = shard.index();
        let buffer = shard
            .buffer()
            .map(|buffer| buffer.as_ref())
            .ok_or(ArrayError::MissingAddressableShardForMove { shard_index, device_id: device.id() })?;
        let shard_bytes = buffer.copy_to_host(None)?.r#await()?;
        let shard_shape = shard.shape();
        let expected_byte_count = ArrayType::new(element_type, shard_shape.into(), None, None)
            .map_err(XlaError::from)?
            .size_in_bytes()?;
        if shard_bytes.len() != expected_byte_count {
            return Err(ArrayError::CopiedShardByteCountMismatch {
                shard_index,
                device_id: device.id(),
                expected_byte_count,
                actual_byte_count: shard_bytes.len(),
            });
        }
        merge_dense_shard_bytes(
            shard_bytes.as_slice(),
            global_shape.as_slice(),
            shard.slice(),
            element_type,
            shard_index,
            &mut global_bytes,
            &mut written,
        )?;
    }

    Ok(global_bytes)
}

/// Merges one shard's dense row-major host bytes into `global_bytes`.
fn merge_dense_shard_bytes(
    shard_bytes: &[u8],
    global_shape: &[usize],
    shard_slices: &[Range<usize>],
    element_type: DataType,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written: &mut [bool],
) -> Result<(), ArrayError> {
    if shard_slices.is_empty() {
        return merge_dense_byte_segment(shard_bytes, 0, shard_index, global_bytes, written);
    }

    let global_strides = row_major_element_strides(global_shape, element_type)?;
    let shard_shape = shard_slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
    let shard_strides = row_major_element_strides(shard_shape.as_slice(), element_type)?;
    let element_size_in_bytes = element_type.to_pjrt().element_size_in_bytes().map_err(XlaError::from)?;
    merge_dense_shard_bytes_recursive(
        shard_bytes,
        shard_slices,
        global_strides.as_slice(),
        shard_strides.as_slice(),
        0,
        0,
        0,
        element_size_in_bytes,
        shard_index,
        global_bytes,
        written,
    )
}

/// Recursively merges one shard's bytes into `global_bytes`.
fn merge_dense_shard_bytes_recursive(
    shard_bytes: &[u8],
    shard_slices: &[Range<usize>],
    global_strides: &[usize],
    shard_strides: &[usize],
    dimension: usize,
    base_global_element_offset: usize,
    base_shard_element_offset: usize,
    element_size_in_bytes: usize,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written: &mut [bool],
) -> Result<(), ArrayError> {
    let slice = &shard_slices[dimension];
    if dimension + 1 == shard_slices.len() {
        let global_element_offset = base_global_element_offset
            + slice.start.checked_mul(global_strides[dimension]).expect("validated global offsets fit");
        let global_byte_offset =
            global_element_offset.checked_mul(element_size_in_bytes).expect("validated global byte offsets fit");
        let shard_byte_offset = base_shard_element_offset
            .checked_mul(element_size_in_bytes)
            .expect("validated shard byte offsets fit");
        let byte_count = slice.len().checked_mul(element_size_in_bytes).expect("validated shard byte counts fit");
        return merge_dense_byte_segment(
            &shard_bytes[shard_byte_offset..shard_byte_offset + byte_count],
            global_byte_offset,
            shard_index,
            global_bytes,
            written,
        );
    }

    for (local_index, global_index) in (slice.start..slice.end).enumerate() {
        let next_global_element_offset = base_global_element_offset
            + global_index.checked_mul(global_strides[dimension]).expect("validated global offsets fit");
        let next_shard_element_offset = base_shard_element_offset
            + local_index.checked_mul(shard_strides[dimension]).expect("validated shard offsets fit");
        merge_dense_shard_bytes_recursive(
            shard_bytes,
            shard_slices,
            global_strides,
            shard_strides,
            dimension + 1,
            next_global_element_offset,
            next_shard_element_offset,
            element_size_in_bytes,
            shard_index,
            global_bytes,
            written,
        )?;
    }
    Ok(())
}

/// Merges `source_bytes` into `global_bytes` starting at `global_byte_offset`.
fn merge_dense_byte_segment(
    source_bytes: &[u8],
    global_byte_offset: usize,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written: &mut [bool],
) -> Result<(), ArrayError> {
    for (offset, &byte) in source_bytes.iter().enumerate() {
        let index = global_byte_offset + offset;
        if written[index] {
            if global_bytes[index] != byte {
                return Err(ArrayError::InconsistentOverlappingShardData { shard_index });
            }
        } else {
            global_bytes[index] = byte;
            written[index] = true;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shard-descriptor computation
// ---------------------------------------------------------------------------

pub(crate) trait DenseHostDevicePutLeaf {
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>);
}

pub(crate) trait DenseHostElement {
    const DATA_TYPE: DataType;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>);
}

macro_rules! impl_dense_host_element {
    ($ty:ty, $data_type:expr) => {
        impl DenseHostElement for $ty {
            const DATA_TYPE: DataType = $data_type;

            fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_ne_bytes());
            }
        }
    };
}

impl DenseHostElement for bool {
    const DATA_TYPE: DataType = DataType::Boolean;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(u8::from(*self));
    }
}

impl DenseHostElement for bf16 {
    const DATA_TYPE: DataType = DataType::BF16;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_bits().to_ne_bytes());
    }
}

impl DenseHostElement for f16 {
    const DATA_TYPE: DataType = DataType::F16;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_bits().to_ne_bytes());
    }
}

impl_dense_host_element!(i8, DataType::I8);
impl_dense_host_element!(i16, DataType::I16);
impl_dense_host_element!(i32, DataType::I32);
impl_dense_host_element!(i64, DataType::I64);
impl_dense_host_element!(u8, DataType::U8);
impl_dense_host_element!(u16, DataType::U16);
impl_dense_host_element!(u32, DataType::U32);
impl_dense_host_element!(u64, DataType::U64);
impl_dense_host_element!(f32, DataType::F32);
impl_dense_host_element!(f64, DataType::F64);

macro_rules! impl_scalar_dense_host_leaf {
    ($ty:ty) => {
        impl DenseHostDevicePutLeaf for $ty {
            fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
                let mut bytes = Vec::with_capacity(size_of::<$ty>());
                self.append_ne_bytes(&mut bytes);
                (Vec::new(), <$ty as DenseHostElement>::DATA_TYPE, bytes)
            }
        }
    };
}

impl_scalar_dense_host_leaf!(bool);
impl_scalar_dense_host_leaf!(i8);
impl_scalar_dense_host_leaf!(i16);
impl_scalar_dense_host_leaf!(i32);
impl_scalar_dense_host_leaf!(i64);
impl_scalar_dense_host_leaf!(u8);
impl_scalar_dense_host_leaf!(u16);
impl_scalar_dense_host_leaf!(u32);
impl_scalar_dense_host_leaf!(u64);
impl_scalar_dense_host_leaf!(bf16);
impl_scalar_dense_host_leaf!(f16);
impl_scalar_dense_host_leaf!(f32);
impl_scalar_dense_host_leaf!(f64);

#[cfg(feature = "ndarray")]
impl<T: Clone + DenseHostElement, D: ndarray::Dimension> DenseHostDevicePutLeaf for ndarray::Array<T, D> {
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
        let standard_layout = self.as_standard_layout().to_owned();
        let element_count = standard_layout.len();
        let mut bytes = Vec::with_capacity(element_count * size_of::<T>());
        for element in standard_layout.iter() {
            element.append_ne_bytes(&mut bytes);
        }
        (standard_layout.shape().to_vec(), T::DATA_TYPE, bytes)
    }
}
