use std::ops::Range;

use ryft_pjrt::Event;

use ryft_core::Typed;
use ryft_core::types::ArrayType;

use crate::arrays::ArrayTypeExtension;
use crate::{Array, Error as XlaError, ToPjrt};

use super::*;

/// Returns row-major element strides for `global_shape`. `strides[i]` is the number of logical
/// elements skipped when incrementing dimension `i` by one in a dense major-to-minor layout.
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

/// One in-flight device-to-host shard copy and the metadata needed to merge it.
struct PendingDenseShardHostCopy {
    /// Completion event carrying the copied shard bytes.
    event: Event<Vec<u8>>,

    /// Stable global shard index used in diagnostics.
    shard_index: ShardIndex,

    /// Device identifier used in diagnostics.
    device_id: DeviceId,

    /// Expected byte count derived from the shard's logical shape.
    expected_byte_count: usize,

    /// Global slices covered by the shard.
    slices: Vec<Range<usize>>,
}

/// In-flight copies for materializing one array as dense row-major host bytes.
///
/// Construction starts every addressable shard copy without waiting. [`Self::finish`] then awaits the already-issued
/// transfers and merges multi-shard layouts. This separation lets callers issue copies for several arrays before
/// blocking on any one transfer.
pub(crate) struct DenseArrayHostCopy {
    /// Global logical shape of the source array.
    global_shape: StaticShape,

    /// Element type of the source array.
    element_type: DataType,

    /// Total byte count of the global dense representation.
    total_byte_count: usize,

    /// In-flight copies, one per global shard.
    shards: Vec<PendingDenseShardHostCopy>,
}

impl DenseArrayHostCopy {
    /// Waits for every previously issued copy and produces one dense row-major byte buffer.
    pub(crate) fn finish(self) -> Result<Vec<u8>, ArrayError> {
        self.finish_with_measurements().map(|(bytes, _)| bytes)
    }

    /// Finishes this copy and reports whether a global merge buffer was allocated.
    pub(crate) fn finish_with_measurements(self) -> Result<(Vec<u8>, bool), ArrayError> {
        if self.shards.is_empty() {
            return Ok((Vec::new(), false));
        }

        let mut shards = self
            .shards
            .into_iter()
            .map(|shard| {
                let PendingDenseShardHostCopy { event, shard_index, device_id, expected_byte_count, slices } = shard;
                let bytes = event.r#await()?;
                if bytes.len() != expected_byte_count {
                    return Err(ArrayError::CopiedShardByteCountMismatch {
                        shard_index,
                        device_id,
                        expected_byte_count,
                        actual_byte_count: bytes.len(),
                    });
                }
                Ok((shard_index, slices, bytes))
            })
            .collect::<Result<Vec<_>, ArrayError>>()?;

        if shards.len() == 1 {
            let (shard_index, slices, bytes) = shards.pop().unwrap();
            let covers_global_shape = (slices.is_empty() && self.global_shape.rank() == 0)
                || (slices.len() == self.global_shape.rank()
                    && slices
                        .iter()
                        .zip(self.global_shape.as_slice())
                        .all(|(slice, extent)| slice.start == 0 && slice.end == *extent));
            if covers_global_shape && bytes.len() == self.total_byte_count {
                return Ok((bytes, false));
            }
            shards.push((shard_index, slices, bytes));
        }

        let mut global_bytes = vec![0u8; self.total_byte_count];
        let mut written_intervals = Vec::new();
        for (shard_index, slices, bytes) in shards {
            merge_dense_shard_bytes(
                bytes.as_slice(),
                self.global_shape.as_slice(),
                slices.as_slice(),
                self.element_type,
                shard_index,
                &mut global_bytes,
                &mut written_intervals,
            )?;
        }
        Ok((global_bytes, true))
    }

    /// Returns the number of already-issued device-to-host shard copies represented by this materialization.
    #[inline]
    pub(crate) fn shard_copy_count(&self) -> usize {
        self.shards.len()
    }
}

/// Starts copies for every addressable shard of `array` without awaiting them. Errors when any global shard is not
/// addressable from the current process.
pub(crate) fn begin_materialize_dense_array_bytes(array: &Array<'_>) -> Result<DenseArrayHostCopy, ArrayError> {
    if array.data_type().is_zero() {
        return Ok(DenseArrayHostCopy {
            global_shape: array.shape(),
            element_type: array.data_type(),
            total_byte_count: 0,
            shards: Vec::new(),
        });
    }

    let global_shape = array.shape();
    let element_type = array.data_type();
    let total_byte_count = array.r#type().size_in_bytes()?;
    let shards = array
        .shards()
        .iter()
        .map(|shard| {
            let device = shard.device();
            let shard_index = shard.index();
            let buffer = shard
                .buffer()
                .map(|buffer| buffer.as_ref())
                .ok_or(ArrayError::MissingAddressableShardForMove { shard_index, device_id: device.id() })?;
            let shard_shape = shard.shape();
            Ok(PendingDenseShardHostCopy {
                event: buffer.copy_to_host(None)?,
                shard_index,
                device_id: device.id(),
                expected_byte_count: ArrayType::new(element_type, shard_shape.into()).size_in_bytes()?,
                slices: shard.slice().to_vec(),
            })
        })
        .collect::<Result<Vec<_>, ArrayError>>()?;
    Ok(DenseArrayHostCopy { global_shape, element_type, total_byte_count, shards })
}

/// Copies every addressable shard of `array` to host and merges them into one dense row-major byte buffer. Errors when
/// any global shard is not addressable from the current process.
///
/// Used by [`Array::to`](crate::Array::to)'s last-resort host fallback when the fast and compiled paths cannot satisfy
/// the requested placement.
pub(crate) fn materialize_dense_array_bytes(array: &Array<'_>) -> Result<Vec<u8>, ArrayError> {
    begin_materialize_dense_array_bytes(array)?.finish()
}

/// Merges one shard's dense row-major host bytes into `global_bytes`.
fn merge_dense_shard_bytes(
    shard_bytes: &[u8],
    global_shape: &[usize],
    shard_slices: &[std::ops::Range<usize>],
    element_type: DataType,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written_intervals: &mut Vec<Range<usize>>,
) -> Result<(), ArrayError> {
    if shard_slices.is_empty() {
        return merge_dense_byte_segment(shard_bytes, 0, shard_index, global_bytes, written_intervals);
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
        written_intervals,
    )
}

/// Recursively merges one shard's bytes into `global_bytes`.
fn merge_dense_shard_bytes_recursive(
    shard_bytes: &[u8],
    shard_slices: &[std::ops::Range<usize>],
    global_strides: &[usize],
    shard_strides: &[usize],
    dimension: usize,
    base_global_element_offset: usize,
    base_shard_element_offset: usize,
    element_size_in_bytes: usize,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written_intervals: &mut Vec<Range<usize>>,
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
            written_intervals,
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
            written_intervals,
        )?;
    }
    Ok(())
}

/// Merges `source_bytes` into `global_bytes` starting at `global_byte_offset`. Errors when two
/// overlapping shards disagree on a byte's value.
fn merge_dense_byte_segment(
    source_bytes: &[u8],
    global_byte_offset: usize,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written_intervals: &mut Vec<Range<usize>>,
) -> Result<(), ArrayError> {
    // Zero-extent shard slices produce empty segments; recording their degenerate `[x, x)` intervals would only
    // bloat the overlap bookkeeping without ever contributing bytes.
    if source_bytes.is_empty() {
        return Ok(());
    }
    let range = global_byte_offset..global_byte_offset + source_bytes.len();
    let insertion_index = written_intervals.partition_point(|written| written.start <= range.start);
    let mut first_overlap = insertion_index.saturating_sub(1);
    while first_overlap < written_intervals.len() && written_intervals[first_overlap].end <= range.start {
        first_overlap += 1;
    }
    let mut overlap_index = first_overlap;
    while overlap_index < written_intervals.len() && written_intervals[overlap_index].start < range.end {
        let written = &written_intervals[overlap_index];
        let overlap = written.start.max(range.start)..written.end.min(range.end);
        if overlap.start < overlap.end {
            let source_offset = overlap.start - range.start;
            if global_bytes[overlap.clone()] != source_bytes[source_offset..source_offset + overlap.len()] {
                return Err(ArrayError::InconsistentOverlappingShardData { shard_index });
            }
        }
        overlap_index += 1;
    }
    global_bytes[range.clone()].copy_from_slice(source_bytes);

    let mut merged_start = range.start;
    let mut merged_end = range.end;
    let mut remove_start = insertion_index;
    if remove_start > 0 && written_intervals[remove_start - 1].end >= merged_start {
        remove_start -= 1;
        merged_start = merged_start.min(written_intervals[remove_start].start);
        merged_end = merged_end.max(written_intervals[remove_start].end);
    }
    let mut remove_end = remove_start;
    while remove_end < written_intervals.len() && written_intervals[remove_end].start <= merged_end {
        merged_end = merged_end.max(written_intervals[remove_end].end);
        remove_end += 1;
    }
    written_intervals.splice(remove_start..remove_end, [merged_start..merged_end]);
    Ok(())
}

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
