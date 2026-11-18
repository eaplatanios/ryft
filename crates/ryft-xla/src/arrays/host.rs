//! Readback of device arrays to host memory, including the merging of shards and of unreduced values.
//!
//! [`Array::begin_copy_to_host`] issues the device-to-host copies of the shards that hold the value of an array without
//! waiting for them, and [`DenseArrayHostCopy::finish`] then waits for those copies and assembles the dense row-major
//! bytes of the array, which is what [`Array::copy_to_host`] returns. Separating the two steps lets callers issue the
//! copies of several arrays before blocking on any one of them. This module also provides the host round trips that
//! must preserve the separate values that devices hold along unreduced mesh axes (i.e.,
//! [`Array::relayout_through_host`] and [`Array::validate_host_round_trip`]).

use std::ops::Range;
use std::sync::Arc;

use ryft_core::{ArrayType, DataType, Layout, ProgramError, StaticShape, Typed, UnreducedKind};
use ryft_pjrt::{DeviceId, ExecutionFence, HostDestination, HostDestinationSlice};

use crate::arrays::{Array, ArrayError, ArrayShard, ArrayTypeExtension, ShardIndex};
use crate::sessions::synchronization::HostCopy;
use crate::{Error, Session, ToPjrt};

// TODO(eaplatanios): Review this module.

/// In-flight copies for materializing one array as dense row-major host bytes.
///
/// Construction starts every needed shard copy without waiting. [`Self::finish`] then awaits the transfers that were
/// already issued and merges the shards when there is more than one. This separation lets callers issue copies for
/// several arrays before blocking on any one transfer.
pub(crate) struct DenseArrayHostCopy {
    /// Global logical shape of the source array.
    global_shape: StaticShape,

    /// Element type of the source array.
    element_type: DataType,

    /// [`UnreducedKind`] that combines the per-device values of the source array along its unreduced axes.
    unreduced_kind: UnreducedKind,

    /// Total byte count of the global dense representation.
    total_byte_count: usize,

    /// In-flight copies of the shards.
    shards: DenseShardHostCopies,

    /// Producer completion, including when no payload copy is needed to observe an execution error.
    completion: Option<ExecutionFence>,
}

impl DenseArrayHostCopy {
    /// Returns the number of device-to-host shard copies that this materialization has already issued.
    #[inline]
    pub(crate) fn shard_copy_count(&self) -> usize {
        match &self.shards {
            DenseShardHostCopies::Merged(shards) => shards.len(),
            DenseShardHostCopies::Direct { copies, .. } => copies.len(),
        }
    }

    /// Waits for every previously issued copy and produces one dense row-major byte buffer.
    pub(crate) fn finish(self) -> Result<Vec<u8>, Error> {
        self.finish_with_measurements().map(|(bytes, _)| bytes)
    }

    /// Finishes this copy and reports whether a global merge buffer was allocated.
    pub(crate) fn finish_with_measurements(self) -> Result<(Vec<u8>, bool), Error> {
        if let Some(completion) = &self.completion {
            completion.block_until_ready()?;
        }
        let shards = match self.shards {
            DenseShardHostCopies::Merged(shards) if shards.is_empty() => return Ok((Vec::new(), false)),
            DenseShardHostCopies::Merged(shards) => shards,
            DenseShardHostCopies::Direct { destination, copies } => {
                // Every slice is dropped once its copy completes, which makes the destination's bytes accessible.
                for copy in copies {
                    drop(copy.wait()?);
                }
                let bytes = destination.into_bytes().ok().unwrap();
                return Ok((bytes, false));
            }
        };
        let shards = shards
            .into_iter()
            .map(|shard| {
                let PendingDenseShardHostCopy {
                    copy,
                    shard_index,
                    device_id,
                    expected_byte_count,
                    slices,
                    unreduced_coordinates,
                } = shard;
                let bytes = copy.wait()?;
                if bytes.len() != expected_byte_count {
                    return Err(Error::from(ArrayError::ShardByteCountMismatch {
                        shard_index,
                        device_id,
                        expected_byte_count,
                        actual_byte_count: bytes.len(),
                    }));
                }
                Ok((shard_index, slices, unreduced_coordinates, bytes))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut shards = combine_unreduced_values(shards, self.element_type, self.unreduced_kind)?;

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
}

impl Array {
    /// Starts copies for the shards of this array that hold its whole value without awaiting them. Those are all the
    /// shards, or only the addressable ones when some shard lives in another process but the addressable ones cover the
    /// array and it has no unreduced axes (e.g., for a replicated array).
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] before issuing any copy when the current process does not hold the
    /// whole value. PJRT transfer failures are returned unchanged.
    pub(crate) fn begin_copy_to_host(&self) -> Result<DenseArrayHostCopy, Error> {
        let completion = self.execution_fence().cloned();
        // Empty arrays have no payload either. Avoid asking PJRT for a dynamic copy of zero bytes, whose fallback for
        // the size query can expose the bound allocation and its trailer instead of the empty logical value.
        if self.data_type().is_zero() || self.shape().as_slice().contains(&0) {
            return Ok(DenseArrayHostCopy {
                global_shape: self.shape(),
                element_type: self.data_type(),
                unreduced_kind: self.sharding().unreduced_kind(),
                total_byte_count: 0,
                shards: DenseShardHostCopies::Merged(Vec::new()),
                completion: Some(match completion {
                    Some(completion) => completion,
                    None => ExecutionFence::new(
                        self.addressable_shards()
                            .filter_map(|shard| shard.buffer())
                            .map(|buffer| buffer.ready())
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                }),
            });
        }

        let global_shape = self.shape();
        let element_type = self.data_type();
        let total_byte_count = self.r#type().size_in_bytes()?;
        let mesh = self.mesh();
        let unreduced_axis_indices = mesh
            .logical_mesh()
            .axes()
            .iter()
            .enumerate()
            .filter(|(_, axis)| self.sharding().unreduced_axes().contains(axis.name()))
            .map(|(axis_index, _)| axis_index)
            .collect::<Vec<_>>();
        // Select the shards to copy before issuing any copy, so that no copy is in flight when the selection fails.
        // Every shard is needed when some shard lives in another process and the array has unreduced axes, whose
        // per-device values must all be combined. Otherwise, the addressable shards suffice when their distinct tiles
        // cover the whole array (e.g., for a replicated array, which JAX also reads when the current process holds it
        // in full), because the tiles of a sharding are either disjoint or identical.
        let all_shards = self.shards();
        let shards = match all_shards.iter().find(|shard| shard.buffer().is_none()) {
            None => all_shards.iter().collect::<Vec<_>>(),
            Some(missing) => {
                let addressable = all_shards.iter().filter(|shard| shard.buffer().is_some()).collect::<Vec<_>>();
                let mut tiles = Vec::<&[Range<usize>]>::new();
                for shard in &addressable {
                    if !tiles.contains(&shard.slice()) {
                        tiles.push(shard.slice());
                    }
                }
                let covered =
                    tiles.iter().map(|slices| slices.iter().map(|slice| slice.len()).product::<usize>()).sum::<usize>();
                if !unreduced_axis_indices.is_empty() || covered != global_shape.as_slice().iter().product::<usize>() {
                    return Err(Error::from(ArrayError::MissingShardBuffer {
                        shard_index: missing.index(),
                        device_id: missing.device().id(),
                    }));
                }
                addressable
            }
        };
        let unreduced_kind = self.sharding().unreduced_kind();
        let mut work = self.session().reserve_work();

        // Without unreduced axes, every device holds its tile of the array itself, so one shard per distinct tile
        // suffices. When every tile occupies one contiguous byte range of the dense representation (e.g., when only the
        // leading dimension is sharded, or for a replicated array), each is copied directly into that range of one
        // destination.
        if unreduced_axis_indices.is_empty() {
            let mut tiles = Vec::<(&ArrayShard, Range<usize>)>::new();
            let mut contiguous = true;
            for shard in &shards {
                if tiles.iter().any(|(tile, _)| tile.slice() == shard.slice()) {
                    continue;
                }
                match contiguous_tile_byte_range(shard.slice(), global_shape.as_slice(), element_type)? {
                    Some(range) => tiles.push((shard, range)),
                    None => {
                        contiguous = false;
                        break;
                    }
                }
            }
            if contiguous {
                let mut destination = HostDestination::new(total_byte_count);
                let ranges = tiles.iter().map(|(_, range)| range.clone()).collect::<Vec<_>>();
                let slices = destination.split(ranges.as_slice())?;
                let copies = tiles
                    .iter()
                    .zip(slices)
                    .map(|((shard, _), slice)| {
                        let layout = ryft_pjrt::Layout::dense_major_to_minor(shard.shape().rank());
                        let event = shard.buffer().unwrap().copy_to_host_buffer(Some(layout), slice)?;
                        Ok(work.track_host_copy(event))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let shards = DenseShardHostCopies::Direct { destination, copies };
                return Ok(DenseArrayHostCopy {
                    global_shape,
                    element_type,
                    unreduced_kind,
                    total_byte_count,
                    shards,
                    completion,
                });
            }
        }

        let shards = shards
            .into_iter()
            .map(|shard| {
                let device = shard.device();
                let shard_index = shard.index();
                let buffer = shard.buffer().map(|buffer| buffer.as_ref()).unwrap();
                let shard_shape = shard.shape();
                // The merge consumes row-major bytes even when the device buffer has another physical layout.
                let event = buffer.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(shard_shape.rank())))?;
                Ok(PendingDenseShardHostCopy {
                    copy: work.track_host_copy(event),
                    shard_index,
                    device_id: device.id(),
                    expected_byte_count: ArrayType::new(element_type, shard_shape.into()).size_in_bytes()?,
                    slices: shard.slice().to_vec(),
                    unreduced_coordinates: match mesh.device_coordinates(shard_index) {
                        Some(coordinates) => unreduced_axis_indices.iter().map(|&axis| coordinates[axis]).collect(),
                        None => Vec::new(),
                    },
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let shards = DenseShardHostCopies::Merged(shards);
        Ok(DenseArrayHostCopy { global_shape, element_type, unreduced_kind, total_byte_count, shards, completion })
    }

    /// Converts this array to `layout` by copying each of its addressable shards to the host and uploading it back to
    /// the same memory in `layout`, so that every device keeps its own shard bytes. This is the host relayout for an
    /// array whose sharding has unreduced axes, whose devices hold separate summands of its value. Materializing such
    /// an array densely would add those summands, and uploading the sum would replace them with a canonical
    /// decomposition of it, which an unchecked `shard_map` body could tell apart, because it observes the summands.
    pub(crate) fn relayout_through_host(&self, session: &Session, layout: &Layout) -> Result<Array, Error> {
        let r#type = self.r#type().into_owned().with_layout(layout.clone());
        let device_layout = layout.to_pjrt()?;
        let buffer_type = Array::physical_buffer_type(self.data_type());
        let client = session.client();
        let mut work = session.reserve_work();
        let mut buffers = Vec::new();
        for shard in self.addressable_shards() {
            let shard_index = shard.index();
            let buffer = shard.buffer().ok_or_else(|| {
                Error::from(ArrayError::MissingShardBuffer { shard_index, device_id: shard.device().id() })
            })?;
            let shape = shard.shape();
            let copy = buffer.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(shape.rank())))?;
            let bytes = work.track_host_copy(copy).wait()?;
            let dimensions = shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let uploaded = client.buffer(
                Arc::<[u8]>::from(bytes),
                buffer_type,
                dimensions.as_slice(),
                None,
                buffer.memory()?,
                Some(device_layout.clone()),
            )?;
            work.track_buffer(&uploaded);
            buffers.push(uploaded);
        }
        Array::from_addressable_buffers(session, r#type, self.mesh().clone(), Some(self.sharding().clone()), buffers)
    }

    /// Rejects a dense host round trip of this array for `purpose` (i.e., materializing its logical value on the host
    /// and uploading a value derived from it) when the sharding of this array has unreduced axes. The devices of such
    /// an array hold separate summands of its value, which the round trip would replace with a canonical decomposition
    /// of their sum, and an unchecked `shard_map` body observes those summands. `purpose` names the round trip in the
    /// error message (e.g., `"pad"`).
    pub(crate) fn validate_host_round_trip(&self, purpose: &str) -> Result<(), Error> {
        if self.sharding().unreduced_axes().is_empty() {
            return Ok(());
        }
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot {purpose} an array of type `{}` through the host, because its devices hold separate summands \
                 along its unreduced axes, which a dense host copy would not preserve",
                self.r#type(),
            ),
        }
        .into())
    }
}

/// One in-flight device-to-host shard copy and the metadata needed to merge it.
struct PendingDenseShardHostCopy {
    /// Tracked copy carrying the shard bytes.
    copy: HostCopy<Vec<u8>>,

    /// Stable global shard index used in diagnostics.
    shard_index: ShardIndex,

    /// Device identifier used in diagnostics.
    device_id: DeviceId,

    /// Expected byte count derived from the shard's logical shape.
    expected_byte_count: usize,

    /// Global slices covered by the shard.
    slices: Vec<Range<usize>>,

    /// Coordinates of the shard's device along the unreduced axes of the array's sharding, in mesh order, which are
    /// empty for an array without unreduced axes. Shards that cover the same slices but differ in these coordinates
    /// hold separate per-device values (e.g., summands) of that tile, while shards that also agree on them are
    /// replicas of one per-device value.
    unreduced_coordinates: Vec<usize>,
}

/// In-flight device-to-host copies of the shards of one array.
enum DenseShardHostCopies {
    /// Copies of the shards into buffers of their own, which [`DenseArrayHostCopy::finish`] combines along unreduced
    /// axes and merges into the dense representation of the array.
    Merged(Vec<PendingDenseShardHostCopy>),

    /// Copies of one shard per distinct tile of the array directly into the range of `destination` that the tile
    /// occupies in the dense representation of the array, which needs neither per-shard buffers nor a merge (refer to
    /// [`contiguous_tile_byte_range`] for when this applies).
    Direct { destination: HostDestination, copies: Vec<HostCopy<HostDestinationSlice>> },
}

/// Returns row-major element strides for `global_shape`. `strides[i]` is the number of logical elements skipped when
/// incrementing dimension `i` by one in a dense major-to-minor layout. `element_type` is only used in the error that
/// reports a stride that overflows [`usize`].
fn row_major_element_strides(global_shape: &[usize], element_type: DataType) -> Result<Vec<usize>, Error> {
    let mut strides = vec![1usize; global_shape.len()];
    let mut stride = 1usize;
    for dimension in (0..global_shape.len()).rev() {
        strides[dimension] = stride;
        stride = stride.checked_mul(global_shape[dimension]).ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "row-major stride for array with shape `{global_shape:?}` and element type `{element_type}` exceeds \
                 the maximum allowed size of {}",
                usize::MAX,
            ),
        })?;
    }
    Ok(strides)
}

/// Returns the byte range that the tile covering `slices` of an array of shape `global_shape` and element type
/// `element_type` occupies in the dense row-major representation of the array, or [`None`] when that range is not
/// contiguous or the element type is not byte-addressable. A tile is contiguous when it spans one index along every
/// dimension before some dimension and the whole extent along every dimension after it.
fn contiguous_tile_byte_range(
    slices: &[Range<usize>],
    global_shape: &[usize],
    element_type: DataType,
) -> Result<Option<Range<usize>>, Error> {
    let element_size = element_type.to_pjrt().element_size_in_bytes()?;
    let tile_shape = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
    let tile_element_count = tile_shape.iter().product::<usize>();
    if ArrayType::new(element_type, StaticShape::new(tile_shape).into()).size_in_bytes()?
        != tile_element_count * element_size
    {
        return Ok(None);
    }
    let first_partial = slices.iter().position(|slice| slice.len() != 1).unwrap_or(slices.len());
    let trailing_dimensions_are_whole = slices
        .iter()
        .zip(global_shape)
        .skip(first_partial + 1)
        .all(|(slice, extent)| slice.start == 0 && slice.end == *extent);
    if !trailing_dimensions_are_whole {
        return Ok(None);
    }
    let strides = row_major_element_strides(global_shape, element_type)?;
    let start = slices.iter().zip(&strides).map(|(slice, stride)| slice.start * stride).sum::<usize>() * element_size;
    Ok(Some(start..start + tile_element_count * element_size))
}

/// Combines the host copies of the shards of an array whose sharding has unreduced axes into one copy per tile, which
/// holds the logical value of that tile, and returns the copies of an array without unreduced axes unchanged. Each
/// copy carries its shard index, the global slices that it covers, its device's coordinates along the unreduced axes,
/// and its dense row-major bytes. Copies that cover the same slices and agree on those coordinates are replicas of
/// one per-device value and must hold the same bytes. The per-device values of a tile, which are the copies of its
/// slices that differ in those coordinates, are combined elementwise with `unreduced_kind` (i.e., added, or reduced to
/// their maximum or minimum).
fn combine_unreduced_values(
    shards: Vec<(ShardIndex, Vec<Range<usize>>, Vec<usize>, Vec<u8>)>,
    element_type: DataType,
    unreduced_kind: UnreducedKind,
) -> Result<Vec<(ShardIndex, Vec<Range<usize>>, Vec<u8>)>, Error> {
    if shards.iter().all(|(_, _, unreduced_coordinates, _)| unreduced_coordinates.is_empty()) {
        return Ok(shards.into_iter().map(|(shard_index, slices, _, bytes)| (shard_index, slices, bytes)).collect());
    }
    let mut summands = Vec::<(Vec<Range<usize>>, Vec<usize>, ShardIndex, Vec<u8>)>::new();
    for (shard_index, slices, unreduced_coordinates, bytes) in shards {
        match summands.iter().find(|(summand_slices, summand_coordinates, _, _)| {
            *summand_slices == slices && *summand_coordinates == unreduced_coordinates
        }) {
            Some((_, _, _, summand_bytes)) if *summand_bytes != bytes => {
                return Err(Error::from(ArrayError::InconsistentOverlappingShardData { shard_index }));
            }
            Some(_) => {}
            None => summands.push((slices, unreduced_coordinates, shard_index, bytes)),
        }
    }
    let mut tiles = Vec::<(ShardIndex, Vec<Range<usize>>, ryft_core::Array)>::new();
    for (slices, _, shard_index, bytes) in summands {
        let shape = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
        let summand = ryft_core::Array::new(ArrayType::new_static(element_type, shape), bytes)?;
        match tiles.iter_mut().find(|(_, tile_slices, _)| *tile_slices == slices) {
            Some((_, _, tile)) => {
                *tile = match unreduced_kind {
                    UnreducedKind::Sum => ryft_core::Add::add(&*tile, &summand)?,
                    UnreducedKind::Max => ryft_core::Max::max(&*tile, &summand)?,
                    UnreducedKind::Min => ryft_core::Min::min(&*tile, &summand)?,
                }
            }
            None => tiles.push((shard_index, slices, summand)),
        }
    }
    Ok(tiles
        .into_iter()
        .map(|(shard_index, slices, tile)| (shard_index, slices, tile.logical_bytes()))
        .collect())
}

/// Merges one shard's dense row-major host bytes into `global_bytes`, which holds the dense row-major bytes of an array
/// of shape `global_shape`. `written_intervals` holds the sorted, disjoint byte ranges of `global_bytes` that earlier
/// shards wrote, which [`merge_dense_byte_segment`] uses to detect inconsistent overlapping data.
fn merge_dense_shard_bytes(
    shard_bytes: &[u8],
    global_shape: &[usize],
    shard_slices: &[Range<usize>],
    element_type: DataType,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written_intervals: &mut Vec<Range<usize>>,
) -> Result<(), Error> {
    if shard_slices.is_empty() {
        return merge_dense_byte_segment(shard_bytes, 0, shard_index, global_bytes, written_intervals);
    }

    let global_strides = row_major_element_strides(global_shape, element_type)?;
    let shard_shape = shard_slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
    let shard_strides = row_major_element_strides(shard_shape.as_slice(), element_type)?;
    let element_size_in_bytes = element_type.to_pjrt().element_size_in_bytes()?;
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

/// Recursively merges one shard's bytes into `global_bytes`, from dimension `dimension` onward and starting at the
/// element offsets `base_global_element_offset` and `base_shard_element_offset` (refer to
/// [`merge_dense_shard_bytes`]).
#[allow(clippy::too_many_arguments)]
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
    written_intervals: &mut Vec<Range<usize>>,
) -> Result<(), Error> {
    let slice = &shard_slices[dimension];
    if dimension + 1 == shard_slices.len() {
        let global_element_offset =
            base_global_element_offset + slice.start.checked_mul(global_strides[dimension]).unwrap();
        let global_byte_offset = global_element_offset.checked_mul(element_size_in_bytes).unwrap();
        let shard_byte_offset = base_shard_element_offset.checked_mul(element_size_in_bytes).unwrap();
        let byte_count = slice.len().checked_mul(element_size_in_bytes).unwrap();
        return merge_dense_byte_segment(
            &shard_bytes[shard_byte_offset..shard_byte_offset + byte_count],
            global_byte_offset,
            shard_index,
            global_bytes,
            written_intervals,
        );
    }

    for (local_index, global_index) in (slice.start..slice.end).enumerate() {
        let next_global_element_offset =
            base_global_element_offset + global_index.checked_mul(global_strides[dimension]).unwrap();
        let next_shard_element_offset =
            base_shard_element_offset + local_index.checked_mul(shard_strides[dimension]).unwrap();
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

/// Merges `source_bytes` into `global_bytes` starting at `global_byte_offset` and records the written range in
/// `written_intervals`, which stays sorted and disjoint. Returns [`ArrayError::InconsistentOverlappingShardData`] when
/// two overlapping shards disagree on the value of a byte.
fn merge_dense_byte_segment(
    source_bytes: &[u8],
    global_byte_offset: usize,
    shard_index: ShardIndex,
    global_bytes: &mut [u8],
    written_intervals: &mut Vec<Range<usize>>,
) -> Result<(), Error> {
    // Shard slices with zero extents produce empty segments, and recording their degenerate `[x, x)` intervals would
    // only bloat the overlap bookkeeping without ever contributing bytes.
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
                return Err(Error::from(ArrayError::InconsistentOverlappingShardData { shard_index }));
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
    // The replacement is one interval, not the elements of a range.
    #[allow(clippy::single_range_in_vec_init)]
    written_intervals.splice(remove_start..remove_end, [merged_start..merged_end]);
    Ok(())
}

// Per-axis tile ranges are lists that often hold a single range, which this lint mistakes for a range to collect.
#[allow(clippy::single_range_in_vec_init)]
#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_pjrt::{Error as PjrtError, ExecutionFence};

    use crate::FromPjrt;
    use crate::sessions::tests::{domain_mesh, session_with_client};
    use crate::tests::{cpu_client, execution_client, values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_array_begin_copy_to_host_preserves_producer_errors_for_empty_arrays() {
        let client = execution_client();
        let session = session_with_client(client);
        let mesh = domain_mesh(client, "x", 1);
        for data_type in [DataType::F32, DataType::Zero] {
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(PjrtError::invalid_argument("empty producer failed"))).unwrap();
            let array =
                Array::from_host_buffer(&session, ArrayType::new_static(data_type, [0]), mesh.clone(), None, [])
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(vec![event]));
            let copy = array.begin_copy_to_host().unwrap();
            assert_eq!(copy.shard_copy_count(), 0);
            assert_eq!(copy.finish().unwrap_err().to_string(), "empty producer failed");
        }
    }

    #[test]
    fn test_array_begin_copy_to_host_copies_contiguous_tiles_in_place() {
        // On a `2x2` mesh, a `[4, 2]` array whose rows are sharded along `x` has two distinct tiles, each one
        // contiguous range of the dense array, so one replica of each is copied directly into place without a merge.
        // Sharding its columns makes the tiles non-contiguous, so every shard is copied into its own buffer and merged,
        // and a replicated array needs a single copy.
        let client = cpu_client(4);
        let session = session_with_client(client);
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let values = [1f32, 2., 3., 4., 5., 6., 7., 8.];
        let materialize = |dimensions: Vec<ShardingDimension>| {
            let sharding = Sharding::new(logical_mesh.clone(), dimensions).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap();
            let array =
                Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&values)).unwrap();
            let copy = array.begin_copy_to_host().unwrap();
            let shard_copy_count = copy.shard_copy_count();
            let (bytes, merged) = copy.finish_with_measurements().unwrap();
            (values_from_bytes::<f32>(bytes.as_slice()), shard_copy_count, merged)
        };
        let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let columns = vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])];
        let replicated = vec![ShardingDimension::replicated(), ShardingDimension::replicated()];
        assert_eq!(materialize(rows), (values.to_vec(), 2, false));
        assert_eq!(materialize(columns), (values.to_vec(), 4, true));
        assert_eq!(materialize(replicated), (values.to_vec(), 1, false));
    }

    #[test]
    fn test_row_major_element_strides() {
        assert_eq!(row_major_element_strides(&[2, 3, 4], DataType::F32).unwrap(), vec![12, 4, 1]);
        assert_eq!(row_major_element_strides(&[], DataType::F32).unwrap(), Vec::<usize>::new());

        // A stride that overflows `usize` is an error.
        let error = row_major_element_strides(&[2, usize::MAX, 2], DataType::F32).unwrap_err();
        let message = format!(
            "row-major stride for array with shape `[2, {}, 2]` and element type `f32` exceeds the maximum allowed size \
             of {}",
            usize::MAX,
            usize::MAX,
        );
        assert!(matches!(&error, Error::SizeLimitExceeded { message: actual } if actual == &message));
        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn test_contiguous_tile_byte_range() {
        // Whole rows of a `[4, 2]` array and a partial row followed by whole trailing dimensions are contiguous.
        assert_eq!(contiguous_tile_byte_range(&[1..3, 0..2], &[4, 2], DataType::F32).unwrap(), Some(8..24));
        assert_eq!(contiguous_tile_byte_range(&[1..2, 2..5, 0..4], &[3, 6, 4], DataType::F32).unwrap(), Some(128..176),);
        assert_eq!(contiguous_tile_byte_range(&[], &[], DataType::F64).unwrap(), Some(0..8));

        // A column of a `[4, 2]` array spans several byte ranges.
        assert_eq!(contiguous_tile_byte_range(&[0..4, 0..1], &[4, 2], DataType::F32).unwrap(), None);
    }

    #[test]
    fn test_combine_unreduced_values() {
        // Copies without unreduced coordinates are returned unchanged.
        let tiles = vec![(0, vec![0..2], Vec::new(), vec![1u8, 2]), (1, vec![2..4], Vec::new(), vec![3u8, 4])];
        assert_eq!(
            combine_unreduced_values(tiles, DataType::U8, UnreducedKind::Sum).unwrap(),
            vec![(0, vec![0..2], vec![1u8, 2]), (1, vec![2..4], vec![3u8, 4])],
        );

        // The per-device values of a tile are combined, while its replicas contribute once.
        let summands = vec![
            (0, vec![0..2], vec![0], values_to_bytes(&[1f32, 2.])),
            (1, vec![0..2], vec![1], values_to_bytes(&[10f32, 20.])),
            (2, vec![0..2], vec![0], values_to_bytes(&[1f32, 2.])),
        ];
        assert_eq!(
            combine_unreduced_values(summands.clone(), DataType::F32, UnreducedKind::Sum).unwrap(),
            vec![(0, vec![0..2], values_to_bytes(&[11f32, 22.]))],
        );
        assert_eq!(
            combine_unreduced_values(summands, DataType::F32, UnreducedKind::Min).unwrap(),
            vec![(0, vec![0..2], values_to_bytes(&[1f32, 2.]))],
        );

        // Replicas of one per-device value must agree.
        let inconsistent = vec![
            (0, vec![0..2], vec![0], values_to_bytes(&[1f32, 2.])),
            (1, vec![0..2], vec![0], values_to_bytes(&[1f32, 3.])),
        ];
        let error = combine_unreduced_values(inconsistent, DataType::F32, UnreducedKind::Sum).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(), ArrayError::InconsistentOverlappingShardData { shard_index: 1 }),
        ));
        assert_eq!(error.to_string(), "array move found inconsistent overlapping data while materializing shard #1");
    }

    #[test]
    fn test_merge_dense_shard_bytes() {
        // The second column of a `[2, 2]` array occupies two separate byte ranges of its dense representation.
        let mut global_bytes = vec![0u8; 16];
        let mut written_intervals = Vec::new();
        merge_dense_shard_bytes(
            values_to_bytes(&[1f32, 2.]).as_slice(),
            &[2, 2],
            &[0..2, 1..2],
            DataType::F32,
            0,
            &mut global_bytes,
            &mut written_intervals,
        )
        .unwrap();
        assert_eq!(values_from_bytes::<f32>(global_bytes.as_slice()), vec![0., 1., 0., 2.]);
        assert_eq!(written_intervals, vec![4..8, 12..16]);

        // A rank-0 shard covers the whole array.
        let mut global_bytes = vec![0u8; 4];
        let mut written_intervals = Vec::new();
        merge_dense_shard_bytes(&[1, 2, 3, 4], &[], &[], DataType::F32, 0, &mut global_bytes, &mut written_intervals)
            .unwrap();
        assert_eq!(global_bytes, vec![1, 2, 3, 4]);
        assert_eq!(written_intervals, vec![0..4]);
    }

    #[test]
    fn test_merge_dense_byte_segment() {
        let mut global_bytes = vec![0u8; 6];
        let mut written_intervals = Vec::new();
        merge_dense_byte_segment(&[1, 2], 0, 0, &mut global_bytes, &mut written_intervals).unwrap();
        merge_dense_byte_segment(&[5, 6], 4, 1, &mut global_bytes, &mut written_intervals).unwrap();
        assert_eq!(written_intervals, vec![0..2, 4..6]);

        // A segment that agrees with the bytes that it overlaps joins the intervals that it touches.
        merge_dense_byte_segment(&[2, 3, 4, 5], 1, 2, &mut global_bytes, &mut written_intervals).unwrap();
        assert_eq!(global_bytes, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(written_intervals, vec![0..6]);

        // Empty segments record nothing.
        merge_dense_byte_segment(&[], 6, 3, &mut global_bytes, &mut written_intervals).unwrap();
        assert_eq!(written_intervals, vec![0..6]);

        // A segment that disagrees with the bytes that it overlaps is rejected.
        let error = merge_dense_byte_segment(&[9], 2, 4, &mut global_bytes, &mut written_intervals).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(), ArrayError::InconsistentOverlappingShardData { shard_index: 4 }),
        ));
        assert_eq!(global_bytes, vec![1, 2, 3, 4, 5, 6]);
    }
}
