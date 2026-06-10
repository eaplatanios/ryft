use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::sync::Arc;

use ryft_core::programs::Value;
use ryft_core::{
    ArrayType, DataType, Device, DeviceId, DeviceMesh, Layout, Parameter, Sharding, ShardingDimension, ShardingError,
    StaticShape, Typed, check_sharding,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Buffer, Client};

use crate::{ArrayError, Error, FromPjrt, ToPjrt};

/// Distributed array with a global [`StaticShape`] and element [`DataType`] as well as [`Sharding`] information.
/// An [`Array`] represents one logical [`ArrayType`] whose elements may be split or replicated across the multiple
/// devices in a [`DeviceMesh`], potentially spanning multiple nodes or processes. The global array is described by
/// its type and sharding metadata, while each physical piece of that global array is represented by an [`ArrayShard`].
/// [`Array::shards`] is a global list: it contains one [`ShardDescriptor`] for each device that participates in the
/// array placement, and not only for the devices that are visible to the current process. In a single-process setup,
/// every shard is normally _addressable_ because the local [`Client`] can directly access every backing [`Buffer`].
/// In a multi-device or multi-node setup, the same logical array can span devices owned by other processes. Shards
/// on the current process's devices are addressable and carry local [`Buffer`]s; shards on remote devices are
/// non-addressable and carry only metadata such as their global [`ShardIndex`], [`DeviceId`], and [`ArrayType`].
/// Keeping both addressable and non-addressable shards in the same [`Array`] lets local code reason about the complete
/// global placement while only transferring, executing with, or materializing buffers that this process can access
/// directly. This distinction is what allows array movement, execution argument assembly, and cross-host transfers to
/// preserve the full global sharding contract without requiring every process to own every shard buffer.
#[derive(Parameter)]
pub struct Array<'o> {
    /// [`ArrayType`] of this [`Array`].
    r#type: ArrayType,

    /// [`ArrayShard`]s that make up this [`Array`].
    shards: Vec<ArrayShard<'o>>,

    /// Lookup table mapping [`DeviceId`]s to their corresponding [`ShardIndex`]es (indexing into [`Self::shards`]).
    shard_index_by_device: HashMap<DeviceId, ShardIndex>,
}

// TODO(eaplatanios): Review this.
impl<'o> Clone for Array<'o> {
    fn clone(&self) -> Self {
        crate::telemetry::array_constructed();
        Self {
            r#type: self.r#type.clone(),
            shards: self.shards.clone(),
            shard_index_by_device: self.shard_index_by_device.clone(),
        }
    }
}

// TODO(eaplatanios): Review this.
impl<'o> Drop for Array<'o> {
    fn drop(&mut self) {
        crate::telemetry::array_dropped();
    }
}

impl<'o> Array<'o> {
    /// Creates an [`Array`] of type `r#type` from the provided _addressable_ [`Buffer`]s using the provided concrete
    /// [`DeviceMesh`] to determine what [`ArrayShard`]s make up the array and which ones correspond to the addressable
    /// buffers. The provided [`ArrayType`] describes the global logical array. Its [`Shape`](ryft_core::Shape) must be
    /// static, and it must normally contain [`Sharding`] metadata whose logical mesh matches `mesh`. As a convenience
    /// for unsharded arrays, callers may omit the sharding information only when exactly one addressable buffer is
    /// provided; in that case the array is treated as replicated over the provided `mesh`.
    ///
    /// Each [`Buffer`] is assigned to a global shard based on the [`DeviceId`] of its owning device. This function will
    /// return an [`Error::MultipleBuffersOnDevice`] if there are multiple buffers provided that are owned by the same
    /// device. It will also return an [`Error::DeviceNotInMesh`] if there are buffer whose device is not present in
    /// `mesh`, an [`Error::NonAddressableDevice`] for buffers whose device belongs to a different process than the
    /// corresponding mesh device, and an [`Error::BufferTypeMismatch`] for buffers whose data type or static shape
    /// does not match the shard type derived from `r#type`, `mesh`, and the effective sharding. Shards that do not
    /// have a local/addressable buffer are retained as non-addressable [`ArrayShard`]s.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Global [`ArrayType`] for the new [`Array`].
    ///   - `mesh`: [`DeviceMesh`] that is used to determine the [`ArrayShard`] placement.
    ///   - `buffers`: [`Buffer`]s for the [`ArrayShard`]s that are addressable from the current process.
    pub fn from_addressable_buffers(
        r#type: ArrayType,
        mesh: DeviceMesh,
        buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, Error> {
        // Normalize the provided `r#type` before deriving shard placement. A single buffer with no sharding metadata
        // is treated as an unsharded replicated array over the caller-provided mesh.
        let r#type = match r#type.sharding() {
            Some(_) => r#type,
            None if buffers.len() == 1 => r#type.replicated(&mesh)?,
            None => {
                return Err(Error::MissingSharding);
            }
        };

        // Compute the global [`ShardLayout`] implied by the normalized array type and concrete device mesh.
        let shape = r#type.static_shape().ok_or_else(|| Error::DynamicShape { shape: r#type.shape().clone() })?;
        let sharding = r#type.sharding().ok_or(Error::MissingSharding)?;
        let layout = ShardLayout::new(&shape, &mesh, sharding)?;
        let (descriptors, shard_index_by_device) = layout.into_parts();

        // Index the provided [`Buffer`]s by device while rejecting duplicate local buffers for the same device.
        let mut buffers_by_device = HashMap::with_capacity(buffers.len());
        for buffer in buffers {
            let device = Device::from_pjrt(buffer.device()?)?;
            let device_id = device.id();
            if buffers_by_device.contains_key(&device_id) {
                return Err(Error::MultipleBuffersOnDevice { device_id });
            }

            let shard_index = shard_index_by_device.get(&device_id).ok_or(Error::DeviceNotInMesh { device_id })?;
            let descriptor = descriptors.get(*shard_index).unwrap();

            // Validate that each buffer is owned by the process expected for the corresponding mesh device.
            let process_index = device.process_index();
            if process_index != descriptor.device().process_index() {
                return Err(Error::DeviceProcessIndexMismatch {
                    device_id,
                    expected_process_index: descriptor.device().process_index(),
                    actual_process_index: process_index,
                });
            }

            // Validate the concrete buffer type against the shard type that the layout assigns to this device.
            let data_type = DataType::from_pjrt(buffer.element_type()?)?;
            let shape = StaticShape::new(
                buffer
                    .dimensions()?
                    .iter()
                    .map(|size| {
                        usize::try_from(*size).map_err(|_| Error::SizeLimitExceeded {
                            message: format!(
                                "buffer dimension size {size} exceeds the maximum allowed size of {}",
                                usize::MAX,
                            ),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let array_type = ArrayType::new(data_type, shape.into(), None, None)?;
            let expected_array_type = ArrayType::new(r#type.data_type(), descriptor.shape().into(), None, None)?;
            if array_type != expected_array_type {
                return Err(Error::BufferTypeMismatch { expected: expected_array_type, actual: array_type });
            }

            buffers_by_device.insert(device_id, Arc::new(buffer));
        }

        // Materialize the global shard list with local buffers attached to the shards addressable from this process.
        let shards = descriptors
            .into_iter()
            .map(|descriptor| {
                let buffer = buffers_by_device.remove(&descriptor.device().id());
                ArrayShard::new(descriptor, buffer)
            })
            .collect::<Vec<_>>();

        // TODO(eaplatanios): Review this.
        crate::telemetry::array_constructed();
        Ok(Self { r#type, shards, shard_index_by_device })
    }

    /// Creates an [`Array`] by transferring a dense row-major host buffer to the local shards implied by `r#type`
    /// and `mesh`, while avoiding redundant allocations external to the PJRT backend of the provided [`Client`].
    /// This function derives the per-device shard slices from the provided type/mesh pair, transfers only the shards
    /// addressable by `client`, and returns an [`Array`] whose global shard metadata covers the full mesh.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT [`Client`] used to transfer the local addressable shard buffers.
    ///   - `r#type`: Global [`ArrayType`] of the constructed array.
    ///   - `mesh`: Destination [`DeviceMesh`] describing the device topology.
    ///   - `buffer`: Dense row-major host bytes for the full logical array.
    pub fn from_host_buffer<B: AsRef<[u8]>>(
        client: &'o Client<'_>,
        r#type: ArrayType,
        mesh: DeviceMesh,
        buffer: B,
    ) -> Result<Self, ArrayError> {
        // Normalize missing sharding metadata before any placement-derived work. A dense host buffer with no explicit
        // sharding is interpreted as one replicated logical array over the provided mesh.
        let r#type = match r#type.sharding() {
            Some(_) => r#type,
            None => r#type.replicated(&mesh)?,
        };

        // Validate that the host buffer contains exactly the dense row-major bytes for the full logical array.
        let buffer = buffer.as_ref();
        let expected_byte_count = r#type.size_in_bytes()?;
        if buffer.len() != expected_byte_count {
            return Err(Error::ByteCountMismatch { expected: expected_byte_count, actual: buffer.len() }.into());
        }

        // Build a lookup table for the PJRT devices that this process can upload to directly.
        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        // Derive the global shard layout and transfer only the shards owned by this process. Remote shards remain
        // represented by metadata when [`Self::from_addressable_buffers`] materializes the final array.
        let data_type = r#type.data_type();
        let shape = r#type.static_shape().ok_or_else(|| Error::DynamicShape { shape: r#type.shape().clone() })?;
        let sharding = r#type.sharding().unwrap();
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let mut addressable_buffers = Vec::new();
        for shard in layout.descriptors() {
            let shard_device = shard.device();
            if shard_device.process_index() != client_process_index {
                continue;
            }

            let shard_device_id = shard_device.id();
            let device = addressable_device_by_id.get(&shard_device_id).ok_or(Error::NonAddressableDevice {
                device_id: shard_device_id,
                process_index: client_process_index,
            })?;

            // Upload the local shard bytes to the PJRT device using the shard-local static shape. For non-scalar
            // shards, the host pointer is moved to the shard's first element in the dense source buffer, and PJRT's
            // host byte-stride support describes how to read the shard without materializing a packed temporary vector.
            let shard_shape = shard.shape();
            let shard_dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let shard_slice = shard.slice();
            let buffer_type = data_type.to_pjrt();
            if shard_slice.is_empty() {
                addressable_buffers.push(client.buffer(
                    buffer,
                    buffer_type,
                    shard_dimensions.as_slice(),
                    None,
                    device.clone(),
                    None,
                )?);
            } else {
                // Compute row-major strides in units of elements so that each shard slice can be translated
                // into flat element offsets in the full dense host buffer.
                let mut strides = vec![1usize; shape.rank()];
                let mut stride = 1usize;
                for dimension in (0..shape.rank()).rev() {
                    strides[dimension] = stride;
                    stride = stride.checked_mul(shape[dimension]).ok_or_else(|| Error::SizeLimitExceeded {
                        message: format!(
                            "row-major stride for array with shape {shape} and element type {buffer_type} exceeds the \
                             maximum allowed size of {}",
                            usize::MAX,
                        ),
                    })?;
                }

                // Convert the shard's starting coordinates into a flat row-major element offset in the full host
                // buffer, and then into the corresponding byte offset.
                let element_size_in_bytes = buffer_type.element_size_in_bytes().map_err(Error::from)?;
                let start_element_offset = shard_slice
                    .iter()
                    .enumerate()
                    .map(|(dimension, slice)| slice.start * strides[dimension])
                    .sum::<usize>();
                let start_byte_offset = start_element_offset * element_size_in_bytes;

                // Give PJRT a host buffer slice that covers every byte it may read for this shard. Empty shards have
                // no last element, and so their byte window is empty and starts at the shard start offset.
                let end_byte_offset = if shard_shape.as_slice().contains(&0) {
                    start_byte_offset
                } else {
                    let last_element_offset = start_element_offset
                        + shard_shape
                            .as_slice()
                            .iter()
                            .enumerate()
                            .map(|(dimension, size)| (size - 1) * strides[dimension])
                            .sum::<usize>();
                    last_element_offset * element_size_in_bytes + element_size_in_bytes
                };

                // If the shard is already contiguous in the dense row-major host buffer, pass no explicit strides and
                // let PJRT read the host buffer slice as a packed dense shard. Otherwise, describe the global row-major
                // strides in bytes so that PJRT can gather the strided shard directly from the original host buffer.
                let (block_dimension, _) = shard.contiguous_inner_block(&shape)?;
                let byte_strides = if block_dimension == 0 {
                    None
                } else {
                    Some(
                        strides
                            .iter()
                            .map(|stride| {
                                let byte_stride = stride * element_size_in_bytes;
                                i64::try_from(byte_stride).map_err(|_| Error::SizeLimitExceeded {
                                    message: format!(
                                        "row-major byte stride for array with shape {shape} and element type \
                                         {buffer_type} exceeds the maximum allowed size of {}",
                                        i64::MAX,
                                    ),
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                };

                // Transfer the shard from the original host buffer. For strided shards, the slice starts at the shard's
                // first element and spans the full byte window that the stride metadata can address.
                addressable_buffers.push(client.buffer(
                    &buffer[start_byte_offset..end_byte_offset],
                    buffer_type,
                    shard_dimensions.as_slice(),
                    byte_strides.as_deref(),
                    device.clone(),
                    None,
                )?);
            }
        }

        // Reuse the buffer-based constructor for final buffer validation and global sharding metadata assembly.
        Ok(Self::from_addressable_buffers(r#type, mesh, addressable_buffers)?)
    }

    /// Returns the [`DataType`] of the elements stored in this [`Array`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.r#type.data_type()
    }

    /// Returns the global [`StaticShape`] of this [`Array`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.r#type
            .static_shape()
            .expect("runtime arrays should only be constructed from array types with static shapes")
    }

    /// Returns the physical memory/storage [`Layout`] of this [`Array`] if it is known.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.r#type.layout()
    }

    /// Returns [`Sharding`] information about this [`Array`].
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        self.r#type
            .sharding()
            .expect("runtime arrays should only be constructed from array types with sharding")
    }

    /// Returns the concrete [`DeviceMesh`] implied by this [`Array`]'s global shard placement metadata.
    #[inline]
    pub fn mesh(&self) -> DeviceMesh {
        DeviceMesh::new(self.sharding().mesh().clone(), self.shards.iter().map(|shard| shard.device()).collect())
            .expect("runtime arrays should always contain one shard descriptor per device")
    }

    /// Returns the [`ArrayShard`]s that make up this [`Array`].
    #[inline]
    pub fn shards(&self) -> &[ArrayShard<'o>] {
        self.shards.as_slice()
    }

    /// Returns an [`Iterator`] over the _addressable_ [`ArrayShard`]s of this [`Array`].
    #[inline]
    pub fn addressable_shards(&self) -> impl Iterator<Item = &ArrayShard<'o>> {
        self.shards.iter().filter(|shard| shard.is_addressable())
    }

    /// Returns the [`ArrayShard`] of this [`Array`] that is placed on the device with the provided
    /// [`DeviceId`], if such a shard exists.
    #[inline]
    pub fn device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.shard_index_by_device.get(&device_id).and_then(|index| self.shards.get(*index))
    }

    /// Returns the _addressable_ [`ArrayShard`] of this [`Array`] that is placed on the device with the provided
    /// [`DeviceId`], if such a shard exists.
    #[inline]
    pub fn addressable_device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.device_shard(device_id).filter(|shard| shard.is_addressable())
    }

    /// Returns the number of bytes that this [`Array`] occupies on device memory, across all devices and processes.
    #[inline]
    pub fn size_in_bytes(&self) -> Result<usize, Error> {
        self.r#type.size_in_bytes()
    }
}

impl Debug for Array<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Array").field("type", &self.r#type).field("shards", &self.shards()).finish()
    }
}

impl Display for Array<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "array(type={})", self.r#type)
    }
}

impl Typed<ArrayType> for Array<'_> {
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl Value<ArrayType> for Array<'_> {}

/// Shard of an [`Array`]. [`ArrayShard`]s always carry global shard metadata through [`ArrayShard::descriptor`].
/// They also carry a PJRT [`Buffer`] when the owning device is addressable from the current process (otherwise
/// [`ArrayShard::buffer`] is set to `None`). This lets an [`Array`] describe its full global layout while storing
/// local buffers only for the shards that the current process can read directly, without moving data around. Note
/// that, [`ArrayShard`]s holds their [`Buffer`]s inside [`Arc`]s so that cloning an array can share addressable
/// PJRT [`Buffer`]s with other array instances. The last [`Arc`] dropped releases the underlying PJRT buffer via
/// [`Buffer`]'s [`Drop`] implementation.
#[derive(Clone)]
pub struct ArrayShard<'o> {
    /// Refer to the documentation of [`Self::descriptor`] for information on this field.
    descriptor: ShardDescriptor,

    /// Refer to the documentation of [`Self::buffer`] for information on this field.
    buffer: Option<Arc<Buffer<'o>>>,
}

impl<'o> ArrayShard<'o> {
    /// Creates a new [`ArrayShard`].
    #[inline]
    pub fn new(descriptor: ShardDescriptor, buffer: Option<Arc<Buffer<'o>>>) -> Self {
        Self { descriptor, buffer }
    }

    /// Returns the [`ShardDescriptor`] of this [`ArrayShard`], which is defined and provided irrespective of whether
    /// this shard is addressable from the current process or not.
    #[inline]
    pub fn descriptor(&self) -> &ShardDescriptor {
        &self.descriptor
    }

    /// Returns the [`Buffer`] underlying this [`ArrayShard`]. This is `None` if the shard is not addressable
    /// from the current process.
    #[inline]
    pub fn buffer(&self) -> Option<&Arc<Buffer<'o>>> {
        self.buffer.as_ref()
    }

    /// Returns this [`ArrayShard`]'s global index.
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.descriptor.index()
    }

    /// Returns the [`Device`] that owns this [`ArrayShard`].
    #[inline]
    pub fn device(&self) -> Device {
        self.descriptor.device()
    }

    /// Returns the [`ShardDescriptor::slice`] covered by this [`ArrayShard`].
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.descriptor.slice()
    }

    /// Returns the local [`StaticShape`] of this [`ArrayShard`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.descriptor.shape()
    }

    /// Returns `true` if this shard is addressable from the current process.
    #[inline]
    pub fn is_addressable(&self) -> bool {
        self.buffer.is_some()
    }

    /// Consumes this shard and returns its [`ShardDescriptor`] and addressable [`Buffer`], if any.
    #[inline]
    pub(crate) fn into_parts(self) -> (ShardDescriptor, Option<Arc<Buffer<'o>>>) {
        (self.descriptor, self.buffer)
    }
}

impl Debug for ArrayShard<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device = self.device();
        formatter
            .debug_struct("ArrayShard")
            .field("index", &self.index())
            .field("device_id", &device.id())
            .field("process_index", &device.process_index())
            .field("shape", &self.shape())
            .field("is_addressable", &self.is_addressable())
            .finish()
    }
}

/// Row-major ordinal index of an [`ArrayShard`] within a [`DeviceMesh`]. Shard indices are assigned using the same
/// row-major ordering as [`DeviceMesh::devices`]. This gives all processes a stable way to refer to the same global
/// shard without depending on whether that shard is locally addressable or not.
pub type ShardIndex = usize;

/// Device ownership and slice metadata for an [`ArrayShard`]. A [`ShardDescriptor`] is intentionally independent of any
/// local [`Buffer`]. It describes which [`Device`] owns the [`ArrayShard`] and which slice of the underlying [`Array`]
/// that shard represents. [`ArrayShard`]s pair this metadata with optional addressable buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// Refer to the documentation of [`Self::index`] for information on this field.
    index: ShardIndex,

    /// Refer to the documentation of [`Self::device`] for information on this field.
    device: Device,

    /// Refer to the documentation of [`Self::slice`] for information on this field.
    slice: Vec<Range<usize>>,
}

impl ShardDescriptor {
    /// Creates a new [`ShardDescriptor`].
    #[inline]
    pub fn new(index: ShardIndex, device: Device, slice: Vec<Range<usize>>) -> Self {
        Self { index, device, slice }
    }

    /// Returns the [`ShardIndex`] of this [`ShardDescriptor`]. This index is stable across processes and matches this
    /// descriptor's position in [`ShardLayout::descriptors`].
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.index
    }

    /// Returns the [`Device`] that owns the shard. Note that ownership does not imply local addressability.
    /// A shard is local to a process when [`Device::process_index`] matches that process.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the per-dimension global index ranges covered by the shard. This vector contains one [`Range`] per array
    /// dimension. Ranges use normal Rust semantics: the start is inclusive and the end is exclusive. Replicated and
    /// unconstrained dimensions span the full dimension range (i.e., `0..dimension_size`) while sharded dimensions span
    /// the contiguous partition selected by this shard's [`Device`] coordinates in the underlying [`DeviceMesh`].
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.slice.as_slice()
    }

    /// Returns the local [`StaticShape`] of the shard described by this [`ShardDescriptor`]. The returned shape is
    /// derived purely from static metadata; it does not inspect any device buffers and is valid for both addressable
    /// and non-addressable shards.
    #[inline]
    pub fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(|slice| slice.len()).collect())
    }

    /// Returns the outermost dimension `d` such that, in a row-major buffer representing the storage of an array with
    /// the provided [`StaticShape`], the elements traced by this [`ShardDescriptor`]'s [`Self::slice`] over dimensions
    /// `d..rank` form a single contiguous span, together with the element count of that contiguous block.
    ///
    /// Trailing dimensions whose slice fully covers the corresponding `shape` extent are absorbed into the contiguous
    /// block; The innermost dimension is always part of the block since its element span at fixed outer indices is
    /// already contiguous in row-major order. Replicated or major-axis-sharded layouts therefore collapse to a single
    /// block, while inner-axis-sharded layouts stop the collapse early. Scalar shards (rank zero) trivially return
    /// `(0, 1)` since their singular element is its own contiguous block. This is most useful for shard byte transfers,
    /// where each block can be copied with one [`Vec::extend_from_slice`] call.
    ///
    /// Consider the following two shard slices over the same `shape = [4, 6]`:
    ///
    /// ```text
    ///   slice = [1..3, 0..6]            slice = [0..4, 2..5]
    ///   returns (0, 12)                 returns (1, 3)
    ///
    ///   . . . . . .                     . . X X X .
    ///   X X X X X X                     . . X X X .
    ///   X X X X X X                     . . X X X .
    ///   . . . . . .                     . . X X X .
    ///
    ///   one contiguous block            four contiguous blocks
    ///   of 12 elements                  of 3 elements each
    /// ```
    ///
    /// The left slice fully covers the inner dimension (i.e., `0..6` over a host extent of `6`), and so the outer
    /// dimension is absorbed into the block, and the whole shard is one contiguous span. The right slice does not cover
    /// the inner dimension, and so each row of the slice becomes its own contiguous block of `3` elements.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`StaticShape`] of the contiguous buffer that this [`ShardDescriptor`]'s [`Self::slice`] refers to.
    ///     Must have the same rank as [`Self::slice`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShardSliceRankMismatch`] if `shape` has a different rank than [`Self::slice`],
    /// or [`Error::SizeLimitExceeded`] if the absorbed block element count overflows [`usize`].
    pub fn contiguous_inner_block(&self, shape: &StaticShape) -> Result<(usize, usize), Error> {
        let rank = self.slice.len();

        if shape.rank() != rank {
            return Err(Error::ShardSliceRankMismatch { shape_rank: shape.rank(), slice_rank: rank });
        }

        if rank == 0 {
            return Ok((0, 1));
        }

        let mut dimension = rank - 1;
        let mut element_count = self.slice[dimension].len();
        while dimension > 0 && self.slice[dimension].len() == shape[dimension] {
            dimension -= 1;
            element_count =
                element_count.checked_mul(self.slice[dimension].len()).ok_or_else(|| Error::SizeLimitExceeded {
                    message: format!(
                        "contiguous shard block element count for shard slice {:?} and shape {shape} exceeds the \
                         maximum allowed size of {}",
                        self.slice,
                        usize::MAX,
                    ),
                })?;
        }
        Ok((dimension, element_count))
    }
}

/// [`ShardLayout`] that is obtained by applying a [`Sharding`] to a [`StaticShape`] over a [`DeviceMesh`].
/// [`ShardLayout`] is the bridge between type-level device placement metadata and runtime array metadata. It expands
/// the logical sharding specification into one [`ShardDescriptor`] per [`Device`] and also records a device-ID
/// lookup table for routing local [`Buffer`]s back to their global shard descriptors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardLayout {
    /// Refer to the documentation of [`Self::descriptors`] for information on this field.
    descriptors: Vec<ShardDescriptor>,

    /// Refer to the documentation of [`Self::shard_index_by_device`] for information on this field.
    shard_index_by_device: HashMap<DeviceId, ShardIndex>,
}

impl ShardLayout {
    /// Constructs a new [`ShardLayout`] by applying the provided [`Sharding`] to the provided [`StaticShape`]
    /// over the provided [`DeviceMesh`]. The returned layout contains one [`ShardDescriptor`] per [`Device`].
    /// For each array dimension, replicated and unconstrained dimensions map every device to the full dimension range,
    /// while sharded dimensions are split across the product of the referenced mesh axes. If the dimension size is not
    /// evenly divisible by the partition count, earlier partitions receive one extra element.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`StaticShape`] of the [`Array`] being sharded/partitioned.
    ///   - `mesh`: [`DeviceMesh`] whose row-major device order determines shard indices.
    ///   - `sharding`: Logical [`Sharding`] specification to apply to `shape`.
    pub fn new(shape: &StaticShape, mesh: &DeviceMesh, sharding: &Sharding) -> Result<Self, Error> {
        check_sharding!(mesh, sharding);

        let sharding_rank = sharding.rank();
        let array_rank = shape.rank();
        if sharding_rank != array_rank {
            return Err(ShardingError::ShardingRankMismatch { sharding_rank, array_rank }.into());
        }

        let dimensions = shape.as_slice();

        // Resolve each sharded array dimension to mesh-axis indices. This keeps the inner per-device loop
        // allocation-free with respect to axis-name lookup and centralizes defensive validation before any shard
        // descriptors are built.
        let mut used_axis_names = HashSet::new();
        let dimension_axis_indices = sharding
            .dimensions()
            .iter()
            .enumerate()
            .map(|(dimension, sharding_dimension)| -> Result<Vec<usize>, Error> {
                let ShardingDimension::Sharded(axis_names) = sharding_dimension else {
                    return Ok(Vec::new());
                };
                if axis_names.is_empty() {
                    return Err(ShardingError::EmptySharding { dimension }.into());
                }

                let mut axis_indices = Vec::with_capacity(axis_names.len());
                for axis_name in axis_names {
                    let axis_index = mesh
                        .logical_mesh()
                        .axis_index(axis_name.as_str())
                        .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                    if !used_axis_names.insert(axis_name.as_str()) {
                        return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() }.into());
                    }
                    axis_indices.push(axis_index);
                }
                Ok(axis_indices)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut descriptors = Vec::with_capacity(mesh.device_count());
        let mut shard_index_by_device = HashMap::with_capacity(mesh.device_count());
        for (index, device) in mesh.devices().iter().copied().enumerate() {
            // `DeviceMesh` stores devices in row-major mesh order, so the loop index is also the global shard index.
            // Convert it back into mesh coordinates so sharded dimensions can pick the partition owned by this device.
            let device_coordinates =
                mesh.device_coordinates(index).expect("mesh coordinate should exist for valid device index");

            let mut slices = Vec::with_capacity(dimensions.len());
            for (dimension, dimension_size) in dimensions.iter().copied().enumerate() {
                let axis_indices = &dimension_axis_indices[dimension];
                let range = if axis_indices.is_empty() {
                    // Replicated and unconstrained dimensions both map every device to the full dimension range.
                    0..dimension_size
                } else {
                    // Linearize this device's coordinates across the dimension's sharding axes. The resulting
                    // partition index selects one contiguous chunk along the global array dimension.
                    let mut partition_index = 0usize;
                    let mut partition_count = 1usize;
                    for &axis_index in axis_indices {
                        let axis_size = mesh.logical_mesh().axes()[axis_index].size();
                        let axis_coordinate = device_coordinates[axis_index];

                        partition_index = partition_index * axis_size + axis_coordinate;
                        partition_count *= axis_size;
                    }

                    let base_size = dimension_size / partition_count;
                    let remainder = dimension_size % partition_count;
                    let extra_before = partition_index.min(remainder);

                    // Split uneven dimensions by assigning one extra element to the earliest partitions.
                    let start = partition_index * base_size + extra_before;
                    let size = base_size + usize::from(partition_index < remainder);
                    start..start + size
                };
                slices.push(range);
            }

            shard_index_by_device.insert(device.id(), index);
            descriptors.push(ShardDescriptor::new(index, device, slices));
        }

        Ok(Self { descriptors, shard_index_by_device })
    }

    /// Returns the [`ShardDescriptor`]s for all global shards in row-major [`DeviceMesh`] order. The position of each
    /// descriptor in this vector is the descriptor's [`ShardDescriptor::index`].
    #[inline]
    pub fn descriptors(&self) -> &[ShardDescriptor] {
        &self.descriptors
    }

    /// Returns the lookup table mapping [`DeviceId`]s to their corresponding [`ShardIndex`]es. This is used when a
    /// [`Buffer`] reports a [`DeviceId`] and the [`Array`] constructor needs to find the global shard metadata that
    /// buffer should satisfy.
    #[inline]
    pub fn shard_index_by_device(&self) -> &HashMap<DeviceId, ShardIndex> {
        &self.shard_index_by_device
    }

    /// Consumes this [`ShardLayout`] and returns its [`ShardDescriptor`]s and device-to-shard lookup table.
    #[inline]
    pub(crate) fn into_parts(self) -> (Vec<ShardDescriptor>, HashMap<DeviceId, ShardIndex>) {
        (self.descriptors, self.shard_index_by_device)
    }
}

/// Provides XLA-specific helpers for [`ArrayType`]s.
pub(crate) trait ArrayTypeExtension {
    /// Returns the number of bytes that dense [`Array`]s of this [`ArrayType`] require/occupy on device memory.
    fn size_in_bytes(&self) -> Result<usize, Error>;
}

impl ArrayTypeExtension for ArrayType {
    #[inline]
    fn size_in_bytes(&self) -> Result<usize, Error> {
        let element_count = self.element_count()?.ok_or_else(|| Error::DynamicShape { shape: self.shape().clone() })?;
        let element_size_in_bytes = self.data_type().to_pjrt().element_size_in_bytes()?;
        element_count.checked_mul(element_size_in_bytes).ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "dense byte size for array type {self} exceeds the maximum allowed size of {}",
                usize::MAX,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{
        ArrayType, DataType, Device, DeviceMesh, Error as CoreError, Layout, LogicalMesh, MeshAxis, MeshAxisType,
        Shape, Sharding, ShardingDimension, ShardingError, Size, StaticShape, TiledLayout, Typed,
    };
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::tests::{device_mesh_2x2, logical_mesh_2x2, values_from_bytes, values_to_bytes};
    use crate::{Error, FromPjrt};

    use super::{Array, ArrayShard, ArrayTypeExtension, ShardDescriptor, ShardLayout};

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let array_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4), Size::Static(4)]),
            None,
            Some(sharding.clone()),
        )
        .unwrap();
        let shard_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(shard_index, device)| {
                let base = (shard_index * 10) as f32;
                let bytes = values_to_bytes::<f32>(&[base, base + 1.0, base + 2.0, base + 3.0]);
                client.buffer(bytes.as_slice(), BufferType::F32, [2u64, 2u64], None, device.clone(), None).unwrap()
            })
            .collect::<Vec<_>>();

        // Construct via [`Array::from_addressable_buffers`] and exercise every canonical [`Array`] accessor.
        let array = Array::from_addressable_buffers(array_type.clone(), mesh.clone(), shard_buffers).unwrap();

        assert_eq!(array.r#type().as_ref(), &array_type);
        assert_eq!(array.data_type(), DataType::F32);
        assert_eq!(array.shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(array.layout(), None);
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(array.mesh(), mesh);
        assert_eq!(array.shards().len(), 4);
        assert_eq!(array.addressable_shards().count(), 4);
        assert!(array.shards().iter().all(|shard| shard.shape() == StaticShape::new(vec![2, 2])));
        assert_eq!(array.size_in_bytes(), Ok(64));

        // Every device in the mesh has a corresponding addressable shard, while device ids absent from the mesh
        // resolve to no shard at all.
        for device in &devices {
            let shard = array.device_shard(device.id()).unwrap();
            assert_eq!(shard.device(), *device);
            assert!(shard.is_addressable());
            assert_eq!(array.addressable_device_shard(device.id()).map(|shard| shard.device()), Some(*device));
        }
        let absent_device_id = devices.iter().map(Device::id).max().unwrap() + 1;
        assert!(array.device_shard(absent_device_id).is_none());
        assert!(array.addressable_device_shard(absent_device_id).is_none());

        // Layout metadata stored on the underlying array type is exposed verbatim by [`Array::layout`].
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()));
        let layout_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4), Size::Static(4)]),
            Some(layout.clone()),
            Some(sharding),
        )
        .unwrap();
        let layout_array = Array::from_addressable_buffers(layout_type, mesh.clone(), Vec::new()).unwrap();
        assert_eq!(layout_array.layout(), Some(&layout));
        assert_eq!(layout_array.addressable_shards().count(), 0);
        assert!(layout_array.shards().iter().all(|shard| shard.buffer().is_none()));

        // Construct via [`Array::from_host_buffer`] and verify that it partitions the dense row-major host bytes
        // over the shard layout implied by the mesh and sharding. With a `[4, 4]` `F32` array sharded over both
        // mesh axes, the first mesh device owns the `2x2` block over rows `0..2` and columns `0..2`.
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let host_array =
            Array::from_host_buffer(&client, array_type, mesh, values_to_bytes::<f32>(values.as_slice()).as_slice())
                .unwrap();

        assert_eq!(host_array.shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(host_array.addressable_shards().count(), 4);
        assert!(host_array.shards().iter().all(|shard| shard.is_addressable()));
        let shard_bytes = host_array
            .device_shard(devices[0].id())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn test_array_debug() {
        let shape = Shape::new(vec![Size::Static(2)]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 1)]).unwrap();
        let sharding = Sharding::replicated(device_mesh.logical_mesh().clone(), 1);
        let array_type = ArrayType::new(DataType::F32, shape, None, Some(sharding)).unwrap();
        let array = Array::from_addressable_buffers(array_type, device_mesh, Vec::new()).unwrap();
        assert_eq!(
            format!("{array:?}"),
            concat!(
                "Array { type: ArrayType { data_type: F32, shape: Shape { dimensions: [Static(2)] }, layout: None, ",
                "sharding: Some(Sharding { mesh: LogicalMesh { axes: [MeshAxis { name: \"x\", size: 1, type: Auto }], ",
                "axis_indices: {\"x\": 0} }, dimensions: [Replicated], unreduced_axes: {}, reduced_manual_axes: {}, ",
                "varying_manual_axes: {} }) }, shards: [ArrayShard { index: 0, device_id: 0, process_index: 1, ",
                "shape: StaticShape { dimensions: [2] }, is_addressable: false }] }",
            ),
        );
    }

    #[test]
    fn test_array_shard() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]);
        let shard = ArrayShard::new(descriptor.clone(), None);

        assert_eq!(shard.descriptor(), &descriptor);
        assert!(shard.buffer().is_none());
        assert_eq!(shard.index(), 3);
        assert_eq!(shard.device(), Device::new(7, 2));
        assert_eq!(shard.slice(), [2..5, 0..4].as_slice());
        assert_eq!(shard.shape(), StaticShape::new(vec![3, 4]));
        assert!(!shard.is_addressable());
        assert_eq!(shard.descriptor, descriptor);
        assert!(shard.buffer.is_none());
    }

    #[test]
    fn test_array_shard_debug() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]);
        let shard = ArrayShard::new(descriptor, None);

        assert_eq!(
            format!("{shard:?}"),
            concat!(
                "ArrayShard { index: 3, device_id: 7, process_index: 2, ",
                "shape: StaticShape { dimensions: [3, 4] }, is_addressable: false }",
            ),
        );
    }

    #[test]
    fn test_shard_descriptor() {
        let device = Device::new(7, 2);
        let descriptor = ShardDescriptor::new(3, device, vec![2..5, 0..4]);

        assert_eq!(descriptor.index(), 3);
        assert_eq!(descriptor.device(), device);
        assert_eq!(descriptor.slice(), [2..5, 0..4].as_slice());
        assert_eq!(descriptor.shape(), StaticShape::new(vec![3, 4]));

        // The inner dimension is not fully covered by the host extent, and so the block stops at the innermost
        // dimension and only spans `slice[1].len()` elements.
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![8, 5])), Ok((1, 4)));

        // When the inner dimension is fully covered, the block absorbs the outer dimension and spans
        // `slice[0].len() * slice[1].len()` elements.
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![8, 4])), Ok((0, 12)));

        // A shape whose rank does not match the descriptor's slice rank reports a rank-mismatch error.
        assert_eq!(
            descriptor.contiguous_inner_block(&StaticShape::new(vec![8, 4, 2])),
            Err(Error::ShardSliceRankMismatch { shape_rank: 3, slice_rank: 2 }),
        );

        // A rank-one descriptor always reports a single block over its slice.
        let descriptor = ShardDescriptor::new(0, device, vec![2..7]);
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![10])), Ok((0, 5)));

        // A rank-three descriptor whose two innermost dimensions are fully covered collapses to a single block
        // over all three dimensions.
        let descriptor = ShardDescriptor::new(0, device, vec![1..2, 0..3, 0..4]);
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![5, 3, 4])), Ok((0, 12)));

        // A scalar (i.e., rank-zero) descriptor trivially reports a single one-element block.
        let descriptor = ShardDescriptor::new(0, device, vec![]);
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![])), Ok((0, 1)));
    }

    #[test]
    fn test_shard_layout_evenly_partitions_two_dimensions() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..4, 0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..4, 3..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![4..8, 0..3]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![4..8, 3..6]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_keeps_replicated_and_unconstrained_dimensions_full_size() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6, 5]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::replicated(),
                ShardingDimension::unconstrained(),
            ],
        )
        .unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![4..8, 0..6, 0..5]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![4..8, 0..6, 0..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_partitions_uneven_dimension() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let shape = StaticShape::new(vec![5]);
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![3..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_partitions_single_dimension_over_multiple_mesh_axes() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![10]);
        let sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![3..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![6..8]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![8..10]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_into_parts() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();
        let layout = ShardLayout::new(&shape, &mesh, &sharding).unwrap();

        assert_eq!(
            layout.into_parts(),
            (
                vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..8, 0..6]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..8, 0..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![0..8, 0..6]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![0..8, 0..6]),
                ],
                HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            ),
        );
    }

    #[test]
    fn test_shard_layout_rejects_mesh_mismatch() {
        let mesh = device_mesh_2x2();
        let expected_mesh = mesh.logical_mesh().clone();
        let actual_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let shape = StaticShape::new(vec![8]);
        let sharding = Sharding::new(actual_mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert_eq!(
            error,
            Error::CoreError(CoreError::Sharding(ShardingError::MeshMismatch {
                expected: expected_mesh.clone(),
                actual: actual_mesh.clone(),
            })),
        );
        let expected_message = format!("mesh mismatch; expected '{expected_mesh:?}' but got '{actual_mesh:?}'");
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_shard_layout_rejects_rank_mismatch() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert_eq!(
            error,
            Error::CoreError(CoreError::Sharding(ShardingError::ShardingRankMismatch {
                sharding_rank: 2,
                array_rank: 1,
            })),
        );
        assert_eq!(error.to_string(), "sharding rank (2) does not match array rank (1)");
    }

    #[test]
    fn test_array_type_size_in_bytes() {
        let matrix_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let scalar_type = ArrayType::scalar(DataType::C128);
        let token_type = ArrayType::new(DataType::Token, Shape::new(vec![Size::Static(4)]), None, None).unwrap();

        assert_eq!(matrix_type.size_in_bytes(), Ok(24));
        assert_eq!(scalar_type.size_in_bytes(), Ok(16));
        assert_eq!(token_type.size_in_bytes(), Ok(0));

        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(None)]), None, None).unwrap();
        let error = dynamic_type.size_in_bytes().unwrap_err();
        assert_eq!(error, Error::DynamicShape { shape: dynamic_type.shape().clone() });
        assert_eq!(error.to_string(), "expected static shape but got [*]");

        let oversized_type =
            ArrayType::new(DataType::U16, Shape::new(vec![Size::Static(usize::MAX)]), None, None).unwrap();
        let error = oversized_type.size_in_bytes().unwrap_err();
        let message = format!(
            "dense byte size for array type {oversized_type} exceeds the maximum allowed size of {}",
            usize::MAX,
        );
        assert_eq!(error, Error::SizeLimitExceeded { message: message.clone() });
        assert_eq!(error.to_string(), message);
    }
}
