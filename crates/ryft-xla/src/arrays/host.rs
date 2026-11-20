use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    Add, Array as CoreArray, ArrayType, DataType, DeviceMesh, Max, Min, Sharding, StaticShape, Typed, UnreducedKind,
};
use ryft_pjrt::{DeviceId, ExecutionFence, HostBuffer, HostDestination};

use crate::arrays::metadata::resolve_placement;
use crate::arrays::shards::{ArrayShard, ShardIndex, ShardLayout};
use crate::arrays::{Array, ArrayError, ArrayTypeExtension};
use crate::devices::PjrtDeviceExtension;
use crate::types::DataTypeExtension;
use crate::{Error, Session, ToPjrt};

impl Array {
    /// Creates an [`Array`] by transferring a dense row-major host buffer to the local shards implied by
    /// `r#type` and `mesh`, while avoiding redundant allocations external to the PJRT backend of the session's
    /// [`Client`](ryft_pjrt::Client). This function derives the per-device shard slices from the provided type and
    /// mesh, transfers only the shards addressable by that client, and returns an [`Array`] that belongs to `session`
    /// and whose global shard metadata covers the full mesh.
    ///
    /// With [`None`] for `sharding`, the function uses the sharding of `r#type` if present and replicates the array
    /// across the selected mesh otherwise. A supplied sharding can place the array over auto mesh axes. After removing
    /// its auto axes with [`Sharding::without_auto_axes`], it must match any sharding in the type, or the function
    /// returns [`ArrayError::PlacementTypeMismatch`]. The type of the resulting array records that projection, and
    /// [`Self::sharding`] returns the full runtime placement.
    ///
    /// Passing [`None`] for `mesh` uses the mesh configured on `session` if present. Otherwise, the logical mesh
    /// from the runtime or type sharding is laid out over the client's addressable devices in enumeration order.
    /// Without sharding, the first addressable device forms a mesh with a single axis `x`.
    ///
    /// Each addressable shard is uploaded to the memory requested by `r#type`.
    /// [`Memory::Device`](ryft_core::Memory::Device) uses the device's default memory. Host placements select
    /// its `pinned_host` or `unpinned_host` memory. A device without the requested host space produces
    /// [`ArrayError::UnsupportedMemory`]. Bufferless zero-space arrays require no allocation.
    ///
    /// Along an unreduced axis of a sharding of kind [`UnreducedKind::Sum`], the devices hold summands of the logical
    /// value, so the devices at coordinate zero along every unreduced axis receive their shards of the host value and
    /// the others receive zeros. With kind [`UnreducedKind::Max`] or [`UnreducedKind::Min`], every device receives its
    /// shard of the host value, whose extremum is that value.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] the constructed array belongs to. Its client transfers the local addressable shard
    ///     buffers.
    ///   - `r#type`: Global [`ArrayType`] of the constructed array.
    ///   - `mesh`: Destination [`DeviceMesh`], or [`None`] to use eager placement.
    ///   - `sharding`: Optional runtime [`Sharding`] over the destination mesh.
    ///   - `buffer`: Dense row-major host bytes for the full logical array.
    ///
    /// # Errors
    ///
    /// Returns an error if placement cannot be resolved, the bytes do not match the type, or a device upload fails.
    pub fn from_host_buffer<M: Into<Option<DeviceMesh>>, B: AsRef<[u8]>>(
        session: &Session,
        r#type: ArrayType,
        mesh: M,
        sharding: Option<Sharding>,
        buffer: B,
    ) -> Result<Self, Error> {
        let logical_mesh = sharding.as_ref().or(r#type.sharding()).map(Sharding::mesh);
        let mesh = match mesh.into() {
            Some(mesh) => mesh,
            None => session.device_mesh_for(&[], logical_mesh)?,
        };
        let buffer = buffer.as_ref();

        // Normalize missing sharding metadata before any placement-derived work. A dense host buffer with no explicit
        // sharding is interpreted as one replicated logical array over the provided mesh.
        let (r#type, sharding) = resolve_placement(r#type, sharding, &mesh)?;

        // Validate that the host buffer contains exactly the dense row-major bytes for the full logical array.
        let expected_byte_count = r#type.size_in_bytes()?;
        if buffer.len() != expected_byte_count {
            return Err(Error::from(ArrayError::HostBufferByteCountMismatch {
                expected: expected_byte_count,
                actual: buffer.len(),
            }));
        }

        if r#type.data_type().is_zero() {
            return Self::zero(session, r#type, Some(sharding), mesh);
        }

        // Build a lookup table for the PJRT devices that this process can upload to directly.
        let client = session.client();
        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        // Derive the global shard layout and transfer only the shards owned by this process. Remote shards remain
        // represented by metadata when `Self::from_canonical_addressable_buffers` materializes the final array.
        let data_type = r#type.data_type();
        let device_layout = r#type.layout().map(ToPjrt::to_pjrt).transpose()?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let no_axes = BTreeSet::new();

        // Along an unreduced axis of kind sum, the devices hold summands of the logical value. The devices at
        // coordinate zero along every unreduced axis receive the host value and the others receive zeros, so the
        // summands add up to it. Copies of the host value are their own maximum and minimum, so the other kinds need
        // no zeros.
        let summed_axes =
            if sharding.unreduced_kind() == UnreducedKind::Sum { sharding.unreduced_axes() } else { &no_axes };
        let unreduced_axis_indices = mesh
            .logical_mesh()
            .axes()
            .iter()
            .enumerate()
            .filter(|(_, axis)| summed_axes.contains(axis.name()))
            .map(|(axis_index, _)| axis_index)
            .collect::<Vec<_>>();
        let mut work = session.reserve_work();
        let mut addressable_buffers = Vec::new();
        for shard in layout.descriptors() {
            let shard_device = shard.device();
            if shard_device.process_index() != client_process_index {
                continue;
            }

            let shard_device_id = shard_device.id();
            let device = addressable_device_by_id.get(&shard_device_id).ok_or_else(|| {
                Error::from(ArrayError::NonAddressableDevice {
                    device_id: shard_device_id,
                    process_index: client_process_index,
                })
            })?;

            // Resolve the requested space explicitly, because passing the device itself would always upload to its
            // default memory and could silently mislabel a host-resident array as pinned or unpinned memory.
            let memory = device.memory(r#type.memory())?;

            // Upload the local shard bytes to the PJRT device using the static shape of the shard. For non-scalar
            // shards, the host pointer is moved to the shard's first element in the dense source buffer, and PJRT's
            // support for host byte strides describes how to read the shard without materializing a packed temporary
            // vector.
            let shard_shape = shard.shape();
            let shard_dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let shard_slice = shard.slice();
            let buffer_type = data_type.executable_buffer_type();
            let zero_summand = mesh
                .device_coordinates(shard.index())
                .is_some_and(|coordinates| unreduced_axis_indices.iter().any(|&axis| coordinates[axis] != 0));
            let uploaded = if zero_summand {
                let byte_count = ArrayType::new(data_type, shard_shape.clone().into()).size_in_bytes()?;
                client.buffer(
                    Arc::<[u8]>::from(vec![0u8; byte_count]),
                    buffer_type,
                    shard_dimensions.as_slice(),
                    None,
                    memory,
                    device_layout.clone(),
                )?
            } else if shard_slice.is_empty() {
                client.buffer(buffer, buffer_type, shard_dimensions.as_slice(), None, memory, device_layout.clone())?
            } else {
                // Compute row-major strides in units of elements so that each shard slice can be translated
                // into flat element offsets in the full dense host buffer.
                let mut strides = vec![1usize; shape.rank()];
                let mut stride = 1usize;
                for dimension in (0..shape.rank()).rev() {
                    strides[dimension] = stride;
                    stride = stride.checked_mul(shape[dimension]).ok_or_else(|| Error::SizeLimitExceeded {
                        message: format!(
                            "row-major stride for array with shape `{}` and element type `{}` exceeds \
                             the maximum allowed size of {}",
                            shape,
                            buffer_type,
                            usize::MAX,
                        ),
                    })?;
                }

                // Convert the shard's starting coordinates into a flat row-major element offset in the full host
                // buffer, and then into the corresponding byte offset.
                let element_size_in_bytes = buffer_type.element_size_in_bytes()?;
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
                                        "row-major byte stride for array with shape `{}` and element type \
                                         `{}` exceeds the maximum allowed size of {}",
                                        shape,
                                        buffer_type,
                                        i64::MAX,
                                    ),
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                };

                // Transfer the shard from the original host buffer. For strided shards, the slice starts at the shard's
                // first element and spans the full byte window that the stride metadata can address.
                client.buffer(
                    &buffer[start_byte_offset..end_byte_offset],
                    buffer_type,
                    shard_dimensions.as_slice(),
                    byte_strides.as_deref(),
                    memory,
                    device_layout.clone(),
                )?
            };

            work.track_buffer(&uploaded);
            addressable_buffers.push(uploaded);
        }

        // Reuse the buffer constructor for final buffer validation and global sharding metadata assembly.
        Self::from_canonical_addressable_buffers(session, r#type, Some(sharding), mesh, addressable_buffers)
    }

    /// Creates an [`Array`] by uploading the dense row-major bytes for each shard owned by this process. The type
    /// describes the global array. Every process supplies bytes only for its own devices and uses the same global
    /// type, mesh, and placement. Each replica needs its own entry, and along unreduced axes the bytes represent
    /// the partial value of that device. Shards of [`DataType::Zero`] take empty buffers and require no upload.
    ///
    /// With [`None`] for `sharding`, the function uses the sharding of `r#type` if present and replicates the
    /// array across `mesh` otherwise. A supplied sharding can place the array over auto mesh axes. After
    /// removing its auto axes with [`Sharding::without_auto_axes`], it must match any sharding in the type.
    ///
    /// The result may contain pending device buffers. Use [`Self::block_until_ready`] to wait synchronously or
    /// [`Self::ready`] to wait asynchronously for the local uploads. A borrowed byte slice can be reused once
    /// this function returns. Pass an [`Arc<[u8]>`] to let PJRT retain the host allocation until it finishes
    /// reading it. Processes can enqueue their uploads independently.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] whose PJRT client uploads the host shards.
    ///   - `r#type`: Global static [`ArrayType`] of the new array.
    ///   - `mesh`: Global [`DeviceMesh`] over which the array is placed.
    ///   - `sharding`: Optional runtime [`Sharding`] over `mesh`.
    ///   - `shards`: One pair of device ID and host bytes per local device in `mesh`, in any order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Array`] if the type has a dynamic shape, the type and runtime sharding disagree, or
    /// the host shards have missing or duplicate entries, unexpected devices, or incorrect byte counts. Other
    /// placement and PJRT errors propagate from the layout and upload.
    pub fn from_host_shards<B: AsRef<[u8]> + HostBuffer, S: IntoIterator<Item = (DeviceId, B)>>(
        session: &Session,
        r#type: ArrayType,
        mesh: DeviceMesh,
        sharding: Option<Sharding>,
        shards: S,
    ) -> Result<Self, Error> {
        let (r#type, sharding) = resolve_placement(r#type, sharding, &mesh)?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let layout = ShardLayout::new(&shape, &mesh, &sharding)?;
        let client = session.client();
        let process_index = client.process_index()?;
        let mut devices = HashMap::new();
        for device in client.addressable_devices()? {
            devices.insert(device.id()?, device);
        }

        let mut host_shards = HashMap::new();
        for (device_id, bytes) in shards {
            let index =
                layout.shard_index_by_device().get(&device_id).ok_or(ArrayError::DeviceNotInMesh { device_id })?;
            let descriptor = &layout.descriptors()[*index];
            if descriptor.device().process_index() != process_index || !devices.contains_key(&device_id) {
                return Err(ArrayError::NonAddressableDevice { device_id, process_index }.into());
            }
            if host_shards.insert(device_id, bytes).is_some() {
                return Err(ArrayError::MultipleHostShardsOnDevice { device_id }.into());
            }
        }

        let data_type = r#type.data_type();
        let mut uploads = Vec::new();
        for descriptor in layout.descriptors() {
            if descriptor.device().process_index() != process_index {
                continue;
            }

            let device_id = descriptor.device().id();
            let device =
                devices.get(&device_id).ok_or(ArrayError::NonAddressableDevice { device_id, process_index })?;
            let bytes = host_shards.remove(&device_id).ok_or(ArrayError::MissingHostShard { device_id })?;
            let shape = descriptor.shape();

            let expected = ArrayType::new(data_type, shape.clone().into()).size_in_bytes()?;
            let actual = bytes.as_ref().len();
            if actual != expected {
                return Err(ArrayError::HostShardByteCountMismatch { device_id, expected, actual }.into());
            }

            if !data_type.is_zero() {
                let memory = device.memory(r#type.memory())?;
                uploads.push((bytes, shape, memory));
            }
        }

        if data_type.is_zero() {
            return Self::zero(session, r#type, Some(sharding), mesh);
        }

        let buffer_type = data_type.executable_buffer_type();
        let device_layout = r#type.layout().map(ToPjrt::to_pjrt).transpose()?;
        let mut work = session.reserve_work();
        let mut buffers = Vec::with_capacity(uploads.len());
        for (bytes, shape, memory) in uploads {
            let dimensions = shape.as_slice().iter().map(|&extent| extent as u64).collect::<Vec<_>>();
            let buffer = client.buffer(bytes, buffer_type, dimensions, None, memory, device_layout.clone())?;
            work.track_buffer(&buffer);
            buffers.push(buffer);
        }

        Self::from_canonical_addressable_buffers(session, r#type, Some(sharding), mesh, buffers)
    }

    /// Copies the value of this [`Array`] to the host and returns it as a host array of the reference backend
    /// (i.e., a [`ryft_core::Array`]) with the data type and shape of this array. The type of the result has no
    /// sharding and no physical layout, because the host value is one dense row-major array. Read its elements with
    /// [`ryft_core::Array::elements`], which checks the requested element type against the data type of the array.
    ///
    /// This function reads the value as [`Self::copy_to_host_bytes`] does, with the same synchronization, shard
    /// selection, and combination of unreduced values, and it then validates the bytes as the storage of the host
    /// array, which visits every element once. Use [`Self::copy_to_host_bytes`] to skip that validation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::copy_to_host_bytes`], and an [`Error::Core`] if the bytes that the devices
    /// returned are not a valid encoding of the data type of this array.
    #[inline]
    pub fn copy_to_host(&self) -> Result<CoreArray, Error> {
        let bytes = self.copy_to_host_bytes()?;
        Ok(CoreArray::new(ArrayType::new(self.data_type(), self.shape().into()), bytes)?)
    }

    /// Copies the value of this [`Array`] to the host and returns its dense row-major bytes, irrespective of the
    /// physical [`Layout`](ryft_core::Layout) of its shard buffers. This is an explicit synchronization boundary.
    /// It blocks the current thread until the copies and the whole execution that produced this array complete, so
    /// it also surfaces the errors of that execution.
    ///
    /// When each tile occupies a contiguous byte range of the dense array, one replica of each tile is copied
    /// directly into the destination. Otherwise, the shard copies are merged on the host. When the [`Sharding`] has
    /// unreduced axes, every shard is copied and the separate per-device values (e.g., summands) are combined with
    /// the [`UnreducedKind`] of the sharding. Arrays with no elements and [`DataType::Zero`] arrays produce no bytes
    /// and issue no copies. Use [`ArrayShard::copy_to_host_bytes`] to read the bytes that a single device holds.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] when the current process does not hold the whole value. That is
    /// the case when a shard lives in another process and either the addressable shards do not cover the array or its
    /// sharding has unreduced axes. This is checked before issuing any copies. PJRT transfer failures and the errors
    /// of the producing execution are returned unchanged.
    pub fn copy_to_host_bytes(&self) -> Result<Vec<u8>, Error> {
        let completion = self.execution_fence();
        let global_shape = self.shape();
        let element_type = self.data_type();

        // A PJRT size query for an empty dynamic buffer can expose its bound allocation and trailer. Skip transfers
        // for arrays with no payload, and wait for the producer or local buffers so their errors still surface.
        if element_type.is_zero() || global_shape.as_slice().contains(&0) {
            if let Some(completion) = completion {
                completion.block_until_ready()?;
            } else {
                let events = self
                    .addressable_shards()
                    .filter_map(|shard| shard.buffer())
                    .map(|buffer| buffer.ready())
                    .collect::<Result<Vec<_>, _>>()?;
                ExecutionFence::new(events).block_until_ready()?;
            }
            return Ok(Vec::new());
        }

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

        let mut work = self.session().reserve_work();

        // Without unreduced axes, every device holds its tile of the array itself, so one shard per distinct tile
        // suffices. When every tile occupies one contiguous byte range of the dense representation (e.g., when only
        // the leading dimension is sharded, or for a replicated array), each is copied directly into that range of
        // one destination.
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

                // Publish the readback to the session after issuing all copies, then check the producer before
                // consuming the transfer results so any execution error takes precedence.
                drop(work);

                if let Some(completion) = completion {
                    completion.block_until_ready()?;
                }

                // Drop each returned slice so the destination can yield its full byte buffer.
                for copy in copies {
                    drop(copy.wait()?);
                }

                return Ok(destination.into_bytes().ok().unwrap());
            }
        }

        // Non-contiguous tiles and per-device values along unreduced axes need separate host buffers. Request
        // row-major bytes for the merge regardless of the source layouts, and issue all copies before waiting.
        let copies = shards
            .into_iter()
            .map(|shard| {
                let buffer = shard.buffer().unwrap();
                let shard_shape = shard.shape();
                let event = buffer.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(shard_shape.rank())))?;
                let copy = work.track_host_copy(event);
                let expected_byte_count = ArrayType::new(element_type, shard_shape.into()).size_in_bytes()?;
                Ok((shard, copy, expected_byte_count))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Publish the readback before waiting, and check the producer before consuming the transfer results.
        drop(work);

        if let Some(completion) = completion {
            completion.block_until_ready()?;
        }

        // Shards with the same slices and unreduced coordinates are replicas of one per-device value. Those
        // with different unreduced coordinates contribute separate values to the tile and must be combined.
        let shards = copies
            .into_iter()
            .map(|(shard, copy, expected_byte_count)| {
                let bytes = copy.wait()?;
                if bytes.len() != expected_byte_count {
                    return Err(Error::from(ArrayError::ShardByteCountMismatch {
                        shard_index: shard.index(),
                        device_id: shard.device().id(),
                        expected: expected_byte_count,
                        actual: bytes.len(),
                    }));
                }
                let unreduced_coordinates = match mesh.device_coordinates(shard.index()) {
                    Some(coordinates) => unreduced_axis_indices.iter().map(|&axis| coordinates[axis]).collect(),
                    None => Vec::new(),
                };
                Ok((shard.index(), shard.slice().to_vec(), unreduced_coordinates, bytes))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut shards = combine_unreduced_values(shards, element_type, self.sharding().unreduced_kind())?;

        // Reuse a single full tile's bytes when the copies and any reduction already produced the complete value.
        if shards.len() == 1 {
            let (shard_index, slices, bytes) = shards.pop().unwrap();
            let covers_global_shape = slices.len() == global_shape.rank()
                && slices
                    .iter()
                    .zip(global_shape.as_slice())
                    .all(|(slice, extent)| slice.start == 0 && slice.end == *extent);
            if covers_global_shape && bytes.len() == total_byte_count {
                return Ok(bytes);
            }
            shards.push((shard_index, slices, bytes));
        }

        // Merge tiles into the global buffer, recording written ranges to detect replicas with inconsistent data.
        let mut global_bytes = vec![0u8; total_byte_count];
        let mut written_intervals = Vec::new();
        for (shard_index, slices, bytes) in shards {
            // A scalar shard occupies one byte range starting at zero.
            if slices.is_empty() {
                merge_dense_byte_segment(bytes.as_slice(), 0, shard_index, &mut global_bytes, &mut written_intervals)?;
                continue;
            }

            // The strides map each packed shard row to its position in the global row-major buffer.
            let global_strides = row_major_element_strides(global_shape.as_slice(), element_type)?;
            let shard_shape = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
            let shard_strides = row_major_element_strides(shard_shape.as_slice(), element_type)?;
            let element_size_in_bytes = element_type.to_pjrt().element_size_in_bytes()?;
            merge_dense_shard_bytes_recursive(
                bytes.as_slice(),
                slices.as_slice(),
                global_strides.as_slice(),
                shard_strides.as_slice(),
                0,
                0,
                0,
                element_size_in_bytes,
                shard_index,
                &mut global_bytes,
                &mut written_intervals,
            )?;
        }

        Ok(global_bytes)
    }
}

// TODO(eaplatanios): Review from here onwards.

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
    let mut tiles = Vec::<(ShardIndex, Vec<Range<usize>>, CoreArray)>::new();
    for (slices, _, shard_index, bytes) in summands {
        let shape = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
        let summand = CoreArray::new(ArrayType::new_static(element_type, shape), bytes)?;
        match tiles.iter_mut().find(|(_, tile_slices, _)| *tile_slices == slices) {
            Some((_, _, tile)) => {
                *tile = match unreduced_kind {
                    UnreducedKind::Sum => Add::add(&*tile, &summand)?,
                    UnreducedKind::Max => Max::max(&*tile, &summand)?,
                    UnreducedKind::Min => Min::min(&*tile, &summand)?,
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

/// Recursively merges one shard's bytes into `global_bytes`, from dimension `dimension` onward and starting at the
/// element offsets `base_global_element_offset` and `base_shard_element_offset`. The strides are row-major element
/// strides. [`merge_dense_byte_segment`] uses the sorted, disjoint byte ranges in `written_intervals` to detect
/// inconsistent overlapping data.
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Device, DeviceMesh, Dimension, Error as CoreError, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        ProgramError, Shape, Sharding, ShardingDimension, TiledLayout,
    };
    use ryft_pjrt::{BufferSpecification, BufferType, Error as PjrtError, ExecutionFence};

    use crate::FromPjrt;
    use crate::sessions::tests::{domain_mesh, replicated_vector_type};
    use crate::tests::{cpu_client, execution_client, logical_mesh_2x2, values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_array_from_host_buffer() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0]);

        // Without a configured mesh or sharding, the first addressable device forms the default mesh.
        let default_mesh = domain_mesh(client, "x", 1);
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, bytes.as_slice()).unwrap();
        assert_eq!(array.r#type().into_owned(), replicated_vector_type(&default_mesh, 2));
        assert_eq!(array.mesh(), &default_mesh);
        assert!(array.session().is_same_session(&session));
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);

        // Without a configured mesh, the type's sharding supplies the logical mesh.
        let inferred_mesh = domain_mesh(client, "y", 1);
        let sharded_type = replicated_vector_type(&inferred_mesh, 2);
        let array = Array::from_host_buffer(&session, sharded_type.clone(), None, None, bytes.as_slice()).unwrap();
        assert_eq!(array.r#type().into_owned(), sharded_type);
        assert_eq!(array.mesh(), &inferred_mesh);
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);

        // An explicit mesh takes precedence over the session's configured mesh.
        let configured_mesh = domain_mesh(client, "z", 1);
        let configured_session = session.with_device_mesh(configured_mesh.clone());
        let array = Array::from_host_buffer(&configured_session, r#type.clone(), None, None, bytes.as_slice()).unwrap();
        assert_eq!(array.r#type().into_owned(), replicated_vector_type(&configured_mesh, 2));
        assert_eq!(array.mesh(), &configured_mesh);
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0, 2.0]);
        let array =
            Array::from_host_buffer(&configured_session, r#type.clone(), default_mesh.clone(), None, bytes.as_slice())
                .unwrap();
        assert_eq!(array.mesh(), &default_mesh);

        // A sharding over more devices than the client has cannot be inferred.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let r#type = r#type.with_sharding(Sharding::replicated(logical_mesh.clone(), 1)).unwrap();
        assert!(matches!(
            Array::from_host_buffer(&session, r#type, None, None, bytes.as_slice()),
            Err(Error::Core(error))
                if *error == CoreError::Program(ProgramError::InvalidArgument {
                    message: format!(
                        "eager XLA placement of a value without array inputs over mesh {logical_mesh:?} requires 2 \
                         addressable device(s) but the client only has 1",
                    ),
                }),
        ));
    }

    #[test]
    #[allow(deprecated)] // PJRT currently exposes actual buffer layout only through this query.
    fn test_array_from_host_buffer_layout() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = domain_mesh(client, "x", 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F64, [2, 3]).with_layout(layout.clone()),
            mesh.clone(),
            None,
            values_to_bytes(&[1f64, 2., 3., 4., 5., 6.]),
        )
        .unwrap();
        assert_eq!(array.layout(), Some(&layout));
        assert_eq!(
            array.addressable_shards().next().unwrap().buffer().unwrap().layout(),
            Ok(layout.to_pjrt().unwrap())
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f64>().unwrap(), vec![1., 2., 3., 4., 5., 6.]);
        let wrong_buffer = client
            .buffer(
                values_to_bytes(&[1f64, 2., 3., 4., 5., 6.]).as_slice(),
                BufferType::F64,
                [2, 3],
                None,
                device.clone(),
                None,
            )
            .unwrap();
        let error = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::F64, [2, 3]).with_layout(layout),
            mesh,
            None,
            vec![wrong_buffer],
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "invalid layout: buffer on device {} uses layout `tiled{{1,0}}` but its type declares `tiled{{0,1}}`",
                device.id().unwrap(),
            )
        );
    }

    #[test]
    fn test_array_from_host_buffer_memory() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap()[0].clone();
        let mesh = domain_mesh(client, "x", 1);
        let values = values_to_bytes::<i64>(&[i64::MIN, i64::MAX]);

        // Verify physical placement and exact contents for every public memory variant. Exercise both scalar and
        // nonscalar uploads because their host buffer paths use different stride handling.
        for memory in [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            for shape in [Shape::new(vec![]), Shape::new(vec![Dimension::Static(2)])] {
                let count = if shape.rank() == 0 { 1 } else { 2 };
                let r#type = ArrayType::new(DataType::I64, shape).with_memory(memory);
                let array =
                    Array::from_host_buffer(&session, r#type, mesh.clone(), None, &values[..count * 8]).unwrap();
                assert_eq!(array.r#type().memory(), memory);
                let buffer = array.addressable_shards().next().unwrap().buffer().unwrap();
                let actual = buffer.memory().unwrap();
                match memory {
                    Memory::Device => assert_eq!(actual, device.default_memory().unwrap()),
                    Memory::Host { pinned: true } => assert_eq!(actual.kind().unwrap(), "pinned_host"),
                    Memory::Host { pinned: false } => assert_eq!(actual.kind().unwrap(), "unpinned_host"),
                }
                assert_eq!(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice(), &values[..count * 8]);
            }
        }
    }

    #[test]
    fn test_array_from_host_buffer_unreduced() {
        // Along the unreduced axis `x`, the devices hold summands of the logical value, and `y` tiles its only
        // dimension. A host upload places the whole value on the devices at coordinate zero along `x` and zeros on the
        // others, and host materialization sums the summands of each tile.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let device_values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| {
                    let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>()
        };
        let materialize = |array: &Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();

        let uploaded =
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        assert_eq!(device_values(&uploaded), vec![vec![1., 2.], vec![3., 4.], vec![0., 0.], vec![0., 0.]]);
        assert_eq!(materialize(&uploaded), vec![1., 2., 3., 4.]);

        let buffers = [[1f32, 2.], [3., 4.], [10., 20.], [30., 40.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let summands = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
        assert_eq!(materialize(&summands), vec![11., 22., 33., 44.]);

        // With kind `max` or `min`, the devices along `x` hold candidates whose maximum or minimum, respectively, is
        // the logical value. A host upload places the whole value on every device, which is its own maximum, and host
        // materialization takes the elementwise maximum of the candidates of each tile.
        for (kind, expected) in [(UnreducedKind::Max, [10., 2., 30., 40.]), (UnreducedKind::Min, [1., -20., 3., 4.])] {
            let r#type = ArrayType::new_static(DataType::F32, [4])
                .with_sharding(sharding.clone().with_unreduced_kind(kind).unwrap())
                .unwrap();
            let uploaded = Array::from_host_buffer(
                &session,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes(&[-1f32, -2., 3., 4.]),
            )
            .unwrap();
            assert_eq!(device_values(&uploaded), vec![vec![-1., -2.], vec![3., 4.], vec![-1., -2.], vec![3., 4.]]);
            assert_eq!(materialize(&uploaded), vec![-1., -2., 3., 4.]);
            let buffers = [[1f32, 2.], [3., 4.], [10., -20.], [30., 40.]]
                .iter()
                .zip(&devices)
                .map(|(values, device)| {
                    client
                        .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                        .unwrap()
                })
                .collect();
            let candidates = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
            assert_eq!(materialize(&candidates), expected.to_vec());
        }
    }

    #[test]
    fn test_array_from_host_buffer_zero_space() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));

        let array = Array::from_host_buffer(&session, r#type, mesh, None, []).unwrap();
        assert_eq!(array.data_type(), DataType::Zero);
        assert_eq!(array.size_in_bytes().unwrap(), 0);
        let shard = array.addressable_shards().next().unwrap();
        assert!(shard.buffer().is_none());
        assert_eq!(array, array.clone());
        assert_eq!(array.copy_to_host_bytes().unwrap(), Vec::<u8>::new());

        let zero_sized_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(0)]));
        let zero_sized = Array::from_host_buffer(&session, zero_sized_type, array.mesh().clone(), None, []).unwrap();
        assert_eq!(zero_sized.shape(), StaticShape::new(vec![0]));
        assert!(zero_sized.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_from_host_buffer_zero_space_remote_shards() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote_device = Device::new(local_device.id() + 1, client.process_index().unwrap() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, vec![local_device, remote_device]).unwrap();
        let r#type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));

        let array = Array::from_host_buffer(&session, r#type, mesh, None, []).unwrap();

        assert_eq!(array.shards().len(), 2);
        assert!(array.shards()[0].is_addressable());
        assert!(array.shards()[0].buffer().is_none());
        assert!(!array.shards()[1].is_addressable());
        assert!(array.shards()[1].buffer().is_none());
    }

    #[test]
    fn test_array_from_host_buffer_uneven_1d_partitioning() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(5)]));

        let array = Array::from_host_buffer(
            &session,
            r#type,
            None,
            Some(sharding),
            values_to_bytes::<f32>(values.as_slice()).as_slice(),
        )
        .unwrap();
        assert_eq!(array.mesh(), &mesh);

        // The first partition receives the extra element of the uneven dimension.
        assert_eq!(array.addressable_shards().count(), 2);
        assert!(array.shards().iter().all(|shard| shard.buffer().is_some()));
        assert_eq!(
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0]],
        );
        assert_eq!(
            array.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
    }

    #[test]
    fn test_array_from_host_buffer_2d_partitioning() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let values = (0..48).map(|value| value as f32).collect::<Vec<_>>();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]));

        let array = Array::from_host_buffer(
            &session,
            r#type,
            mesh,
            Some(sharding),
            values_to_bytes::<f32>(values.as_slice()).as_slice(),
        )
        .unwrap();

        assert_eq!(array.addressable_shards().count(), 4);
        assert!(array.shards().iter().all(|shard| shard.buffer().is_some()));
        assert_eq!(
            array.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │     │     │
                    │  0  │  1  │
                    │     │     │
                    ├─────┼─────┤
                    │     │     │
                    │  2  │  3  │
                    │     │     │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        // Shards are ordered like the mesh devices, and each one holds its row-major `4x3` tile of the host value.
        assert_eq!(
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0],
                vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 15.0, 16.0, 17.0, 21.0, 22.0, 23.0],
                vec![24.0, 25.0, 26.0, 30.0, 31.0, 32.0, 36.0, 37.0, 38.0, 42.0, 43.0, 44.0],
                vec![27.0, 28.0, 29.0, 33.0, 34.0, 35.0, 39.0, 40.0, 41.0, 45.0, 46.0, 47.0],
            ],
        );
    }

    #[test]
    fn test_array_from_host_shards() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mut devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let first = devices[0].id();
        let second = devices[1].id();
        let remote = Device::new(first.max(second) + 1, devices[0].process_index() + 1);
        devices.push(remote);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [5]).with_sharding(sharding).unwrap();
        let first_bytes = [1, 2].as_slice();
        let second_bytes = [3, 4].as_slice();

        let array = Array::from_host_shards(
            &session,
            r#type.clone(),
            mesh.clone(),
            None,
            [(second, second_bytes), (first, first_bytes)],
        )
        .unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(array.shape(), StaticShape::new(vec![5]));
        assert_eq!(array.addressable_shards().count(), 2);
        assert_eq!(array.shards()[0].copy_to_host_bytes().unwrap(), first_bytes);
        assert_eq!(array.shards()[1].copy_to_host_bytes().unwrap(), second_bytes);
        assert!(!array.shards()[2].is_addressable());

        let missing =
            Array::from_host_shards(&session, r#type.clone(), mesh.clone(), None, [(first, first_bytes)]).unwrap_err();
        assert!(matches!(
            &missing,
            Error::Array(error) if matches!(error.as_ref(), ArrayError::MissingHostShard { device_id } if *device_id == second)
        ));
        assert_eq!(missing.to_string(), format!("missing host shard for device {second}"));

        let duplicate = Array::from_host_shards(
            &session,
            r#type.clone(),
            mesh.clone(),
            None,
            [(first, first_bytes), (first, first_bytes), (second, second_bytes)],
        )
        .unwrap_err();
        assert!(matches!(
            &duplicate,
            Error::Array(error) if matches!(error.as_ref(), ArrayError::MultipleHostShardsOnDevice { device_id } if *device_id == first)
        ));
        assert_eq!(duplicate.to_string(), format!("got multiple host shards for device {first}"));

        let wrong_size = Array::from_host_shards(
            &session,
            r#type.clone(),
            mesh.clone(),
            None,
            [(first, first_bytes), (second, &second_bytes[..1])],
        )
        .unwrap_err();
        assert!(matches!(
            &wrong_size,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::HostShardByteCountMismatch { device_id, expected: 2, actual: 1 } if *device_id == second
            )
        ));
        assert_eq!(wrong_size.to_string(), format!("host shard on device {second} needs 2 byte(s) but got 1"));

        assert!(matches!(
            Array::from_host_shards(
                &session,
                r#type,
                mesh,
                None,
                [(first, first_bytes), (second, second_bytes), (remote.id(), [5].as_slice())],
            ),
            Err(Error::Array(error)) if matches!(error.as_ref(), ArrayError::NonAddressableDevice { device_id, .. }
                if *device_id == remote.id()),
        ));
    }

    #[test]
    fn test_array_from_host_shards_with_auto_axis() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [3]);
        let first = mesh.devices()[0].id();
        let second = mesh.devices()[1].id();
        let array = Array::from_host_shards(
            &session,
            r#type,
            mesh,
            Some(sharding.clone()),
            [(first, Arc::<[u8]>::from([1, 2])), (second, Arc::<[u8]>::from([3]))],
        )
        .unwrap();
        array.block_until_ready().unwrap();
        assert_eq!(array.sharding(), &sharding);
        assert_eq!(array.r#type().sharding(), Some(&sharding.without_auto_axes()));
        assert_eq!(array.copy_to_host_bytes().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_array_copy_to_host() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);

        // Sharded and replicated arrays read back as the dense row-major bytes of their whole value.
        for sharding in [sharded.clone(), replicated.clone()] {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            let array = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values.as_slice()).unwrap();
            assert_eq!(array.copy_to_host_bytes().unwrap(), values);
        }

        // An array with no elements produces no bytes.
        let r#type = ArrayType::new_static(DataType::F32, [0]).with_sharding(sharded.clone()).unwrap();
        let empty = Array::from_host_buffer(&session, r#type, mesh, None, []).unwrap();
        assert_eq!(empty.copy_to_host_bytes().unwrap(), Vec::<u8>::new());

        // When the second mesh device lives in another process, the current process holds a replicated value in full,
        // but only half of a sharded value and only one of the summands of an unreduced value.
        let remote_device =
            Device::new(devices.iter().map(Device::id).max().unwrap() + 1, client.process_index().unwrap() + 1);
        let remote_mesh = DeviceMesh::new(logical_mesh, vec![devices[0], remote_device]).unwrap();
        let with_remote_shard = |sharding: Sharding, local_values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            let dimensions = [local_values.len() as u64];
            let bytes = values_to_bytes(local_values);
            let buffer =
                client.buffer(bytes.as_slice(), BufferType::F32, dimensions, None, client_devices[0].clone(), None);
            Array::from_addressable_buffers(&session, r#type, remote_mesh.clone(), None, vec![buffer.unwrap()]).unwrap()
        };
        assert_eq!(with_remote_shard(replicated, &[1., 2., 3., 4.]).copy_to_host_bytes().unwrap(), values);
        for (sharding, local_values) in [(sharded, [1f32, 2.].as_slice()), (unreduced, [1f32, 2., 3., 4.].as_slice())] {
            let error = with_remote_shard(sharding, local_values).copy_to_host_bytes().unwrap_err();
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(error.as_ref(),
                    ArrayError::MissingShardBuffer { shard_index: 1, device_id } if *device_id == remote_device.id(),
                ),
            ));
            assert_eq!(
                error.to_string(),
                format!("array shard #1 on device {} has no addressable buffer", remote_device.id()),
            );
        }
    }

    #[test]
    fn test_array_copy_to_host_bytes() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
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
        let shape = [4, 2];
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let replicated = vec![ShardingDimension::replicated(), ShardingDimension::replicated()];
        // Leave one readable shard per tile. Any extra replica copy would fail the readback.
        for (dimensions, tile_count) in [(rows, 2), (replicated, 1)] {
            let sharding = Sharding::new(logical_mesh.clone(), dimensions).unwrap();
            let r#type = ArrayType::new_static(DataType::U8, shape).with_sharding(sharding).unwrap();
            let tile_shape = [(shape[0] / tile_count) as u64, shape[1] as u64];
            let replicas_per_tile = devices.len() / tile_count;
            let tiles = bytes.chunks_exact(bytes.len() / tile_count).collect::<Vec<_>>();
            let buffers = devices
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, device)| {
                    if index.is_multiple_of(replicas_per_tile) {
                        client.buffer(tiles[index / replicas_per_tile], BufferType::U8, tile_shape, None, device, None)
                    } else {
                        client.error_buffer(
                            PjrtError::aborted("unexpected replica copy"),
                            BufferSpecification::new(BufferType::U8, tile_shape),
                            device,
                        )
                    }
                    .unwrap()
                })
                .collect();
            let array = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
            assert_eq!(array.copy_to_host_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn test_array_copy_to_host_bytes_preserves_producer_errors_without_payload() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        for (data_type, shape) in [(DataType::F32, [0]), (DataType::Zero, [2])] {
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(PjrtError::invalid_argument("empty producer failed"))).unwrap();
            let array =
                Array::from_host_buffer(&session, ArrayType::new_static(data_type, shape), mesh.clone(), None, [])
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(vec![event]));
            assert_eq!(array.copy_to_host_bytes().unwrap_err().to_string(), "empty producer failed");
        }
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
    fn test_merge_dense_shard_bytes_recursive() {
        // The second column of a `[2, 2]` array occupies two separate byte ranges of its dense representation.
        let mut global_bytes = vec![0u8; 16];
        let mut written_intervals = Vec::new();
        merge_dense_shard_bytes_recursive(
            values_to_bytes(&[1f32, 2.]).as_slice(),
            &[0..2, 1..2],
            &[2, 1],
            &[1, 1],
            0,
            0,
            0,
            size_of::<f32>(),
            0,
            &mut global_bytes,
            &mut written_intervals,
        )
        .unwrap();
        assert_eq!(values_from_bytes::<f32>(global_bytes.as_slice()), vec![0., 1., 0., 2.]);
        assert_eq!(written_intervals, vec![4..8, 12..16]);
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
