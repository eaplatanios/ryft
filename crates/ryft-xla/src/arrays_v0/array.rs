use std::collections::HashMap;
use std::sync::Arc;

use ryft_core::{ArrayType, DeviceMesh, Shape, Sharding, Size, check_sharding};
use ryft_pjrt::{Buffer, Client, DeviceId};

use crate::arrays::ArrayTypeExtension;
use crate::arrays_v0::{
    DevicePutTarget, ExecuteArguments, copy_addressable_destination_shards_from_exact_source_shards,
    materialize_dense_array_bytes,
};
use crate::{Array, ArrayError, Error, ShardLayout, ToMlir, ToPjrt};

impl<'o> Array<'o> {
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
            return Err(Error::ByteCountMismatch { expected: expected_byte_count, got: buffer.len() }.into());
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

            let device = addressable_device_by_id.get(&shard_device.id()).ok_or(Error::NonAddressableDevice {
                device_id: shard_device.id(),
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

    /// Returns the global array sharding.
    pub fn sharding(&self) -> &Sharding {
        self.r#type
            .sharding()
            .expect("runtime arrays should only be constructed from array types with sharding")
    }

    /// Returns the concrete mesh implied by this array's global shard placement metadata.
    pub fn mesh(&self) -> DeviceMesh {
        DeviceMesh::new(self.sharding().mesh().clone(), self.shards.iter().map(|shard| shard.device()).collect())
            .expect("runtime arrays should always contain one shard descriptor per device")
    }

    /// Moves or copies this array to the provided placement.
    ///
    /// This is the `ryft` analogue of applying JAX's `device_put(array, sharding)` or
    /// `Array.to_device(sharding)` to an existing array. The method first tries to satisfy every
    /// local destination shard from one exact matching source shard. Exact matches on the current
    /// host use direct device-to-device copies, and exact matches on remote hosts use the PJRT
    /// cross-host transfers extension when it is available. When the destination requires
    /// repartitioning, concatenating shards, or exact remote moves without the extension, the
    /// method falls back to materializing the full logical array as dense row-major host bytes on
    /// the current process and then reuses [`Array::from_host_buffer`] to upload the destination
    /// shards. That
    /// fallback requires every global shard of `self` to be addressable from the current process.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to upload the destination local shards.
    ///   - `mesh`: Concrete destination mesh describing the device topology.
    ///   - `sharding`: Sharding to apply over `mesh`.
    pub fn to_placement(
        &self,
        client: &'o Client<'_>,
        mesh: DeviceMesh,
        sharding: Sharding,
    ) -> Result<Self, ArrayError> {
        check_sharding!(&mesh, &sharding);
        let global_shape = self.shape();
        let global_dimensions = global_shape.as_slice();
        if let Some(addressable_buffers) =
            copy_addressable_destination_shards_from_exact_source_shards(self, client, &global_shape, &mesh, &sharding)?
        {
            let shape = Shape::new(global_dimensions.iter().copied().map(Size::Static).collect());
            let array_type = ArrayType::new(self.data_type(), shape, None, Some(sharding))?;
            return Ok(Self::from_addressable_buffers(array_type, mesh, addressable_buffers)?);
        }

        let host_bytes = materialize_dense_array_bytes(self)?;
        let r#type = ArrayType::new(self.data_type(), self.r#type.shape().clone(), None, Some(sharding))?;
        Self::from_host_buffer(client, r#type, mesh, host_bytes.as_slice())
    }

    /// Moves or copies this array to the provided placement, consuming `self`.
    ///
    /// This is the closest `ryft` analogue to JAX's
    /// [`jax.Array.to_device`](https://docs.jax.dev/en/latest/_autosummary/jax.Array.to_device.html).
    /// When the resolved placement matches the current placement, the method returns `self`
    /// unchanged. Otherwise it falls back to [`Array::to_placement`] to produce a newly placed array.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to materialize any new destination buffers.
    ///   - `device`: Destination placement for this array.
    pub fn to_device(self, client: &'o Client<'_>, device: DevicePutTarget) -> Result<Self, ArrayError> {
        let current_mesh = self.mesh();
        let current_sharding = self.sharding().clone();
        let (target_mesh, target_sharding) = device.resolve(current_sharding.rank())?;
        if current_mesh == target_mesh && current_sharding == target_sharding {
            Ok(self)
        } else {
            self.to_placement(client, target_mesh, target_sharding)
        }
    }

    /// Renders the Shardy tensor sharding attribute (`#sdy.sharding<...>`) implied by this array.
    ///
    /// Uses the canonical `@mesh` symbol name.
    pub fn to_shardy_tensor_sharding_attribute(&self) -> Result<String, ryft_mlir::Error> {
        let context = ryft_mlir::Context::new();
        self.sharding().to_mlir(context.unknown_location()).map(|attribute| attribute.to_string())
    }

    /// Converts distributed arrays to per-device execution arguments for [`ryft_pjrt::LoadedExecutable::execute`].
    ///
    /// Inputs are generated in `addressable_device_ids` order. The resulting [`ExecuteArguments`] can be converted
    /// to `Vec<ExecutionDeviceInputs>` via [`ExecuteArguments::as_execution_device_inputs`].
    pub fn into_execute_arguments(
        arrays: Vec<Self>,
        addressable_device_ids: &[DeviceId],
    ) -> Result<ExecuteArguments<'o>, ArrayError> {
        let donation_flags = vec![false; arrays.len()];
        ExecuteArguments::from_arrays_with_donation(arrays, addressable_device_ids, donation_flags.as_slice())
    }

    /// Same as [`Array::into_execute_arguments`] but with explicit per-input donation flags.
    pub fn into_execute_arguments_with_donation(
        arrays: Vec<Self>,
        addressable_device_ids: &[DeviceId],
        donation_flags: &[bool],
    ) -> Result<ExecuteArguments<'o>, ArrayError> {
        ExecuteArguments::from_arrays_with_donation(arrays, addressable_device_ids, donation_flags)
    }

    pub(crate) fn into_addressable_buffers_by_device(self) -> HashMap<DeviceId, Arc<Buffer<'o>>> {
        self.shards
            .into_iter()
            .filter_map(|shard| {
                let (descriptor, buffer) = shard.into_parts();
                let device_id = descriptor.device().id();
                buffer.map(|buffer| (device_id, buffer))
            })
            .collect()
    }
}
