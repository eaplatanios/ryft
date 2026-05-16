use std::collections::HashMap;
use std::sync::Arc;

use ryft_core::{ArrayType, DeviceMesh, Shape, Sharding, Size, check_sharding};
use ryft_pjrt::{Buffer, Client, DeviceId};

use crate::arrays::ArrayTypeExtension;
use crate::arrays_v0::{
    DevicePutTarget, ExecuteArguments, copy_addressable_destination_shards_from_exact_source_shards,
};
use crate::{Array, ArrayError, Error as XlaError, ToMlir, ToPjrt};

impl<'o> Array<'o> {
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

        let element_type = self.data_type();
        let total_byte_count = self.size_in_bytes()?;
        let mut host_bytes = vec![0u8; total_byte_count];
        let mut written = vec![false; total_byte_count];

        // Row-major element strides over `global_shape` translate each shard slice into flat
        // element offsets in the dense host buffer.
        let element_size_in_bytes = element_type.to_pjrt().element_size_in_bytes().map_err(XlaError::from)?;
        let mut global_strides = vec![1usize; global_dimensions.len()];
        let mut stride = 1usize;
        for dimension in (0..global_dimensions.len()).rev() {
            global_strides[dimension] = stride;
            stride = stride.checked_mul(global_dimensions[dimension]).ok_or_else(|| XlaError::SizeLimitExceeded {
                message: format!(
                    "row-major stride for array with shape {global_shape} and element type {element_type} exceeds \
                     the maximum allowed size of {}",
                    usize::MAX,
                ),
            })?;
        }

        for shard in self.shards() {
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

            // Each shard slice decomposes into `outer_iteration_count` contiguous row-major spans
            // of `block_byte_count` bytes; outer dimensions `[0..block_dim)` are walked by an
            // odometer over `counters`, while inner dimensions `[block_dim..rank)` form one block
            // per iteration. The shard buffer is dense and consumed sequentially, while
            // `global_element_offset` is updated incrementally on each counter mutation to keep
            // per-iteration arithmetic O(1) in the rank.
            let descriptor = shard.descriptor();
            let shard_slices = descriptor.slice();
            let (block_dim, block_element_count) = descriptor.contiguous_inner_block(&global_shape)?;
            let block_byte_count = block_element_count * element_size_in_bytes;
            let mut counters: Vec<usize> = shard_slices[..block_dim].iter().map(|slice| slice.start).collect();
            let mut global_element_offset = shard_slices
                .iter()
                .enumerate()
                .map(|(dimension, slice)| slice.start * global_strides[dimension])
                .sum::<usize>();
            let outer_iteration_count: usize = shard_slices[..block_dim].iter().map(|slice| slice.len()).product();
            for iteration_index in 0..outer_iteration_count {
                let global_byte_offset = global_element_offset * element_size_in_bytes;
                let shard_byte_offset = iteration_index * block_byte_count;
                for (offset, &byte) in
                    shard_bytes[shard_byte_offset..shard_byte_offset + block_byte_count].iter().enumerate()
                {
                    let index = global_byte_offset + offset;
                    if written[index] {
                        if host_bytes[index] != byte {
                            return Err(ArrayError::InconsistentOverlappingShardData { shard_index });
                        }
                    } else {
                        host_bytes[index] = byte;
                        written[index] = true;
                    }
                }
                let mut dimension = block_dim;
                while dimension > 0 {
                    dimension -= 1;
                    counters[dimension] += 1;
                    global_element_offset += global_strides[dimension];
                    if counters[dimension] < shard_slices[dimension].end {
                        break;
                    }
                    let span = shard_slices[dimension].len();
                    counters[dimension] = shard_slices[dimension].start;
                    global_element_offset -= span * global_strides[dimension];
                }
            }
        }

        let r#type = ArrayType::new(self.data_type(), self.shape().clone().into(), None, Some(sharding))?;
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
        self.shards()
            .iter()
            .filter_map(|shard| {
                let (descriptor, buffer) = shard.clone().into_parts();
                let device_id = descriptor.device().id();
                buffer.map(|buffer| (device_id, buffer))
            })
            .collect()
    }
}
