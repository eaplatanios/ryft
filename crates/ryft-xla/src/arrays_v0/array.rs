use std::collections::HashMap;
use std::sync::Arc;

use ryft_core::{ArrayType, DataType, DeviceMesh, Shape, Sharding, Size, StaticShape, check_sharding};
use ryft_pjrt::{Buffer, Client, DeviceId};

use crate::arrays_v0::{
    DevicePutTarget, ExecuteArguments, checked_byte_count,
    copy_addressable_destination_shards_from_exact_source_shards, extract_dense_shard_bytes,
    materialize_dense_array_bytes,
};
use crate::{Array, ArrayError, ArrayShard, Error, ShardLayout, ToMlir, ToPjrt};

impl<'o> Array<'o> {
    /// Creates an [`Array`] by uploading one dense row-major host buffer to the local shards implied by `mesh` and
    /// `sharding`.
    ///
    /// This is the low-level host-buffer constructor used by the higher-level
    /// [`device_put`](crate::arrays_v0::device_put::device_put)
    /// surface. The constructor derives the per-device shard slices from the provided mesh/sharding pair, uploads only
    /// the shards addressable by `client`, and returns an [`Array`] whose global shard metadata covers the full mesh.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to upload the local addressable shard buffers.
    ///   - `buffer`: Dense row-major host bytes for the full logical array.
    ///   - `global_shape`: Global logical array shape.
    ///   - `element_type`: Element type stored in `buffer`.
    ///   - `mesh`: Concrete destination mesh describing the device topology.
    ///   - `sharding`: Sharding to apply over `mesh`.
    pub fn from_host_buffer<B: AsRef<[u8]>, D: AsRef<[usize]>>(
        client: &'o Client<'_>,
        buffer: B,
        global_shape: D,
        element_type: DataType,
        mesh: DeviceMesh,
        sharding: Sharding,
    ) -> Result<Self, ArrayError> {
        check_sharding!(&mesh, &sharding);

        let buffer = buffer.as_ref();
        let global_dimensions = global_shape.as_ref();
        let global_shape = StaticShape::new(global_dimensions.to_vec());
        let expected_byte_count = checked_byte_count(global_dimensions, element_type)?;
        if buffer.len() != expected_byte_count {
            return Err(Error::ByteCountMismatch { expected: expected_byte_count, got: buffer.len() }.into());
        }

        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        let layout = ShardLayout::new(&global_shape, &mesh, &sharding)?;
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
            let shard_bytes = extract_dense_shard_bytes(buffer, global_dimensions, shard.slice(), element_type)?;
            let shard_shape = shard.shape();
            let shard_dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let addressable_buffer = client.buffer(
                shard_bytes.as_slice(),
                element_type.to_pjrt(),
                shard_dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )?;
            addressable_buffers.push(addressable_buffer);
        }

        let shape = Shape::new(global_dimensions.iter().copied().map(Size::Static).collect());
        let array_type = ArrayType::new(element_type, shape, None, Some(sharding))?;
        Ok(Self::from_addressable_buffers(array_type, mesh, addressable_buffers)?)
    }

    /// Returns the global array type metadata.
    pub fn array_type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns the global array element type.
    pub fn element_type(&self) -> DataType {
        self.r#type.data_type()
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

    /// Returns metadata for all global shards.
    pub fn shards(&self) -> &[ArrayShard<'o>] {
        self.shards.as_slice()
    }

    /// Returns an iterator over the addressable local shards.
    pub fn addressable_shards(&self) -> impl Iterator<Item = &ArrayShard<'o>> {
        self.shards.iter().filter(|shard| shard.is_addressable())
    }

    /// Returns global shard metadata for `device_id`, if it exists in the mesh.
    pub fn device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.shard_index_by_device.get(&device_id).and_then(|index| self.shards.get(*index))
    }

    /// Returns the addressable shard for `device_id`, if local.
    pub fn addressable_device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.device_shard(device_id).filter(|shard| shard.is_addressable())
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
            let array_type = ArrayType::new(self.element_type(), shape, None, Some(sharding))?;
            return Ok(Self::from_addressable_buffers(array_type, mesh, addressable_buffers)?);
        }

        let host_bytes = materialize_dense_array_bytes(self)?;
        Self::from_host_buffer(client, host_bytes.as_slice(), global_dimensions, self.element_type(), mesh, sharding)
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
