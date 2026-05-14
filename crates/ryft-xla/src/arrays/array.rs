use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ryft_core::{ArrayType, DataType, DeviceMesh, Parameter, Shape, Sharding, Size, StaticShape, Typed};
use ryft_macros::Parameter;
use ryft_mlir::Location;
use ryft_mlir::dialects::shardy::DetachedMeshOperation;
use ryft_pjrt::{Buffer, Client, DeviceId};

use crate::arrays::{
    ArrayPlacement, DevicePutTarget, ExecuteArguments, checked_byte_count,
    copy_addressable_destination_shards_from_exact_source_shards, extract_dense_shard_bytes,
    materialize_dense_array_bytes, static_shape,
};
use crate::{ArrayError, ArrayShard, FromPjrt, ShardIndex, ShardLayout, ToMlir, ToPjrt};

/// Distributed array backed by local addressable PJRT buffers together with global array metadata.
#[derive(Clone, Parameter)]
pub struct Array<'o> {
    /// Global array metadata carried by this distributed array handle.
    array_type: ArrayType,

    /// All global shards in mesh order together with their device ownership.
    shards: Vec<ArrayShard<'o>>,

    /// Lookup table from device id to the corresponding shard index in [`Self::shards`].
    shard_index_by_device: HashMap<DeviceId, ShardIndex>,

    /// Indices of the shards addressable from the current process.
    addressable_shard_indices: Vec<ShardIndex>,
}

impl Typed<ArrayType> for Array<'_> {
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.array_type)
    }
}

impl<'o> Array<'o> {
    /// Creates an [`Array`] from global array metadata, a concrete mesh, and local addressable buffers.
    ///
    /// `array_type.shape` must be fully static. Each buffer is mapped to a shard using its device ID, and its shape
    /// and element type are validated against the computed shard metadata.
    pub fn from_addressable_buffers(
        array_type: ArrayType,
        mesh: DeviceMesh,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let sharding = array_type.sharding().ok_or(ArrayError::MissingArraySharding)?;
        let global_shape = StaticShape::new(static_shape(&array_type)?);
        let layout = ShardLayout::new(&global_shape, &mesh, sharding)?;
        let (descriptors, shard_index_by_device) = layout.into_parts();

        let mut seen_devices = HashSet::with_capacity(addressable_buffers.len());
        let mut buffers_by_device = HashMap::with_capacity(addressable_buffers.len());

        for buffer in addressable_buffers {
            let device = buffer.device()?;
            let device_id = device.id()?;
            if !seen_devices.insert(device_id) {
                return Err(ArrayError::DuplicateAddressableBufferDevice { device_id });
            }

            let shard_index = shard_index_by_device
                .get(&device_id)
                .copied()
                .ok_or(ArrayError::AddressableBufferDeviceNotInMesh { device_id })?;
            let descriptor = descriptors
                .get(shard_index)
                .expect("shard index should exist for valid mesh device-to-shard mapping");

            let process_index = device.process_index()?;
            if process_index != descriptor.device().process_index() {
                return Err(ArrayError::BufferProcessIndexMismatch {
                    device_id,
                    expected_process_index: descriptor.device().process_index(),
                    actual_process_index: process_index,
                });
            }

            let actual_element_type = DataType::from_pjrt(buffer.element_type()?)?;
            if actual_element_type != array_type.data_type() {
                return Err(ArrayError::BufferElementTypeMismatch {
                    device_id,
                    expected: array_type.data_type(),
                    actual: actual_element_type,
                });
            }

            let actual_shape = StaticShape::new(
                buffer
                    .dimensions()?
                    .iter()
                    .enumerate()
                    .map(|(dimension, size)| {
                        usize::try_from(*size).map_err(|_| ArrayError::BufferShapeDimensionTooLarge {
                            device_id,
                            dimension,
                            size: *size,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let expected_shape = descriptor.shape();
            if actual_shape != expected_shape {
                return Err(ArrayError::BufferShapeMismatch { device_id, shard_index, expected_shape, actual_shape });
            }

            buffers_by_device.insert(device_id, Arc::new(buffer));
        }

        let mut addressable_shard_indices = Vec::with_capacity(buffers_by_device.len());
        let shards = descriptors
            .into_iter()
            .map(|descriptor| {
                let buffer = buffers_by_device.remove(&descriptor.device().id());
                if buffer.is_some() {
                    addressable_shard_indices.push(descriptor.index());
                }
                ArrayShard::new(descriptor, buffer)
            })
            .collect::<Vec<_>>();

        Ok(Self { array_type, shards, shard_index_by_device, addressable_shard_indices })
    }

    /// Creates an [`Array`] by uploading one dense row-major host buffer to the local shards
    /// implied by `placement`.
    ///
    /// This is the low-level host-buffer constructor used by the higher-level [`device_put`]
    /// surface. The constructor derives the per-device shard slices from the provided placement,
    /// uploads only the shards addressable by `client`, and returns an [`Array`] whose global
    /// shard metadata covers the full mesh.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to upload the local addressable shard buffers.
    ///   - `buffer`: Dense row-major host bytes for the full logical array.
    ///   - `global_shape`: Global logical array shape.
    ///   - `element_type`: Element type stored in `buffer`.
    ///   - `placement`: Concrete mesh and sharding for the global logical array.
    pub fn from_host_buffer<B: AsRef<[u8]>, D: AsRef<[usize]>>(
        client: &'o Client<'_>,
        buffer: B,
        global_shape: D,
        element_type: DataType,
        placement: ArrayPlacement,
    ) -> Result<Self, ArrayError> {
        let buffer = buffer.as_ref();
        let global_dimensions = global_shape.as_ref();
        let global_shape = StaticShape::new(global_dimensions.to_vec());
        let expected_byte_count = checked_byte_count(global_dimensions, element_type)?;
        if buffer.len() != expected_byte_count {
            return Err(ArrayError::HostDataLengthMismatch { expected_byte_count, actual_byte_count: buffer.len() });
        }

        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        let layout = ShardLayout::new(&global_shape, placement.mesh(), placement.sharding())?;
        let mut addressable_buffers = Vec::new();
        for shard in layout.descriptors() {
            let shard_device = shard.device();
            if shard_device.process_index() != client_process_index {
                continue;
            }

            let device = addressable_device_by_id.get(&shard_device.id()).ok_or(
                ArrayError::MissingClientDeviceForLocalMeshDevice {
                    device_id: shard_device.id(),
                    process_index: client_process_index,
                },
            )?;
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

        Self::from_shape_and_placement(global_dimensions.to_vec(), element_type, placement, addressable_buffers)
    }

    pub(crate) fn from_shape_and_placement(
        global_shape: Vec<usize>,
        element_type: DataType,
        placement: ArrayPlacement,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let shape = Shape::new(global_shape.iter().copied().map(Size::Static).collect());
        let (mesh, sharding) = placement.into_parts();
        let array_type = ArrayType::new(element_type, shape, None, Some(sharding))?;
        Self::from_addressable_buffers(array_type, mesh, addressable_buffers)
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
    ///   - `placement`: Concrete destination placement.
    pub fn to_placement(&self, client: &'o Client<'_>, placement: ArrayPlacement) -> Result<Self, ArrayError> {
        let global_dimensions = self.shape();
        let global_shape = StaticShape::new(global_dimensions.clone());
        if let Some(addressable_buffers) = copy_addressable_destination_shards_from_exact_source_shards(
            self,
            client,
            &global_shape,
            placement.mesh(),
            placement.sharding(),
        )? {
            return Self::from_shape_and_placement(
                global_dimensions,
                self.element_type(),
                placement,
                addressable_buffers,
            );
        }

        let host_bytes = materialize_dense_array_bytes(self)?;
        Self::from_host_buffer(
            client,
            host_bytes.as_slice(),
            global_dimensions.as_slice(),
            self.element_type(),
            placement,
        )
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
        let current_placement = ArrayPlacement::from_parts_unchecked(self.mesh(), self.sharding().clone());
        let target_placement = device.resolve(self.sharding().rank())?;
        if current_placement == target_placement { Ok(self) } else { self.to_placement(client, target_placement) }
    }

    /// Returns the global array type metadata.
    pub fn array_type(&self) -> &ArrayType {
        &self.array_type
    }

    /// Returns the concrete global array shape.
    pub fn shape(&self) -> Vec<usize> {
        static_shape(&self.array_type)
            .expect("runtime arrays should only be constructed from array types with static shapes")
    }

    /// Returns the global array element type.
    pub fn element_type(&self) -> DataType {
        self.array_type.data_type()
    }

    /// Returns the global array sharding.
    pub fn sharding(&self) -> &Sharding {
        self.array_type
            .sharding()
            .expect("runtime arrays should only be constructed from array types with sharding")
    }

    /// Returns the concrete placement implied by this array's global shard metadata.
    pub fn placement(&self) -> ArrayPlacement {
        ArrayPlacement::from_parts_unchecked(self.mesh(), self.sharding().clone())
    }

    /// Returns the concrete mesh implied by this array's global shard placement metadata.
    pub fn mesh(&self) -> DeviceMesh {
        DeviceMesh::new(self.sharding().mesh().clone(), self.shards.iter().map(|shard| shard.device()).collect())
            .expect("runtime arrays should always contain one shard descriptor per mesh device")
    }

    /// Returns metadata for all global shards.
    pub fn shards(&self) -> &[ArrayShard<'o>] {
        self.shards.as_slice()
    }

    /// Returns an iterator over the addressable local shards.
    pub fn addressable_shards(&self) -> impl ExactSizeIterator<Item = &ArrayShard<'o>> {
        self.addressable_shard_indices.iter().map(|index| &self.shards[*index])
    }

    /// Returns the addressable shard for `device_id`, if local.
    pub fn addressable_shard_for_device(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.shard_for_device(device_id).filter(|shard| shard.is_addressable())
    }

    /// Returns global shard metadata for `device_id`, if it exists in the mesh.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.shard_index_by_device.get(&device_id).and_then(|index| self.shards.get(*index))
    }

    /// Returns global shard metadata for a local addressable shard index.
    pub fn shard_for_addressable_index(&self, addressable_shard_index: usize) -> Option<&ArrayShard<'o>> {
        self.addressable_shard_indices
            .get(addressable_shard_index)
            .and_then(|index| self.shards.get(*index))
    }

    /// Builds the detached Shardy mesh declaration (`sdy.mesh`) implied by this array's sharding.
    ///
    /// # Parameters
    ///
    ///   - `location`: MLIR location attached to the emitted mesh operation.
    ///
    /// Uses the canonical `@mesh` symbol name.
    pub fn to_shardy_mesh_operation<'c, 't, L>(
        &self,
        location: L,
    ) -> Result<DetachedMeshOperation<'c, 't>, ryft_mlir::Error>
    where
        't: 'c,
        L: Location<'c, 't>,
    {
        self.sharding().mesh().to_mlir(location)
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

impl std::fmt::Debug for Array<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Array")
            .field("array_type", &self.array_type)
            .field("shape", &self.shape())
            .field("element_type", &self.element_type())
            .field("global_shard_count", &self.shards().len())
            .field("addressable_shard_count", &self.addressable_shard_indices.len())
            .finish()
    }
}
