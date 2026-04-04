//! Runtime sharded-array data structures for XLA execution.
//!
//! This module builds on the sharding metadata from [`super::sharding`] to provide runtime
//! array types that pair global [`crate::types::ArrayType`] metadata, global shard placement
//! metadata, and local PJRT buffers:
//!
//! - [`Array`] corresponds to `jax.Array` / IFRT `Array`: global type and shard-placement
//!   metadata plus local addressable device buffers.
//! - [`ArrayShard`] corresponds to one entry in JAX's `array.global_shards`, with
//!   [`ArrayShard::buffer`] identifying the addressable local subset.
//! - [`ExecuteArguments`] marshals distributed arrays into per-device execution inputs for PJRT.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

#[cfg(test)]
use ryft_mlir::Block;
use ryft_mlir::{Location, dialects::shardy::DetachedMeshOperation};
use ryft_pjrt::{Buffer, DeviceId, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput};

use crate::sharding::{DeviceMesh, MeshDevice, Sharding, ShardingError};
use crate::types::data_types::{DataType, DataTypeError};
use crate::types::{ArrayType, Shape, Size};

use super::sharding::{Shard, ShardSlice, compute_shard_descriptors};

/// Error type for [`Array`] construction and execution-input preparation.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ArrayError {
    /// Underlying error returned by PJRT.
    #[error("{0}")]
    PjrtError(#[from] PjrtError),

    /// Underlying sharding error.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Underlying data-type conversion error.
    #[error("{0}")]
    DataTypeError(#[from] DataTypeError),

    /// Error returned when the array type is missing sharding metadata.
    #[error("array type is missing sharding metadata")]
    MissingArraySharding,

    /// Error returned when the array type shape is not fully static.
    #[error("array type dimension #{dimension} must be static, but got {size}")]
    DynamicArrayShape { dimension: usize, size: Size },

    /// Error returned when an addressable buffer is placed on a device not present in the array mesh.
    #[error("addressable buffer is placed on device {device_id}, but that device is not in the mesh")]
    AddressableBufferDeviceNotInMesh { device_id: DeviceId },

    /// Error returned when more than one addressable buffer is provided for the same device.
    #[error("got multiple addressable buffers for device {device_id}")]
    DuplicateAddressableBufferDevice { device_id: DeviceId },

    /// Error returned when a buffer element type does not match the array element type.
    #[error("buffer on device {device_id} has element type {actual}, but array element type is {expected}")]
    BufferElementTypeMismatch { device_id: DeviceId, expected: DataType, actual: DataType },

    /// Error returned when a buffer shape dimension cannot be represented as `usize`.
    #[error("buffer on device {device_id} has shape dimension #{dimension}={size}, which does not fit in usize")]
    BufferShapeDimensionTooLarge { device_id: DeviceId, dimension: usize, size: u64 },

    /// Error returned when a buffer shape does not match the expected shard shape.
    #[error(
        "buffer on device {device_id} has shape {actual_shape:?}, but shard #{shard_index} expects {expected_shape:?}"
    )]
    BufferShapeMismatch {
        device_id: DeviceId,
        shard_index: usize,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },

    /// Error returned when a buffer process index does not match the process index encoded in the mesh.
    #[error(
        "buffer on device {device_id} reports process index {actual_process_index}, but the mesh expects {expected_process_index}"
    )]
    BufferProcessIndexMismatch { device_id: DeviceId, expected_process_index: usize, actual_process_index: usize },

    /// Error returned when the number of donation flags does not match the number of arrays.
    #[error("got {actual_count} donation flag(s), but expected {expected_count}")]
    DonationFlagCountMismatch { expected_count: usize, actual_count: usize },

    /// Error returned when the device list for execution contains duplicate IDs.
    #[error("device {device_id} appears multiple times in the execution device order")]
    DuplicateExecutionDeviceId { device_id: DeviceId },

    /// Error returned when an array does not have an addressable shard for a required device.
    #[error("input array #{array_index} has no addressable shard for device {device_id}")]
    MissingArrayShardForDevice { array_index: usize, device_id: DeviceId },

    /// Error returned when an array has an addressable shard for a device that is not in the execution device order.
    #[error("input array #{array_index} has an unexpected addressable shard for device {device_id}")]
    UnexpectedArrayShardDevice { array_index: usize, device_id: DeviceId },
}

/// Returns the concrete shape encoded by `array_type`.
fn static_shape(array_type: &ArrayType) -> Result<Vec<usize>, ArrayError> {
    array_type
        .shape
        .dimensions
        .iter()
        .enumerate()
        .map(|(dimension, size)| match size {
            Size::Static(value) => Ok(*value),
            _ => Err(ArrayError::DynamicArrayShape { dimension, size: *size }),
        })
        .collect()
}

/// One global shard of an [`Array`].
///
/// This corresponds to one entry in JAX's `array.global_shards`. When [`Self::buffer`] returns
/// `Some(_)`, the shard is addressable from the current process and corresponds to one entry in
/// `array.addressable_shards`.
pub struct ArrayShard<'o> {
    descriptor: Shard,
    buffer: Option<Buffer<'o>>,
}

impl<'o> ArrayShard<'o> {
    /// Global shard descriptor.
    pub fn descriptor(&self) -> &Shard {
        &self.descriptor
    }

    /// Global shard index in row-major mesh order.
    pub fn shard_index(&self) -> usize {
        self.descriptor.shard_index()
    }

    /// Device that owns this shard.
    pub fn device(&self) -> MeshDevice {
        self.descriptor.device()
    }

    /// Device ID on which this buffer is placed.
    pub fn device_id(&self) -> DeviceId {
        self.device().id
    }

    /// Process index owning this shard's device.
    pub fn process_index(&self) -> usize {
        self.device().process_index
    }

    /// Row-major mesh coordinate of this shard.
    pub fn mesh_coordinate(&self) -> &[usize] {
        self.descriptor.mesh_coordinate()
    }

    /// Per-dimension logical slices for this shard.
    pub fn slices(&self) -> &[ShardSlice] {
        self.descriptor.slices()
    }

    /// Logical shape of this shard.
    pub fn shape(&self) -> &[usize] {
        self.descriptor.shape()
    }

    /// Whether this shard is backed by a local PJRT buffer on the current process.
    pub fn is_addressable(&self) -> bool {
        self.buffer.is_some()
    }

    /// Local PJRT buffer for this shard, if the shard is addressable from the current process.
    pub fn buffer(&self) -> Option<&Buffer<'o>> {
        self.buffer.as_ref()
    }

    fn into_addressable_buffer(self) -> Option<(DeviceId, Buffer<'o>)> {
        let device_id = self.device_id();
        self.buffer.map(|buffer| (device_id, buffer))
    }
}

impl std::fmt::Debug for ArrayShard<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ArrayShard")
            .field("shard_index", &self.shard_index())
            .field("device_id", &self.device_id())
            .field("process_index", &self.process_index())
            .field("is_addressable", &self.is_addressable())
            .finish()
    }
}

/// Distributed array backed by local addressable PJRT buffers together with global array metadata.
///
/// This is conceptually aligned with JAX/IFRT arrays:
/// - `array_type` stores element type, abstract shape metadata, and sharding.
/// - `shards` describes all global shards across the full mesh together with concrete device ownership.
/// - addressable shards are the subset of [`Self::shards`] whose [`ArrayShard::buffer`] is present.
///
/// In JAX terminology, this is the runtime pairing of:
/// - mesh-bound global array metadata, and
/// - addressable device buffers (the local portion of an IFRT array).
pub struct Array<'o> {
    array_type: ArrayType,
    shards: Vec<ArrayShard<'o>>,
    shard_index_by_device: HashMap<DeviceId, usize>,
    addressable_shard_indices: Vec<usize>,
}

impl<'o> Array<'o> {
    /// Creates an [`Array`] from global array metadata, a concrete mesh, and local addressable buffers.
    ///
    /// `array_type.shape` must be fully static. Each buffer is mapped to a shard using its device ID, and its shape
    /// and element type are validated against the computed shard metadata.
    pub fn new(
        array_type: ArrayType,
        mesh: DeviceMesh,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let shape = static_shape(&array_type)?;
        let sharding = array_type.sharding.as_ref().ok_or(ArrayError::MissingArraySharding)?;
        let (descriptors, shard_index_by_device) = compute_shard_descriptors(shape.as_slice(), &mesh, sharding)?;

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
            if process_index != descriptor.device().process_index {
                return Err(ArrayError::BufferProcessIndexMismatch {
                    device_id,
                    expected_process_index: descriptor.device().process_index,
                    actual_process_index: process_index,
                });
            }

            let actual_element_type = DataType::from_pjrt_buffer_type(buffer.element_type()?)?;
            if actual_element_type != array_type.data_type {
                return Err(ArrayError::BufferElementTypeMismatch {
                    device_id,
                    expected: array_type.data_type,
                    actual: actual_element_type,
                });
            }

            let actual_shape = buffer
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
                .collect::<Result<Vec<_>, _>>()?;
            if actual_shape != descriptor.shape() {
                return Err(ArrayError::BufferShapeMismatch {
                    device_id,
                    shard_index,
                    expected_shape: descriptor.shape().to_vec(),
                    actual_shape,
                });
            }

            buffers_by_device.insert(device_id, buffer);
        }

        let mut addressable_shard_indices = Vec::with_capacity(buffers_by_device.len());
        let shards = descriptors
            .into_iter()
            .map(|descriptor| {
                let shard_index = descriptor.shard_index();
                let buffer = buffers_by_device.remove(&descriptor.device().id);
                if buffer.is_some() {
                    addressable_shard_indices.push(shard_index);
                }
                ArrayShard { descriptor, buffer }
            })
            .collect::<Vec<_>>();

        Ok(Self { array_type, shards, shard_index_by_device, addressable_shard_indices })
    }

    /// Creates an [`Array`] from shape/type/sharding metadata and local addressable buffers.
    pub fn from_sharding(
        global_shape: Vec<usize>,
        element_type: DataType,
        mesh: DeviceMesh,
        sharding: Sharding,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let shape = Shape::new(global_shape.iter().copied().map(Size::Static).collect());
        let array_type = ArrayType::new(element_type, shape, None, Some(sharding))?;
        Self::new(array_type, mesh, addressable_buffers)
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
        self.array_type.data_type
    }

    /// Returns the global array sharding.
    pub fn sharding(&self) -> &Sharding {
        self.array_type
            .sharding
            .as_ref()
            .expect("runtime arrays should only be constructed from array types with sharding")
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
    pub fn to_shardy_mesh_operation<'c, 't, L>(&self, location: L) -> DetachedMeshOperation<'c, 't>
    where
        't: 'c,
        L: Location<'c, 't>,
    {
        self.sharding().mesh.to_shardy(location)
    }

    /// Renders the Shardy tensor sharding attribute (`#sdy.sharding<...>`) implied by this array.
    ///
    /// Uses the canonical `@mesh` symbol name.
    pub fn to_shardy_tensor_sharding_attribute(&self) -> String {
        let context = ryft_mlir::Context::new();
        self.sharding().to_shardy(context.unknown_location()).to_string()
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

    fn into_addressable_buffers_by_device(self) -> HashMap<DeviceId, Buffer<'o>> {
        self.shards.into_iter().filter_map(ArrayShard::into_addressable_buffer).collect()
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

/// Prepared execution inputs for calling [`ryft_pjrt::LoadedExecutable::execute`].
///
/// This stores one `Vec<ExecutionInput>` per addressable device (in caller-provided device order).
pub struct ExecuteArguments<'o> {
    addressable_device_ids: Vec<DeviceId>,
    inputs_by_device: Vec<Vec<ExecutionInput<'o>>>,
}

impl<'o> ExecuteArguments<'o> {
    /// Returns addressable device IDs corresponding to [`Self::inputs_by_device`].
    pub fn addressable_device_ids(&self) -> &[DeviceId] {
        self.addressable_device_ids.as_slice()
    }

    /// Returns execution inputs grouped by device.
    pub fn inputs_by_device(&self) -> &[Vec<ExecutionInput<'o>>] {
        self.inputs_by_device.as_slice()
    }

    /// Creates PJRT `ExecutionDeviceInputs` in the same device order as [`Self::addressable_device_ids`].
    pub fn as_execution_device_inputs<'l>(&'l self) -> Vec<ExecutionDeviceInputs<'o, 'l>> {
        self.inputs_by_device.iter().map(|inputs| ExecutionDeviceInputs::from(inputs.as_slice())).collect()
    }

    fn from_arrays_with_donation(
        arrays: Vec<Array<'o>>,
        addressable_device_ids: &[DeviceId],
        donation_flags: &[bool],
    ) -> Result<Self, ArrayError> {
        if donation_flags.len() != arrays.len() {
            return Err(ArrayError::DonationFlagCountMismatch {
                expected_count: arrays.len(),
                actual_count: donation_flags.len(),
            });
        }

        let mut seen_devices = HashSet::with_capacity(addressable_device_ids.len());
        for &device_id in addressable_device_ids {
            if !seen_devices.insert(device_id) {
                return Err(ArrayError::DuplicateExecutionDeviceId { device_id });
            }
        }

        let mut buffers_by_array =
            arrays.into_iter().map(Array::into_addressable_buffers_by_device).collect::<Vec<_>>();

        let mut inputs_by_device = Vec::with_capacity(addressable_device_ids.len());
        for &device_id in addressable_device_ids {
            let mut device_inputs = Vec::with_capacity(buffers_by_array.len());
            for (array_index, array_buffers_by_device) in buffers_by_array.iter_mut().enumerate() {
                let buffer = array_buffers_by_device
                    .remove(&device_id)
                    .ok_or(ArrayError::MissingArrayShardForDevice { array_index, device_id })?;
                device_inputs.push(ExecutionInput { buffer, donatable: donation_flags[array_index] });
            }
            inputs_by_device.push(device_inputs);
        }

        for (array_index, array_buffers_by_device) in buffers_by_array.iter().enumerate() {
            if let Some(device_id) = array_buffers_by_device.keys().next().copied() {
                return Err(ArrayError::UnexpectedArrayShardDevice { array_index, device_id });
            }
        }

        Ok(Self { addressable_device_ids: addressable_device_ids.to_vec(), inputs_by_device })
    }
}

impl std::fmt::Debug for ExecuteArguments<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_counts = self.inputs_by_device.iter().map(Vec::len).collect::<Vec<_>>();
        formatter
            .debug_struct("ExecuteArguments")
            .field("addressable_device_ids", &self.addressable_device_ids)
            .field("input_counts_per_device", &input_counts)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

    use crate::sharding::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, Sharding, ShardingDimension};
    use crate::types::data_types::DataType;
    use crate::types::{ArrayType, Shape, Size};

    use super::*;

    fn test_spmd_compilation_options(partition_count: usize) -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: partition_count as i64,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn two_f32s_from_bytes(bytes: &[u8]) -> [f32; 2] {
        assert_eq!(bytes.len(), 2 * size_of::<f32>());
        let first = f32::from_ne_bytes(bytes[..size_of::<f32>()].try_into().unwrap());
        let second = f32::from_ne_bytes(bytes[size_of::<f32>()..].try_into().unwrap());
        [first, second]
    }

    #[test]
    fn test_array_new_requires_sharding() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(0, 1)],
        )
        .unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        assert!(matches!(Array::new(array_type, mesh, Vec::new()), Err(ArrayError::MissingArraySharding),));
    }

    #[test]
    fn test_array_shape_returns_static_shape() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(0, 1)],
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh.clone(), 1);
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(7)]), None, Some(sharding)).unwrap();

        let array = Array::new(array_type.clone(), mesh.clone(), Vec::new()).unwrap();

        assert_eq!(array.array_type(), &array_type);
        assert_eq!(array.shape(), vec![7]);
        assert_eq!(array.shards().len(), 1);
        assert_eq!(array.addressable_shards().count(), 0);
        assert_eq!(array.shards()[0].shape(), &[7]);
        assert!(!array.shards()[0].is_addressable());
    }

    #[test]
    fn test_array_new_rejects_dynamic_shape() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(0, 1)],
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh.clone(), 1);
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(Some(10))]), None, Some(sharding)).unwrap();

        assert!(matches!(
            Array::new(array_type, mesh, Vec::new()),
            Err(ArrayError::DynamicArrayShape { dimension: 0, size: Size::Dynamic(Some(10)) }),
        ));
    }

    #[test]
    fn test_array_driven_shardy_jit_sharded_matmul_on_cpu() {
        // Use the same 8-device CPU setup as `ryft_pjrt` tests.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8) }))
            .expect("failed to create 8-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        // Build mesh used for runtime arrays. In a JIT setting, we derive StableHLO Shardy
        // annotations directly from these arrays.
        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Auto).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();

        let lhs_sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap();
        let rhs_sharding = Sharding::replicated(mesh.logical_mesh.clone(), 2);

        // Global lhs matrix is 8x4, split by rows across 8 devices (each shard is 1x4).
        // Row i is [i, i+1, i+2, i+3].
        let lhs_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(row_index, device)| {
                let row = row_index as f32;
                client
                    .buffer(
                        f32_values_to_bytes(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
                        BufferType::F32,
                        [1u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // Global rhs matrix is replicated on each device.
        // [[1, 2], [0, 1], [1, 0], [2, 1]]
        let rhs_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let rhs_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        f32_values_to_bytes(rhs_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let lhs_array_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8), Size::Static(4)]),
            None,
            Some(lhs_sharding.clone()),
        )
        .unwrap();
        let rhs_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]), None, Some(rhs_sharding))
                .unwrap();
        let lhs_array = Array::new(lhs_array_type, mesh.clone(), lhs_buffers).unwrap();
        let rhs_array = Array::new(rhs_array_type, mesh.clone(), rhs_buffers).unwrap();

        assert_eq!(lhs_array.element_type(), DataType::F32);
        assert_eq!(rhs_array.element_type(), DataType::F32);
        assert_eq!(lhs_array.addressable_shards().count(), 8);
        assert!(lhs_array.shards().iter().all(|shard| shard.is_addressable()));

        // Derive Shardy attributes from runtime arrays (JIT-style).
        let context = ryft_mlir::Context::new();
        let mesh_module = context.module(context.unknown_location());
        let mesh_operation = mesh_module
            .body()
            .append_operation(lhs_array.to_shardy_mesh_operation(context.unknown_location()))
            .to_string();
        let lhs_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute();
        let rhs_sharding_attribute = rhs_array.to_shardy_tensor_sharding_attribute();
        let output_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute();

        assert_eq!(mesh_operation, "sdy.mesh @mesh = <[\"x\"=8]>");
        assert_eq!(lhs_sharding_attribute, "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
        assert_eq!(rhs_sharding_attribute, "#sdy.sharding<@mesh, [{}, {}]>");

        let mlir_program = format!(
            r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8x4xf32> {{sdy.sharding = {lhs_sharding_attribute}}},
                        %arg1: tensor<4x2xf32> {{sdy.sharding = {rhs_sharding_attribute}}}
                    ) -> (tensor<8x2xf32> {{sdy.sharding = {output_sharding_attribute}}}) {{
                        %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0]
                            : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                        return %0 : tensor<8x2xf32>
                    }}
                }}
            "#
        );
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = lhs_array.shard_for_device(*device_id).unwrap().slices()[0].start;
                (*device_id, row_start)
            })
            .collect::<HashMap<_, _>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        // Validate each output shard: row r should be [4r + 8, 4r + 4].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values = two_f32s_from_bytes(output_bytes.as_slice());
            let row = *row_start_by_device.get(&device_id).unwrap() as f32;
            assert_eq!(values[0], 4.0 * row + 8.0);
            assert_eq!(values[1], 4.0 * row + 4.0);
        }
    }
}
