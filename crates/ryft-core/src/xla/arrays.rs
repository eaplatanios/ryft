//! Runtime sharded-array data structures for XLA execution.
//!
//! This module builds on the sharding metadata from [`super::sharding`] to provide runtime
//! array types that pair global sharding metadata with local PJRT buffers:
//!
//! - [`Array`] corresponds to `jax.Array` / IFRT `Array`: global array metadata plus local
//!   addressable device buffers.
//! - [`AddressableShard`] corresponds to one entry in JAX's `array.addressable_shards`.
//! - [`ExecuteArguments`] marshals distributed arrays into per-device execution inputs for PJRT.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use ryft_pjrt::{Buffer, BufferType, DeviceId, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput};

use super::sharding::{Mesh, PartitionSpec, ShardDescriptor, ShardingContext, ShardingError, ShardingLayout};

/// Error type for [`Array`] construction and execution-input preparation.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ArrayError {
    /// Underlying error returned by PJRT.
    #[error("{0}")]
    PjrtError(#[from] PjrtError),

    /// Underlying sharding/layout error.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when an addressable buffer is placed on a device not present in the array mesh.
    #[error("addressable buffer is placed on device {device_id}, but that device is not in the mesh")]
    AddressableBufferDeviceNotInMesh { device_id: DeviceId },

    /// Error returned when more than one addressable buffer is provided for the same device.
    #[error("got multiple addressable buffers for device {device_id}")]
    DuplicateAddressableBufferDevice { device_id: DeviceId },

    /// Error returned when a buffer element type does not match the array element type.
    #[error("buffer on device {device_id} has element type {actual}, but array element type is {expected}")]
    BufferElementTypeMismatch { device_id: DeviceId, expected: BufferType, actual: BufferType },

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

/// Addressable shard on the current host.
///
/// Each entry ties one local [`Buffer`] to one global shard index.
/// This corresponds to one entry in JAX's `array.addressable_shards`.
pub struct AddressableShard<'o> {
    shard_index: usize,
    device_id: DeviceId,
    process_index: usize,
    buffer: Buffer<'o>,
}

impl<'o> AddressableShard<'o> {
    /// Global shard index for this buffer.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Device ID on which this buffer is placed.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Process index owning the device on which this buffer is placed.
    pub fn process_index(&self) -> usize {
        self.process_index
    }

    /// Addressable shard buffer.
    pub fn buffer(&self) -> &Buffer<'o> {
        &self.buffer
    }
}

impl std::fmt::Debug for AddressableShard<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AddressableShard")
            .field("shard_index", &self.shard_index)
            .field("device_id", &self.device_id)
            .field("process_index", &self.process_index)
            .finish()
    }
}

/// Distributed array backed by local addressable PJRT buffers and global sharding metadata.
///
/// This is conceptually aligned with JAX/IFRT arrays:
/// - `layout` describes all global shards across the full mesh.
/// - `addressable_shards` contains only shards local to the current host process.
/// - each addressable buffer is mapped to its global shard index.
///
/// In JAX terminology, this is the runtime pairing of:
/// - sharding metadata (mesh + partition spec), and
/// - addressable device buffers (the local portion of an IFRT array).
pub struct Array<'o> {
    layout: ShardingLayout,
    element_type: BufferType,
    addressable_shards: Vec<AddressableShard<'o>>,
    addressable_shard_index_by_device: HashMap<DeviceId, usize>,
}

impl<'o> Array<'o> {
    /// Creates an [`Array`] from precomputed sharding metadata and local addressable buffers.
    ///
    /// Each buffer is mapped to a shard using its device ID. Buffer shape and element type are validated against
    /// shard metadata.
    pub fn new(
        layout: ShardingLayout,
        element_type: BufferType,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let mut seen_devices = HashSet::with_capacity(addressable_buffers.len());
        let mut addressable_shards = Vec::with_capacity(addressable_buffers.len());

        for buffer in addressable_buffers {
            let device = buffer.device()?;
            let device_id = device.id()?;
            if !seen_devices.insert(device_id) {
                return Err(ArrayError::DuplicateAddressableBufferDevice { device_id });
            }

            let shard_index = layout
                .shard_index_for_device(device_id)
                .ok_or(ArrayError::AddressableBufferDeviceNotInMesh { device_id })?;
            let shard = layout
                .shard(shard_index)
                .expect("layout shard index should exist for valid layout device-to-shard mapping");

            let process_index = device.process_index()?;
            if process_index != shard.device().process_index() {
                return Err(ArrayError::BufferProcessIndexMismatch {
                    device_id,
                    expected_process_index: shard.device().process_index(),
                    actual_process_index: process_index,
                });
            }

            let actual_element_type = buffer.element_type()?;
            if actual_element_type != element_type {
                return Err(ArrayError::BufferElementTypeMismatch {
                    device_id,
                    expected: element_type,
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
            if actual_shape != shard.shape() {
                return Err(ArrayError::BufferShapeMismatch {
                    device_id,
                    shard_index,
                    expected_shape: shard.shape().to_vec(),
                    actual_shape,
                });
            }

            addressable_shards.push(AddressableShard { shard_index, device_id, process_index, buffer });
        }

        addressable_shards.sort_by_key(AddressableShard::shard_index);
        let addressable_shard_index_by_device = addressable_shards
            .iter()
            .enumerate()
            .map(|(addressable_shard_index, shard)| (shard.device_id(), addressable_shard_index))
            .collect::<HashMap<_, _>>();

        Ok(Self { layout, element_type, addressable_shards, addressable_shard_index_by_device })
    }

    /// Creates an [`Array`] from shape/type/sharding metadata and local addressable buffers.
    pub fn from_sharding(
        global_shape: Vec<usize>,
        element_type: BufferType,
        mesh: Mesh,
        partition_spec: PartitionSpec,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let layout = ShardingLayout::new(global_shape, mesh, partition_spec)?;
        Self::new(layout, element_type, addressable_buffers)
    }

    /// Returns global sharding layout metadata.
    pub fn layout(&self) -> &ShardingLayout {
        &self.layout
    }

    /// Returns the global array shape.
    pub fn global_shape(&self) -> &[usize] {
        self.layout.global_shape()
    }

    /// Returns the global array element type.
    pub fn element_type(&self) -> BufferType {
        self.element_type
    }

    /// Returns metadata for all global shards.
    pub fn shards(&self) -> &[ShardDescriptor] {
        self.layout.shards()
    }

    /// Returns addressable local shards.
    pub fn addressable_shards(&self) -> &[AddressableShard<'o>] {
        self.addressable_shards.as_slice()
    }

    /// Returns the addressable shard for `device_id`, if local.
    pub fn addressable_shard_for_device(&self, device_id: DeviceId) -> Option<&AddressableShard<'o>> {
        self.addressable_shard_index_by_device
            .get(&device_id)
            .and_then(|index| self.addressable_shards.get(*index))
    }

    /// Returns global shard metadata for `device_id`, if it exists in the mesh.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ShardDescriptor> {
        self.layout.shard_for_device(device_id)
    }

    /// Returns global shard metadata for a local addressable shard index.
    pub fn shard_for_addressable_index(&self, addressable_shard_index: usize) -> Option<&ShardDescriptor> {
        self.addressable_shards
            .get(addressable_shard_index)
            .and_then(|addressable_shard| self.layout.shard(addressable_shard.shard_index()))
    }

    /// Renders the Shardy mesh declaration (`sdy.mesh`) implied by this array's sharding.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used in MLIR (without or with leading `'@'`).
    pub fn to_shardy_mesh_operation<S: AsRef<str>>(&self, mesh_symbol_name: S) -> Result<String, ShardingError> {
        self.layout.mesh().abstract_mesh().to_shardy_mesh_operation(mesh_symbol_name)
    }

    /// Renders the Shardy tensor sharding attribute (`#sdy.sharding<...>`) implied by this array.
    ///
    /// Uses [`ShardingContext::ExplicitSharding`] because runtime arrays have fully determined shardings.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the corresponding `sdy.mesh` op (without or with leading `'@'`).
    pub fn to_shardy_tensor_sharding_attribute<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
    ) -> Result<String, ShardingError> {
        let mesh_symbol_name_str = {
            let s = mesh_symbol_name.as_ref().trim();
            let s = s.strip_prefix('@').unwrap_or(s);
            s.to_string()
        };
        let dim_shardings = self
            .layout
            .partition_spec()
            .to_shardy_dimension_shardings_literal(ShardingContext::ExplicitSharding);
        if mesh_symbol_name_str.is_empty() || mesh_symbol_name_str.chars().any(char::is_whitespace) {
            return Err(if mesh_symbol_name_str.is_empty() {
                ShardingError::EmptyMeshSymbolName
            } else {
                ShardingError::InvalidMeshSymbolName { mesh_symbol_name: mesh_symbol_name_str }
            });
        }
        Ok(format!("#sdy.sharding<@{mesh_symbol_name_str}, {dim_shardings}>"))
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
        self.addressable_shards
            .into_iter()
            .map(|addressable_shard| (addressable_shard.device_id(), addressable_shard.buffer))
            .collect()
    }
}

impl std::fmt::Debug for Array<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Array")
            .field("global_shape", &self.global_shape())
            .field("element_type", &self.element_type())
            .field("global_shard_count", &self.shards().len())
            .field("addressable_shard_count", &self.addressable_shards.len())
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

    use crate::xla::sharding::{Mesh, MeshAxis, MeshDevice, PartitionDimension, PartitionSpec};

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
        let mesh = Mesh::new(vec![MeshAxis::new("x", 8).unwrap()], mesh_devices).unwrap();

        let lhs_partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let rhs_partition_spec = PartitionSpec::replicated(2);

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

        let lhs_array =
            Array::from_sharding(vec![8, 4], BufferType::F32, mesh.clone(), lhs_partition_spec.clone(), lhs_buffers)
                .unwrap();
        let rhs_array =
            Array::from_sharding(vec![4, 2], BufferType::F32, mesh.clone(), rhs_partition_spec, rhs_buffers).unwrap();

        // Derive Shardy attributes from runtime arrays (JIT-style).
        let mesh_operation = lhs_array.to_shardy_mesh_operation("mesh").unwrap();
        let lhs_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute("mesh").unwrap();
        let rhs_sharding_attribute = rhs_array.to_shardy_tensor_sharding_attribute("mesh").unwrap();
        let output_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute("mesh").unwrap();

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
                let row_start = lhs_array.shard_for_device(*device_id).unwrap().slices()[0].start();
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
