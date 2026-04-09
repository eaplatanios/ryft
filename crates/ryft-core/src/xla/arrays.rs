//! Runtime sharded-array data structures for XLA execution.
//!
//! This module builds on the sharding metadata from [`super::sharding`] to provide runtime
//! array types that pair global [`crate::types::ArrayType`] metadata, global shard placement
//! metadata, and local PJRT buffers:
//!
//! - [`device_put`] is the higher-level `ryft` analogue of JAX's
//!   [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html) over
//!   supported host leaves, [`Array`] leaves, and `Parameterized` trees of those leaves.
//! - [`Array::from_host_buffer`] is the lower-level dense-host-buffer constructor used by
//!   [`device_put`] and by tests that need explicit byte-level control.
//! - [`Array`] corresponds to `jax.Array` / IFRT `Array`: global type and shard-placement
//!   metadata plus local addressable device buffers.
//! - [`ArrayShard`] corresponds to one entry in JAX's `array.global_shards`, with
//!   [`ArrayShard::buffer`] identifying the addressable local subset.
//! - [`ExecuteArguments`] marshals distributed arrays into per-device execution inputs for PJRT.

use std::collections::{HashMap, HashSet};

use half::{bf16, f16};
use thiserror::Error;

#[cfg(test)]
use ryft_mlir::Block;
use ryft_mlir::{Location, dialects::shardy::DetachedMeshOperation};
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{Buffer, Client, DeviceId, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput};

use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::sharding::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, Sharding, ShardingError};
use crate::types::data_types::{DataType, DataTypeError};
use crate::types::{ArrayType, Shape, Size};

use super::sharding::{Shard, ShardSlice, compute_shard_descriptors};

/// Concrete mesh/sharding target used by the higher-level [`device_put`] API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePutSharding {
    /// Concrete destination mesh describing the device topology.
    pub mesh: DeviceMesh,

    /// Sharding to apply over [`Self::mesh`].
    pub sharding: Sharding,
}

impl DevicePutSharding {
    /// Creates a new [`DevicePutSharding`].
    #[inline]
    pub fn new(mesh: DeviceMesh, sharding: Sharding) -> Self {
        Self { mesh, sharding }
    }
}

impl Parameter for DevicePutSharding {}

/// Placement leaf accepted by the higher-level [`device_put`] API.
///
/// This models the current `ryft` subset of JAX's `device` / `src` arguments:
/// - [`Self::Device`] commits one leaf to a single concrete device, represented internally as a
///   size-1 mesh with fully replicated sharding, and
/// - [`Self::Sharding`] commits one leaf to an explicit mesh/sharding pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DevicePutPlacement {
    /// Commit the value to one concrete device.
    Device(MeshDevice),

    /// Commit the value to the provided mesh/sharding pair.
    Sharding(DevicePutSharding),
}

impl DevicePutPlacement {
    /// Creates a single-device placement.
    #[inline]
    pub fn device(device: MeshDevice) -> Self {
        Self::Device(device)
    }

    /// Creates an explicit mesh/sharding placement.
    #[inline]
    pub fn sharding(mesh: DeviceMesh, sharding: Sharding) -> Self {
        Self::Sharding(DevicePutSharding::new(mesh, sharding))
    }
}

impl From<MeshDevice> for DevicePutPlacement {
    fn from(value: MeshDevice) -> Self {
        Self::Device(value)
    }
}

impl From<DevicePutSharding> for DevicePutPlacement {
    fn from(value: DevicePutSharding) -> Self {
        Self::Sharding(value)
    }
}

impl Parameter for DevicePutPlacement {}

/// Options for the higher-level [`device_put`] API.
///
/// Each field follows JAX's tree-prefix semantics: when a field is present, its structure is
/// broadcast over the input tree and applied leafwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePutOptions<
    Device = DevicePutPlacement,
    Src = DevicePutPlacement,
    Donate = bool,
    MayAlias = Option<bool>,
> {
    /// Destination placement tree prefix. When absent, host leaves are committed to the default
    /// local device and [`Array`] leaves preserve their current placement.
    pub device: Option<Device>,

    /// Source placement tree prefix. This is validated for [`Array`] leaves and ignored for host
    /// leaves, which do not carry runtime placement metadata before upload.
    pub src: Option<Src>,

    /// Donation tree prefix. This is best-effort in the current `ryft` runtime.
    pub donate: Option<Donate>,

    /// May-alias tree prefix. `Some(false)` forces a fresh array result when possible.
    pub may_alias: Option<MayAlias>,
}

impl<Device, Src, Donate, MayAlias> DevicePutOptions<Device, Src, Donate, MayAlias> {
    /// Creates a new [`DevicePutOptions`] with all fields unset.
    #[inline]
    pub fn new() -> Self {
        Self { device: None, src: None, donate: None, may_alias: None }
    }
}

impl<Device, Src, Donate, MayAlias> Default for DevicePutOptions<Device, Src, Donate, MayAlias> {
    fn default() -> Self {
        Self::new()
    }
}

impl DevicePutOptions<DevicePutPlacement, DevicePutPlacement, bool, Option<bool>> {
    /// Creates default high-level [`device_put`] options without requiring generic type inference.
    #[inline]
    pub fn defaults() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedDevicePutPlacement {
    mesh: DeviceMesh,
    sharding: Sharding,
}

impl ResolvedDevicePutPlacement {
    fn new(mesh: DeviceMesh, sharding: Sharding) -> Self {
        Self { mesh, sharding }
    }

    fn into_public(self) -> DevicePutSharding {
        DevicePutSharding::new(self.mesh, self.sharding)
    }
}

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

    /// Underlying parameter-tree broadcasting error.
    #[error("{0}")]
    ParameterError(#[from] ParameterError),

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

    /// Error returned when `device_put` receives a host buffer whose dense size does not match the logical array.
    #[error("device_put expected {expected_byte_count} host byte(s), but got {actual_byte_count}")]
    HostDataLengthMismatch { expected_byte_count: usize, actual_byte_count: usize },

    /// Error returned when `device_put` is asked to shard an element type without a supported dense host encoding.
    #[error("device_put does not support dense host bytes for element type {element_type}")]
    UnsupportedDevicePutElementType { element_type: DataType },

    /// Error returned when `device_put` cannot represent the dense host size of the requested array.
    #[error("array with shape {shape:?} and element type {element_type} is too large for device_put")]
    DevicePutArrayTooLarge { shape: Vec<usize>, element_type: DataType },

    /// Error returned when a mesh device local to the current process is not addressable by the PJRT client.
    #[error("mesh device {device_id} is local to process {process_index}, but the PJRT client cannot address it")]
    MissingClientDeviceForLocalMeshDevice { device_id: DeviceId, process_index: usize },

    /// Error returned when the higher-level [`device_put`] API needs a default device but the client has no local devices.
    #[error("device_put needs a default local device, but the PJRT client has no addressable devices")]
    MissingDefaultDevice,

    /// Error returned when a provided `src` placement does not match an array leaf's current placement.
    #[error("device_put src placement {expected:?} does not match the array's current placement {actual:?}")]
    SourcePlacementMismatch { expected: DevicePutSharding, actual: DevicePutSharding },

    /// Error returned when a device ID cannot be represented by the PJRT cross-host transfers extension.
    #[error("device {device_id} cannot be represented as a PJRT global device ID for cross-host transfers")]
    CrossHostTransferDeviceIdTooLarge { device_id: DeviceId },

    /// Error returned when a shard dimension cannot be represented by the PJRT cross-host transfers extension.
    #[error(
        "cross-host transfer for shard #{shard_index} has shape dimension #{dimension}={size}, which does not fit in i64"
    )]
    CrossHostTransferShapeDimensionTooLarge { shard_index: usize, dimension: usize, size: usize },

    /// Error returned when an exact-shard cross-host transfer key cannot be represented in PJRT.
    #[error(
        "exact-shard transfer key for source shard #{source_shard_index} and destination shard #{destination_shard_index} \
         does not fit in i64"
    )]
    CrossHostTransferKeyTooLarge { source_shard_index: usize, destination_shard_index: usize },

    /// Error returned when [`Array::put`] needs a source shard that is not addressable locally.
    #[error(
        "array move requires shard #{shard_index} on device {device_id} to be addressable from the current process"
    )]
    MissingAddressableShardForMove { shard_index: usize, device_id: DeviceId },

    /// Error returned when copying a source shard to host yields an unexpected byte count.
    #[error(
        "copied shard #{shard_index} from device {device_id} to host and got {actual_byte_count} byte(s), but expected {expected_byte_count}"
    )]
    CopiedShardByteCountMismatch {
        shard_index: usize,
        device_id: DeviceId,
        expected_byte_count: usize,
        actual_byte_count: usize,
    },

    /// Error returned when overlapping source shards disagree while materializing a dense host array.
    #[error("array move found inconsistent overlapping data while materializing shard #{shard_index}")]
    InconsistentOverlappingShardData { shard_index: usize },

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

/// Returns the dense host-storage size in bytes for one `element_type` value.
///
/// [`Array::from_host_buffer`] accepts raw dense host bytes rather than a typed host container. It
/// therefore supports only element types whose host representation is byte-addressable and whose
/// packing is unambiguous.
fn device_put_element_size_in_bytes(element_type: DataType) -> Result<usize, ArrayError> {
    match element_type {
        DataType::Boolean
        | DataType::I8
        | DataType::U8
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU => Ok(1),
        DataType::I16 | DataType::U16 | DataType::BF16 | DataType::F16 => Ok(2),
        DataType::I32 | DataType::U32 | DataType::F32 => Ok(4),
        DataType::I64 | DataType::U64 | DataType::F64 | DataType::C64 => Ok(8),
        DataType::C128 => Ok(16),
        DataType::Token
        | DataType::I1
        | DataType::I2
        | DataType::I4
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::F4E2M1FN => Err(ArrayError::UnsupportedDevicePutElementType { element_type }),
    }
}

/// Returns the product of `dimensions`, rejecting shapes whose element count does not fit in `usize`.
fn checked_element_count(dimensions: &[usize], element_type: DataType) -> Result<usize, ArrayError> {
    dimensions.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| ArrayError::DevicePutArrayTooLarge { shape: dimensions.to_vec(), element_type })
    })
}

/// Returns the dense host byte count for a row-major array with `global_shape` and `element_type`.
fn checked_byte_count(global_shape: &[usize], element_type: DataType) -> Result<usize, ArrayError> {
    let element_count = checked_element_count(global_shape, element_type)?;
    let element_size_in_bytes = device_put_element_size_in_bytes(element_type)?;
    element_count
        .checked_mul(element_size_in_bytes)
        .ok_or_else(|| ArrayError::DevicePutArrayTooLarge { shape: global_shape.to_vec(), element_type })
}

fn single_device_put_placement(device: MeshDevice, rank: usize) -> Result<ResolvedDevicePutPlacement, ArrayError> {
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
    let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
    let sharding = Sharding::replicated(mesh.logical_mesh.clone(), rank);
    Ok(ResolvedDevicePutPlacement::new(mesh, sharding))
}

fn default_device_put_placement(client: &Client<'_>, rank: usize) -> Result<ResolvedDevicePutPlacement, ArrayError> {
    let device = client.addressable_devices()?.into_iter().next().ok_or(ArrayError::MissingDefaultDevice)?;
    single_device_put_placement(MeshDevice::new(device.id()?, device.process_index()?), rank)
}

fn resolve_device_put_placement(
    placement: DevicePutPlacement,
    rank: usize,
) -> Result<ResolvedDevicePutPlacement, ArrayError> {
    match placement {
        DevicePutPlacement::Device(device) => single_device_put_placement(device, rank),
        DevicePutPlacement::Sharding(sharding) => Ok(ResolvedDevicePutPlacement::new(sharding.mesh, sharding.sharding)),
    }
}

/// Returns row-major element strides for `global_shape`.
///
/// The returned vector has the same rank as `global_shape`, with `strides[i]` giving the number of
/// logical elements skipped when incrementing dimension `i` by one in a dense major-to-minor
/// layout.
fn row_major_element_strides(global_shape: &[usize], element_type: DataType) -> Result<Vec<usize>, ArrayError> {
    let mut strides = vec![1usize; global_shape.len()];
    let mut stride = 1usize;
    for dimension in (0..global_shape.len()).rev() {
        strides[dimension] = stride;
        stride = stride
            .checked_mul(global_shape[dimension])
            .ok_or_else(|| ArrayError::DevicePutArrayTooLarge { shape: global_shape.to_vec(), element_type })?;
    }
    Ok(strides)
}

/// Extracts the dense row-major bytes corresponding to `shard_slices` from `host_data`.
fn extract_dense_shard_bytes(
    host_data: &[u8],
    global_shape: &[usize],
    shard_slices: &[ShardSlice],
    element_type: DataType,
) -> Result<Vec<u8>, ArrayError> {
    if shard_slices.is_empty() {
        return Ok(host_data.to_vec());
    }

    let strides = row_major_element_strides(global_shape, element_type)?;
    let shard_shape = shard_slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
    let shard_byte_count = checked_byte_count(shard_shape.as_slice(), element_type)?;
    let element_size_in_bytes = device_put_element_size_in_bytes(element_type)?;
    let mut shard_bytes = Vec::with_capacity(shard_byte_count);
    append_dense_shard_bytes(
        host_data,
        shard_slices,
        strides.as_slice(),
        0,
        0,
        element_size_in_bytes,
        &mut shard_bytes,
    );
    Ok(shard_bytes)
}

/// Appends the row-major bytes for the shard slice at `dimension` to `shard_bytes`.
fn append_dense_shard_bytes(
    host_data: &[u8],
    shard_slices: &[ShardSlice],
    strides: &[usize],
    dimension: usize,
    base_element_offset: usize,
    element_size_in_bytes: usize,
    shard_bytes: &mut Vec<u8>,
) {
    let slice = &shard_slices[dimension];
    if dimension + 1 == shard_slices.len() {
        let start_element_offset =
            base_element_offset + slice.start.checked_mul(strides[dimension]).expect("validated shard offsets fit");
        let end_element_offset =
            base_element_offset + slice.end.checked_mul(strides[dimension]).expect("validated shard offsets fit");
        let start_byte_offset =
            start_element_offset.checked_mul(element_size_in_bytes).expect("validated shard byte offsets fit");
        let end_byte_offset =
            end_element_offset.checked_mul(element_size_in_bytes).expect("validated shard byte offsets fit");
        shard_bytes.extend_from_slice(&host_data[start_byte_offset..end_byte_offset]);
        return;
    }

    for index in slice.start..slice.end {
        let element_offset =
            base_element_offset + index.checked_mul(strides[dimension]).expect("validated shard offsets fit");
        append_dense_shard_bytes(
            host_data,
            shard_slices,
            strides,
            dimension + 1,
            element_offset,
            element_size_in_bytes,
            shard_bytes,
        );
    }
}

/// Deterministic exact-shard transfer plan for one [`Array::put`] call on the current process.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactShardPutPlan {
    /// Local destination shards that can be satisfied via intra-host device-to-device copies.
    local_copies: Vec<LocalShardCopyPlan>,

    /// Cross-host sends that this process must initiate for remote destination shards.
    cross_host_sends: Vec<CrossHostShardSendPlan>,

    /// Cross-host receives that this process must initiate for local destination shards.
    cross_host_receives: Vec<CrossHostShardReceivePlan>,
}

/// One exact-shard local device-to-device copy in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalShardCopyPlan {
    /// Source shard index in the source array.
    source_shard_index: usize,

    /// Source device ID.
    source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    destination_shard_index: usize,

    /// Destination device ID.
    destination_device_id: DeviceId,
}

/// One exact-shard cross-host send in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct CrossHostShardSendPlan {
    /// Source shard index in the source array.
    source_shard_index: usize,

    /// Source device ID.
    source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    destination_shard_index: usize,

    /// Destination device ID.
    destination_device_id: DeviceId,

    /// Deterministic transfer rendezvous key shared with the matching receive.
    transfer_key: CrossHostTransferKey,
}

/// One exact-shard cross-host receive in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct CrossHostShardReceivePlan {
    /// Source shard index in the source array.
    source_shard_index: usize,

    /// Source device ID.
    source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    destination_shard_index: usize,

    /// Destination device ID.
    destination_device_id: DeviceId,

    /// Logical destination shard shape.
    destination_shape: Vec<usize>,

    /// Deterministic transfer rendezvous key shared with the matching send.
    transfer_key: CrossHostTransferKey,
}

/// Returns the deterministic exact-shard cross-host transfer key for one source/destination pair.
fn exact_shard_transfer_key(
    source_shard_index: usize,
    destination_shard_index: usize,
    destination_shard_count: usize,
) -> Result<CrossHostTransferKey, ArrayError> {
    let transfer_key = source_shard_index
        .checked_mul(destination_shard_count)
        .and_then(|key| key.checked_add(destination_shard_index))
        .ok_or(ArrayError::CrossHostTransferKeyTooLarge { source_shard_index, destination_shard_index })?;
    i64::try_from(transfer_key)
        .map_err(|_| ArrayError::CrossHostTransferKeyTooLarge { source_shard_index, destination_shard_index })
}

/// Returns the PJRT cross-host global device ID for `device_id`.
fn cross_host_global_device_id(device_id: DeviceId) -> Result<GlobalDeviceId, ArrayError> {
    i32::try_from(device_id).map_err(|_| ArrayError::CrossHostTransferDeviceIdTooLarge { device_id })
}

/// Returns the PJRT cross-host shape for one destination shard.
fn cross_host_shape(plan: &CrossHostShardReceivePlan) -> Result<Vec<i64>, ArrayError> {
    plan.destination_shape
        .iter()
        .enumerate()
        .map(|(dimension, &size)| {
            i64::try_from(size).map_err(|_| ArrayError::CrossHostTransferShapeDimensionTooLarge {
                shard_index: plan.destination_shard_index,
                dimension,
                size,
            })
        })
        .collect()
}

/// Returns the preferred exact matching source shard for `destination_shard`.
///
/// Preference is deterministic and depends only on global shard metadata so that every process
/// chooses the same source shard for a given destination shard:
/// 1. prefer a source shard on the destination process,
/// 2. then prefer the same device ID as the destination shard, and
/// 3. finally break ties by device ID and shard index.
fn preferred_exact_source_shard<'a, 'o>(
    source_shards: &[&'a ArrayShard<'o>],
    destination_shard: &Shard,
) -> &'a ArrayShard<'o> {
    let destination_device = destination_shard.device();
    source_shards
        .iter()
        .min_by_key(|source_shard| {
            (
                source_shard.process_index() != destination_device.process_index,
                source_shard.device_id() != destination_device.id,
                source_shard.device_id(),
                source_shard.shard_index(),
            )
        })
        .copied()
        .expect("preferred exact source shard selection requires at least one candidate")
}

/// Plans exact whole-shard moves for one [`Array::put`] call on the current process.
///
/// Returns `Ok(None)` when any destination shard requires repartitioning or concatenating multiple
/// source shards, which means the exact-shard fast path cannot satisfy the requested sharding.
fn plan_exact_shard_put<'o>(
    array: &Array<'o>,
    client_process_index: usize,
    global_shape: &[usize],
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<Option<ExactShardPutPlan>, ArrayError> {
    let mut source_shards_by_slices = HashMap::<Vec<ShardSlice>, Vec<&ArrayShard<'o>>>::new();
    for shard in array.shards() {
        source_shards_by_slices.entry(shard.slices().to_vec()).or_default().push(shard);
    }

    let (destination_shards, _) = compute_shard_descriptors(global_shape, mesh, sharding)?;
    let destination_shard_count = destination_shards.len();
    let mut plan =
        ExactShardPutPlan { local_copies: Vec::new(), cross_host_sends: Vec::new(), cross_host_receives: Vec::new() };
    for destination_shard in &destination_shards {
        let source_shards = match source_shards_by_slices.get(destination_shard.slices()) {
            Some(source_shards) => source_shards,
            None => return Ok(None),
        };
        let source_shard = preferred_exact_source_shard(source_shards.as_slice(), destination_shard);
        let source_process_index = source_shard.process_index();
        let destination_process_index = destination_shard.device().process_index;

        if destination_process_index == client_process_index {
            if source_process_index == client_process_index {
                if !source_shard.is_addressable() {
                    return Ok(None);
                }
                plan.local_copies.push(LocalShardCopyPlan {
                    source_shard_index: source_shard.shard_index(),
                    source_device_id: source_shard.device_id(),
                    destination_shard_index: destination_shard.shard_index(),
                    destination_device_id: destination_shard.device().id,
                });
            } else {
                plan.cross_host_receives.push(CrossHostShardReceivePlan {
                    source_shard_index: source_shard.shard_index(),
                    source_device_id: source_shard.device_id(),
                    destination_shard_index: destination_shard.shard_index(),
                    destination_device_id: destination_shard.device().id,
                    destination_shape: destination_shard.shape().to_vec(),
                    transfer_key: exact_shard_transfer_key(
                        source_shard.shard_index(),
                        destination_shard.shard_index(),
                        destination_shard_count,
                    )?,
                });
            }
        } else if source_process_index == client_process_index {
            if !source_shard.is_addressable() {
                return Ok(None);
            }
            plan.cross_host_sends.push(CrossHostShardSendPlan {
                source_shard_index: source_shard.shard_index(),
                source_device_id: source_shard.device_id(),
                destination_shard_index: destination_shard.shard_index(),
                destination_device_id: destination_shard.device().id,
                transfer_key: exact_shard_transfer_key(
                    source_shard.shard_index(),
                    destination_shard.shard_index(),
                    destination_shard_count,
                )?,
            });
        }
    }

    Ok(Some(plan))
}

/// Tries to build the destination local shard buffers via exact whole-shard transfers.
///
/// This fast path succeeds only when every destination shard addressable by `client` already
/// exists as one source shard with the exact same logical slices. Local source shards are copied
/// directly between devices, and remote source shards are transferred with the PJRT cross-host
/// transfers extension when it is available. When the destination requires repartitioning or the
/// cross-host extension is unavailable for a needed remote move, the function returns `Ok(None)`
/// so that [`Array::put`] can fall back to the dense host path.
fn copy_addressable_destination_shards_from_exact_source_shards<'o>(
    array: &Array<'o>,
    client: &'o Client<'_>,
    global_shape: &[usize],
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<Option<Vec<Buffer<'o>>>, ArrayError> {
    let client_process_index = client.process_index()?;
    let plan = match plan_exact_shard_put(array, client_process_index, global_shape, mesh, sharding)? {
        Some(plan) => plan,
        None => return Ok(None),
    };
    let needs_cross_host_transfers = !plan.cross_host_sends.is_empty() || !plan.cross_host_receives.is_empty();
    if needs_cross_host_transfers {
        match client.cross_host_transfers_extension() {
            Ok(_) => {}
            Err(PjrtError::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        }
    }

    let addressable_devices = client.addressable_devices()?;
    let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
    for device in addressable_devices {
        addressable_device_by_id.insert(device.id()?, device);
    }

    if !plan.cross_host_sends.is_empty() {
        let mut send_buffers = Vec::with_capacity(plan.cross_host_sends.len());
        let mut destination_devices = Vec::with_capacity(plan.cross_host_sends.len());
        let mut transfer_keys = Vec::with_capacity(plan.cross_host_sends.len());
        for send_plan in &plan.cross_host_sends {
            let source_buffer = array
                .addressable_shard_for_device(send_plan.source_device_id)
                .ok_or(ArrayError::MissingAddressableShardForMove {
                    shard_index: send_plan.source_shard_index,
                    device_id: send_plan.source_device_id,
                })?
                .buffer()
                .expect("addressable shard lookups should always return a local buffer");
            send_buffers.push(source_buffer);
            destination_devices.push(cross_host_global_device_id(send_plan.destination_device_id)?);
            transfer_keys.push(send_plan.transfer_key);
        }
        let _send_events = client.cross_host_send_buffers(
            send_buffers.as_slice(),
            destination_devices.as_slice(),
            transfer_keys.as_slice(),
        )?;
    }

    let mut addressable_buffers = Vec::new();
    for local_copy_plan in &plan.local_copies {
        let source_buffer = array
            .addressable_shard_for_device(local_copy_plan.source_device_id)
            .ok_or(ArrayError::MissingAddressableShardForMove {
                shard_index: local_copy_plan.source_shard_index,
                device_id: local_copy_plan.source_device_id,
            })?
            .buffer()
            .expect("addressable shard lookups should always return a local buffer");
        let destination_device = addressable_device_by_id.get(&local_copy_plan.destination_device_id).ok_or(
            ArrayError::MissingClientDeviceForLocalMeshDevice {
                device_id: local_copy_plan.destination_device_id,
                process_index: client_process_index,
            },
        )?;
        if local_copy_plan.source_device_id == local_copy_plan.destination_device_id {
            addressable_buffers.push(source_buffer.bitcast(ryft_pjrt::BufferSpecification {
                element_type: source_buffer.element_type()?,
                dimensions: source_buffer.dimensions()?,
                layout: None,
            })?);
        } else {
            addressable_buffers.push(source_buffer.copy_to_device(destination_device.clone())?);
        }
    }

    let mut receive_plans_by_device = HashMap::<DeviceId, Vec<&CrossHostShardReceivePlan>>::new();
    for receive_plan in &plan.cross_host_receives {
        receive_plans_by_device.entry(receive_plan.destination_device_id).or_default().push(receive_plan);
    }
    let mut receive_device_ids = receive_plans_by_device.keys().copied().collect::<Vec<_>>();
    receive_device_ids.sort_unstable();
    for receive_device_id in receive_device_ids {
        let receive_plans = receive_plans_by_device
            .get(&receive_device_id)
            .expect("grouped receive plans should exist for every grouped destination device");
        let destination_device = addressable_device_by_id.get(&receive_device_id).ok_or(
            ArrayError::MissingClientDeviceForLocalMeshDevice {
                device_id: receive_device_id,
                process_index: client_process_index,
            },
        )?;
        let element_types =
            receive_plans.iter().map(|_| array.element_type().to_pjrt_buffer_type()).collect::<Vec<_>>();
        let dimensions = receive_plans
            .iter()
            .map(|receive_plan| cross_host_shape(receive_plan))
            .collect::<Result<Vec<_>, _>>()?;
        let dimension_slices = dimensions.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let source_devices = receive_plans
            .iter()
            .map(|receive_plan| cross_host_global_device_id(receive_plan.source_device_id))
            .collect::<Result<Vec<_>, _>>()?;
        let transfer_keys = receive_plans.iter().map(|receive_plan| receive_plan.transfer_key).collect::<Vec<_>>();
        let received_buffers = client.cross_host_receive_buffers(
            element_types.as_slice(),
            dimension_slices.as_slice(),
            destination_device,
            source_devices.as_slice(),
            transfer_keys.as_slice(),
        )?;
        addressable_buffers.extend(received_buffers);
    }

    Ok(Some(addressable_buffers))
}

/// Materializes a fully addressable [`Array`] as dense row-major host bytes.
fn materialize_dense_array_bytes(array: &Array<'_>) -> Result<Vec<u8>, ArrayError> {
    let global_shape = array.shape();
    let element_type = array.element_type();
    let total_byte_count = checked_byte_count(global_shape.as_slice(), element_type)?;
    let mut global_bytes = vec![0u8; total_byte_count];
    let mut written = vec![false; total_byte_count];

    for shard in array.shards() {
        let buffer = shard.buffer().ok_or(ArrayError::MissingAddressableShardForMove {
            shard_index: shard.shard_index(),
            device_id: shard.device_id(),
        })?;
        let shard_bytes = buffer.copy_to_host(None)?.r#await()?;
        let expected_byte_count = checked_byte_count(shard.shape(), element_type)?;
        if shard_bytes.len() != expected_byte_count {
            return Err(ArrayError::CopiedShardByteCountMismatch {
                shard_index: shard.shard_index(),
                device_id: shard.device_id(),
                expected_byte_count,
                actual_byte_count: shard_bytes.len(),
            });
        }
        merge_dense_shard_bytes(
            shard_bytes.as_slice(),
            global_shape.as_slice(),
            shard.slices(),
            element_type,
            shard.shard_index(),
            &mut global_bytes,
            &mut written,
        )?;
    }

    Ok(global_bytes)
}

/// Merges one shard's dense row-major host bytes into `global_bytes`.
fn merge_dense_shard_bytes(
    shard_bytes: &[u8],
    global_shape: &[usize],
    shard_slices: &[ShardSlice],
    element_type: DataType,
    shard_index: usize,
    global_bytes: &mut [u8],
    written: &mut [bool],
) -> Result<(), ArrayError> {
    if shard_slices.is_empty() {
        return merge_dense_byte_segment(shard_bytes, 0, shard_index, global_bytes, written);
    }

    let global_strides = row_major_element_strides(global_shape, element_type)?;
    let shard_shape = shard_slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
    let shard_strides = row_major_element_strides(shard_shape.as_slice(), element_type)?;
    let element_size_in_bytes = device_put_element_size_in_bytes(element_type)?;
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
        written,
    )
}

/// Recursively merges one shard's bytes into `global_bytes`.
fn merge_dense_shard_bytes_recursive(
    shard_bytes: &[u8],
    shard_slices: &[ShardSlice],
    global_strides: &[usize],
    shard_strides: &[usize],
    dimension: usize,
    base_global_element_offset: usize,
    base_shard_element_offset: usize,
    element_size_in_bytes: usize,
    shard_index: usize,
    global_bytes: &mut [u8],
    written: &mut [bool],
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
            written,
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
            written,
        )?;
    }
    Ok(())
}

/// Merges `source_bytes` into `global_bytes` starting at `global_byte_offset`.
fn merge_dense_byte_segment(
    source_bytes: &[u8],
    global_byte_offset: usize,
    shard_index: usize,
    global_bytes: &mut [u8],
    written: &mut [bool],
) -> Result<(), ArrayError> {
    for (offset, &byte) in source_bytes.iter().enumerate() {
        let index = global_byte_offset + offset;
        if written[index] {
            if global_bytes[index] != byte {
                return Err(ArrayError::InconsistentOverlappingShardData { shard_index });
            }
        } else {
            global_bytes[index] = byte;
            written[index] = true;
        }
    }
    Ok(())
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

impl Parameter for Array<'_> {}

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

    /// Creates an [`Array`] by uploading one dense row-major host buffer to the local shards
    /// implied by `mesh` and `sharding`.
    ///
    /// This is the low-level host-buffer constructor used by the higher-level [`device_put`]
    /// surface. The constructor derives the per-device shard slices from the provided
    /// [`Sharding`], uploads only the shards addressable by `client`, and returns an [`Array`]
    /// whose global shard metadata covers the full mesh.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to upload the local addressable shard buffers.
    ///   - `buffer`: Dense row-major host bytes for the full logical array.
    ///   - `global_shape`: Global logical array shape.
    ///   - `element_type`: Element type stored in `buffer`.
    ///   - `mesh`: Concrete device mesh describing the global device topology.
    ///   - `sharding`: Sharding to apply to the global logical array over `mesh`.
    pub fn from_host_buffer<B: AsRef<[u8]>, D: AsRef<[usize]>>(
        client: &'o Client<'_>,
        buffer: B,
        global_shape: D,
        element_type: DataType,
        mesh: DeviceMesh,
        sharding: Sharding,
    ) -> Result<Self, ArrayError> {
        let buffer = buffer.as_ref();
        let global_shape = global_shape.as_ref();
        let expected_byte_count = checked_byte_count(global_shape, element_type)?;
        if buffer.len() != expected_byte_count {
            return Err(ArrayError::HostDataLengthMismatch { expected_byte_count, actual_byte_count: buffer.len() });
        }

        let client_process_index = client.process_index()?;
        let addressable_devices = client.addressable_devices()?;
        let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
        for device in addressable_devices {
            addressable_device_by_id.insert(device.id()?, device);
        }

        let (shards, _) = compute_shard_descriptors(global_shape, &mesh, &sharding)?;
        let mut addressable_buffers = Vec::new();
        for shard in &shards {
            let mesh_device = shard.device();
            if mesh_device.process_index != client_process_index {
                continue;
            }

            let device = addressable_device_by_id.get(&mesh_device.id).ok_or(
                ArrayError::MissingClientDeviceForLocalMeshDevice {
                    device_id: mesh_device.id,
                    process_index: client_process_index,
                },
            )?;
            let shard_bytes = extract_dense_shard_bytes(buffer, global_shape, shard.slices(), element_type)?;
            let shard_dimensions = shard.shape().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let addressable_buffer = client.buffer(
                shard_bytes.as_slice(),
                element_type.to_pjrt_buffer_type(),
                shard_dimensions.as_slice(),
                None,
                device.clone(),
                None,
            )?;
            addressable_buffers.push(addressable_buffer);
        }

        Self::from_sharding(global_shape.to_vec(), element_type, mesh, sharding, addressable_buffers)
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

    /// Moves or copies this array to the provided `mesh` and `sharding`.
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
    ///   - `mesh`: Concrete destination device mesh.
    ///   - `sharding`: Destination sharding over `mesh`.
    pub fn put(&self, client: &'o Client<'_>, mesh: DeviceMesh, sharding: Sharding) -> Result<Self, ArrayError> {
        let global_shape = self.shape();
        if let Some(addressable_buffers) = copy_addressable_destination_shards_from_exact_source_shards(
            self,
            client,
            global_shape.as_slice(),
            &mesh,
            &sharding,
        )? {
            return Self::from_sharding(global_shape, self.element_type(), mesh, sharding, addressable_buffers);
        }

        let host_bytes = materialize_dense_array_bytes(self)?;
        Self::from_host_buffer(
            client,
            host_bytes.as_slice(),
            global_shape.as_slice(),
            self.element_type(),
            mesh,
            sharding,
        )
    }

    /// Moves or copies this array to the provided placement, consuming `self`.
    ///
    /// This is the closest `ryft` analogue to JAX's
    /// [`jax.Array.to_device`](https://docs.jax.dev/en/latest/_autosummary/jax.Array.to_device.html).
    /// When the resolved placement matches the current placement, the method returns `self`
    /// unchanged. Otherwise it falls back to [`Array::put`] to produce a newly placed array.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to materialize any new destination buffers.
    ///   - `device`: Destination placement for this array.
    pub fn to_device(self, client: &'o Client<'_>, device: DevicePutPlacement) -> Result<Self, ArrayError> {
        let current_placement = DevicePutSharding::new(self.mesh(), self.sharding().clone());
        let target_placement = resolve_device_put_placement(device, self.sharding().rank())?.into_public();
        if current_placement == target_placement {
            Ok(self)
        } else {
            self.put(client, target_placement.mesh, target_placement.sharding)
        }
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

    /// Returns the concrete mesh implied by this array's global shard placement metadata.
    pub fn mesh(&self) -> DeviceMesh {
        DeviceMesh::new(self.sharding().mesh.clone(), self.shards.iter().map(ArrayShard::device).collect())
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

trait DenseHostDevicePutLeaf {
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>);
}

trait DenseHostElement {
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

/// Leaf types accepted by the higher-level [`device_put`] API.
///
/// A [`DevicePutLeaf`] consumes one input leaf and materializes one runtime [`Array`] leaf. `ryft`
/// currently provides implementations for:
/// - runtime [`Array`] leaves,
/// - primitive scalar host values (`bool`, integers, `bf16`, `f16`, `f32`, `f64`), and
/// - owned `ndarray::Array`s when the `ndarray` feature is enabled.
pub trait DevicePutLeaf<'c>: Parameter {
    /// Converts `self` into one runtime [`Array`] using the provided leafwise placement options.
    ///
    /// # Parameters
    ///
    ///   - `client`: PJRT client used to materialize any needed destination buffers.
    ///   - `device`: Destination placement for this leaf, if one was specified.
    ///   - `src`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Best-effort donation flag for this leaf.
    ///   - `may_alias`: Best-effort aliasing hint for this leaf.
    fn device_put_leaf(
        self,
        client: &'c Client<'_>,
        device: Option<DevicePutPlacement>,
        src: Option<DevicePutPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, ArrayError>;
}

impl<'c, T: DenseHostDevicePutLeaf + Parameter> DevicePutLeaf<'c> for T {
    fn device_put_leaf(
        self,
        client: &'c Client<'_>,
        device: Option<DevicePutPlacement>,
        _src: Option<DevicePutPlacement>,
        _donate: bool,
        _may_alias: Option<bool>,
    ) -> Result<Array<'c>, ArrayError> {
        let (shape, element_type, bytes) = self.into_dense_host_array();
        let resolved_placement = match device {
            Some(device) => resolve_device_put_placement(device, shape.len())?,
            None => default_device_put_placement(client, shape.len())?,
        };
        Array::from_host_buffer(
            client,
            bytes.as_slice(),
            shape.as_slice(),
            element_type,
            resolved_placement.mesh,
            resolved_placement.sharding,
        )
    }
}

impl<'c> DevicePutLeaf<'c> for Array<'c> {
    fn device_put_leaf(
        self,
        client: &'c Client<'_>,
        device: Option<DevicePutPlacement>,
        src: Option<DevicePutPlacement>,
        _donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, ArrayError> {
        let current_placement = DevicePutSharding::new(self.mesh(), self.sharding().clone());
        if let Some(src) = src {
            let expected = resolve_device_put_placement(src, self.sharding().rank())?.into_public();
            if expected != current_placement {
                return Err(ArrayError::SourcePlacementMismatch { expected, actual: current_placement.clone() });
            }
        }

        let target_placement = match device {
            Some(device) => resolve_device_put_placement(device, self.sharding().rank())?.into_public(),
            None => current_placement.clone(),
        };
        if target_placement == current_placement && may_alias != Some(false) {
            Ok(self)
        } else {
            self.put(client, target_placement.mesh, target_placement.sharding)
        }
    }
}

/// Higher-level `ryft` analogue of JAX's
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
///
/// The input `x` may be one supported leaf or a `Parameterized` tree of supported leaves. Any
/// provided `device`, `src`, `donate`, and `may_alias` fields follow tree-prefix broadcasting
/// semantics over `x`.
///
/// Host leaves are committed to the default local device when `options.device` is absent. Existing
/// [`Array`] leaves preserve their current placement when `options.device` is absent.
pub fn device_put<'c, P, Input, Device, Src, Donate, MayAlias>(
    client: &'c Client<'_>,
    x: Input,
    options: DevicePutOptions<Device, Src, Donate, MayAlias>,
) -> Result<<Input as Parameterized<P>>::To<Array<'c>>, ArrayError>
where
    P: DevicePutLeaf<'c>,
    Input: Parameterized<P, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Array<'c>>
        + ParameterizedFamily<DevicePutPlacement>
        + ParameterizedFamily<bool>
        + ParameterizedFamily<Option<bool>>,
    Device: Parameterized<DevicePutPlacement>,
    Src: Parameterized<DevicePutPlacement>,
    Donate: Parameterized<bool>,
    MayAlias: Parameterized<Option<bool>>,
{
    let structure = x.parameter_structure();
    let leaf_count = structure.parameter_count();

    let device_values = match options.device {
        Some(device) => Input::To::<DevicePutPlacement>::from_broadcasted_named_parameters(
            structure.clone(),
            device.into_named_parameters(),
        )?
        .into_parameters()
        .map(Some)
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };
    let src_values = match options.src {
        Some(src) => Input::To::<DevicePutPlacement>::from_broadcasted_named_parameters(
            structure.clone(),
            src.into_named_parameters(),
        )?
        .into_parameters()
        .map(Some)
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };
    let donate_values = match options.donate {
        Some(donate) => {
            Input::To::<bool>::from_broadcasted_named_parameters(structure.clone(), donate.into_named_parameters())?
                .into_parameters()
                .collect::<Vec<_>>()
        }
        None => vec![false; leaf_count],
    };
    let may_alias_values = match options.may_alias {
        Some(may_alias) => Input::To::<Option<bool>>::from_broadcasted_named_parameters(
            structure.clone(),
            may_alias.into_named_parameters(),
        )?
        .into_parameters()
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };

    let mut output_parameters = Vec::with_capacity(leaf_count);
    let mut device_values = device_values.into_iter();
    let mut src_values = src_values.into_iter();
    let mut donate_values = donate_values.into_iter();
    let mut may_alias_values = may_alias_values.into_iter();
    for parameter in x.into_parameters() {
        output_parameters.push(
            parameter.device_put_leaf(
                client,
                device_values
                    .next()
                    .expect("device tree-prefix broadcasting should produce one placement per input leaf"),
                src_values.next().expect("src tree-prefix broadcasting should produce one placement per input leaf"),
                donate_values
                    .next()
                    .expect("donate tree-prefix broadcasting should produce one flag per input leaf"),
                may_alias_values
                    .next()
                    .expect("may_alias tree-prefix broadcasting should produce one flag per input leaf"),
            )?,
        );
    }
    Input::To::<Array<'c>>::from_parameters(structure, output_parameters).map_err(Into::into)
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

    use indoc::indoc;
    use pretty_assertions::assert_eq;
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

    fn f32_values_from_bytes(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len() % size_of::<f32>(), 0);
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
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
    fn test_device_put_visualizes_uneven_1d_partitioning() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let values = [0.0f32, 1.0, 2.0, 3.0, 4.0];

        let array = Array::from_host_buffer(
            &client,
            f32_values_to_bytes(values.as_slice()).as_slice(),
            [5usize],
            DataType::F32,
            mesh.clone(),
            sharding,
        )
        .unwrap();

        assert_eq!(array.addressable_shards().count(), 2);
        assert!(array.shards().iter().all(|shard| shard.is_addressable()));
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
    fn test_device_put_visualizes_2d_partitioning() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            ])
            .unwrap(),
            mesh_devices,
        )
        .unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let values = (0..48).map(|value| value as f32).collect::<Vec<_>>();

        let array = Array::from_host_buffer(
            &client,
            f32_values_to_bytes(values.as_slice()).as_slice(),
            [8usize, 6usize],
            DataType::F32,
            mesh.clone(),
            sharding,
        )
        .unwrap();

        assert_eq!(array.addressable_shards().count(), 4);
        assert!(array.shards().iter().all(|shard| shard.is_addressable()));
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
    }

    #[test]
    fn test_array_put_reshards_fully_addressable_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh.clone(), 1);
        let source_values = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let source_array = Array::from_host_buffer(
            &client,
            f32_values_to_bytes(source_values.as_slice()).as_slice(),
            [5usize],
            DataType::F32,
            source_mesh,
            source_sharding,
        )
        .unwrap();

        let target_mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            target_mesh_devices,
        )
        .unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_array = source_array.put(&client, target_mesh.clone(), target_sharding).unwrap();

        assert_eq!(moved_array.addressable_shards().count(), 2);
        assert_eq!(
            moved_array.sharding().visualize().unwrap().render(false),
            indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string()
        );
        let first_shard_bytes = moved_array.shard_for_device(client_devices[0].id().unwrap()).unwrap();
        let second_shard_bytes = moved_array.shard_for_device(client_devices[1].id().unwrap()).unwrap();
        assert_eq!(
            f32_values_from_bytes(
                first_shard_bytes.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap().as_slice()
            ),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(
            f32_values_from_bytes(
                second_shard_bytes.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap().as_slice()
            ),
            vec![3.0, 4.0]
        );
    }

    #[test]
    fn test_array_put_copies_matching_local_shards_without_full_source_addressability() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![
                MeshDevice::new(local_device_id, local_device.process_index().unwrap()),
                MeshDevice::new(remote_device_id, 1),
            ],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array = Array::from_sharding(
            vec![4usize],
            DataType::F32,
            mesh.clone(),
            sharding.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array.put(&client, mesh.clone(), sharding).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        assert_eq!(
            f32_values_from_bytes(
                copied_array
                    .shard_for_device(local_device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap()
                    .as_slice()
            ),
            vec![0.0, 1.0]
        );
        assert!(copied_array.shard_for_device(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_plan_exact_shard_put_uses_cross_host_send_and_receive_for_remote_exact_moves() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![
                MeshDevice::new(local_device_id, local_device.process_index().unwrap()),
                MeshDevice::new(remote_device_id, 1),
            ],
        )
        .unwrap();
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array =
            Array::from_sharding(vec![4usize], DataType::F32, source_mesh, source_sharding, vec![local_source_buffer])
                .unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![
                MeshDevice::new(remote_device_id, 1),
                MeshDevice::new(local_device_id, client.process_index().unwrap()),
            ],
        )
        .unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let plan = plan_exact_shard_put(
            &source_array,
            client.process_index().unwrap(),
            source_array.shape().as_slice(),
            &target_mesh,
            &target_sharding,
        )
        .unwrap();

        assert_eq!(
            plan,
            Some(ExactShardPutPlan {
                local_copies: Vec::new(),
                cross_host_sends: vec![CrossHostShardSendPlan {
                    source_shard_index: 0,
                    source_device_id: local_device_id,
                    destination_shard_index: 0,
                    destination_device_id: remote_device_id,
                    transfer_key: 0,
                }],
                cross_host_receives: vec![CrossHostShardReceivePlan {
                    source_shard_index: 1,
                    source_device_id: remote_device_id,
                    destination_shard_index: 1,
                    destination_device_id: local_device_id,
                    destination_shape: vec![2],
                    transfer_key: 3,
                }],
            })
        );
    }

    #[test]
    fn test_array_put_rejects_non_addressable_source_shards() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(0, 0), MeshDevice::new(1, 1)],
        )
        .unwrap();
        let source_sharding =
            Sharding::new(source_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source_array =
            Array::from_sharding(vec![4usize], DataType::F32, source_mesh, source_sharding, Vec::new()).unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(0, 0)],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh.clone(), 1);

        assert!(matches!(
            source_array.put(&client, target_mesh, target_sharding),
            Err(ArrayError::MissingAddressableShardForMove { shard_index: 0, device_id: 0 }),
        ));
    }

    #[test]
    fn test_device_put_broadcasts_root_placement_over_array_tuple() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![MeshDevice::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh.clone(), 1);
        let first_source_array = Array::from_host_buffer(
            &client,
            f32_values_to_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]).as_slice(),
            [5usize],
            DataType::F32,
            source_mesh.clone(),
            source_sharding.clone(),
        )
        .unwrap();
        let second_source_array = Array::from_host_buffer(
            &client,
            f32_values_to_bytes(&[10.0, 11.0, 12.0, 13.0, 14.0]).as_slice(),
            [5usize],
            DataType::F32,
            source_mesh,
            source_sharding,
        )
        .unwrap();

        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            client_devices
                .iter()
                .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
                .collect(),
        )
        .unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_arrays = device_put(
            &client,
            (first_source_array, second_source_array),
            DevicePutOptions {
                device: Some(DevicePutPlacement::sharding(target_mesh.clone(), target_sharding.clone())),
                ..DevicePutOptions::defaults()
            },
        )
        .unwrap();

        assert_eq!(
            moved_arrays.0.sharding().visualize().unwrap().render(false),
            indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string()
        );
        assert_eq!(
            moved_arrays.1.sharding().visualize().unwrap().render(false),
            indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string()
        );
        assert_eq!(
            f32_values_from_bytes(
                moved_arrays
                    .0
                    .shard_for_device(client_devices[1].id().unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap()
                    .as_slice()
            ),
            vec![3.0, 4.0]
        );
        assert_eq!(
            f32_values_from_bytes(
                moved_arrays
                    .1
                    .shard_for_device(client_devices[0].id().unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap()
                    .as_slice()
            ),
            vec![10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_device_put_preserves_partially_addressable_array_when_device_is_absent() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![
                MeshDevice::new(local_device_id, local_device.process_index().unwrap()),
                MeshDevice::new(remote_device_id, 1),
            ],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array = Array::from_sharding(
            vec![4usize],
            DataType::F32,
            mesh.clone(),
            sharding.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = device_put(&client, source_array, DevicePutOptions::defaults()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        assert_eq!(
            f32_values_from_bytes(
                copied_array
                    .shard_for_device(local_device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap()
                    .as_slice()
            ),
            vec![0.0, 1.0]
        );
        assert!(copied_array.shard_for_device(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_to_device_preserves_same_partially_addressable_placement() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![
                MeshDevice::new(local_device_id, local_device.process_index().unwrap()),
                MeshDevice::new(remote_device_id, 1),
            ],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array = Array::from_sharding(
            vec![4usize],
            DataType::F32,
            mesh.clone(),
            sharding.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = source_array
            .to_device(&client, DevicePutPlacement::sharding(mesh.clone(), sharding.clone()))
            .unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        assert_eq!(
            f32_values_from_bytes(
                copied_array
                    .shard_for_device(local_device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap()
                    .as_slice()
            ),
            vec![0.0, 1.0]
        );
        assert!(copied_array.shard_for_device(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_device_put_rejects_mismatched_src_for_array_leaf() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let client_device = client.addressable_devices().unwrap().remove(0);
        let source_device = MeshDevice::new(client_device.id().unwrap(), client_device.process_index().unwrap());
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![source_device],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh.clone(), 1);
        let source_array = Array::from_host_buffer(
            &client,
            f32_values_to_bytes(&[0.0, 1.0]).as_slice(),
            [2usize],
            DataType::F32,
            source_mesh.clone(),
            source_sharding.clone(),
        )
        .unwrap();
        let expected_src =
            resolve_device_put_placement(DevicePutPlacement::device(MeshDevice::new(source_device.id + 1, 0)), 1)
                .unwrap()
                .into_public();

        assert!(matches!(
            device_put(
                &client,
                source_array,
                DevicePutOptions {
                    src: Some(DevicePutPlacement::device(MeshDevice::new(source_device.id + 1, 0))),
                    ..DevicePutOptions::defaults()
                },
            ),
            Err(ArrayError::SourcePlacementMismatch { expected, actual })
                if expected == expected_src
                    && actual == DevicePutSharding::new(source_mesh, source_sharding),
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
