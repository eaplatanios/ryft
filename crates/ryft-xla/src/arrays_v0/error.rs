use super::*;

use ryft_core::{DataTypeError, ParameterError};

/// Error type for [`Array`](crate::Array) construction and execution-input preparation.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ArrayError {
    /// Underlying crate-level error.
    #[error(transparent)]
    Error(#[from] crate::Error),

    /// Error returned when `device_put` is asked to shard an element type without a supported dense host encoding.
    #[error("device_put does not support dense host bytes for element type {element_type}")]
    UnsupportedDevicePutElementType { element_type: DataType },

    /// Error returned when the higher-level [`device_put()`] API needs a default device but the client has no local devices.
    #[error("device_put needs a default local device, but the PJRT client has no addressable devices")]
    MissingDefaultDevice,

    /// Error returned when a provided `src` placement does not match an array leaf's current placement.
    #[error(
        "device_put src mesh {expected_mesh:?} with sharding {expected_sharding:?} does not match the array's current \
         mesh {actual_mesh:?} with sharding {actual_sharding:?}"
    )]
    SourcePlacementMismatch {
        expected_mesh: DeviceMesh,
        expected_sharding: Sharding,
        actual_mesh: DeviceMesh,
        actual_sharding: Sharding,
    },

    /// Error returned when a device ID cannot be represented by the PJRT cross-host transfers extension.
    #[error("device {device_id} cannot be represented as a PJRT global device ID for cross-host transfers")]
    CrossHostTransferDeviceIdTooLarge { device_id: DeviceId },

    /// Error returned when a shard dimension cannot be represented by the PJRT cross-host transfers extension.
    #[error(
        "cross-host transfer for shard #{shard_index} has shape dimension #{dimension}={size}, which does not fit in i64"
    )]
    CrossHostTransferShapeDimensionTooLarge { shard_index: ShardIndex, dimension: usize, size: usize },

    /// Error returned when an exact-shard cross-host transfer key cannot be represented in PJRT.
    #[error(
        "exact-shard transfer key for source shard #{source_shard_index} and destination shard #{destination_shard_index} \
         does not fit in i64"
    )]
    CrossHostTransferKeyTooLarge { source_shard_index: ShardIndex, destination_shard_index: ShardIndex },

    /// Error returned when [`Array::to_placement`](crate::Array::to_placement) needs a source shard that is not
    /// addressable locally.
    #[error(
        "array move requires shard #{shard_index} on device {device_id} to be addressable from the current process"
    )]
    MissingAddressableShardForMove { shard_index: ShardIndex, device_id: DeviceId },

    /// Error returned when copying a source shard to host yields an unexpected byte count.
    #[error(
        "copied shard #{shard_index} from device {device_id} to host and got {actual_byte_count} byte(s), but expected {expected_byte_count}"
    )]
    CopiedShardByteCountMismatch {
        shard_index: ShardIndex,
        device_id: DeviceId,
        expected_byte_count: usize,
        actual_byte_count: usize,
    },

    /// Error returned when overlapping source shards disagree while materializing a dense host array.
    #[error("array move found inconsistent overlapping data while materializing shard #{shard_index}")]
    InconsistentOverlappingShardData { shard_index: ShardIndex },

    /// Error returned when the destination [`DeviceMesh`] has a mesh axis with
    /// [`MeshAxisType::Manual`](ryft_core::sharding::MeshAxisType::Manual). `Manual` axes are
    /// managed explicitly by the user (e.g. inside
    /// [`shard_map`](crate::experimental::shard_map)); the compiled reshard path uses the SPMD
    /// partitioner from the top level and can only plan `Auto` and `Explicit` axes.
    #[error("compiled reshard does not support `Manual` mesh axis {axis_name} (type {axis_type:?})")]
    UnsupportedMeshAxisType { axis_name: String, axis_type: ryft_core::sharding::MeshAxisType },

    /// Error returned when a destination device of a compiled reshard is not addressable from the
    /// current process. Cross-host destinations are not yet supported.
    #[error(
        "compiled reshard destination device {device_id} is not addressable from the current process (its process index is {process_index})"
    )]
    NonAddressableDestinationDevice { device_id: DeviceId, process_index: usize },

    /// Error returned when the trace/lower/execute pipeline of a compiled reshard fails for
    /// reasons that don't map cleanly to a more specific error variant. The wrapped message
    /// carries the underlying diagnostic so callers can log it.
    #[error("compiled reshard failed: {message}")]
    CompiledReshardInternalError { message: String },

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

impl From<PjrtError> for ArrayError {
    fn from(error: PjrtError) -> Self {
        Self::Error(error.into())
    }
}

impl From<ShardingError> for ArrayError {
    fn from(error: ShardingError) -> Self {
        Self::Error(error.into())
    }
}

impl From<DataTypeError> for ArrayError {
    fn from(error: DataTypeError) -> Self {
        Self::Error(error.into())
    }
}

impl From<ParameterError> for ArrayError {
    fn from(error: ParameterError) -> Self {
        Self::Error(error.into())
    }
}
