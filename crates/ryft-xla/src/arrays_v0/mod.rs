use std::collections::{HashMap, HashSet};
use std::ops::Range;

use half::{bf16, f16};
use ryft_macros::Parameter;
use thiserror::Error;

#[cfg(test)]
use ryft_mlir::Block;
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{Buffer, Client, DeviceId, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput};

use ryft_core::parameters::{Parameter, Parameterized, ParameterizedFamily};
use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingError};
use ryft_core::types::StaticShape;
use ryft_core::types::data::DataType;

pub mod array;
pub mod compiled_reshard;
pub mod device_put;
pub mod error;
pub mod execution;
pub mod host;
mod materialization;
pub mod placement;
pub mod transfers;

#[cfg(test)]
pub mod tests;

pub use crate::arrays::{ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
pub use device_put::{DevicePutLeaf, device_put};
pub use error::ArrayError;
pub use execution::ExecuteArguments;
pub(crate) use host::DenseHostDevicePutLeaf;
pub(crate) use materialization::{
    BoundedMaterializationCache, BoundedMaterializationKey, BoundedMaterializationProbe,
    BoundedMaterializationProducer, BoundedMaterializationWaiter, LOGICAL_EXTENT_SCALAR_CACHE_CAPACITY,
};
pub use placement::{DevicePutOptions, DevicePutTarget};

pub(crate) use transfers::copy_addressable_destination_shards_from_exact_source_shards;
#[cfg(test)]
pub(crate) use transfers::{
    CrossHostShardReceivePlan, CrossHostShardSendPlan, ExactShardPutPlan, plan_exact_shard_put,
};
