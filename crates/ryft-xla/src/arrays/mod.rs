use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use half::{bf16, f16};
use ryft_macros::Parameter;
use thiserror::Error;

#[cfg(test)]
use ryft_mlir::Block;
use ryft_mlir::Location;
use ryft_mlir::dialects::shardy::DetachedMeshOperation;
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{Buffer, Client, DeviceId, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput};

use ryft_core::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use ryft_core::sharding::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, Sharding, ShardingError};
use ryft_core::types::data_types::{DataType, DataTypeError};
use ryft_core::types::{ArrayType, Shape, Size, StaticShape, Typed};

use crate::mlir::ToMlir;
use crate::pjrt::{FromPjrt, ToPjrt};

mod array;
mod device_put;
mod error;
mod execution;
mod host;
mod placement;
mod shards;
mod transfers;

#[cfg(test)]
mod tests;

pub use array::Array;
pub use device_put::{DevicePutLeaf, device_put};
pub use error::ArrayError;
pub use execution::ExecuteArguments;
#[cfg(test)]
pub(crate) use host::device_put_element_size_in_bytes;
#[cfg(test)]
pub(crate) use host::static_shape_dimensions;
pub(crate) use host::{
    DenseHostDevicePutLeaf, checked_byte_count, extract_dense_shard_bytes, materialize_dense_array_bytes, static_shape,
};
pub use placement::{ArrayPlacement, DevicePutOptions, DevicePutTarget};
pub use shards::{ShardIndex, ShardLayout, ShardDescriptor, ArrayShard};

pub(crate) use transfers::copy_addressable_destination_shards_from_exact_source_shards;
#[cfg(test)]
pub(crate) use transfers::{
    CrossHostShardReceivePlan, CrossHostShardSendPlan, ExactShardPutPlan, plan_exact_shard_put,
};
