#[cfg(feature = "xla")]
pub mod arrays;
#[cfg(all(feature = "benchmarking", feature = "xla"))]
pub(crate) mod benchmark_support;
#[cfg(feature = "xla")]
pub(crate) mod lowering;
#[cfg(feature = "xla")]
pub(crate) mod shard_map;
pub mod sharding;

#[cfg(feature = "xla")]
pub use arrays::{
    Array, ArrayError, ArrayShard, DevicePutLeaf, DevicePutOptions, DevicePutPlacement, DevicePutSharding,
    ExecuteArguments, device_put,
};
#[cfg(feature = "xla")]
pub use shard_map::{
    ShardMapTraceError, TracedShardMap, TracedXlaProgram, shard_map, shard_map_with_options, trace,
    with_sharding_constraint,
};

pub use crate::sharding::Sharding;
pub use sharding::{Shard, ShardSlice};
