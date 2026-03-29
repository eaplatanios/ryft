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
pub use arrays::{AddressableShard, Array, ArrayError, ExecuteArguments};
#[cfg(feature = "xla")]
pub use shard_map::{ShardMapTraceError, TracedShardMap, TracedXlaProgram, shard_map, trace, with_sharding_constraint};

pub use sharding::{ShardDescriptor, ShardSlice, Sharding, ShardingLayout};
