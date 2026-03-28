pub mod arrays;
#[cfg(feature = "benchmarking")]
pub(crate) mod benchmark_support;
pub(crate) mod lowering;
pub(crate) mod shard_map;
pub mod sharding;

pub use arrays::{AddressableShard, Array, ArrayError, ExecuteArguments};
pub use shard_map::{ShardMapTraceError, TracedShardMap, TracedXlaProgram, shard_map, trace};

pub use sharding::{ShardDescriptor, ShardSlice, ShardingLayout, ShardingSpecification};
