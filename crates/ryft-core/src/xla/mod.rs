pub mod arrays;
pub(crate) mod lowering;
mod shard_map;
pub mod sharding;

pub use arrays::{AddressableShard, Array, ArrayError, ExecuteArguments};
pub use shard_map::{ShardMapTraceError, TracedShardMap, shard_map};

pub use sharding::{
    DeviceMesh, LogicalMesh, MeshAxis, MeshDevice, NamedSharding, PartitionDimension, PartitionSpec, ShardDescriptor,
    ShardSlice, ShardingContext, ShardingError, ShardingLayout,
};
