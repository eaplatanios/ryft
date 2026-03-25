pub mod arrays;
pub(crate) mod lowering;
mod shard_map;
pub mod sharding;

pub use arrays::{AddressableShard, Array, ArrayError, ExecuteArguments};
pub use shard_map::{ShardMapTraceError, TracedShardMap, TracedXlaProgram, shard_map, trace};
pub(crate) use shard_map::{
    try_apply_shard_map_program_jvp_rule, try_grad_traced_xla_inputs, try_transpose_shard_map_program_op,
    uses_traced_xla_grad,
};

pub use sharding::{
    DeviceMesh, LogicalMesh, MeshAxis, MeshDevice, NamedSharding, PartitionDimension, PartitionSpec, ShardDescriptor,
    ShardSlice, ShardingContext, ShardingError, ShardingLayout,
};
