/// XLA-specific runtime data structures.
///
/// The [`arrays`] module models JAX/IFRT-style distributed arrays and their sharding metadata for PJRT execution.
pub mod arrays;

pub use arrays::{
    AddressableShard, Array, ArrayError, ExecuteArguments, Mesh, MeshAxis, MeshDevice, NamedSharding,
    PartitionDimension, PartitionSpecification, ShardDescriptor, ShardSlice, ShardingError, ShardingLayout,
};
