/// XLA-specific runtime data structures.
///
/// The [`sharding`] module defines device mesh and array sharding metadata (mirroring JAX's
/// sharding model and Shardy MLIR attributes). The [`arrays`] module builds on that to provide
/// distributed array types backed by local PJRT buffers for execution.
pub mod arrays;
pub mod sharding;

pub use arrays::{AddressableShard, Array, ArrayError, ExecuteArguments};

pub use sharding::{
    AbstractMesh, Mesh, MeshAxis, MeshDevice, NamedSharding, PartitionDimension, PartitionSpec, ShardDescriptor,
    ShardSlice, ShardingContext, ShardingError, ShardingLayout,
};
