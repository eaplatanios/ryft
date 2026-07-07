/// Reshape support for abstract shard-map tensor leaves.
mod reshape;
/// Higher-order shard-map operations used during traced XLA staging and differentiation.
mod shard_map;

pub use shard_map::{ShardMapOperation, transpose_primal_shard_map};
