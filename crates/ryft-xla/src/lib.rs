/// PJRT-backed runtime arrays and `device_put` support.
pub mod arrays;
/// Experimental traced-XLA surface, including lowering and `shard_map`.
pub mod experimental;
pub mod mlir;
pub mod pjrt;
pub mod sharding;
pub mod types;

pub use arrays::{Array, ArrayShard, ArrayError, ShardDescriptor, ShardLayout, ShardIndex};
