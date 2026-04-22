/// Reshape support for abstract shard-map tensor leaves.
pub mod reshape;
/// Higher-order shard-map operations used during traced XLA staging and differentiation.
pub mod shard_map;
/// Identity-like traced op that records an explicit Shardy sharding constraint.
pub mod with_sharding_constraint;

pub use shard_map::{LinearShardMapEvalMode, LinearShardMapOperation, ShardMapOperation};
pub use with_sharding_constraint::WithShardingConstraintOperation;
