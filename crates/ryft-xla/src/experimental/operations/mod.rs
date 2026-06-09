/// Reshape support for abstract shard-map tensor leaves.
mod reshape;
/// Higher-order shard-map operations used during traced XLA staging and differentiation.
mod shard_map;
/// Identity-like traced op that records an explicit Shardy sharding constraint.
mod with_sharding_constraint;

pub(crate) use shard_map::{
    FactorizedTransposeOutputSource, FactorizedTransposeResidualSource, LinearShardMapEvalMode,
};
pub use shard_map::{LinearShardMapOperation, ShardMapOperation};
pub use with_sharding_constraint::WithShardingConstraintOperation;
