/// Reshape support for abstract shard-map tensor leaves.
mod reshape;
/// Higher-order shard-map operations used during traced XLA staging and differentiation.
mod shard_map;

pub(crate) use shard_map::{
    FactorizedTransposeOutputSource, FactorizedTransposeResidualSource, LinearShardMapEvalMode,
};
pub use shard_map::{LinearShardMapOperation, ShardMapOperation};
