/// Runtime PJRT-backed array types and `device_put` support for the experimental XLA surface.
pub mod arrays;
#[cfg(all(feature = "benchmarking"))]
/// XLA-specific IR benchmark cases layered on top of `ryft_core::tracing_v2::benchmarking`.
pub mod benchmark_support;
/// Backend token used for traced XLA staging and PJRT-backed execution.
pub mod engines;
/// StableHLO and Shardy lowering helpers for traced XLA programs.
pub mod lowering;
/// Experimental XLA-only higher-order primitives and staged operation helpers.
pub mod operations;
/// Backend-owned staged operation carriers for traced XLA programs.
pub mod ops;
/// Tracing-backed `shard_map` surface and the supporting manual-computation metadata model.
pub mod shard_map;

pub use arrays::{
    Array, ArrayError, ArrayShard, DevicePutLeaf, DevicePutOptions, DevicePutPlacement, DevicePutSharding,
    ExecuteArguments, device_put,
};

pub use engines::{XlaEngine, XlaEngineError};

pub use shard_map::{
    ShardMapTraceError, TracedShardMap, TracedXlaProgram, shard_map, shard_map_with_options, trace,
    with_sharding_constraint,
};
