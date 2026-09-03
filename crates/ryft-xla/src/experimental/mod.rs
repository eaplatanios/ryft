/// Runtime assertion support for compiled first-class-dimension programs.
pub(crate) mod assertions;
/// Host-callback debugging support: the `ryft.print` XLA FFI handler and its capturable print sink.
pub mod debugging;
/// Backend token used for traced XLA staging and PJRT-backed execution.
pub mod domains;
/// StableHLO and Shardy lowering helpers for traced XLA programs.
pub mod lowering;
/// Experimental XLA-only higher-order primitives and staged operation helpers.
pub mod operations;
/// Backend-owned staged operation types for traced XLA programs.
pub mod ops;
/// Experimental preserved-reference kernel boundary for externally stateful XLA kernels.
pub mod reference_kernels;
/// Tracing-backed `shard_map` surface and the supporting manual-computation metadata model.
pub mod shard_map;

pub use lowering::RaggedDotLoweringStrategy;

pub use domains::{
    XlaAnalysisValue, XlaCompilationAnalysis, XlaDomain, XlaDomainError, XlaFeedbackDirectedProfile, XlaMemoryAnalysis,
    XlaOptimizedProgram,
};

pub use shard_map::{
    ShardMapTraceError, ShardMapTracer, TracedShardMap, TracedXlaProgram, reshard, shard_map, shard_map_with_options,
    sharding_constraint, trace,
};
