/// Backend token used for traced XLA staging and PJRT-backed execution.
pub mod domains;
/// StableHLO and Shardy lowering helpers for traced XLA programs.
pub mod lowering;
/// Backend-owned staged operation types for traced XLA programs.
pub mod ops;
/// General XLA program tracing: the [`trace`] entry point, its [`TracedXlaProgram`] result, which lowers the traced
/// program to StableHLO/Shardy MLIR for compilation and execution, and the [`TraceError`] of both stages.
pub mod tracing;

pub use domains::{
    XlaAnalysisValue, XlaCompilationAnalysis, XlaDomain, XlaDomainError, XlaFeedbackDirectedProfile, XlaMemoryAnalysis,
    XlaOptimizedProgram, XlaSession,
};

pub use tracing::{TraceError, TracedXlaProgram, XlaArrayTracer, trace};
