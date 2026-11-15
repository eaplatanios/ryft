//! Compilation of staged programs for the XLA [`Domain`](crate::Domain).
//!
//! This module mirrors [`ryft_core::compilation`]. [`functions`] holds the typed entry points (i.e., [`stage`],
//! [`compile`], and [`jit`]) and the function handles that they return. [`domains`] implements the lifecycle of
//! [`ryft_core::CompilationDomain`] for the domain, which lowers a staged program into a [`LoweredProgram`] and
//! compiles it into a [`CompiledProgram`]. [`options`] configures a compilation, [`keys`] identifies retained
//! specializations and cached artifacts, [`persistence`] serializes compiled programs, [`analysis`] exposes compiler
//! diagnostics, and [`profile_guided`] recompiles a function from execution profiles.

// TODO(eaplatanios): Review this module.

pub mod analysis;
pub mod domains;
pub mod functions;
pub mod keys;
pub mod options;
pub mod persistence;
pub mod profile_guided;
pub mod programs;

pub use analysis::{AnalysisValue, CompilationAnalysis, MemoryAnalysis, OptimizedProgram};
pub use domains::with_input_shardings;
pub use functions::{
    CompileTracer, CompiledFunction, CompiledFunctionDispatcher, ExecutableFunction, StagedFunction,
    StatefulCompileTracer, StatefulCompiledFunction, StatefulCompiledFunctionDispatcher, StatefulValue, compile,
    compile_statefully, compile_statefully_with_captures, compile_with_captures, compile_with_options,
    infer_output_types, jit, jit_statefully, jit_statefully_with_options, jit_with_options, stage, stage_with_captures,
};
pub use keys::{CompilationKey, DispatchKey};
pub use options::{FeedbackDirectedProfile, InputBoundBucketing, Options};
pub use profile_guided::{
    AdaptiveProfileGuidedFunction, AdaptiveProfileGuidedOptions, AdaptiveProfileGuidedState,
    AdaptiveProfileGuidedStatistics,
};
pub use programs::{CompiledProgram, LoweredProgram};

/// XLA option, executable compatibility, persistence, and analysis failures.
#[derive(Clone, Debug, thiserror::Error)]
pub enum CompilationError {
    /// A requested compilation option combination is invalid.
    #[error("invalid compilation options: {reason}")]
    InvalidOptions { reason: String },

    /// Signature sharding metadata has a different arity than the values it describes.
    #[error("invalid {kind} sharding count; expected {expected} but got {actual}")]
    ShardingCountMismatch { kind: &'static str, expected: usize, actual: usize },

    /// A valid external-reference configuration cannot be represented by this backend's executable ABI.
    #[error("unsupported XLA reference ABI: {reason}")]
    UnsupportedReferenceAbi { reason: String },

    /// A compiled executable does not preserve its declared invocation contract.
    #[error("incompatible XLA executable: {reason}")]
    IncompatibleExecutable { reason: String },

    /// Persistent executable bytes or metadata violate their versioned transport contract.
    #[error("invalid persistent XLA executable: {reason}")]
    InvalidPersistentExecutable { reason: String },

    /// Compiler analysis properties cannot be interpreted according to their declared schema.
    #[error("invalid XLA compilation analysis: {reason}")]
    InvalidAnalysis { reason: String },
}
