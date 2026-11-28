//! Compilation of staged programs for the XLA [`Session`](crate::Session).
//!
//! This module mirrors [`ryft_core::compilation`]. [`functions`] holds the typed entry points (i.e., [`stage`],
//! [`compile`], and [`jit`]) and the function handles that they return. [`domains`] implements the lifecycle of
//! [`ryft_core::CompilationDomain`] for the session, which lowers a staged program into a [`LoweredProgram`] and
//! compiles it into a [`CompiledProgram`]. [`options`] configures a compilation, [`keys`] identifies retained
//! specializations and cached artifacts, [`persistence`] serializes compiled programs, [`analysis`] exposes compiler
//! diagnostics, and [`profile_guided`] recompiles a function from execution profiles. Compilation and StableHLO
//! export share placement preparation and analysis through the [`lowering`](crate::lowering) module.

// TODO(eaplatanios): Review this module.

use ryft_core::{CALL_OPERATION_NAME, Device, DeviceMesh, PLACE_OPERATION_NAME};

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

/// Represents errors produced while validating compilation [`Options`], checking or persisting an XLA executable, or
/// interpreting compilation analysis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, thiserror::Error)]
pub enum CompilationError {
    #[error("invalid compilation options: {reason}")]
    InvalidOptions { reason: String },

    #[error("invalid {kind} sharding count; expected {expected} but got {actual}")]
    ShardingCountMismatch { kind: &'static str, expected: usize, actual: usize },

    #[error(
        "the sharding of {kind} {index} splits dimension {dimension} of extent {extent} into {shard_count} shards, but \
         the boundary of an executable requires shards of equal extent, while XLA pads the last shards of an uneven \
         split"
    )]
    UnevenBoundarySharding { kind: &'static str, index: usize, dimension: usize, extent: usize, shard_count: usize },

    #[error(
        "cannot `{}` a value on the devices [{}] of mesh `{}` inside a program compiled for the devices [{}] of mesh \
         `{}`",
        PLACE_OPERATION_NAME,
        .target.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .target.logical_mesh(),
        .mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .mesh.logical_mesh(),
    )]
    PlacementMeshMismatch { target: DeviceMesh, mesh: DeviceMesh },

    #[error(
        "cannot `{}` a value on the device {} inside a program compiled for the devices [{}] of mesh `{}`",
        PLACE_OPERATION_NAME,
        .device,
        .mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .mesh.logical_mesh(),
    )]
    PlacementDeviceMismatch { device: Device, mesh: DeviceMesh },

    #[error(
        "cannot `{}` a function compiled for the devices [{}] of mesh `{}` inside a program compiled for the devices \
         [{}] of mesh `{}`",
        CALL_OPERATION_NAME,
        .callee.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .callee.logical_mesh(),
        .mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .mesh.logical_mesh(),
    )]
    CalleeMeshMismatch { callee: DeviceMesh, mesh: DeviceMesh },

    #[error("unsupported XLA reference ABI: {reason}")]
    UnsupportedReferenceAbi { reason: String },

    #[error("incompatible XLA executable: {reason}")]
    IncompatibleExecutable { reason: String },

    #[error(
        "incompatible XLA executable: executable {kind} {index} uses memory kind `{actual}` but its type declares \
         `{expected}`"
    )]
    ExecutableMemoryKindMismatch { kind: &'static str, index: usize, expected: String, actual: String },

    #[error("invalid persistent XLA executable: {reason}")]
    InvalidPersistentExecutable { reason: String },

    #[error("invalid XLA compilation analysis: {reason}")]
    InvalidAnalysis { reason: String },
}

#[cfg(test)]
mod tests {
    use ryft_core::{ArrayType, FlatProgram, Placeholder, ProgramBuilder, ReferenceReadOperation, ReferenceType};

    use crate::{Constant, Operation, Session, Type};

    /// Builds a flat program that reads and returns one externally supplied reference.
    pub(super) fn external_reference_read_program(referent_type: ArrayType) -> FlatProgram<Session> {
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let reference = builder.add_input(Type::Reference(ReferenceType::new(referent_type)));
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }
}
