//! Experimental backend-independent kernel definitions over ordinary Ryft programs.
//!
//! Kernel bodies use canonical array, dimension, reference, and operation semantics. Validation is explicit at the
//! kernel boundary; ordinary programs do not acquire a mandatory whole-program analysis. Native compilation and
//! execution bindings belong to adapters and execution backends.

// TODO(eaplatanios): Review this module and all of its submodules.

pub mod authoring;
pub mod calls;
pub mod compilation;
pub mod grids;
pub mod indexing;
pub mod initialization;
pub mod interpretation;
pub mod mappings;
pub mod memory;
pub mod operations;
pub mod scheduling;
pub mod serialization;
pub mod validation;

#[cfg(test)]
mod tests;

pub use ryft_macros::kernel;

pub use authoring::{
    KernelCall, condition, dot, for_loop, shape_div_ceil, static_extent, tile_load, tile_store, tiled_call,
    whole_array_parameter, zeros,
};
pub use calls::{
    KERNEL_CALL_OPERATION_NAME, KERNEL_SCHEMA_VERSION, KernelCallOperation, KernelDefinition, KernelError,
    KernelParameter,
};
pub use compilation::{KernelCompilationError, KernelCompiler, KernelSchedule, VerifiedKernel};
pub use grids::{Grid, GridDimension, GridError, GridExecution, GridPoints};
pub use indexing::TileLoadOperation;
pub use initialization::{KernelInitializationError, validate_kernel_initialization};
pub use interpretation::{
    DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS, DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS, KernelDebugOptions,
    KernelInterpretationError, KernelTraceAccess, KernelTraceEntry,
};
pub use mappings::{BlockMapping, BlockMappingError, BlockWindow, BoundaryPolicy};
pub use memory::{
    AsyncCopyOperation, KernelMemoryError, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation,
    ScratchOperation, WaitOperation,
};
pub use operations::{KernelOperation, NoKernelExtension};
pub use scheduling::{KernelInterleaving, KernelSchedulingError};
pub use serialization::KernelSerializationError;

pub use validation::{
    KernelBoundaryContract, KernelParameterAccess, KernelParameterSummary, KernelReferenceOperation,
    KernelReferenceSummary, KernelSwapLowering, KernelValidationError, validate_kernel_body,
};
