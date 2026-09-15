//! Backend-independent kernel authoring over ordinary Ryft programs.
//!
//! Start with [`kernel`] and ordinary [`Array`](crate::arrays::Array) annotations. The macro constructs canonical
//! array, dimension, reference, and control-flow operations; it does not run a separate tracing graph or select a
//! compiler. Native compilation belongs to a [`KernelCompiler`], and execution belongs to its integration backend.
//! The portable contract is the common authoring surface. Adapter APIs and exact GPU extensions remain experimental;
//! this documentation does not promise a stabilized upstream ABI or unrestricted lowering on every target.
//!
//! # Author, inspect, and verify
//!
//! ```
//! use ryft_core::{Array, ArrayType, DataType};
//! use ryft_core::kernels::{kernel, VerifiedKernel};
//!
//! #[kernel(requires = left.shape()[0] == right.shape()[0])]
//! fn add(
//!     #[input(data_type = F32, rank = 1)] left: &Array,
//!     #[input(data_type = F32, rank = 1)] right: &Array,
//!     #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
//! ) {
//!     output.store(left.load() + right.load());
//! }
//!
//! let left = Array::vector(vec![1.0f32, 2.0]).unwrap();
//! let right = Array::vector(vec![3.0f32, 5.0]).unwrap();
//! assert_eq!(add(&left, &right).unwrap(), Array::vector(vec![4.0f32, 7.0]).unwrap());
//! let vector = ArrayType::new_static(DataType::F32, [2]);
//! let definition = add::definition(&vector, &vector).unwrap();
//! let verified = VerifiedKernel::new(&definition, 1).unwrap();
//! assert_eq!(verified.definition().semantic_key().unwrap(), definition.semantic_key().unwrap());
//! ```
//!
//! The callable interprets concrete host arrays or stages through an existing supported value context. The generated
//! `add::definition` function accepts logical types, validates annotations and `requires` constraints, and returns
//! an inspectable [`KernelDefinition`]. Choose a different compiler by passing the same verified definition to
//! [`VerifiedKernel::compile`]; backend options do not rewrite the macro body. XLA owns its selected compiler binding,
//! custom-call ABI, and executable lifecycle. Importing this module requires no XLA, CUDA, PJRT, or Python installation.
//!
//! # Tiles and control flow
//!
//! Input annotations constrain dtype and rank; output annotations specify shape, tile shape, and boundary policy.
//! `output.tile_index()` exposes logical grid coordinates. `input.tiles([width]).pad(0.0).load([index])` expresses a
//! clipped load with explicit padding. `output.store(value)` uses the declared output window. [`BoundaryPolicy`]
//! distinguishes proved in-bounds access from masked edge windows: inactive lanes must never read invalid memory.
//! Shapes and annotation expressions specialize the definition. Supported loops and conditions in the body stage
//! as canonical control flow, not arbitrary host callbacks. Loop-carried arrays retain simultaneous-update semantics;
//! checked scalar indices preserve observable bounds assertions. Unsupported syntax produces source-span diagnostics.
//!
//! [`zeros`], [`dot`], reduction expressions, and tile loads reuse canonical array operations. The macro supports a
//! deliberate syntax subset; a Rust expression accepted outside a kernel is not automatically admitted inside one.
//! Use [`KernelDefinition::trace`] with [`KernelCallOperation`], [`KernelParameter`], [`Grid`], and [`BlockMapping`]
//! for explicit builder control. Both authoring paths share initialization, alias, effect, and write-conflict checks.
//!
//! # Transforms and debugging
//!
//! [`KernelDefinition::batched`] adds a canonical grid axis and requalifies write ownership. Unmapped read-only
//! inputs may be shared; unsupported shared writes and dynamic batching fail explicitly. Specialize scalar-prefetch
//! values using [`KernelDefinition::specialize_prefetch`]; transforms do not read device buffers back to the host.
//! Mutable bodies have no implicit differentiation rule. Execution integrations may accept explicit custom JVP/VJP
//! rules or a pure fallback using their existing transform machinery. Rematerialization preserves the whole call.
//!
//! [`KernelDebugOptions`] bounds interpretation and exposes canonical source/access traces. A successful interpreter
//! run proves only the interpreted case, not target support or native performance. [`VerifiedKernel`] rechecks the
//! current immutable body before adapter admission. Preserve its semantic identity when comparing compiler outputs.
//!
//! See the repository's `crates/ryft-core/src/kernels/SUPPORT.md` for the support and qualification matrix, existing
//! runnable examples, upgrade procedure, and the distinction between hardware measurements and unavailable counters.

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
pub mod transforms;
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
pub use operations::{KernelExtension, KernelExtensionMemory, KernelOperation, NoKernelExtension};
pub use scheduling::{KernelInterleaving, KernelSchedulingError};
pub use serialization::KernelSerializationError;
pub use transforms::KernelTransformError;

pub use validation::{
    KernelBoundaryContract, KernelParameterAccess, KernelParameterSummary, KernelReferenceOperation,
    KernelReferenceSummary, KernelSwapLowering, KernelValidationError, validate_kernel_body,
};
