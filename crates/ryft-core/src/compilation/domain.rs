use std::hash::Hash;

use crate::contexts::Domain;
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError};
use crate::types::Type;

use super::context::CompilationContext;

/// Backend contract for lowering, compiling, and executing staged [`Program`]s.
///
/// Compilation is deliberately split into three semantic stages:
///
/// 1. Ryft traces a Rust closure into a backend-independent [`Program`] expressed in this domain's constant and
///    operation universe.
/// 2. [`Self::lower`] translates the flat program into [`Self::LoweredProgram`], the backend's compiler input.
/// 3. [`Self::compile`] turns that lowering into [`Self::CompiledProgram`], which [`Self::execute`] can invoke.
///
/// Keeping lowered and compiled artifacts backend-owned lets `ryft-core` provide a common lifecycle without exposing
/// StableHLO, PJRT, XLA, or any other compiler-specific representation. It also makes cache identity precise:
/// [`Self::compilation_key`] receives the complete lowering and must account for every option and piece of backend
/// state that can change the executable.
pub trait CompilationDomain: Domain + Clone {
    /// Backend-owned compiler input produced by [`Self::lower`].
    type LoweredProgram;

    /// Backend-owned executable produced by [`Self::compile`]. Core stores this artifact behind an [`Arc`](std::sync::Arc)
    /// so cache hits and compiled-function clones do not require a potentially expensive backend clone.
    type CompiledProgram;

    /// Backend-specific compilation options. Meshes, sharding and layout overrides, donation declarations, compiler
    /// flags, and similar target-specific state belong here rather than in `ryft-core`.
    type Options;

    /// Backend error channel. Staging and call-boundary errors flow through it as [`ProgramError`]s.
    type Error: std::error::Error + From<ProgramError>;

    /// Exact in-memory cache key for one lowered compilation.
    ///
    /// Equality must mean that the corresponding compiled artifacts are interchangeable for execution. In
    /// particular, the key must include the complete computation represented by [`Self::LoweredProgram`] as well as
    /// every compile-relevant option, target property, compiler version, and backend setting. A source location or a
    /// hash of input types is not a sufficient computation identity.
    type CompilationKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Applies options that affect abstract input types before tracing.
    ///
    /// Most domains leave input types unchanged. A sharded backend can override this to apply explicit input sharding
    /// or layout declarations before the closure observes its abstract arguments.
    #[inline]
    fn prepare_input_types<Input: Parameterized<Self::Type>>(
        &self,
        input_types: Input,
        _options: &Self::Options,
    ) -> Result<Input, Self::Error> {
        Ok(input_types)
    }

    /// Validates that options affecting abstract inputs are already represented by a staged signature.
    ///
    /// This hook protects the explicit `stage(...).lower(options)` path: a backend whose input options affect tracing
    /// must reject a staged signature that was not prepared consistently. [`Self::prepare_input_types`] remains the
    /// mechanism used by the combined compile entry point before tracing.
    #[inline]
    fn validate_staged_input_types(
        &self,
        _input_types: &[Self::Type],
        _options: &Self::Options,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Lowers a flat source program into the backend's compiler input.
    ///
    /// `capture_count` identifies the leading inputs that came from a closed program's runtime capture table. Backends
    /// use it to keep capture-only policy (for example, non-donation) separate from public input policy.
    fn lower(
        &self,
        program: &Program<Self::Constant, Self::Operation, Vec<Self::Constant>, Vec<Self::Constant>>,
        capture_count: usize,
        options: &Self::Options,
    ) -> Result<Self::LoweredProgram, Self::Error>;

    /// Returns the effective flat output types produced by `program` after lowering-time option rewrites.
    fn lowered_output_types<'a>(&self, program: &'a Self::LoweredProgram) -> &'a [Self::Type];

    /// Constructs the exact in-memory cache key for `program` and `options`.
    fn compilation_key(
        &self,
        program: &Self::LoweredProgram,
        options: &Self::Options,
    ) -> Result<Self::CompilationKey, Self::Error>;

    /// Compiles one lowered program into a backend executable.
    fn compile(
        &self,
        program: &Self::LoweredProgram,
        options: &Self::Options,
    ) -> Result<Self::CompiledProgram, Self::Error>;

    /// Returns the effective flat output types produced by `program` at execution.
    fn compiled_output_types<'a>(&self, program: &'a Self::CompiledProgram) -> &'a [Self::Type];

    /// Validates one runtime input type against the declared staged type.
    ///
    /// The default accepts exact refinements. Backends that implement an explicit call-boundary conversion, such as
    /// implicit resharding, may accept a broader relation here as long as [`Self::execute`] performs that conversion
    /// before invoking the executable.
    #[inline]
    fn validate_input_type(&self, declared: &Self::Type, actual: &Self::Type) -> Result<(), Self::Error> {
        if declared.is_refined_by(actual) {
            Ok(())
        } else {
            Err(ProgramError::InvalidArgument {
                message: format!("runtime input type {actual} does not refine declared type {declared}"),
            }
            .into())
        }
    }

    /// Validates one runtime output type against the executable's declared output type.
    #[inline]
    fn validate_output_type(&self, declared: &Self::Type, actual: &Self::Type) -> Result<(), Self::Error> {
        if declared.is_refined_by(actual) {
            Ok(())
        } else {
            Err(ProgramError::InvalidArgument {
                message: format!("output type {actual} does not refine declared type {declared}"),
            }
            .into())
        }
    }

    /// Executes `program` against flat runtime inputs and returns flat runtime outputs.
    fn execute(
        &self,
        program: &Self::CompiledProgram,
        inputs: Vec<Self::Value>,
    ) -> Result<Vec<Self::Value>, Self::Error>;

    /// Returns stable canonical bytes for persistent cache identity, or `None` when this domain does not support
    /// persistent executable caching.
    ///
    /// These bytes must remain stable across processes that may share the cache and must cover the complete
    /// computation, options, compiler/backend versions, and target topology. Core hashes them only for filename-safe
    /// addressing; it does not add missing semantic state.
    #[inline]
    fn persistent_cache_key(&self, _key: &Self::CompilationKey) -> Option<Vec<u8>> {
        None
    }

    /// Serializes a compiled program for persistent caching. `Ok(None)` means unsupported.
    #[inline]
    fn serialize_program(&self, _program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(None)
    }

    /// Deserializes a persistent-cache payload. `Ok(None)` means unsupported or incompatible.
    #[inline]
    fn deserialize_program(&self, _bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        Ok(None)
    }

    /// Returns the process-local compilation context used by this domain, when caching is enabled.
    #[inline]
    fn cache(&self) -> Option<&CompilationContext<Self>> {
        None
    }
}
