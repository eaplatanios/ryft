use std::hash::Hash;
use std::sync::Arc;

use crate::captures::CaptureConstant;
use crate::compilation::functions::{ExecutableFunction, ReferenceExecution, StagedProgram};
use crate::contexts::Domain;
use crate::parameters::Parameterized;
use crate::programs::ProgramError;

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four phases:
///
///   1. [`StagedProgram::stage`] traces a flat closure into a [`StagedProgram`], typically through
///      [`StagedFunction::stage`](crate::StagedFunction::stage) or [`stage`](crate::stage) for typed callers.
///   2. [`Self::lower`] lowers a [`StagedProgram`] into a [`Self::LoweredProgram`], typically through
///      [`StagedFunction::lower`](crate::StagedFunction::lower).
///   3. [`Self::compile`] compiles a [`Self::LoweredProgram`] into a [`Self::CompiledProgram`], typically through
///      [`LoweredFunction::compile`](crate::LoweredFunction::compile).
///   4. [`Self::execute`] executes a [`Self::CompiledProgram`] on flat runtime arguments, typically through
///      [`ExecutableFunction::call`].
///
/// Core owns staging and everything that depends on the structured input and output types of a function. It traces
/// closures, lifts runtime captures into leading program inputs, checks input counts and compiled output types, and
/// reconstructs structured outputs. Backends implement [`Self::lower`], [`Self::compile`], and [`Self::execute`] over
/// flat programs, types, and values, including any backend validation, caching, or runtime conversion. Staging traces
/// exactly the input types that it receives. Backends apply options that change input types before staging (e.g., in
/// [`Self::dispatch_signature`] for retained dispatch), apply options that change output types during lowering, and
/// reject staged functions that they cannot execute during lowering as well.
pub trait CompilationDomain: Domain<Constant: CaptureConstant> + Clone {
    /// Backend-defined key for one input specialization of a retained Just-In-Time (JIT) dispatcher.
    type DispatchKey: Clone + Eq + Hash;

    /// Backend-specific lowered program representation produced by [`Self::lower`].
    type LoweredProgram;

    /// Backend-specific compiled program representation produced by [`Self::compile`].
    type CompiledProgram;

    /// Backend-specific compilation options type. Meshes, sharding and layout overrides, donation declarations,
    /// compiler flags, etc., are all represented as part of this type.
    type Options;

    /// Backend-specific error type. Staging and call boundary errors flow through it as [`ProgramError`]s,
    /// which is why it requires [`From<ProgramError>`].
    type Error: std::error::Error + From<ProgramError>;

    /// Derives the effective staged input types and their dispatch key from one runtime input signature. Most domains
    /// return the input types unchanged and use them as the key. Backends may instead apply a semantics preserving
    /// dispatch policy such as bounded shape bucketing. The returned key must compare equal exactly when the returned
    /// effective types are interchangeable for tracing, lowering, compilation, and calls. The effective types are
    /// returned as a shared slice so that key construction on the hot path of retained dispatch never clones the type
    /// vector deeply. Domains whose key embeds the effective types (e.g., exact dispatch) share one allocation between
    /// the key and the returned signature, and the caller clones individual types only on a specialization miss, when
    /// it prepares the staging request.
    fn dispatch_signature(
        &self,
        input_types: Vec<Self::Type>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Self::Type]>), Self::Error>;

    /// Lowers `staged` into the lowered program representation of this backend and returns it together with its
    /// effective flat output types. Callers typically use [`StagedFunction::lower`](crate::StagedFunction::lower)
    /// rather than calling this function directly. The lowered program must capture every option, compiler, target,
    /// and topology property that affects compilation, because caching backends derive the identity of compiled
    /// programs from the lowered program alone.
    ///
    /// # Parameters
    ///
    ///   - `staged`: Owned staging artifact borrowed for lowering. Use [`StagedProgram::lifted_program`] to obtain the
    ///     source program with captures lifted into leading inputs.
    fn lower(&self, staged: &StagedProgram<Self>) -> Result<(Self::LoweredProgram, Vec<Self::Type>), Self::Error>;

    /// Compiles a lowered program into an executable program of this backend and returns it together with its effective
    /// flat output types. Callers use [`LoweredFunction::compile`](crate::LoweredFunction::compile) rather than calling
    /// this function directly, which also checks that the returned output types refine the lowered ones. Caching
    /// backends usually implement this function via
    /// [`CompilationContext::compile`](crate::CompilationContext::compile).
    fn compile(
        &self,
        program: &Self::LoweredProgram,
    ) -> Result<(Arc<Self::CompiledProgram>, Vec<Self::Type>), Self::Error>;

    /// Executes a compiled program on flat runtime `arguments` and returns its flat outputs. Callers typically use
    /// [`ExecutableFunction::call`] rather than calling this function directly.
    ///
    /// # Parameters
    ///
    ///   - `program`: Compiled program to execute.
    ///   - `input_types`: Flat declared public input types of the compiled function, excluding captures.
    ///   - `arguments`: Runtime captures of the compiled function followed by exactly one public input per entry
    ///     of `input_types`, in the `[captures..., public inputs...]` order of lifted programs.
    fn execute(
        &self,
        program: &Self::CompiledProgram,
        input_types: &[Self::Type],
        arguments: Vec<Self::Value>,
    ) -> Result<Vec<Self::Value>, Self::Error>;
}

// TODO(eaplatanios): Review from here onwards.

/// Optional execution capability for compiled functions that bind external mutable references. Implementations must
/// keep hidden state results out of the outputs that they return and preserve their reference transaction guarantees
/// even when the public output has no leaves or fails reconstruction.
pub trait StatefulCompilationDomain: CompilationDomain {
    /// Executes a compiled program that binds external mutable references and returns its flat outputs together with
    /// the completion of the whole invocation, without waiting for that completion. The parameters are the same as for
    /// [`CompilationDomain::execute`]. Callers use [`ExecutableFunction::call_statefully_async`] rather than calling
    /// this function directly.
    fn execute_statefully_async(
        &self,
        program: &Self::CompiledProgram,
        input_types: &[Self::Type],
        arguments: Vec<Self::Value>,
    ) -> ReferenceExecution<Vec<Self::Value>, Self::Error>;
}

/// Optional capability for caching backend-compiled programs in a [`CompilationContext`](crate::CompilationContext).
///
/// Cache identity is derived from the lowered program alone. [`CompilationDomain::lower`] folds every option, compiler,
/// target, and topology property that affects compilation into [`CompilationDomain::LoweredProgram`], so key equality
/// always means that compiled programs are interchangeable.
pub trait CompilationCacheDomain: CompilationDomain {
    /// Exact structural cache key for interchangeable compiled programs.
    type CacheKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Constructs the exact cache key for `program`.
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error>;

    /// Returns stable canonical bytes for persistent cache identity, or [`None`] when this domain does not support
    /// persistent executable caching.
    ///
    /// These bytes must remain stable across processes that may share the cache and must cover the complete
    /// computation, options, compiler and backend versions, and target topology. Core hashes them only to obtain safe
    /// filenames and does not add any missing semantic state.
    #[inline]
    fn persistent_cache_key(&self, _key: &Self::CacheKey) -> Option<Vec<u8>> {
        None
    }

    /// Serializes a compiled program for persistent caching and artifact exchange. `Ok(None)` means unsupported.
    #[inline]
    fn serialize_program(&self, _program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(None)
    }

    /// Deserializes a persistent cache or artifact exchange payload. `Ok(None)` means unsupported or incompatible.
    #[inline]
    fn deserialize_program(&self, _bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        Ok(None)
    }
}

/// Optional capability for inspecting a compiled program without recompiling it.
///
/// Analysis is deliberately separate from [`CompilationDomain`] because not every backend or runtime plugin can
/// expose compiler cost and memory information. Implementations should cache immutable backend results when querying
/// them is expensive. Calling this function must never trigger compilation.
pub trait AnalyzableCompilationDomain: CompilationDomain {
    /// Backend-owned typed analysis report.
    type Analysis;

    /// Analyzes the compiled program carried by an [`ExecutableFunction`] without recompiling it.
    fn analyze<Input: Parameterized<Self::Type>, Output: Parameterized<Self::Type>>(
        &self,
        executable_function: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error>
    where
        Self: Sized;
}
