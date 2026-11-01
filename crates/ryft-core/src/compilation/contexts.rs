use std::hash::Hash;
use std::sync::Arc;

use crate::captures::CaptureConstant;
use crate::compilation::functions::{
    CallRequest, CompileRequest, CompiledFunction, ExecutableFunction, LoweredFunction, LoweringRequest,
    ReferenceExecution, StageRequest, StagedFunction,
};
use crate::contexts::{Domain, StagingContext};
use crate::parameters::Parameterized;
use crate::programs::ProgramError;
use crate::tracing::DomainTracingContext;

// TODO(eaplatanios): Review from here onwards.

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four semantic stages:
///
///   1. [`Self::stage`] traces a closure using options fixed before tracing.
///   2. [`Self::lower`] transforms the staged artifact into [`Self::LoweredProgram`].
///   3. [`Self::compile`] transforms the lowered artifact into [`Self::CompiledProgram`].
///   4. [`Self::call`] executes an [`ExecutableFunction`] against structured runtime inputs.
///
/// The domain is the active service for every transition, while staged, lowered, compiled, and executable artifacts
/// remain passive values. Each function owns its complete backend-specific transition, including any validation,
/// caching, or runtime conversion policy that the backend chooses to apply.
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
    /// effective types are interchangeable for tracing, lowering, compilation, and calls.
    ///
    /// The effective types are returned as a shared slice so that key construction on the hot path of retained dispatch
    /// never clones the type vector deeply. Domains whose key embeds the effective types (e.g., exact dispatch) share
    /// one allocation between the key and the returned signature, and the caller clones individual types only on a
    /// specialization miss, when it prepares the staging request.
    fn dispatch_signature(
        &self,
        input_types: Vec<Self::Type>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Self::Type]>), Self::Error>;

    /// Traces a fallible function with explicit runtime captures and their symbolic references. This is the only
    /// staging primitive. Implementations apply all options that affect tracing to the effective input and output
    /// signatures and retain those same options in the returned artifact.
    fn stage<Request>(
        &self,
        request: Request,
    ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Request: StageRequest<Self>,
        Self: Sized,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>;

    /// Performs the complete backend-specific lowering transition. The returned lowered artifact must be
    /// self-describing. Every option, compiler, target, and topology property that affects compilation must be folded
    /// into [`Self::LoweredProgram`] during this transition, because caching backends derive compiled program identity
    /// from the lowered program alone.
    fn lower<Request>(
        &self,
        staged: Request,
    ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: LoweringRequest<Self>;

    /// Performs the complete backend-specific compilation transition.
    fn compile<Request>(
        &self,
        lowered: Request,
    ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: CompileRequest<Self>;

    /// Performs the complete backend-specific structured execution transition.
    fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>;
}

/// Optional execution capability for compiled functions that bind external mutable references.
///
/// Implementations must keep hidden state results out of [`CallRequest::RuntimeOutput`] and preserve their reference
/// transaction guarantees even when the public output has no leaves or fails reconstruction.
pub trait StatefulCompilationDomain: CompilationDomain {
    /// Performs a stateful structured execution transition that returns a completion bearing result.
    fn call_statefully_async<Request>(
        &self,
        request: Request,
    ) -> ReferenceExecution<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>;

    /// Executes a stateful call and waits for the whole invocation to complete before returning.
    #[inline]
    fn call_statefully<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>,
    {
        self.call_statefully_async(request).r#await()
    }
}

/// Optional capability for caching backend-compiled programs in a
/// [`CompilationContext`](super::caching::CompilationContext).
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
