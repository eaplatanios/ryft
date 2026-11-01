use std::hash::Hash;
use std::sync::Arc;

use crate::captures::CaptureConstant;
use crate::compilation::functions::{
    CompilationTracer, CompiledFunction, ExecutableFunction, FlatCompilationProgram, LoweredFunction,
    ReferenceExecution, StagedFunction,
};
use crate::contexts::{Domain, StagingContext};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::ProgramError;
use crate::tracing::DomainTracingContext;

// TODO(eaplatanios): Review from here onwards.

/// Staging request consumed by [`CompilationDomain::stage`]. Callers construct requests with
/// [`CompilationStagingRequest::new`](super::functions::CompilationStagingRequest::new). Refer to the
/// [`CompilationDomain`] documentation for why the domain functions accept request traits rather than concrete types.
pub trait StagingRequest<C: CompilationDomain>: Sized
where
    DomainTracingContext<C, C::Value>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    /// Structured abstract input type.
    type Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>>;

    /// Structured abstract output type.
    type Output: Parameterized<
            C::Type,
            Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
            To<CompilationTracer<C>>: Parameterized<
                CompilationTracer<C>,
                To<C::Type> = Self::Output,
                To<C::Constant> = <Self::Output as Parameterized<C::Type>>::To<C::Constant>,
            >,
        >;

    /// Returns the structured abstract input signature.
    fn input_types(&self) -> &Self::Input;

    /// Returns the fixed compilation options.
    fn options(&self) -> &C::Options;

    /// Replaces the flat input signature while preserving its parameter structure.
    fn replace_input_types(&mut self, input_types: Vec<C::Type>) -> Result<(), C::Error>;

    /// Traces this request and normalizes the effective flat output signature.
    fn trace<NormalizeOutput>(
        self,
        normalize_output_types: NormalizeOutput,
    ) -> Result<StagedFunction<C, Self::Input, Self::Output>, C::Error>
    where
        NormalizeOutput: FnOnce(&C::Options, Vec<C::Type>) -> Result<Vec<C::Type>, C::Error>;
}

/// Lowering request consumed by [`CompilationDomain::lower`] and implemented by [`StagedFunction`]. Refer to the
/// [`CompilationDomain`] documentation for why the domain functions accept request traits rather than concrete types.
pub trait LoweringRequest<C: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>;

    /// Structured abstract output type.
    type Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>;

    /// Returns the staged artifact.
    fn staged(&self) -> &StagedFunction<C, Self::Input, Self::Output>;

    /// Opens runtime captures as leading flat inputs.
    fn lifted_program(&self) -> Result<Arc<FlatCompilationProgram<C>>, ProgramError>;

    /// Assembles the backend-established lowering.
    fn into_lowered(
        self,
        program: C::LoweredProgram,
        output_types: Vec<C::Type>,
    ) -> LoweredFunction<C, Self::Input, Self::Output>;
}

/// Compilation request consumed by [`CompilationDomain::compile`] and implemented by [`LoweredFunction`]. Refer to
/// the [`CompilationDomain`] documentation for why the domain functions accept request traits rather than concrete
/// types.
pub trait CompilationRequest<C: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>;

    /// Structured abstract output type.
    type Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>;

    /// Returns the lowered artifact.
    fn lowered(&self) -> &LoweredFunction<C, Self::Input, Self::Output>;

    /// Assembles the backend-established compiled artifact.
    fn into_compiled(
        self,
        program: Arc<C::CompiledProgram>,
        output_types: Vec<C::Type>,
    ) -> CompiledFunction<C, Self::Input, Self::Output>;
}

/// Execution request consumed by [`CompilationDomain::call`] and implemented by
/// [`CompilationCall`](super::functions::CompilationCall). Refer to the [`CompilationDomain`] documentation for why the
/// domain functions accept request traits rather than concrete types.
pub trait CallRequest<C: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Value>>;

    /// Structured abstract output type.
    type Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Value>>;

    /// Structured runtime output type.
    type RuntimeOutput;

    /// Returns the executable artifact.
    fn executable(&self) -> &ExecutableFunction<C, Self::Input, Self::Output>;

    /// Returns flat public runtime inputs, excluding captures.
    fn inputs(&self) -> &[C::Value];

    /// Consumes the request and prepends runtime captures to its flat inputs.
    fn into_arguments(self) -> Vec<C::Value>;

    /// Reconstructs structured runtime outputs.
    fn reconstruct(
        executable: &ExecutableFunction<C, Self::Input, Self::Output>,
        outputs: Vec<C::Value>,
    ) -> Result<Self::RuntimeOutput, C::Error>;
}

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four semantic stages:
///
///   1. [`Self::stage`] traces a closure using options fixed before tracing.
///   2. [`Self::lower`] transforms the staged artifact into [`Self::LoweredProgram`].
///   3. [`Self::compile`] transforms the lowered artifact into [`Self::CompiledProgram`].
///   4. [`Self::call`] executes an [`ExecutableFunction`] against structured runtime inputs.
///
/// The domain performs every stage, while staged, lowered, compiled, and executable functions remain passive values.
/// Each stage is implemented entirely by the backend, including any validation, caching, or runtime conversion that
/// the backend chooses to apply.
///
/// # Request Traits
///
/// The stage functions accept [`StagingRequest`], [`LoweringRequest`], [`CompilationRequest`], and [`CallRequest`]
/// implementations rather than the concrete
/// [`CompilationStagingRequest`](super::functions::CompilationStagingRequest), [`StagedFunction`],
/// [`LoweredFunction`], and [`CompilationCall`](super::functions::CompilationCall) types. Each request trait carries
/// the structured input and output bounds of its stage as bounds on its associated types and exposes only the
/// operations that a backend needs to complete that stage.
///
/// The request traits also work around a limitation of the Rust compiler (as of version 1.99), and so they cannot be
/// replaced by explicit bounds on the concrete types. When a trait function bounds an associated type of one of its
/// generic parameters through a trait whose arguments mention an associated type of `Self` (e.g.,
/// `Input: Parameterized<Self::Type, Family: ParameterizedFamily<Self::Constant>>`), implementations of that function
/// for concrete domains fail to compile with an unsatisfied `Input: Parameterized<...>` bound (E0277), even when they
/// repeat the bound verbatim. Bounds on the associated types of a request trait are not affected.
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
        Request: StagingRequest<Self>,
        Self: Sized,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>;

    /// Lowers a staged function into the program representation of this backend (i.e., [`Self::LoweredProgram`]).
    /// The lowered program must capture every option, compiler, target, and topology property that affects
    /// compilation, because caching backends derive the identity of compiled programs from the lowered program alone.
    fn lower<Request>(
        &self,
        staged: Request,
    ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: LoweringRequest<Self>;

    /// Compiles a lowered function into an executable program of this backend (i.e., [`Self::CompiledProgram`]).
    /// Caching backends usually implement this function via
    /// [`CompilationContext::compile_request`](super::caching::CompilationContext::compile_request).
    fn compile<Request>(
        &self,
        lowered: Request,
    ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: CompilationRequest<Self>;

    /// Executes a compiled function on structured runtime inputs and returns its structured outputs. Callers
    /// typically use [`ExecutableFunction::call`] rather than calling this function directly.
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
    /// Executes a compiled function that binds external mutable references and returns its structured outputs
    /// together with the completion of the whole invocation, without waiting for that completion.
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
