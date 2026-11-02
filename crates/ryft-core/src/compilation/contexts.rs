use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::captures::CaptureConstant;
use crate::compilation::functions::{
    CompilationTracer, CompiledFunctionDispatcher, DEFAULT_JIT_CACHE_CAPACITY, ExecutableFunction, StagedFunction,
    StagedProgram, StatefulExecution,
};
use crate::contexts::{Domain, StagingContext};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, Value};
use crate::tracing::DomainTracingContext;

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four phases:
///
///   1. [`StagedProgram::stage`] traces a flat closure into a [`StagedProgram`], typically through
///      [`StagedFunction::stage`] or [`Compile::stage`] for typed callers.
///   2. [`Self::lower`] lowers a [`StagedProgram`] into a [`Self::LoweredProgram`], typically through
///      [`StagedFunction::lower`](crate::StagedFunction::lower).
///   3. [`Self::compile`] compiles a [`Self::LoweredProgram`] into a [`Self::CompiledProgram`], typically
///      through [`LoweredFunction::compile`](crate::LoweredFunction::compile).
///   4. [`Self::execute`] executes a [`Self::CompiledProgram`] on flat runtime inputs, typically through
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

    /// Executes a compiled program on flat runtime `inputs` and returns its flat outputs. Callers typically use
    /// [`ExecutableFunction::call`] rather than calling this function directly. The backend retains any declaration
    /// metadata it needs in [`Self::CompiledProgram`] and validates inputs according to its execution contract.
    ///
    /// # Parameters
    ///
    ///   - `program`: Compiled program to execute.
    ///   - `inputs`: Runtime captures of the compiled function followed by its public inputs, in the
    ///     `[captures..., public inputs...]` order of lifted programs.
    fn execute(
        &self,
        program: &Self::CompiledProgram,
        inputs: Vec<Self::Value>,
    ) -> Result<Vec<Self::Value>, Self::Error>;
}

/// Optional execution capability for compiled functions that bind external mutable references. Implementations must
/// keep hidden state results out of the outputs that they return and preserve their reference transaction guarantees
/// even when the public output has no leaves or fails reconstruction.
pub trait StatefulCompilationDomain: CompilationDomain {
    /// Executes a compiled program that binds external mutable references and returns its flat outputs together with
    /// the completion of the whole invocation, _without_ waiting for that completion. The parameters are the same as
    /// for [`CompilationDomain::execute`]. Callers use [`ExecutableFunction::call_statefully_async`] rather than
    /// calling this function directly.
    fn execute_statefully(
        &self,
        program: &Self::CompiledProgram,
        inputs: Vec<Self::Value>,
    ) -> StatefulExecution<Vec<Self::Value>, Self::Error>;
}

/// Optional capability for caching backend-compiled programs in a [`CompilationContext`](crate::CompilationContext).
/// Cache identity is derived from the lowered program alone. [`CompilationDomain::lower`] folds every option, compiler,
/// target, and topology property that affects compilation into [`CompilationDomain::LoweredProgram`], so key equality
/// always means that compiled programs are interchangeable.
///
/// In-memory caching and coordination of concurrent compilations for the same key require only
/// [`Self::compilation_key`]. Persistent caching and artifact exchange additionally use the optional
/// stable [`Self::persistent_cache_key`] and serialization hooks (i.e., [`Self::serialize_program`] and
/// [`Self::deserialize_program`]). Their availability may depend on the active runtime plugin or the
/// particular compiled program, so implementing this trait does not guarantee serialization support.
pub trait CompilationCacheDomain: CompilationDomain {
    /// Exact structural cache key for interchangeable compiled programs.
    type CacheKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Constructs the exact compilation cache key for `program`.
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error>;

    /// Returns stable canonical bytes for persistent compilation cache identity, or [`None`] when this domain does not
    /// support persistent executable caching. These bytes must remain stable across processes that may share the cache
    /// and must cover the complete computation, options, compiler and backend versions, and target topology. Core
    /// hashes them only to obtain safe filenames and does not add any missing semantic state.
    #[inline]
    fn persistent_cache_key(&self, key: &Self::CacheKey) -> Option<Vec<u8>> {
        let _ = key;
        None
    }

    /// Serializes a compiled program for persistent caching and artifact exchange. Returns `Ok(Some(bytes))` on success
    /// or `Ok(None)` when serialization is unavailable for this backend instance or compiled program (e.g., when the
    /// active runtime plugin does not implement executable serialization). This runtime capability check allows a
    /// domain to retain in-memory caching even when it cannot serialize executables.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization fails. Unavailable serialization is reported as `Ok(None)` instead.
    #[inline]
    fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        let _ = program;
        Ok(None)
    }

    /// Deserializes a persistent cache or artifact exchange payload. Returns `Ok(Some(program))` on success or
    /// `Ok(None)` when deserialization is unavailable for this backend instance or the payload is incompatible with the
    /// current backend, compiler, or target. An incompatible payload is a cache miss rather than a decoding failure.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed payloads or failures while restoring an otherwise compatible compiled program.
    #[inline]
    fn deserialize_program(&self, bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        let _ = bytes;
        Ok(None)
    }
}

/// Optional capability for inspecting a compiled program without recompiling it. Analysis is deliberately separate from
/// [`CompilationDomain`] because not every backend or runtime plugin can expose compiler cost and memory information.
/// Implementations should cache immutable backend results when querying them is expensive. Calling this function must
/// never trigger compilation.
pub trait AnalyzableCompilationDomain: CompilationDomain {
    /// Backend-owned typed analysis report type.
    type Analysis;

    /// Analyzes the compiled program carried by the provided [`ExecutableFunction`] without recompiling it.
    fn analyze<Input: Parameterized<Self::Type>, Output: Parameterized<Self::Type>>(
        &self,
        executable: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error>
    where
        Self: Sized;
}

/// Typed staging and retained compilation through an explicit [`CompilationDomain`]. Retained dispatchers bind to
/// this domain when they are constructed. The global [`stage`] and [`jit`] functions infer their domain from runtime
/// input values instead.
pub trait Compile: CompilationDomain {
    /// Stages an infallible capture-free function for this domain with the given input types and options. Staging
    /// depends only on the types of this domain, so `self` only selects the domain type. Use [`StagedFunction::stage`]
    /// when the function is fallible or closes over runtime captures.
    #[inline]
    fn stage<Input, Output, F>(
        &self,
        function: F,
        input_types: Input,
        options: Self::Options,
    ) -> Result<StagedFunction<Self, Input, Output>, Self::Error>
    where
        Input: Parameterized<Self::Type>,
        Input::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Output: Parameterized<Self::Type>,
        Output::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Output::To<CompilationTracer<Self>>: Parameterized<
                CompilationTracer<Self>,
                To<Self::Type> = Output,
                To<Self::Constant> = Output::To<Self::Constant>,
            >,
        F: FnOnce(Input::To<CompilationTracer<Self>>) -> Output::To<CompilationTracer<Self>>,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>,
    {
        StagedFunction::stage(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, options)
    }

    /// Constructs a retained Just-In-Time (JIT) compilation dispatcher for `function` through this domain with default
    /// options and cache capacity. The function is fallible during tracing, and an infallible function returns its
    /// outputs wrapped in [`Ok`]. Staging, lowering, and compilation happen on the first call for each specialization.
    /// Refer to [`CompiledFunctionDispatcher`] for the dispatch semantics.
    #[inline]
    fn jit<Static, Input, Output, F>(&self, function: F) -> CompiledFunctionDispatcher<Self, F, Static, Input, Output>
    where
        Self::Type: Eq + Hash,
        Self::Options: Default,
        Static: Clone + Debug + Eq + Hash,
        Input: Parameterized<Self::Type>,
        Input::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Input::ParameterStructure: Eq + Hash,
        Output: Parameterized<Self::Type>,
        Output::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        F: Fn(Static, Input::To<CompilationTracer<Self>>) -> Result<Output::To<CompilationTracer<Self>>, Self::Error>,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>,
    {
        self.jit_with_options(function, Self::Options::default())
    }

    /// Constructs a retained Just-In-Time (JIT) compilation dispatcher for `function` through this domain with explicit
    /// options and default cache capacity. The function is fallible during tracing, and an infallible function returns
    /// its outputs wrapped in [`Ok`]. Staging, lowering, and compilation happen on the first call for each
    /// specialization. Refer to [`CompiledFunctionDispatcher`] for the dispatch semantics.
    #[inline]
    fn jit_with_options<Static, Input, Output, F>(
        &self,
        function: F,
        options: Self::Options,
    ) -> CompiledFunctionDispatcher<Self, F, Static, Input, Output>
    where
        Self::Type: Eq + Hash,
        Static: Clone + Debug + Eq + Hash,
        Input: Parameterized<Self::Type>,
        Input::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Input::ParameterStructure: Eq + Hash,
        Output: Parameterized<Self::Type>,
        Output::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        F: Fn(Static, Input::To<CompilationTracer<Self>>) -> Result<Output::To<CompilationTracer<Self>>, Self::Error>,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>,
    {
        self.jit_with_options_and_capacity(function, options, DEFAULT_JIT_CACHE_CAPACITY)
    }

    /// Constructs a retained Just-In-Time (JIT) compilation dispatcher for `function` through this domain with explicit
    /// options that retains at most `capacity` compiled specializations. The function is fallible during tracing, and
    /// an infallible function returns its outputs wrapped in [`Ok`]. Staging, lowering, and compilation happen on the
    /// first call for each specialization. Refer to [`CompiledFunctionDispatcher`] for the dispatch semantics.
    #[inline]
    fn jit_with_options_and_capacity<Static, Input, Output, F>(
        &self,
        function: F,
        options: Self::Options,
        capacity: usize,
    ) -> CompiledFunctionDispatcher<Self, F, Static, Input, Output>
    where
        Self::Type: Eq + Hash,
        Static: Clone + Debug + Eq + Hash,
        Input: Parameterized<Self::Type>,
        Input::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Input::ParameterStructure: Eq + Hash,
        Output: Parameterized<Self::Type>,
        Output::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        F: Fn(Static, Input::To<CompilationTracer<Self>>) -> Result<Output::To<CompilationTracer<Self>>, Self::Error>,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>,
    {
        CompiledFunctionDispatcher::new(self, function, options, capacity)
    }
}

impl<C: CompilationDomain> Compile for C {}

/// Stages an infallible capture-free function using the domain and abstract types of runtime `inputs`. The inputs are
/// used only to recover their domain and signature, and the function runs on fresh symbolic tracers. The first input
/// leaf supplies the domain through [`Value::domain`]. Use [`Compile::stage`] for abstract input types or inputs with
/// no leaves, and [`StagedFunction::stage`] for fallible functions or explicit runtime captures.
///
/// # Errors
///
/// Returns an error if the inputs have no leaves or staging fails.
pub fn stage<V, Input, Output, F>(
    function: F,
    inputs: Input,
    options: <V::Domain as CompilationDomain>::Options,
) -> Result<StagedFunction<V::Domain, Input::To<V::Type>, Output>, <V::Domain as CompilationDomain>::Error>
where
    V: Value<Domain: CompilationDomain>,
    Input: Parameterized<V>,
    Input::Family: ParameterizedFamily<V::Type>
        + ParameterizedFamily<<V::Domain as Domain>::Constant>
        + ParameterizedFamily<CompilationTracer<V::Domain>>,
    Input::To<V::Type>:
        Parameterized<V::Type, To<CompilationTracer<V::Domain>> = Input::To<CompilationTracer<V::Domain>>>,
    Output: Parameterized<V::Type>,
    Output::Family:
        ParameterizedFamily<<V::Domain as Domain>::Constant> + ParameterizedFamily<CompilationTracer<V::Domain>>,
    Output::To<CompilationTracer<V::Domain>>: Parameterized<
            CompilationTracer<V::Domain>,
            To<V::Type> = Output,
            To<<V::Domain as Domain>::Constant> = Output::To<<V::Domain as Domain>::Constant>,
        >,
    F: FnOnce(Input::To<CompilationTracer<V::Domain>>) -> Output::To<CompilationTracer<V::Domain>>,
    DomainTracingContext<V::Domain, V>: StagingContext<
            Type = V::Type,
            Constant = <V::Domain as Domain>::Constant,
            Operation = <V::Domain as Domain>::Operation,
        >,
{
    let domain = inputs.parameters().next().map(Value::domain).ok_or_else(|| ProgramError::InvalidArgument {
        message: "staging requires an input leaf to infer a domain; use an explicit domain for empty inputs"
            .to_string(),
    })?;
    let input_types = Input::To::<V::Type>::from_parameters(
        inputs.parameter_structure(),
        inputs.parameters().map(|input| input.r#type().into_owned()),
    )
    .map_err(ProgramError::from)?;
    domain.stage(function, input_types, options)
}

/// Constructs a retained Just-In-Time (JIT) compilation dispatcher for `function` with default options and cache
/// capacity, inferring its domain on the first nonempty call. `function` is fallible so that it can reject its inputs
/// while it is traced, and an infallible function returns its outputs wrapped in [`Ok`]. Staging, lowering, and
/// compilation happen on the first call for each specialization. The first input leaf supplies the fixed domain
/// through [`Value::domain`], even if that first call later fails. Use [`Compile::jit`] to select a domain explicitly
/// or support a nullary function. Refer to [`CompiledFunctionDispatcher`] for shared domain selection and dispatch
/// semantics.
#[inline]
pub fn jit<C, Static, Input, Output, F>(function: F) -> CompiledFunctionDispatcher<C, F, Static, Input, Output>
where
    C: CompilationDomain<Options: Default>,
    C::Type: Eq + Hash,
    C::Value: Value<Domain = C>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
    F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
    DomainTracingContext<C, C::Value>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    jit_with_options(function, C::Options::default())
}

/// Constructs a retained Just-In-Time (JIT) compilation dispatcher for `function` with explicit options and the default
/// cache capacity, inferring its domain on the first nonempty call. `function` is fallible so that it can reject its
/// inputs while it is traced, and an infallible function returns its outputs wrapped in [`Ok`]. Staging, lowering, and
/// compilation happen on the first call for each specialization. The first input leaf supplies the fixed domain through
/// [`Value::domain`], even if that first call later fails. Use [`Compile::jit_with_options`] to select a domain
/// explicitly or support a nullary function. Refer to [`CompiledFunctionDispatcher`] for shared domain selection
/// and dispatch semantics.
#[inline]
pub fn jit_with_options<C, Static, Input, Output, F>(
    function: F,
    options: C::Options,
) -> CompiledFunctionDispatcher<C, F, Static, Input, Output>
where
    C: CompilationDomain,
    C::Type: Eq + Hash,
    C::Value: Value<Domain = C>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
    F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
    DomainTracingContext<C, C::Value>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    jit_with_options_and_capacity(function, options, DEFAULT_JIT_CACHE_CAPACITY)
}

/// Constructs a retained Just-In-Time (JIT) compilation dispatcher for `function` with explicit options that retains at
/// most `capacity` compiled specializations and infers its domain on the first nonempty call. `function` is fallible so
/// that it can reject its inputs while it is traced, and an infallible function returns its outputs wrapped in [`Ok`].
/// Staging, lowering, and compilation happen on the first call for each specialization. The first input leaf supplies
/// the fixed domain through [`Value::domain`], even if that first call later fails. Use
/// [`Compile::jit_with_options_and_capacity`] to select a domain explicitly or support a nullary function.
/// Refer to [`CompiledFunctionDispatcher`] for shared domain selection and dispatch semantics.
#[inline]
pub fn jit_with_options_and_capacity<C, Static, Input, Output, F>(
    function: F,
    options: C::Options,
    capacity: usize,
) -> CompiledFunctionDispatcher<C, F, Static, Input, Output>
where
    C: CompilationDomain,
    C::Type: Eq + Hash,
    C::Value: Value<Domain = C>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant> + ParameterizedFamily<CompilationTracer<C>>,
    F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
    DomainTracingContext<C, C::Value>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    CompiledFunctionDispatcher::inferred(function, options, capacity)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::compilation::functions::tests::{
        CarriedArray, CarriedDomain, CarriedOptions, TestDomain, compile_from_one_call_site,
    };
    use crate::compilation::functions::{CompilationTracer, CompiledFunctionDispatcher};
    use crate::operations::NegOperation;

    use super::*;

    #[test]
    fn test_compile_stage() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let staged: StagedFunction<CarriedDomain, ArrayType, ArrayType> = domain
            .stage(
                |input: CompilationTracer<CarriedDomain>| input.unary(NegOperation::new()),
                ArrayType::scalar(DataType::F64),
                options.clone(),
            )
            .unwrap();
        assert_eq!(staged.input_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(staged.output_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(staged.options(), &options);
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = neg %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(domain.compilation_count(), 0);
    }

    #[test]
    fn test_compile_stage_distinguishes_computations_from_one_call_site() {
        // Staging keys nothing on the Rust call site, so closures from one call site that capture different host
        // values produce distinct programs, which the compilation cache keeps apart.
        let domain = TestDomain::new();
        let identity = compile_from_one_call_site(&domain, false);
        let negate = compile_from_one_call_site(&domain, true);
        assert_eq!(
            identity.executable_function().call(&domain, Array::scalar(3.0).unwrap()),
            Ok(Array::scalar(3.0).unwrap())
        );
        assert_eq!(
            negate.executable_function().call(&domain, Array::scalar(3.0).unwrap()),
            Ok(Array::scalar(-3.0).unwrap())
        );
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compile_jit() {
        let domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            domain.jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.domain(), Some(&domain));
        assert_eq!(function.options(), &CarriedOptions::default());
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(domain.compilation_count(), 0);
        assert_eq!(function.call((), domain.scalar(2.0)), Ok(domain.scalar(-2.0)));
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compile_jit_with_options() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            domain.jit_with_options(|(), input: CompilationTracer<CarriedDomain>| Ok(input), options.clone());
        assert_eq!(function.domain(), Some(&domain));
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.call((), domain.scalar(2.0)), Ok(domain.scalar(2.0)));
    }

    #[test]
    fn test_compile_jit_with_options_and_capacity() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, _, bool, ArrayType, ArrayType> = domain
            .jit_with_options_and_capacity(
                |negate, input: CompilationTracer<CarriedDomain>| {
                    Ok(if negate { input.unary(NegOperation::new()) } else { input })
                },
                options.clone(),
                1,
            );
        assert_eq!(function.domain(), Some(&domain));
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), 1);

        // Alternating static values evict the only retained specialization, so every call traces and compiles again.
        assert_eq!(function.call(false, domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.call(true, domain.scalar(2.0)), Ok(domain.scalar(-2.0)));
        assert_eq!(function.call(false, domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_misses, 3);
        assert_eq!(domain.compilation_count(), 3);
    }

    #[test]
    fn test_stage() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let staged: StagedFunction<CarriedDomain, ArrayType, ArrayType> = stage(
            |input: CompilationTracer<CarriedDomain>| input.unary(NegOperation::new()),
            domain.scalar(2.0),
            options.clone(),
        )
        .unwrap();
        assert_eq!(staged.input_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(staged.output_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(staged.options(), &options);
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = neg %0
                in (%1)
            "}
            .trim_end(),
        );

        // The staged function runs on fresh tracers and is compiled through the domain recovered from its inputs.
        assert_eq!(domain.compilation_count(), 0);
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        assert_eq!(compiled.executable_function().call(&domain, domain.scalar(3.0)), Ok(domain.scalar(-3.0)));
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_stage_rejects_inputs_without_leaves() {
        assert!(matches!(
            stage::<CarriedArray, _, Vec<ArrayType>, _>(
                |inputs| inputs,
                Vec::<CarriedArray>::new(),
                CarriedOptions::default(),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "staging requires an input leaf to infer a domain; use an explicit domain for empty \
                               inputs",
        ));
    }

    #[test]
    fn test_jit() {
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), Vec<ArrayType>, ArrayType> =
            jit(|(), inputs: Vec<CompilationTracer<CarriedDomain>>| Ok(inputs[0].unary(NegOperation::new())));
        assert_eq!(function.domain(), None);
        assert_eq!(function.options(), &CarriedOptions::default());
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);

        // The first input leaf of the first call selects the domain, even when other leaves carry another domain.
        assert_eq!(
            function.call((), vec![first_domain.scalar(2.0), second_domain.scalar(3.0)]),
            Ok(first_domain.scalar(-2.0)),
        );
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(first_domain.compilation_count(), 1);
        assert_eq!(second_domain.compilation_count(), 0);
    }

    #[test]
    fn test_jit_with_options() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit_with_options(|(), input: CompilationTracer<CarriedDomain>| Ok(input), options.clone());
        assert_eq!(function.domain(), None);
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.call((), domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.domain(), Some(&domain));
    }

    #[test]
    fn test_jit_with_options_and_capacity() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit_with_options_and_capacity(|(), input: CompilationTracer<CarriedDomain>| Ok(input), options.clone(), 1);
        assert_eq!(function.domain(), None);
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), 1);
        assert_eq!(function.call((), domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.domain(), Some(&domain));
    }
}
