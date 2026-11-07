use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::captures::CaptureConstant;
use crate::compilation::functions::{
    CompilationTracer, CompiledFunctionDispatcher, ExecutableFunction, StagedFunction, StagedProgram, StatefulExecution,
};
use crate::contexts::{Domain, StagingContext};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, TypedProgram, Value};
use crate::tracing::DomainTracingContext;

/// Default number of compiled specializations retained by one [`CompiledFunctionDispatcher`].
const DEFAULT_JIT_CACHE_CAPACITY: usize = 256;

/// Inputs used to derive a [`CompilationDomain::dispatch_signature`]. [`Self::Types`] requests a staged call within an
/// enclosing computation. [`Self::Values`] requests an independently dispatched executable and makes runtime metadata
/// available to its specialization policy. The iterator and its values have separate borrow lifetimes so a short-lived
/// iterator can visit longer-lived values.
pub enum DispatchInputs<'o, 'v, V: Value + 'v> {
    /// Abstract input types of a staged call. Executable policies such as input-bound bucketing do not apply.
    Types(Vec<V::Type>),

    /// Runtime inputs of an independently dispatched executable and their effective types.
    Values {
        /// Runtime input values, in the same order and number as `types`.
        values: &'o mut dyn Iterator<Item = &'v V>,

        /// Effective types validated by the caller against `values`, including any
        /// [`declared staging types`](CompiledFunctionDispatcher::with_staged_input_types). A concrete extent may
        /// therefore be represented by a symbolic dimension here, which extracting raw value types cannot recover.
        types: Vec<V::Type>,
    },
}

impl<V: Value> DispatchInputs<'_, '_, V> {
    /// Consumes this request and returns its effective input types, discarding any runtime value iterator.
    /// Backends whose dispatch policies depend only on types can use this function for both request variants.
    #[inline]
    pub fn into_types(self) -> Vec<V::Type> {
        match self {
            Self::Types(types) | Self::Values { types, .. } => types,
        }
    }
}

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four phases:
///
///   1. [`StagedProgram::stage`] traces a flat closure into a [`StagedProgram`], typically through
///      [`StagedFunction::stage`] or [`Compile::stage`] for typed callers.
///   2. [`Self::lower`] lowers a [`StagedProgram`] into a [`Self::LoweredProgram`], typically through
///      [`StagedFunction::lower`](StagedFunction::lower).
///   3. [`Self::compile`] compiles a [`Self::LoweredProgram`] into a [`Self::CompiledProgram`], typically
///      through [`LoweredFunction::compile`](crate::LoweredFunction::compile).
///   4. [`Self::execute`] executes a [`Self::CompiledProgram`] on flat runtime inputs, typically through
///      [`ExecutableFunction::execute`].
///
/// Core owns staging and everything that depends on the structured input and output types of a function. It traces
/// closures, lifts runtime captures into leading program inputs, checks input counts and compiled output types, and
/// reconstructs structured outputs. Backends implement [`Self::lower`], [`Self::compile`], and [`Self::execute`] over
/// flat programs, types, and values, including any backend validation, caching, or runtime conversion. Staging traces
/// exactly the input types that it receives. Backends apply options that change input types before staging (i.e., in
/// [`Self::dispatch_signature`] for retained dispatch), apply options that change output types during lowering, and
/// reject staged functions that they cannot execute during lowering as well.
pub trait CompilationDomain: Domain<Constant: CaptureConstant> + Clone {
    /// Backend-defined key for one input specialization of a retained Just-In-Time (JIT) dispatcher.
    type DispatchKey: Clone + Eq + Hash;

    /// Backend-specific lowered program representation produced by [`Self::lower`].
    type LoweredProgram: TypedProgram<Type = Self::Type>;

    /// Backend-specific compiled program representation produced by [`Self::compile`].
    type CompiledProgram: TypedProgram<Type = Self::Type>;

    /// Backend-specific compilation options type. Meshes, sharding and layout overrides, donation declarations,
    /// compiler flags, etc., are all represented as part of this type.
    type Options;

    /// Backend-specific error type. Staging and call boundary errors flow through it as [`ProgramError`]s,
    /// which is why it requires [`From<ProgramError>`].
    type Error: std::error::Error + From<ProgramError>;

    /// Derives the dispatch key, effective staged input types, and optional specialized options for `inputs`.
    /// [`DispatchInputs::Types`] requests a staged call that is lowered with its enclosing computation, so its
    /// signature applies semantic declarations from `options` but no executable policy. Its returned dispatch key
    /// is unused because staged specializations are keyed by their effective types. [`DispatchInputs::Values`]
    /// requests an independently dispatched executable, so it may additionally apply a semantics preserving policy
    /// such as input-bound bucketing or specialize on runtime metadata absent from the types, such as placement over
    /// auto mesh axes. Such metadata must participate in both the key and the specialized options used for lowering.
    ///
    /// Returning [`None`] for the specialized options uses `options` unchanged. Equal keys must identify types
    /// and options that are interchangeable for tracing, lowering, compilation, and calls. The shared type slice lets
    /// exact dispatch share one allocation with its key, avoiding deep type clones on specialization hits. Most
    /// backends preserve the requested types and return no specialized options. For type requests, equal effective
    /// types must also imply interchangeable specialized options because staged calls are keyed by those types.
    ///
    /// This function needs no domain instance because staging a call does not require a retained dispatcher to have
    /// selected its execution domain.
    ///
    /// # Errors
    ///
    /// Returns an error when `options` are incompatible with `inputs` or their dispatch policy.
    fn dispatch_signature(
        inputs: DispatchInputs<'_, '_, Self::Value>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Self::Type]>, Option<Self::Options>), Self::Error>;

    /// Lowers `staged` into the lowered program representation of this backend. Callers typically use
    /// [`StagedFunction::lower`] rather than calling this function directly. The lowered program must capture every
    /// option, compiler, target, and topology property that affects compilation, because caching backends derive the
    /// identity of compiled programs from the lowered program alone.
    ///
    /// # Parameters
    ///
    ///   - `staged`: Owned staging artifact borrowed for lowering. Use [`StagedProgram::lifted_program`] to obtain the
    ///     source program with captures lifted into leading inputs.
    fn lower(&self, staged: &StagedProgram<Self>) -> Result<Self::LoweredProgram, Self::Error>;

    /// Compiles a lowered program into an executable program of this backend, and returns it together with the runtime
    /// values of the hidden captures that lowering introduced in addition to the captures of its staged source. A
    /// backend introduces such captures for runtime values that the staged program refers to through lifetime-free
    /// constants retained outside of it (e.g., by a runtime session), so that the executable depends only on their
    /// types, and most backends return none. A [`CompiledFunction`](crate::CompiledFunction) supplies them after the
    /// source captures on every call. Callers use [`LoweredFunction::compile`](crate::LoweredFunction::compile) rather
    /// than calling this function directly, which also checks that the output types of the compiled program refine
    /// those of the lowered program.
    ///
    /// Caching backends usually obtain the executable through
    /// [`CompilationContext::compile`](crate::CompilationContext::compile). The cache must store only the executable,
    /// which callers with different captured values share, while the hidden capture values belong to `program` and are
    /// resolved for every request, including cache hits.
    fn compile(
        &self,
        program: &Self::LoweredProgram,
    ) -> Result<(Arc<Self::CompiledProgram>, Vec<Self::Value>), Self::Error>;

    /// Executes a compiled program on flat runtime `inputs` and returns its flat outputs. Callers typically use
    /// [`ExecutableFunction::execute`] rather than calling this function directly. The backend retains any declaration
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
    /// for [`CompilationDomain::execute`]. Callers use [`ExecutableFunction::execute_statefully`] rather than calling
    /// this function directly.
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
/// [`Self::persistent_cache_key`] and serialization hooks (i.e., [`Self::serialize_program`] and
/// [`Self::deserialize_program`]). Their availability may depend on the active runtime plugin or the
/// particular compiled program, so implementing this trait does not guarantee serialization support.
pub trait CompilationCacheDomain: CompilationDomain {
    /// Exact structural cache key for interchangeable compiled programs.
    type CacheKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Constructs the exact compilation cache key for `program`. Equal keys must mean that compiled programs are
    /// interchangeable, so that [`Self::validate_cached_program`] can only reject a program from the in-memory tier
    /// when this function violates that contract.
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error>;

    /// Compiles `program` without consulting any cache.
    /// [`CompilationContext::compile`](crate::CompilationContext::compile) invokes this function only when no cached or
    /// exchanged program can serve `program`, and callers otherwise go through that function so that every compilation
    /// shares the cache.
    fn compile_uncached(&self, program: &Self::LoweredProgram) -> Result<Self::CompiledProgram, Self::Error>;

    /// Validates that `compiled`, which a [`CompilationContext`](crate::CompilationContext) resolved from its in-memory
    /// tier, another producer, persistent storage, or the artifact exchange instead of compiling it for this request,
    /// can serve the requesting `program`. Backends use it to guard the invocation contract that their calls trust
    /// (e.g., signatures and donation flags) against colliding or corrupted artifacts. A rejected persistent or
    /// exchanged artifact is an incompatible miss that falls through to the remaining sources, while a rejected
    /// in-memory or shared in-flight program fails the compilation with the returned error. The default accepts
    /// every program.
    #[inline]
    fn validate_cached_program(
        &self,
        program: &Self::LoweredProgram,
        compiled: &Self::CompiledProgram,
    ) -> Result<(), Self::Error> {
        let _ = (program, compiled);
        Ok(())
    }

    /// Returns canonical bytes identifying the compilation for persistent caching, or [`None`] when this domain does
    /// not support persistent executable caching. For the same compilation, every process that may share a cache entry
    /// must produce identical bytes. The key must cover the complete computation, options, compiler and backend
    /// versions, and target topology. Core hashes these bytes only to obtain safe filenames and does not add any
    /// missing semantic state.
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
    ) -> Result<Self::Analysis, Self::Error>;
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
    fn jit<Static, Input, Output, F>(&self, function: F) -> CompiledFunctionDispatcher<Self, Static, Input, Output, F>
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
    ) -> CompiledFunctionDispatcher<Self, Static, Input, Output, F>
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
    ) -> CompiledFunctionDispatcher<Self, Static, Input, Output, F>
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
pub fn jit<C, Static, Input, Output, F>(function: F) -> CompiledFunctionDispatcher<C, Static, Input, Output, F>
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
) -> CompiledFunctionDispatcher<C, Static, Input, Output, F>
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
) -> CompiledFunctionDispatcher<C, Static, Input, Output, F>
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

    use crate::arrays::{Array, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::compilation::functions::{CompilationTracer, CompiledFunctionDispatcher};
    use crate::compilation::tests::{
        CarriedArray, CarriedDomain, CarriedOptions, TestDomain, compile_from_one_call_site,
    };
    use crate::operations::NegOperation;

    use super::*;

    #[test]
    fn test_dispatch_inputs_into_types() {
        let input = Array::vector(vec![1f32, 2.0]).unwrap();
        let declared = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "items",
                DimensionBounds::new(0, Some(5)).unwrap(),
            ))]),
        );
        assert_eq!(DispatchInputs::<Array>::Types(vec![declared.clone()]).into_types(), vec![declared.clone()]);

        // Runtime dispatch retains the validated declaration rather than re-extracting a concrete extent of 2.
        let mut values = std::iter::once(&input);
        assert_eq!(
            DispatchInputs::Values { values: &mut values, types: vec![declared.clone()] }.into_types(),
            vec![declared],
        );
        assert_eq!(values.next(), Some(&input));

        assert_eq!(DispatchInputs::<Array>::Types(Vec::new()).into_types(), Vec::<ArrayType>::new());
        assert_eq!(
            DispatchInputs::<Array>::Values { values: &mut std::iter::empty(), types: Vec::new() }.into_types(),
            Vec::<ArrayType>::new(),
        );
    }

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
            identity.executable().execute(&domain, Array::scalar(3.0).unwrap()),
            Ok(Array::scalar(3.0).unwrap())
        );
        assert_eq!(negate.executable().execute(&domain, Array::scalar(3.0).unwrap()), Ok(Array::scalar(-3.0).unwrap()));
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compile_jit() {
        let domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            domain.jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.domain(), Some(&domain));
        assert_eq!(function.options(), &CarriedOptions::default());
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(domain.compilation_count(), 0);
        assert_eq!(function.execute((), domain.scalar(2.0)), Ok(domain.scalar(-2.0)));
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compile_jit_with_options() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            domain.jit_with_options(|(), input: CompilationTracer<CarriedDomain>| Ok(input), options.clone());
        assert_eq!(function.domain(), Some(&domain));
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.execute((), domain.scalar(2.0)), Ok(domain.scalar(2.0)));
    }

    #[test]
    fn test_compile_jit_with_options_and_capacity() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, bool, ArrayType, ArrayType, _> = domain
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
        assert_eq!(function.execute(false, domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.execute(true, domain.scalar(2.0)), Ok(domain.scalar(-2.0)));
        assert_eq!(function.execute(false, domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.executable_specialization_count(), 1);
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
        assert_eq!(compiled.executable().execute(&domain, domain.scalar(3.0)), Ok(domain.scalar(-3.0)));
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
        let function: CompiledFunctionDispatcher<CarriedDomain, (), Vec<ArrayType>, ArrayType, _> =
            jit(|(), inputs: Vec<CompilationTracer<CarriedDomain>>| Ok(inputs[0].unary(NegOperation::new())));
        assert_eq!(function.domain(), None);
        assert_eq!(function.options(), &CarriedOptions::default());
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);

        // The first input leaf of the first call selects the domain, even when other leaves carry another domain.
        assert_eq!(
            function.execute((), vec![first_domain.scalar(2.0), second_domain.scalar(3.0)]),
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
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit_with_options(|(), input: CompilationTracer<CarriedDomain>| Ok(input), options.clone());
        assert_eq!(function.domain(), None);
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.execute((), domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.domain(), Some(&domain));
    }

    #[test]
    fn test_jit_with_options_and_capacity() {
        let domain = CarriedDomain::new();
        let options = CarriedOptions { marker: 1 };
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit_with_options_and_capacity(|(), input: CompilationTracer<CarriedDomain>| Ok(input), options.clone(), 1);
        assert_eq!(function.domain(), None);
        assert_eq!(function.options(), &options);
        assert_eq!(function.cache_capacity(), 1);
        assert_eq!(function.execute((), domain.scalar(2.0)), Ok(domain.scalar(2.0)));
        assert_eq!(function.domain(), Some(&domain));
    }
}
