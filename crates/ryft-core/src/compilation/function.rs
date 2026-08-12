use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::captures::{CapturingContext, ClosedProgram};
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::operations::Constant;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::transforms::{Transform, TransformCache};
use crate::programs::{CalleeRegionDriver, Operation, Program, ProgramError, Typed, Value};
use crate::specialization::SpecializationCacheEntry;
use crate::tracing::{DomainTracingContext, Tracer};

use super::contexts::CompilationDomain;

/// Cache key identifying one specialization of a retained function. Cache reuse is authorized only when
/// all three components below agree. Equal keys must identify calls that can safely share the same staged
/// [`Program`]. Unequal keys remain separate even if they happen to stage structurally equivalent programs.
///
///   - **Static Parameters:** The host values the traced function may branch on, read, or embed as literals. Unequal
///     static parameters can stage arbitrarily different programs, so they must separate specializations. Static
///     parameters must be `Clone + Debug + Eq + Hash`. Runtime arrays and other backend values should be represented
///     as dynamic inputs rather than static parameters.
///   - **Input Structure:** The [`Parameterized::ParameterStructure`] shape of the dynamic function input. Tracing
///     rebuilds the function's argument from this structure, so a function may legitimately branch on container arity.
///     Keying on the structure rather than on the flattened leaves also distinguishes inputs that differ only in
///     _empty_ substructure, which flat leaf paths and flat leaf types cannot see.
///   - **Dispatch Key:** An owner-defined key used to select an interchangeable retained specialization for the dynamic
///     inputs. It may be an exact abstract input signature or a normalized equivalence-class key such as a shape
///     bucket. Equal dispatch keys must guarantee that the retained artifact can safely serve either call; unequal keys
///     conservatively separate specializations. Retained Just-In-Time (JIT) compilation dispatch uses
///     [`CompilationDomain::DispatchKey`], for example.
///
/// Everything else that affects staging (e.g., the closure itself, its captures, the domain, any fixed options, etc.)
/// is implicit in the cache's owner, because a cache is scoped to exactly one retained callable. That is why this key
/// carries no fragile function-pointer identity. All three components are call-level (i.e., function-level) concepts,
/// since a bare program has neither static parameters nor a structured input. That is what separates function-level
/// specialization caching from structural region transformation through [`Transform`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionSpecializationKey<P, I, D> {
    /// Host values declared static for the keyed specialization, such as an axis number, mode enum, or boolean option.
    static_parameters: P,

    /// Parameter structure of the dynamic input, such as the shape of a nested tuple or named parameter container.
    input_structure: I,

    /// Owner-defined cache key for the dynamic input, such as an exact abstract signature or normalized shape bucket.
    dispatch_key: D,
}

impl<P, I, D> FunctionSpecializationKey<P, I, D> {
    /// Creates a new [`FunctionSpecializationKey`].
    #[inline]
    pub fn new(static_parameters: P, input_structure: I, dispatch_key: D) -> Self {
        Self { static_parameters, input_structure, dispatch_key }
    }

    /// Returns the static parameters (i.e., the host values declared static) for the keyed specialization.
    #[inline]
    pub fn static_parameters(&self) -> &P {
        &self.static_parameters
    }

    /// Returns the parameter structure of the dynamic input for the keyed specialization.
    #[inline]
    pub fn input_structure(&self) -> &I {
        &self.input_structure
    }

    /// Returns the owner-defined dispatch key for the dynamic input.
    #[inline]
    pub fn dispatch_key(&self) -> &D {
        &self.dispatch_key
    }
}

/// Flat source-program representation of a compiled call's callee, supplied through the operation's region driver and
/// interned as a shared callee root region on the staged instruction.
pub type FlatCompilationProgram<D> = Program<
    <D as Domain>::Constant,
    <D as Domain>::Operation,
    Vec<<D as Domain>::Constant>,
    Vec<<D as Domain>::Constant>,
>;

/// Tracer used while staging a compiled function in `D`'s operation universe.
pub type CompilationTracer<D> = Tracer<DomainTracingContext<D, <D as Domain>::Value>>;

/// Complete typed request consumed by [`CompilationDomain::stage`].
pub struct CompilationStagingRequest<D: CompilationDomain, F, Input, Output> {
    /// Function being traced.
    function: F,

    /// Concrete runtime captures exposed symbolically while tracing.
    captures: Vec<D::Value>,

    /// Structured abstract input signature.
    input_types: Input,

    /// Options fixed before tracing begins.
    options: D::Options,

    /// Staged output signature marker.
    output: PhantomData<fn() -> Output>,
}

impl<D: CompilationDomain, F, Input, Output> CompilationStagingRequest<D, F, Input, Output> {
    /// Creates a complete staging request.
    #[inline]
    pub fn new(function: F, captures: Vec<D::Value>, input_types: Input, options: D::Options) -> Self {
        Self { function, captures, input_types, options, output: PhantomData }
    }
}

/// Typed staging-request behavior used by [`CompilationDomain::stage`].
///
/// This trait encapsulates the structured generic bounds so concrete domains can implement staging without refining
/// the trait method's requirements. Construct requests with [`CompilationStagingRequest::new`].
pub trait StageRequest<D: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>;

    /// Structured abstract output type.
    type Output: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
            To<CompilationTracer<D>>: Parameterized<
                CompilationTracer<D>,
                To<D::Type> = Self::Output,
                To<D::Constant> = <Self::Output as Parameterized<D::Type>>::To<D::Constant>,
            >,
        >;

    /// Returns the structured abstract input signature.
    fn input_types(&self) -> &Self::Input;

    /// Returns the fixed compilation options.
    fn options(&self) -> &D::Options;

    /// Replaces the flat input signature while preserving its parameter structure.
    fn replace_input_types(&mut self, input_types: Vec<D::Type>) -> Result<(), D::Error>;

    /// Traces this request and normalizes the effective flat output signature.
    fn trace<NormalizeOutput>(
        self,
        normalize_output_types: NormalizeOutput,
    ) -> Result<StagedFunction<D, Self::Input, Self::Output>, D::Error>
    where
        NormalizeOutput: FnOnce(&D::Options, Vec<D::Type>) -> Result<Vec<D::Type>, D::Error>;
}

/// Typed staged-artifact behavior used by [`CompilationDomain::lower`].
pub trait LoweringRequest<D: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>;

    /// Structured abstract output type.
    type Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>;

    /// Returns the staged artifact.
    fn staged(&self) -> &StagedFunction<D, Self::Input, Self::Output>;

    /// Opens runtime captures as leading flat inputs.
    fn lifted_program(&self) -> Result<Arc<FlatCompilationProgram<D>>, ProgramError>;

    /// Assembles the backend-established lowering.
    fn into_lowered(
        self,
        program: D::LoweredProgram,
        output_types: Vec<D::Type>,
    ) -> LoweredFunction<D, Self::Input, Self::Output>;
}

/// Typed lowered-artifact behavior used by [`CompilationDomain::compile`].
pub trait CompileRequest<D: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>;

    /// Structured abstract output type.
    type Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>;

    /// Returns the lowered artifact.
    fn lowered(&self) -> &LoweredFunction<D, Self::Input, Self::Output>;

    /// Assembles the backend-established compiled artifact.
    fn into_compiled(
        self,
        program: Arc<D::CompiledProgram>,
        output_types: Vec<D::Type>,
    ) -> CompiledFunction<D, Self::Input, Self::Output>;
}

/// Complete typed request consumed by [`CompilationDomain::call`].
pub struct CompilationCall<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    /// Executable being invoked.
    executable: ExecutableFunction<D, Input, Output>,

    /// Flat public runtime inputs, excluding captures.
    inputs: Vec<D::Value>,
}

impl<D, Input, Output> CompilationCall<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Input::To<D::Value>: Parameterized<D::Value>,
{
    /// Creates a structured execution request.
    pub fn new(executable: &ExecutableFunction<D, Input, Output>, inputs: Input::To<D::Value>) -> Self {
        Self { executable: executable.clone(), inputs: inputs.into_parameters().collect() }
    }
}

/// Typed execution-request behavior used by [`CompilationDomain::call`].
pub trait CallRequest<D: CompilationDomain>: Sized {
    /// Structured abstract input type.
    type Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>;

    /// Structured abstract output type.
    type Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>;

    /// Structured runtime output type.
    type RuntimeOutput;

    /// Returns the executable artifact.
    fn executable(&self) -> &ExecutableFunction<D, Self::Input, Self::Output>;

    /// Returns flat public runtime inputs, excluding captures.
    fn inputs(&self) -> &[D::Value];

    /// Consumes the request and prepends runtime captures to its flat inputs.
    fn into_arguments(self) -> Vec<D::Value>;

    /// Reconstructs structured runtime outputs.
    fn reconstruct(
        executable: &ExecutableFunction<D, Self::Input, Self::Output>,
        outputs: Vec<D::Value>,
    ) -> Result<Self::RuntimeOutput, D::Error>;
}

impl<D, F, Input, Output> StageRequest<D> for CompilationStagingRequest<D, F, Input, Output>
where
    D: CompilationDomain,
    F: FnOnce(
        Vec<D::Constant>,
        Vec<CompilationTracer<D>>,
        Input::To<CompilationTracer<D>>,
    ) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Output::To<CompilationTracer<D>>:
        Parameterized<CompilationTracer<D>, To<D::Type> = Output, To<D::Constant> = Output::To<D::Constant>>,
{
    type Input = Input;
    type Output = Output;

    #[inline]
    fn input_types(&self) -> &Input {
        &self.input_types
    }

    #[inline]
    fn options(&self) -> &D::Options {
        &self.options
    }

    fn replace_input_types(&mut self, input_types: Vec<D::Type>) -> Result<(), D::Error> {
        self.input_types = Input::from_parameters(self.input_types.parameter_structure(), input_types)
            .map_err(ProgramError::from)
            .map_err(D::Error::from)?;
        Ok(())
    }

    fn trace<NormalizeOutput>(
        self,
        normalize_output_types: NormalizeOutput,
    ) -> Result<StagedFunction<D, Input, Output>, D::Error>
    where
        NormalizeOutput: FnOnce(&D::Options, Vec<D::Type>) -> Result<Vec<D::Type>, D::Error>,
    {
        trace_with_capture_references(
            self.function,
            self.captures,
            self.input_types,
            self.options,
            normalize_output_types,
        )
    }
}

/// Snapshot of one retained [`CompiledFunctionDispatcher`]'s dispatch activity.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct JitCacheStatistics {
    /// Calls served by an already compiled specialization.
    pub dispatch_hits: u64,

    /// Calls for which no compiled specialization was retained.
    pub dispatch_misses: u64,

    /// Closure traces performed after dispatch misses.
    pub traces: u64,

    /// Backend lowerings performed after new traces.
    pub lowerings: u64,

    /// Requests made to the domain compilation context after lowering.
    pub compilation_requests: u64,

    /// Total host nanoseconds spent flattening inputs and deriving their dispatch key and effective signature.
    pub input_abstractification_duration_ns: u64,

    /// Total host nanoseconds spent looking up retained specializations.
    pub dispatch_duration_ns: u64,

    /// Total host nanoseconds spent tracing specialization misses.
    pub tracing_duration_ns: u64,

    /// Total host nanoseconds spent lowering newly traced specializations.
    pub lowering_duration_ns: u64,
}

struct JitCacheStatisticsState {
    dispatch_hits: AtomicU64,
    dispatch_misses: AtomicU64,
    traces: AtomicU64,
    lowerings: AtomicU64,
    compilation_requests: AtomicU64,
    input_abstractification_duration_ns: AtomicU64,
    dispatch_duration_ns: AtomicU64,
    tracing_duration_ns: AtomicU64,
    lowering_duration_ns: AtomicU64,
}

impl JitCacheStatisticsState {
    fn new() -> Self {
        Self {
            dispatch_hits: AtomicU64::new(0),
            dispatch_misses: AtomicU64::new(0),
            traces: AtomicU64::new(0),
            lowerings: AtomicU64::new(0),
            compilation_requests: AtomicU64::new(0),
            input_abstractification_duration_ns: AtomicU64::new(0),
            dispatch_duration_ns: AtomicU64::new(0),
            tracing_duration_ns: AtomicU64::new(0),
            lowering_duration_ns: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> JitCacheStatistics {
        JitCacheStatistics {
            dispatch_hits: self.dispatch_hits.load(Ordering::Relaxed),
            dispatch_misses: self.dispatch_misses.load(Ordering::Relaxed),
            traces: self.traces.load(Ordering::Relaxed),
            lowerings: self.lowerings.load(Ordering::Relaxed),
            compilation_requests: self.compilation_requests.load(Ordering::Relaxed),
            input_abstractification_duration_ns: self.input_abstractification_duration_ns.load(Ordering::Relaxed),
            dispatch_duration_ns: self.dispatch_duration_ns.load(Ordering::Relaxed),
            tracing_duration_ns: self.tracing_duration_ns.load(Ordering::Relaxed),
            lowering_duration_ns: self.lowering_duration_ns.load(Ordering::Relaxed),
        }
    }

    fn clear(&self) {
        self.dispatch_hits.store(0, Ordering::Relaxed);
        self.dispatch_misses.store(0, Ordering::Relaxed);
        self.traces.store(0, Ordering::Relaxed);
        self.lowerings.store(0, Ordering::Relaxed);
        self.compilation_requests.store(0, Ordering::Relaxed);
        self.input_abstractification_duration_ns.store(0, Ordering::Relaxed);
        self.dispatch_duration_ns.store(0, Ordering::Relaxed);
        self.tracing_duration_ns.store(0, Ordering::Relaxed);
        self.lowering_duration_ns.store(0, Ordering::Relaxed);
    }

    fn increment(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn add_duration(counter: &AtomicU64, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        counter.fetch_add(nanoseconds, Ordering::Relaxed);
    }
}

/// Default number of compiled specializations retained by one [`CompiledFunctionDispatcher`].
const DEFAULT_JIT_CACHE_CAPACITY: usize = 256;

/// Operation-family capability for representing a call to a staged program.
///
/// The call operation is metadata-only: the callee is a flat program (whose captures have been lifted into leading
/// inputs) composed into the region driver passed to [`Context::bind`], which interns it as a shared callee root
/// region by [`Arc`] identity. The concrete operation family decides how that boundary lowers and how batching,
/// differentiation, partial evaluation, and other transforms rewrite it. This keeps higher-order call semantics with
/// the operation that owns them while allowing the lifecycle and capture plumbing to remain backend-neutral.
pub trait CompiledCallOperation<Constant: Value>: Operation<Type = Constant::Type> + Sized {
    /// Constructs a call operation. The accompanying [`Context::bind`] supplies its callee through a shared-callee
    /// region driver.
    fn compiled_call() -> Self;
}

/// Staged, unlowered form of one compiled function.
///
/// A staged function is the first callable package in the lifecycle: it owns the typed source [`ClosedProgram`]
/// together with the call-level state a bare program never carries, namely its runtime capture table, exact public
/// input/output structures, and compilation options. It can be inspected, lowered by its domain, or embedded as a
/// compiled-call boundary in an enclosing trace. No backend lowering or executable compilation has happened yet.
pub struct StagedFunction<
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Shared immutable staging metadata. Staged-handle clones are therefore constant-time even for large programs.
    state: Arc<StagedFunctionState<D, Input, Output>>,
}

struct StagedFunctionState<
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Source program and concrete runtime captures produced by tracing.
    source_program: ClosedProgram<D::Value, D::Constant, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>>,

    /// Flat declared public input types, excluding hidden captures.
    input_types: Vec<D::Type>,

    /// Flat declared output types in source-program order.
    output_types: Vec<D::Type>,

    /// Output parameter structure used to reconstruct structured values.
    output_structure: Output::ParameterStructure,

    /// Options applied before tracing and retained for lowering and compilation.
    options: Arc<D::Options>,

    /// Memoized source program with captures lifted into leading flat inputs.
    lifted_program: OnceLock<Arc<FlatCompilationProgram<D>>>,
}

impl<D, Input, Output> Clone for StagedFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

impl<D, Input, Output> StagedFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    /// Returns the source program and its runtime captures.
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<D::Value, D::Constant, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>> {
        &self.state.source_program
    }

    /// Returns the flat declared public input types, excluding hidden captures.
    #[inline]
    pub fn input_types(&self) -> &[D::Type] {
        self.state.input_types.as_slice()
    }

    /// Returns the flat source-program output types.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        self.state.output_types.as_slice()
    }

    /// Reconstructs the structured abstract input signature retained by the source program.
    pub fn input_signature(&self) -> Result<Input, ParameterError>
    where
        Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Type, To = Input>, To<D::Type> = Input>,
    {
        let program = self.state.source_program.program();
        let input_types =
            program.input_ids().iter().map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned());
        Input::from_parameters(program.input_structure().clone(), input_types)
    }

    /// Returns the output parameter structure retained by this staged function.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.state.output_structure
    }

    /// Returns the compilation options bound to this staged function.
    #[inline]
    pub fn options(&self) -> &D::Options {
        self.state.options.as_ref()
    }

    /// Stages a call to this function into the active context recovered from `inputs`.
    ///
    /// For nullary functions use [`Self::call_in_context`], because no input value exists from which to recover the
    /// active context.
    pub fn call<V>(&self, inputs: Input::To<V>) -> Result<Output::To<V>, ProgramError>
    where
        D::Operation: CompiledCallOperation<D::Constant>,
        V: Value<Type = D::Type>,
        V::DispatchDomain: Context<Type = D::Type, Constant = D::Constant, Operation = D::Operation>
            + CapturingContext<Capture = D::Value>
            + Constant<V, D::Constant>,
        Input: Parameterized<D::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<D::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let context = inputs
            .parameters()
            .next()
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
            .dispatch_domain();
        self.call_in_context(&context, inputs)
    }

    /// Stages a call to this function through an explicitly supplied active `context`.
    pub fn call_in_context<C, V>(&self, context: &C, inputs: Input::To<V>) -> Result<Output::To<V>, ProgramError>
    where
        D::Operation: CompiledCallOperation<D::Constant>,
        V: Value<Type = D::Type>,
        C: Context<Type = D::Type, Value = V, Constant = D::Constant, Operation = D::Operation>
            + CapturingContext<Capture = D::Value>
            + Constant<V, D::Constant>,
        Input: Parameterized<D::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<D::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let capture_references = self
            .state
            .source_program
            .captures()
            .iter()
            .cloned()
            .map(|capture| context.capture(capture))
            .collect::<Result<Vec<_>, _>>()?;
        let mut flat_inputs = capture_references
            .into_iter()
            .map(|capture| context.constant(capture))
            .collect::<Result<Vec<_>, _>>()?;
        flat_inputs.extend(inputs.into_parameters());
        let outputs = context.bind(
            D::Operation::compiled_call(),
            CalleeRegionDriver::new(&[self.lifted_program()?]),
            flat_inputs.as_slice(),
        )?;
        Output::To::<V>::from_parameters(self.state.output_structure.clone(), outputs).map_err(Into::into)
    }

    /// Binds this staged function using capture references already registered in an enclosing capture table.
    pub fn call_with_flat_capture_references<V>(
        &self,
        capture_references: &[D::Constant],
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        D::Operation: CompiledCallOperation<D::Constant>,
        V: Value<Type = D::Type>,
        V::DispatchDomain:
            Context<Type = D::Type, Constant = D::Constant, Operation = D::Operation> + Constant<V, D::Constant>,
    {
        let context = inputs
            .first()
            .map(|input| input.dispatch_domain())
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        self.call_with_flat_capture_references_in_context(&context, capture_references, inputs)
    }

    /// Binds this staged function through `context` using already-registered flat capture references.
    ///
    /// Unlike [`Self::call_with_flat_capture_references`], this form also supports nullary and capture-only calls,
    /// because it does not need a public input from which to recover the active context.
    pub fn call_with_flat_capture_references_in_context<C, V>(
        &self,
        context: &C,
        capture_references: &[D::Constant],
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        D::Operation: CompiledCallOperation<D::Constant>,
        V: Value<Type = D::Type>,
        C: Context<Type = D::Type, Value = V, Constant = D::Constant, Operation = D::Operation>
            + Constant<V, D::Constant>,
    {
        // Capture-reference indices belong to the caller's capture table and may differ from the source program's
        // local indices, and so only positional arity and types can be validated against the source program's
        // encapsulated capture table.
        let captures = self.state.source_program.captures();
        check_count!("input", capture_references, captures.len(), ProgramError);
        for (index, (expected, actual)) in captures.iter().zip(capture_references).enumerate() {
            let expected_type = expected.r#type();
            let actual_type = actual.r#type();
            if expected_type.as_ref() != actual_type.as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "capture input #{index} has type {}, but capture #{index} has type {}",
                    actual_type, expected_type,
                )));
            }
        }
        let mut flat_inputs = capture_references
            .iter()
            .cloned()
            .map(|capture| context.constant(capture))
            .collect::<Result<Vec<_>, _>>()?;
        flat_inputs.extend(inputs);
        context.bind(
            D::Operation::compiled_call(),
            CalleeRegionDriver::new(&[self.lifted_program()?]),
            flat_inputs.as_slice(),
        )
    }

    /// Returns the source program with runtime captures lifted into leading flat inputs.
    #[doc(hidden)]
    pub fn lifted_program(&self) -> Result<Arc<FlatCompilationProgram<D>>, ProgramError> {
        if let Some(program) = self.state.lifted_program.get() {
            return Ok(program.clone());
        }
        let program = Arc::new(self.state.source_program.to_program_with_lifted_captures()?);
        Ok(self.state.lifted_program.get_or_init(|| program).clone())
    }
}

impl<D, Input, Output> LoweringRequest<D> for StagedFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    type Input = Input;
    type Output = Output;

    fn staged(&self) -> &Self {
        self
    }

    fn lifted_program(&self) -> Result<Arc<FlatCompilationProgram<D>>, ProgramError> {
        StagedFunction::lifted_program(self)
    }

    fn into_lowered(self, program: D::LoweredProgram, output_types: Vec<D::Type>) -> LoweredFunction<D, Input, Output> {
        LoweredFunction::from_parts(self, program, output_types)
    }
}

/// Backend lowering of a [`StagedFunction`], ready for executable compilation.
///
/// It stays a function rather than a program: the backend's lowered program is one artifact it carries, alongside the
/// captures, signatures, and options inherited from its staged source.
pub struct LoweredFunction<
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Backend-owned lowered program.
    program: Arc<D::LoweredProgram>,

    /// Staged source and structured metadata retained across compilation.
    staged: StagedFunction<D, Input, Output>,

    /// Effective flat output types after lowering-time rewrites.
    output_types: Vec<D::Type>,
}

impl<D, Input, Output> Clone for LoweredFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), staged: self.staged.clone(), output_types: self.output_types.clone() }
    }
}

impl<D, Input, Output> LoweredFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    /// Assembles a lowering whose output signature has already been established by its backend.
    #[doc(hidden)]
    pub fn from_parts(
        staged: StagedFunction<D, Input, Output>,
        program: D::LoweredProgram,
        output_types: Vec<D::Type>,
    ) -> Self {
        Self { program: Arc::new(program), staged, output_types }
    }

    /// Returns the backend-owned lowered program.
    #[inline]
    pub fn lowered_program(&self) -> &D::LoweredProgram {
        self.program.as_ref()
    }

    /// Returns the staged function from which this lowering was produced.
    #[inline]
    pub fn staged(&self) -> &StagedFunction<D, Input, Output> {
        &self.staged
    }

    /// Returns the compilation options associated with this lowering.
    #[inline]
    pub fn options(&self) -> &D::Options {
        self.staged.options()
    }

    /// Returns the effective flat output types after lowering-time rewrites.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        &self.output_types
    }
}

impl<D, Input, Output> CompileRequest<D> for LoweredFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    type Input = Input;
    type Output = Output;

    fn lowered(&self) -> &Self {
        self
    }

    fn into_compiled(
        self,
        program: Arc<D::CompiledProgram>,
        output_types: Vec<D::Type>,
    ) -> CompiledFunction<D, Input, Output> {
        CompiledFunction::from_parts(self, program, output_types)
    }
}

/// Runtime-only handle for one compiled executable.
///
/// This handle retains only the state required to validate and execute calls: the backend executable, concrete
/// captures, flat signatures, and structured output shape. That call-level state is what still makes it a function
/// rather than a program, even though it has shed every transformable artifact: it deliberately does not retain the
/// staged or lowered programs, so it becomes [`Send`] and [`Sync`] automatically whenever those runtime fields are
/// `Send + Sync`. In contrast, [`CompiledFunction`] remains transformable and may retain `Rc`-backed source programs.
pub struct ExecutableFunction<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    state: Arc<ExecutableFunctionState<D, Input, Output>>,
}

struct ExecutableFunctionState<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    program: Arc<D::CompiledProgram>,
    captures: Vec<D::Value>,
    input_types: Vec<D::Type>,
    output_types: Vec<D::Type>,
    output_structure: Output::ParameterStructure,
    input: PhantomData<fn(Input)>,
}

impl<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> Clone
    for ExecutableFunction<D, Input, Output>
{
    #[inline]
    fn clone(&self) -> Self {
        Self { state: Arc::clone(&self.state) }
    }
}

/// Type-level source universe for retained JIT specialization.
///
/// The owning [`CompiledFunctionDispatcherState`] fixes the closure identity, lexical captures, domain instance, and
/// immutable compilation options. This token stores none of them; it only makes the descriptor's key and artifact
/// families coherent at the type level. If any owner-fixed semantic becomes variable per call, it must enter the
/// transform arguments instead.
struct JitCompilationSource<D, StaticParameters, Input, Output>(
    PhantomData<fn() -> (D, StaticParameters, Input, Output)>,
);

/// Transform descriptor for the executable specializations retained by one [`CompiledFunctionDispatcher`].
struct JitCompilationTransform;

impl<D, StaticParameters, Input, Output> Transform<JitCompilationSource<D, StaticParameters, Input, Output>>
    for JitCompilationTransform
where
    D: CompilationDomain,
    StaticParameters: Clone + Eq + Hash,
    Input: Parameterized<D::Type>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type>,
{
    type Arguments = FunctionSpecializationKey<StaticParameters, Input::ParameterStructure, D::DispatchKey>;
    type Artifact = ExecutableFunction<D, Input, Output>;

    const DEFAULT_CACHE_CAPACITY: usize = DEFAULT_JIT_CACHE_CAPACITY;
}

impl<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>>
    ExecutableFunction<D, Input, Output>
{
    /// Replaces the backend payload after the backend has established call-boundary compatibility.
    #[doc(hidden)]
    pub fn with_compiled_program(&self, program: Arc<D::CompiledProgram>, output_types: Vec<D::Type>) -> Self {
        Self {
            state: Arc::new(ExecutableFunctionState {
                program,
                captures: self.state.captures.clone(),
                input_types: self.state.input_types.clone(),
                output_types,
                output_structure: self.state.output_structure.clone(),
                input: PhantomData,
            }),
        }
    }

    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &D::CompiledProgram {
        &self.state.program
    }

    /// Returns the concrete runtime captures supplied before public inputs on every call.
    #[inline]
    pub fn captures(&self) -> &[D::Value] {
        &self.state.captures
    }

    /// Returns the flat declared public input types, excluding hidden captures.
    #[inline]
    pub fn input_types(&self) -> &[D::Type] {
        &self.state.input_types
    }

    /// Returns the effective flat output types produced by the executable.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        &self.state.output_types
    }

    /// Returns the output parameter structure used to reconstruct structured runtime values.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.state.output_structure
    }

    /// Validates that `inputs` matches the declared flat public input arity, excluding hidden captures.
    pub fn validate_flat_input_count(&self, inputs: &[D::Value]) -> Result<(), ProgramError> {
        if inputs.len() != self.state.input_types.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.state.input_types.len(),
                actual: inputs.len(),
            });
        }
        Ok(())
    }

    /// Validates that `outputs` matches the declared flat output arity.
    pub fn validate_flat_output_count(&self, outputs: &[D::Value]) -> Result<(), ProgramError> {
        if outputs.len() != self.state.output_types.len() {
            return Err(ProgramError::InvalidOutputCount {
                expected: self.state.output_types.len(),
                actual: outputs.len(),
            });
        }
        Ok(())
    }

    /// Prepends the retained runtime captures to the flat public `inputs`, producing the complete flat argument
    /// list in the `[captures..., public inputs...]` order every execution expects.
    pub fn arguments_with_captures(&self, inputs: Vec<D::Value>) -> Vec<D::Value> {
        let mut arguments = self.state.captures.to_vec();
        arguments.extend(inputs);
        arguments
    }
}

impl<D, Input, Output> ExecutableFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Output::To<D::Value>:
        Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
{
    /// Reconstructs structured runtime outputs from the executable's flat `outputs` using the retained output
    /// parameter structure.
    pub fn reconstruct_outputs(&self, outputs: Vec<D::Value>) -> Result<Output::To<D::Value>, ProgramError> {
        Output::To::<D::Value>::from_parameters(self.state.output_structure.clone(), outputs).map_err(Into::into)
    }
}

impl<D, Input, Output> CallRequest<D> for CompilationCall<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Input::To<D::Value>: Parameterized<D::Value>,
    Output::To<D::Value>:
        Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
{
    type Input = Input;
    type Output = Output;
    type RuntimeOutput = Output::To<D::Value>;

    fn executable(&self) -> &ExecutableFunction<D, Input, Output> {
        &self.executable
    }

    fn inputs(&self) -> &[D::Value] {
        &self.inputs
    }

    fn into_arguments(self) -> Vec<D::Value> {
        self.executable.arguments_with_captures(self.inputs)
    }

    fn reconstruct(
        executable: &ExecutableFunction<D, Input, Output>,
        outputs: Vec<D::Value>,
    ) -> Result<Self::RuntimeOutput, D::Error> {
        executable.reconstruct_outputs(outputs).map_err(D::Error::from)
    }
}

/// Compiled executable plus the staged and lowered metadata required to invoke it safely.
///
/// The backend's compiled program is the artifact this function carries; the captures, structured signatures, and
/// options carried with it are what make the handle callable as well as transformable.
pub struct CompiledFunction<
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Runtime-only executable handle.
    executable_function: ExecutableFunction<D, Input, Output>,

    /// Lowered function retaining source metadata and options.
    lowered: LoweredFunction<D, Input, Output>,
}

impl<D, Input, Output> Clone for CompiledFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    fn clone(&self) -> Self {
        Self { executable_function: self.executable_function.clone(), lowered: self.lowered.clone() }
    }
}

impl<D, Input, Output> CompiledFunction<D, Input, Output>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    /// Assembles a compiled function from backend-validated parts.
    #[doc(hidden)]
    pub fn from_parts(
        lowered: LoweredFunction<D, Input, Output>,
        program: Arc<D::CompiledProgram>,
        output_types: Vec<D::Type>,
    ) -> Self {
        let executable_function = ExecutableFunction {
            state: Arc::new(ExecutableFunctionState {
                program,
                captures: lowered.staged.source_program().captures().to_vec(),
                input_types: lowered.staged.input_types().to_vec(),
                output_types,
                output_structure: lowered.staged.output_structure().clone(),
                input: PhantomData,
            }),
        };
        Self { executable_function, lowered }
    }

    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &D::CompiledProgram {
        self.executable_function.compiled_program()
    }

    /// Returns the runtime-only handle, which omits staged and lowered transform metadata.
    #[inline]
    pub fn executable_function(&self) -> &ExecutableFunction<D, Input, Output> {
        &self.executable_function
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable_function(self) -> ExecutableFunction<D, Input, Output> {
        self.executable_function
    }

    /// Returns the lowering from which this executable was compiled.
    #[inline]
    pub fn lowered(&self) -> &LoweredFunction<D, Input, Output> {
        &self.lowered
    }

    /// Returns the staged source function.
    #[inline]
    pub fn staged(&self) -> &StagedFunction<D, Input, Output> {
        self.lowered.staged()
    }

    /// Returns the source program and runtime captures.
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<D::Value, D::Constant, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>> {
        self.staged().source_program()
    }

    /// Returns the effective flat output types produced by the executable.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        self.executable_function.output_types()
    }
}

/// Retained JIT dispatcher that caches compiled specializations of one Rust closure.
///
/// Unlike [`CompiledFunction`], which represents one already-specialized executable, a
/// `CompiledFunctionDispatcher` holds no program of its own: it is a retained dispatcher over a Rust closure that
/// produces one program, and one function around that program, per specialization from explicit host-side static
/// parameters and runtime dynamic inputs. Its first call for a specialization traces, lowers, and requests
/// compilation; later calls with the same static values, input parameter structure, and runtime-derived dispatch key
/// dispatch directly to the retained executable. Domain staging may normalize distinct runtime signatures to the
/// same staged signature, which can produce harmless duplicate dispatch specializations.
///
/// Tracing executes Rust host code only on a specialization miss. Host side effects inside `function` therefore run
/// once per retained specialization, not once per runtime call; observable per-call work must be represented by staged
/// effectful operations. High-cardinality static values can cause repeated tracing and LRU eviction, so arrays and
/// frequently changing data should remain dynamic inputs.
///
/// The dispatcher identity only namespaces this process-local specialization cache. For domains that cache compiled
/// programs, executable correctness and reuse still depend on
/// [`CompilationCacheDomain::compilation_key`](super::contexts::CompilationCacheDomain::compilation_key), which
/// receives the complete lowering.
///
/// Cloned handles share one specialization cache, and the dispatcher is usable from multiple threads whenever the
/// domain, closure, static parameters, and domain artifact types are thread-safe: `Send`/`Sync` derive structurally
/// from those field types. Recursive same-thread dispatch of the specialization currently being produced is rejected
/// with an error, while concurrent cold misses for one specialization on different threads deliberately produce
/// duplicate frontend work (tracing and lowering are cheap and inserts are idempotent); the domain's shared
/// [`CompilationContext`](super::contexts::CompilationContext) still coordinates the expensive backend compilation.
///
/// # Retained JIT Lifecycle
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   call["&lt;code&gt;CompiledFunctionDispatcher::call&lt;/code&gt; with Static Parameters and Dynamic Inputs"]
///   call --> key["Derive Function Specialization Key"]
///   key --> frontend_cache["Per-Function Bounded Specialization Cache"]
///   frontend_cache -->|"hit"| executable["Shared &lt;code&gt;ExecutableFunction&lt;/code&gt;"]
///   frontend_cache -->|"miss"| closure["Invoke Rust Closure"]
///   closure -->|"trace"| staged["&lt;code&gt;StagedFunction&lt;/code&gt; and Closed Source Program"]
///   staged -->|"lower"| lowered["&lt;code&gt;LoweredFunction&lt;/code&gt; and Backend Lowered Program"]
///   staged -.->|"&lt;code&gt;call&lt;/code&gt; inside an outer trace"| nested["Nested Call Operation"]
///   lowered --> compilation_context["Shared &lt;code&gt;CompilationContext&lt;/code&gt;"]
///   compilation_context -->|"restore or compile"| compiled["&lt;code&gt;CompiledFunction&lt;/code&gt;"]
///   compiled -->|"&lt;code&gt;into_executable_function&lt;/code&gt;"| executable
///   executable -->|"insert after cold production"| frontend_cache
///   executable -->|"call"| outputs["Structured Runtime Outputs"]
/// ```
///
/// The cache-hit edge skips tracing, lowering, and compilation. The dotted edge is the alternative staging path: a
/// staged function can become a nested call boundary instead of continuing immediately toward an executable.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct CompiledFunctionDispatcher<
    D: CompilationDomain<Type: Eq + Hash>,
    F,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> where
    Input::ParameterStructure: Eq + Hash,
{
    state: Arc<CompiledFunctionDispatcherState<D, F, Static, Input, Output>>,
}

struct CompiledFunctionDispatcherState<
    D: CompilationDomain<Type: Eq + Hash>,
    F,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> where
    Input::ParameterStructure: Eq + Hash,
{
    domain: D,
    function: F,
    options: D::Options,
    specializations: TransformCache<JitCompilationTransform, JitCompilationSource<D, Static, Input, Output>>,
    statistics: JitCacheStatisticsState,
}

impl<D, F, Static: Clone + Debug + Eq + Hash, Input, Output> Clone
    for CompiledFunctionDispatcher<D, F, Static, Input, Output>
where
    D: CompilationDomain,
    D::Type: Eq + Hash,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

impl<D, F, Static: Clone + Debug + Eq + Hash, Input, Output> CompiledFunctionDispatcher<D, F, Static, Input, Output>
where
    D: CompilationDomain,
    D::Type: Eq + Hash,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant>,
{
    fn new(domain: &D, function: F, options: D::Options, capacity: usize) -> Self {
        Self {
            state: Arc::new(CompiledFunctionDispatcherState {
                domain: domain.clone(),
                function,
                options,
                specializations: TransformCache::<
                    JitCompilationTransform,
                    JitCompilationSource<D, Static, Input, Output>,
                >::new(capacity),
                statistics: JitCacheStatisticsState::new(),
            }),
        }
    }

    /// Returns the compilation domain used by this dispatcher.
    #[inline]
    pub fn domain(&self) -> &D {
        &self.state.domain
    }

    /// Returns the fixed compilation options used for every specialization.
    #[inline]
    pub fn options(&self) -> &D::Options {
        &self.state.options
    }

    /// Returns the number of compiled specializations currently retained by this dispatcher.
    #[inline]
    pub fn specialization_count(&self) -> usize {
        self.state.specializations.len()
    }

    /// Returns the maximum number of compiled specializations retained by this dispatcher.
    #[inline]
    pub fn cache_capacity(&self) -> usize {
        self.state.specializations.capacity()
    }

    /// Returns a snapshot of this dispatcher's cache activity.
    #[inline]
    pub fn statistics(&self) -> JitCacheStatistics {
        self.state.statistics.snapshot()
    }

    /// Resets dispatch statistics without clearing compiled specializations.
    #[inline]
    pub fn clear_statistics(&self) {
        self.state.statistics.clear();
    }

    /// Clears every retained specialization without changing statistics.
    #[inline]
    pub fn clear_cache(&self) {
        self.state.specializations.clear();
    }

    /// Invalidates every retained specialization for `static_parameters` and returns the number removed.
    pub fn invalidate_static(&self, static_parameters: &Static) -> usize {
        self.state.specializations.invalidate_entries_if(|key| key.static_parameters() == static_parameters)
    }

    /// Calls this dispatcher with explicit host-side `static_parameters` and dynamic runtime `inputs`.
    pub fn call(&self, static_parameters: Static, inputs: Input::To<D::Value>) -> Result<Output::To<D::Value>, D::Error>
    where
        D::Options: Clone,
        F: Fn(Static, Input::To<CompilationTracer<D>>) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
        Input::Family: ParameterizedFamily<D::Value> + ParameterizedFamily<CompilationTracer<D>>,
        Input::To<D::Value>: Parameterized<
                D::Value,
                Family = Input::Family,
                ParameterStructure = Input::ParameterStructure,
                To<D::Type> = Input,
            >,
        Output::Family: ParameterizedFamily<D::Value> + ParameterizedFamily<CompilationTracer<D>>,
        Output::To<D::Value>:
            Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        Output::To<CompilationTracer<D>>:
            Parameterized<CompilationTracer<D>, To<D::Type> = Output, To<D::Constant> = Output::To<D::Constant>>,
    {
        let abstractification_start = Instant::now();
        let input_structure = inputs.parameter_structure();
        let runtime_input_types = inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let (dispatch_key, effective_input_types) =
            self.state.domain.dispatch_signature(runtime_input_types, &self.state.options)?;
        JitCacheStatisticsState::add_duration(
            &self.state.statistics.input_abstractification_duration_ns,
            abstractification_start.elapsed(),
        );
        let key = FunctionSpecializationKey::new(static_parameters.clone(), input_structure, dispatch_key);

        let dispatch_start = Instant::now();
        let entry = self.state.specializations.try_entry(key);
        JitCacheStatisticsState::add_duration(&self.state.statistics.dispatch_duration_ns, dispatch_start.elapsed());
        let producer = match entry {
            Ok(SpecializationCacheEntry::Occupied(executable)) => {
                JitCacheStatisticsState::increment(&self.state.statistics.dispatch_hits);
                return call_function(&self.state.domain, &executable, inputs);
            }
            Ok(SpecializationCacheEntry::Vacant(producer)) => {
                JitCacheStatisticsState::increment(&self.state.statistics.dispatch_misses);
                producer
            }
            Err(_) => {
                JitCacheStatisticsState::increment(&self.state.statistics.dispatch_misses);
                return Err(ProgramError::InvalidArgument {
                    message: "recursive JIT dispatch requested a specialization that is already being produced".into(),
                }
                .into());
            }
        };

        let input_types =
            Input::from_parameters(producer.key().input_structure().clone(), effective_input_types.iter().cloned())
                .map_err(|error| D::Error::from(ProgramError::from(error)))?;
        JitCacheStatisticsState::increment(&self.state.statistics.traces);
        let tracing_start = Instant::now();
        let staged = match self.state.domain.stage(CompilationStagingRequest::new(
            |_, _, traced_inputs| (self.state.function)(static_parameters, traced_inputs),
            Vec::new(),
            input_types,
            self.state.options.clone(),
        )) {
            Ok(staged) => {
                let duration = tracing_start.elapsed();
                JitCacheStatisticsState::add_duration(&self.state.statistics.tracing_duration_ns, duration);
                staged
            }
            Err(error) => {
                let duration = tracing_start.elapsed();
                JitCacheStatisticsState::add_duration(&self.state.statistics.tracing_duration_ns, duration);
                return Err(error);
            }
        };
        JitCacheStatisticsState::increment(&self.state.statistics.lowerings);
        let lowering_start = Instant::now();
        let lowered = match self.state.domain.lower(staged) {
            Ok(lowered) => {
                let duration = lowering_start.elapsed();
                JitCacheStatisticsState::add_duration(&self.state.statistics.lowering_duration_ns, duration);
                lowered
            }
            Err(error) => {
                let duration = lowering_start.elapsed();
                JitCacheStatisticsState::add_duration(&self.state.statistics.lowering_duration_ns, duration);
                return Err(error);
            }
        };
        JitCacheStatisticsState::increment(&self.state.statistics.compilation_requests);
        let compiled = self.state.domain.compile(lowered)?;
        let executable = producer.insert(compiled.into_executable_function());
        call_function(&self.state.domain, &executable, inputs)
    }
}

/// Constructs a retained dispatcher for a fallible closure using explicit options and cache capacity.
pub fn try_jit_with_options_and_capacity<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
    capacity: usize,
) -> CompiledFunctionDispatcher<D, F, Static, Input, Output>
where
    D: CompilationDomain,
    D::Type: Eq + Hash,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    CompiledFunctionDispatcher::new(domain, function, options, capacity)
}

/// Constructs a retained dispatcher for a fallible closure using explicit options.
#[inline]
pub fn try_jit_with_options<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
) -> CompiledFunctionDispatcher<D, F, Static, Input, Output>
where
    D: CompilationDomain,
    D::Type: Eq + Hash,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    try_jit_with_options_and_capacity(domain, function, options, DEFAULT_JIT_CACHE_CAPACITY)
}

/// Constructs a retained dispatcher for a fallible closure using default options.
#[inline]
pub fn try_jit<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
) -> CompiledFunctionDispatcher<D, F, Static, Input, Output>
where
    D: CompilationDomain<Options: Default>,
    D::Type: Eq + Hash,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    try_jit_with_options(domain, function, D::Options::default())
}

/// Constructs a retained dispatcher for an infallible closure using explicit options.
pub fn jit_with_options<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
) -> CompiledFunctionDispatcher<
    D,
    impl Fn(Static, Input::To<CompilationTracer<D>>) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Static,
    Input,
    Output,
>
where
    D: CompilationDomain,
    D::Type: Eq + Hash,
    F: Fn(Static, Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
{
    try_jit_with_options(domain, move |static_parameters, inputs| Ok(function(static_parameters, inputs)), options)
}

/// Constructs a retained dispatcher for an infallible closure using default options.
#[inline]
pub fn jit<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
) -> CompiledFunctionDispatcher<
    D,
    impl Fn(Static, Input::To<CompilationTracer<D>>) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Static,
    Input,
    Output,
>
where
    D: CompilationDomain<Options: Default>,
    D::Type: Eq + Hash,
    F: Fn(Static, Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
{
    jit_with_options(domain, function, D::Options::default())
}

/// Stages an infallible capture-free function through `domain`.
pub fn stage_function<D, F, Input, Output>(
    domain: &D,
    function: F,
    input_types: Input,
    options: D::Options,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain,
    F: FnOnce(Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Output::To<CompilationTracer<D>>:
        Parameterized<CompilationTracer<D>, To<D::Type> = Output, To<D::Constant> = Output::To<D::Constant>>,
{
    domain.stage(CompilationStagingRequest::new(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, options))
}

/// Executes a structured runtime call through `domain`.
pub fn call_function<D, Input, Output>(
    domain: &D,
    executable: &ExecutableFunction<D, Input, Output>,
    inputs: Input::To<D::Value>,
) -> Result<Output::To<D::Value>, D::Error>
where
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Value>>,
    Input::To<D::Value>: Parameterized<D::Value>,
    Output::To<D::Value>:
        Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
{
    domain.call(CompilationCall::new(executable, inputs))
}

/// Constructs a staged function from types already prepared by a compilation domain.
pub(crate) fn trace_with_capture_references<D, F, Input, Output, NormalizeOutput>(
    function: F,
    captures: Vec<D::Value>,
    input_types: Input,
    options: D::Options,
    normalize_output_types: NormalizeOutput,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain,
    F: FnOnce(
        Vec<D::Constant>,
        Vec<CompilationTracer<D>>,
        Input::To<CompilationTracer<D>>,
    ) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Input: Parameterized<D::Type>,
    Input::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Output: Parameterized<D::Type>,
    Output::Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
    Output::To<CompilationTracer<D>>:
        Parameterized<CompilationTracer<D>, To<D::Type> = Output, To<D::Constant> = Output::To<D::Constant>>,
    NormalizeOutput: FnOnce(&D::Options, Vec<D::Type>) -> Result<Vec<D::Type>, D::Error>,
{
    let context = DomainTracingContext::<D, D::Value>::new();
    let capture_table = context.captures().clone();
    let builder = context.builder().clone();
    let capture_references =
        captures.into_iter().map(|capture| context.capture(capture)).collect::<Result<Vec<_>, _>>()?;
    let capture_tracers = capture_references
        .iter()
        .cloned()
        .map(|capture| StagingContext::constant(&context, capture))
        .collect::<Vec<_>>();
    let input_structure = input_types.parameter_structure();
    let input_type_values = input_types.parameters().cloned().collect::<Vec<_>>();
    let inputs = input_types.map_parameters(|input_type| context.input(input_type)).map_err(ProgramError::from)?;
    let outputs = function(capture_references, capture_tracers, inputs)?;
    check_builders!(&builder, [outputs.parameters().map(|output| output.builder())])?;
    if let Some(error) = builder.borrow_mut().error.take() {
        return Err(error.into());
    }
    let output_structure = outputs.parameter_structure();
    let output_ids = outputs.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let output_types = outputs.parameters().map(|output| output.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = normalize_output_types(&options, output_types)?;
    drop(outputs);
    drop(context);
    let captures = Rc::try_unwrap(capture_table).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let program = builder.build(output_ids, input_structure, output_structure.clone())?.into_simplified()?;
    let source_program = ClosedProgram::new(program, captures)?;
    let source_program = source_program.without_unused_captures()?;
    Ok(StagedFunction {
        state: Arc::new(StagedFunctionState {
            source_program,
            input_types: input_type_values,
            output_types,
            output_structure,
            options: Arc::new(options),
            lifted_program: OnceLock::new(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::captures::CaptureReference;
    use crate::compilation::contexts::{CompilationCacheDomain, CompilationContext};
    use crate::parameters::Placeholder;
    use crate::programs::{Operation, RegionInterface, Type, TypeError};

    use super::*;

    #[derive(Clone, Debug)]
    struct NegateOperation;

    impl Operation for NegateOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            "test_negate"
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            _region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError::invalid(format!("test_negate expects 1 input but got {}", input_types.len())));
            }
            Ok(input_types.to_vec())
        }
    }

    #[derive(Clone)]
    struct TestLoweredProgram {
        program: FlatCompilationProgram<TestDomain>,
        capture_count: usize,
        output_types: Vec<ArrayType>,
        options: TestOptions,
    }

    struct TestCompiledProgram {
        program: FlatCompilationProgram<TestDomain>,
        output_types: Vec<ArrayType>,
    }

    #[derive(Clone, Debug, Default)]
    struct TestOptions {
        staged_input_type: Option<ArrayType>,
        lowered_output_type: Option<ArrayType>,
        compiled_output_type: Option<ArrayType>,
        fail_lowering: bool,
        fail_compilation: bool,
    }

    #[derive(Clone)]
    struct TestDomain {
        cache: Arc<CompilationContext<Self>>,
        compilations: Arc<AtomicUsize>,
    }

    impl TestDomain {
        fn new() -> Self {
            Self { cache: Arc::new(CompilationContext::new()), compilations: Arc::new(AtomicUsize::new(0)) }
        }

        fn compilation_count(&self) -> usize {
            self.compilations.load(Ordering::Relaxed)
        }
    }

    impl Domain for TestDomain {
        type Type = ArrayType;
        type Value = Array;
        type Constant = CaptureReference<ArrayType>;
        type Operation = NegateOperation;
    }

    impl CompilationDomain for TestDomain {
        type DispatchKey = Arc<[ArrayType]>;
        type LoweredProgram = TestLoweredProgram;
        type CompiledProgram = TestCompiledProgram;
        type Options = TestOptions;
        type Error = ProgramError;

        fn dispatch_signature(
            &self,
            input_types: Vec<ArrayType>,
            _options: &Self::Options,
        ) -> Result<(Self::DispatchKey, Arc<[ArrayType]>), Self::Error> {
            let input_types: Arc<[ArrayType]> = input_types.into();
            Ok((input_types.clone(), input_types))
        }

        fn stage<Request>(
            &self,
            mut request: Request,
        ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: StageRequest<Self>,
        {
            if let Some(input_type) = &request.options().staged_input_type {
                request
                    .replace_input_types(request.input_types().parameters().map(|_| input_type.clone()).collect())?;
            }
            request.trace(|_, output_types| Ok(output_types))
        }

        fn lower<Request>(
            &self,
            staged: Request,
        ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: LoweringRequest<Self>,
        {
            if staged.staged().options().fail_lowering {
                return Err(ProgramError::InvalidArgument { message: "expected lowering failure".into() });
            }
            let program = staged.lifted_program()?;
            let mut output_types: Vec<ArrayType> = program
                .output_ids()
                .iter()
                .map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned())
                .collect();
            if let Some(output_type) = &staged.staged().options().lowered_output_type {
                output_types.fill(output_type.clone());
            }
            if output_types.len() != staged.staged().output_types().len() {
                return Err(ProgramError::InvalidOutputCount {
                    expected: staged.staged().output_types().len(),
                    actual: output_types.len(),
                });
            }
            for (declared, actual) in staged.staged().output_types().iter().zip(&output_types) {
                if !declared.is_refined_by(actual) {
                    return Err(ProgramError::InvalidArgument {
                        message: format!("output type {actual} does not refine declared type {declared}"),
                    });
                }
            }
            let capture_count = staged.staged().source_program().captures().len();
            let options = staged.staged().options().clone();
            Ok(staged.into_lowered(
                TestLoweredProgram {
                    program: program.as_ref().clone(),
                    capture_count,
                    output_types: output_types.clone(),
                    options,
                },
                output_types,
            ))
        }

        fn compile<Request>(
            &self,
            lowered: Request,
        ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: CompileRequest<Self>,
        {
            self.cache.compile_request(
                self,
                lowered,
                |program| {
                    self.compilations.fetch_add(1, Ordering::Relaxed);
                    if program.options.fail_compilation {
                        return Err(ProgramError::InvalidArgument { message: "expected compilation failure".into() });
                    }
                    let mut output_types = program.output_types.clone();
                    if let Some(output_type) = &program.options.compiled_output_type {
                        output_types.fill(output_type.clone());
                    }
                    Ok(TestCompiledProgram { program: program.program.clone(), output_types })
                },
                |program| program.output_types.clone(),
            )
        }

        fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
        where
            Request: CallRequest<Self>,
        {
            if request.inputs().len() != request.executable().input_types().len() {
                return Err(ProgramError::InvalidInputCount {
                    expected: request.executable().input_types().len(),
                    actual: request.inputs().len(),
                });
            }
            for (declared, actual) in
                request.executable().input_types().iter().zip(request.inputs().iter().map(Typed::r#type))
            {
                if !declared.is_refined_by(actual.as_ref()) {
                    return Err(ProgramError::InvalidArgument {
                        message: format!("runtime input type {actual} does not refine declared type {declared}"),
                    });
                }
            }
            let executable = request.executable().clone();
            let outputs = executable.compiled_program().program.interpret_with(
                request.into_arguments(),
                |_, capture| {
                    Err(ProgramError::MalformedProgram(format!(
                        "lifted test program retained capture {}",
                        capture.index(),
                    )))
                },
                |_, inputs| {
                    if inputs.len() != 1 {
                        return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() });
                    }
                    Ok(vec![-inputs[0].clone()])
                },
            )?;
            Request::reconstruct(&executable, outputs)
        }
    }

    impl CompilationCacheDomain for TestDomain {
        type CacheKey = Vec<String>;

        fn compilation_key(&self, program: &TestLoweredProgram) -> Result<Vec<String>, ProgramError> {
            let mut key = vec![
                format!("captures:{}", program.capture_count),
                format!("atoms:{:?}", program.program.atoms()),
                format!("inputs:{:?}", program.program.input_ids()),
                format!("outputs:{:?}", program.program.output_ids()),
                format!("options:{:?}", program.options),
            ];
            key.extend(program.program.instructions().iter().map(|instruction| {
                format!("{}:{:?}:{:?}", instruction.operation().name(), instruction.inputs(), instruction.outputs(),)
            }));
            Ok(key)
        }
    }

    fn compile_from_one_call_site(
        domain: &TestDomain,
        negate: bool,
    ) -> CompiledFunction<TestDomain, ArrayType, ArrayType> {
        let staged = stage_function(
            domain,
            |input| if negate { input.unary(NegateOperation) } else { input },
            ArrayType::scalar(DataType::F64),
            TestOptions::default(),
        )
        .unwrap();
        domain.compile(domain.lower(staged).unwrap()).unwrap()
    }

    #[test]
    fn test_function_specialization_key() {
        let key: FunctionSpecializationKey<(&str, usize), Vec<()>, &str> =
            FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]");
        assert_eq!(key.static_parameters(), &("training", 4));
        assert_eq!(key.input_structure(), &vec![(), ()]);
        assert_eq!(key.dispatch_key(), &"f32[2,3]");

        // Every component participates in equality.
        assert_eq!(key, FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("inference", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("training", 4), vec![()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[4,3]"));
    }

    #[test]
    fn test_jit_compilation_transform_contract() {
        type Source = JitCompilationSource<TestDomain, usize, Vec<ArrayType>, Vec<ArrayType>>;
        type Arguments = FunctionSpecializationKey<usize, Vec<Placeholder>, Arc<[ArrayType]>>;
        type Artifact = ExecutableFunction<TestDomain, Vec<ArrayType>, Vec<ArrayType>>;

        fn assert_contract<T, S>()
        where
            T: Transform<S, Arguments = Arguments, Artifact = Artifact>,
        {
        }

        assert_contract::<JitCompilationTransform, Source>();
        assert_eq!(<JitCompilationTransform as Transform<Source>>::DEFAULT_CACHE_CAPACITY, DEFAULT_JIT_CACHE_CAPACITY,);
    }

    #[test]
    fn test_compilation_key_distinguishes_computations_from_one_call_site() {
        let domain = TestDomain::new();

        let identity = compile_from_one_call_site(&domain, false);
        let negate = compile_from_one_call_site(&domain, true);

        assert_eq!(
            call_function(&domain, identity.executable_function(), Array::scalar(3.0)).unwrap(),
            Array::scalar(3.0)
        );
        assert_eq!(
            call_function(&domain, negate.executable_function(), Array::scalar(3.0)).unwrap(),
            Array::scalar(-3.0)
        );
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_staged_and_lowered_handles_reuse_one_compilation() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = stage_function(
            &domain,
            |input: CompilationTracer<TestDomain>| input.unary(NegateOperation),
            ArrayType::scalar(DataType::F64),
            TestOptions::default(),
        )
        .unwrap();
        let staged_clone = staged.clone();
        assert!(std::ptr::eq(staged.options(), staged_clone.options()));

        let first = domain.compile(domain.lower(staged_clone).unwrap()).unwrap();
        let second = domain.compile(domain.lower(staged).unwrap()).unwrap();

        assert_eq!(
            call_function(&domain, first.executable_function(), Array::scalar(2.0)).unwrap(),
            Array::scalar(-2.0)
        );
        assert_eq!(
            call_function(&domain, second.executable_function(), Array::scalar(4.0)).unwrap(),
            Array::scalar(-4.0)
        );
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(domain.cache.statistics().memory_hits, 1);
    }

    #[test]
    fn test_staging_options_apply_before_tracing() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = stage_function(
            &domain,
            |input: CompilationTracer<TestDomain>| input,
            ArrayType::scalar(DataType::F64),
            options,
        )
        .unwrap();

        assert_eq!(staged.input_types(), &[ArrayType::scalar(DataType::I64)]);
        assert_eq!(staged.output_types(), &[ArrayType::scalar(DataType::I64)]);
        assert_eq!(staged.options().staged_input_type, Some(ArrayType::scalar(DataType::I64)));
    }

    #[test]
    fn test_compiled_function_rejects_runtime_input_type_mismatch() {
        let domain = TestDomain::new();
        let compiled = compile_from_one_call_site(&domain, false);

        assert!(matches!(
            call_function(&domain, compiled.executable_function(), Array::scalar(3_i64)),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type i64[] does not refine declared type f64[]",
        ));
    }

    #[test]
    fn test_compiled_function_executes_explicit_capture() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), ArrayType> = domain
            .stage(CompilationStagingRequest::new(
                |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
                vec![Array::scalar(7.0)],
                (),
                TestOptions::default(),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();

        assert_eq!(call_function(&domain, compiled.executable_function(), ()).unwrap(), Array::scalar(7.0));
        assert_eq!(compiled.source_program().captures(), &[Array::scalar(7.0)]);
    }

    #[test]
    fn test_executable_function_outlives_transform_metadata_and_executes_captures() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), ArrayType> = domain
            .stage(CompilationStagingRequest::new(
                |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
                vec![Array::scalar(7.0)],
                (),
                TestOptions::default(),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
        let executable = compiled.into_executable_function();

        assert_eq!(executable.captures(), &[Array::scalar(7.0)]);
        assert!(executable.input_types().is_empty());
        assert_eq!(executable.output_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(call_function(&domain, &executable, ()).unwrap(), Array::scalar(7.0));
    }

    #[test]
    fn test_executable_function_is_send_and_sync_for_thread_safe_runtime_state() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        assert_send_and_sync::<ExecutableFunction<TestDomain, ArrayType, ArrayType>>();

        let domain = TestDomain::new();
        let executable = compile_from_one_call_site(&domain, true).into_executable_function();
        let second = executable.clone();
        let first_domain = domain.clone();
        let second_domain = domain.clone();
        let first_thread =
            std::thread::spawn(move || call_function(&first_domain, &executable, Array::scalar(3.0)).unwrap());
        let second_thread =
            std::thread::spawn(move || call_function(&second_domain, &second, Array::scalar(4.0)).unwrap());

        assert_eq!(first_thread.join().unwrap(), Array::scalar(-3.0));
        assert_eq!(second_thread.join().unwrap(), Array::scalar(-4.0));
    }

    #[test]
    fn test_fallible_staging_propagates_closure_error() {
        let domain = TestDomain::new();
        let result: Result<StagedFunction<TestDomain, ArrayType, ArrayType>, ProgramError> =
            domain.stage(CompilationStagingRequest::new(
                |_, _, _| Err(ProgramError::InvalidArgument { message: "staging failed".into() }),
                Vec::new(),
                ArrayType::scalar(DataType::F64),
                TestOptions::default(),
            ));

        assert!(matches!(result, Err(ProgramError::InvalidArgument { message }) if message == "staging failed"));
    }

    #[test]
    fn test_staging_prunes_unused_captures() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = domain
            .stage(CompilationStagingRequest::new(
                |_, _, input: CompilationTracer<TestDomain>| Ok(input),
                vec![Array::scalar(7.0)],
                ArrayType::scalar(DataType::F64),
                TestOptions::default(),
            ))
            .unwrap();

        assert!(staged.source_program().captures().is_empty());
    }

    #[test]
    fn test_lower_rejects_incompatible_output_type() {
        let domain = TestDomain::new();
        let options =
            TestOptions { lowered_output_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };

        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = stage_function(
            &domain,
            |input: CompilationTracer<TestDomain>| input,
            ArrayType::scalar(DataType::F64),
            options,
        )
        .unwrap();
        assert!(matches!(
            domain.lower(staged),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64[] does not refine declared type f64[]",
        ));
    }

    #[test]
    fn test_compile_rejects_incompatible_output_type() {
        let domain = TestDomain::new();
        let options =
            TestOptions { compiled_output_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = stage_function(
            &domain,
            |input: CompilationTracer<TestDomain>| input,
            ArrayType::scalar(DataType::F64),
            options,
        )
        .unwrap();
        let lowered = domain.lower(staged).unwrap();

        assert!(matches!(
            domain.compile(lowered),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64[] does not refine declared type f64[]",
        ));
    }

    #[test]
    fn test_compiled_function_dispatcher_reuses_warm_specializations() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );

        assert_eq!(function.call(true, Array::scalar(2.0)).unwrap(), Array::scalar(-2.0));
        assert_eq!(function.call(true, Array::scalar(3.0)).unwrap(), Array::scalar(-3.0));
        assert_eq!(function.call(false, Array::scalar(4.0)).unwrap(), Array::scalar(4.0));
        assert_eq!(function.specialization_count(), 2);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_applies_staging_options_before_tracing() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            jit_with_options(&domain, |(), input: CompilationTracer<TestDomain>| input, options);

        assert!(matches!(
            function.call((), Array::scalar(2.0)),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type f64[] does not refine declared type i64[]",
        ));
        assert_eq!(function.statistics().traces, 1);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CollidingStatic(bool);

    impl Hash for CollidingStatic {
        fn hash<H: Hasher>(&self, state: &mut H) {
            0_u8.hash(state);
        }
    }

    #[test]
    fn test_compiled_function_dispatcher_distinguishes_hash_colliding_static_values() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, CollidingStatic, ArrayType, ArrayType> = jit(
            &domain,
            |static_parameters: CollidingStatic, input: CompilationTracer<TestDomain>| {
                if static_parameters.0 { input.unary(NegateOperation) } else { input }
            },
        );

        assert_eq!(function.call(CollidingStatic(false), Array::scalar(2.0)).unwrap(), Array::scalar(2.0));
        assert_eq!(function.call(CollidingStatic(true), Array::scalar(2.0)).unwrap(), Array::scalar(-2.0));
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_input_structures_partition_specializations() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), Vec<ArrayType>, ArrayType> =
            try_jit(&domain, |(), mut inputs: Vec<CompilationTracer<TestDomain>>| {
                inputs.drain(..1).next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        assert_eq!(function.call((), vec![Array::scalar(2.0)]).unwrap(), Array::scalar(2.0));
        assert_eq!(function.call((), vec![Array::scalar(2.0), Array::scalar(3.0)]).unwrap(), Array::scalar(2.0),);
        assert_eq!(function.specialization_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_distinguishes_inputs_differing_only_in_empty_substructure() {
        // Both calls flatten to zero leaves, so they share static parameters and an empty dispatch signature. Only
        // the input parameter structure distinguishes them, while the traced closure branches on the observed
        // container arity. Key reuse across these calls would therefore silently return the first call's staged
        // behavior for the second call's structurally different input.
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), Vec<Vec<ArrayType>>, Vec<ArrayType>> =
            try_jit(&domain, |(), inputs: Vec<Vec<CompilationTracer<TestDomain>>>| {
                if inputs.len() == 1 {
                    Ok(inputs.into_iter().next().unwrap())
                } else {
                    Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() })
                }
            });

        assert_eq!(function.call((), vec![Vec::new()]).unwrap(), Vec::new());
        assert!(matches!(
            function.call((), Vec::new()),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_invalidates_one_static_specialization() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );
        function.call(true, Array::scalar(2.0)).unwrap();
        function.call(false, Array::scalar(2.0)).unwrap();

        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.invalidate_static(&true), 1);
        assert_eq!(function.specialization_count(), 1);
        function.call(true, Array::scalar(2.0)).unwrap();
        assert_eq!(function.statistics().traces, 3);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_retries_failed_specialization() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            try_jit(&domain, |fail, input: CompilationTracer<TestDomain>| {
                if fail {
                    Err(ProgramError::InvalidArgument { message: "expected trace failure".into() })
                } else {
                    Ok(input)
                }
            });

        for _ in 0..2 {
            assert!(matches!(
                function.call(true, Array::scalar(2.0)),
                Err(ProgramError::InvalidArgument { message }) if message == "expected trace failure",
            ));
        }
        assert_eq!(function.call(false, Array::scalar(2.0)).unwrap(), Array::scalar(2.0));
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 3);
    }

    #[test]
    fn test_compiled_function_dispatcher_retries_lowering_and_compilation_failures() {
        let lowering_domain = TestDomain::new();
        let lowering_options = TestOptions { fail_lowering: true, ..TestOptions::default() };
        let lowering_function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            jit_with_options(&lowering_domain, |(), input: CompilationTracer<TestDomain>| input, lowering_options);
        for _ in 0..2 {
            assert!(matches!(
                lowering_function.call((), Array::scalar(2.0)),
                Err(ProgramError::InvalidArgument { message }) if message == "expected lowering failure",
            ));
        }
        assert_eq!(lowering_function.specialization_count(), 0);
        let statistics = lowering_function.statistics();
        assert_eq!((statistics.traces, statistics.lowerings, statistics.compilation_requests), (2, 2, 0));

        let compilation_domain = TestDomain::new();
        let compilation_options = TestOptions { fail_compilation: true, ..TestOptions::default() };
        let compilation_function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            jit_with_options(
                &compilation_domain,
                |(), input: CompilationTracer<TestDomain>| input,
                compilation_options,
            );
        for _ in 0..2 {
            assert!(matches!(
                compilation_function.call((), Array::scalar(2.0)),
                Err(ProgramError::InvalidArgument { message }) if message == "expected compilation failure",
            ));
        }
        assert_eq!(compilation_function.specialization_count(), 0);
        let statistics = compilation_function.statistics();
        assert_eq!((statistics.traces, statistics.lowerings, statistics.compilation_requests), (2, 2, 2));
        assert_eq!(compilation_domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_owners_isolate_frontend_specializations() {
        let domain = TestDomain::new();
        let first: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            jit(&domain, |(), input: CompilationTracer<TestDomain>| input);
        let second: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            jit(&domain, |(), input: CompilationTracer<TestDomain>| input);

        assert_eq!(first.call((), Array::scalar(1.0)).unwrap(), Array::scalar(1.0));
        assert_eq!(second.call((), Array::scalar(2.0)).unwrap(), Array::scalar(2.0));
        assert_eq!(first.specialization_count(), 1);
        assert_eq!(second.specialization_count(), 1);
        assert_eq!(first.statistics().traces, 1);
        assert_eq!(second.statistics().traces, 1);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_lru_capacity_retraces_evicted_specialization() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            try_jit_with_options_and_capacity(
                &domain,
                |negate, input: CompilationTracer<TestDomain>| {
                    Ok(if negate { input.unary(NegateOperation) } else { input })
                },
                TestOptions::default(),
                1,
            );

        function.call(false, Array::scalar(1.0)).unwrap();
        function.call(true, Array::scalar(1.0)).unwrap();
        function.call(false, Array::scalar(1.0)).unwrap();

        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_misses, 3);
        assert_eq!(function.statistics().traces, 3);
        assert_eq!(domain.compilation_count(), 2, "the global exact compilation cache should reuse the evicted entry");
    }

    #[test]
    fn test_compiled_function_dispatcher_map_inputs_flatten_deterministically() {
        // Map-like containers flatten in key order, so two runtime maps with the same keys must produce one
        // specialization regardless of construction order, and replay must bind flat arguments to the same leaves.
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), BTreeMap<&'static str, ArrayType>, ArrayType> =
            try_jit(&domain, |(), inputs: BTreeMap<&'static str, CompilationTracer<TestDomain>>| {
                inputs.into_values().next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        let first = BTreeMap::from([("a", Array::scalar(1.0)), ("b", Array::scalar(2.0))]);
        let second = BTreeMap::from([("b", Array::scalar(4.0)), ("a", Array::scalar(3.0))]);
        assert_eq!(function.call((), first).unwrap(), Array::scalar(1.0));
        assert_eq!(function.call((), second).unwrap(), Array::scalar(3.0));
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_rejects_same_key_recursive_dispatch() {
        let domain = TestDomain::new();
        let recursive: Rc<RefCell<Option<Box<dyn Fn() -> Result<Array, ProgramError>>>>> = Rc::new(RefCell::new(None));
        let closure_recursive = recursive.clone();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            try_jit(&domain, move |_, input: CompilationTracer<TestDomain>| {
                if let Some(call) = closure_recursive.borrow().as_ref() {
                    call()?;
                }
                Ok(input)
            });
        let function_clone = function.clone();
        *recursive.borrow_mut() = Some(Box::new(move || function_clone.call(true, Array::scalar(1.0))));

        assert!(matches!(
            function.call(true, Array::scalar(1.0)),
            Err(ProgramError::InvalidArgument { message })
                if message == "recursive JIT dispatch requested a specialization that is already being produced",
        ));
        assert_eq!(function.specialization_count(), 0);
        assert_eq!(function.statistics().dispatch_misses, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_allows_different_key_recursive_dispatch() {
        let domain = TestDomain::new();
        let recursive: Rc<RefCell<Option<Box<dyn Fn() -> Result<Array, ProgramError>>>>> = Rc::new(RefCell::new(None));
        let closure_recursive = recursive.clone();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            try_jit(&domain, move |nested, input: CompilationTracer<TestDomain>| {
                if nested {
                    if let Some(call) = closure_recursive.borrow().as_ref() {
                        call()?;
                    }
                }
                Ok(input)
            });
        let function_clone = function.clone();
        *recursive.borrow_mut() = Some(Box::new(move || function_clone.call(false, Array::scalar(1.0))));

        assert_eq!(function.call(true, Array::scalar(2.0)).unwrap(), Array::scalar(2.0));
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(function.statistics().traces, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_retains_specialization_after_runtime_execution_failure() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            jit_with_options(&domain, |(), input: CompilationTracer<TestDomain>| input, options);

        for _ in 0..2 {
            assert!(matches!(
                function.call((), Array::scalar(2.0)),
                Err(ProgramError::InvalidArgument { message })
                    if message == "runtime input type f64[] does not refine declared type i64[]",
            ));
        }

        // The compiled entry is inserted before runtime execution, so an execution failure does not evict it and the
        // second call dispatches directly to the retained specialization.
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_and_staged_functions_are_send_and_sync_for_thread_safe_state() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        fn assert_dispatcher_send_and_sync<F: Send + Sync>(
            _function: &CompiledFunctionDispatcher<TestDomain, F, bool, ArrayType, ArrayType>,
        ) {
            assert_send_and_sync::<CompiledFunctionDispatcher<TestDomain, F, bool, ArrayType, ArrayType>>();
        }

        assert_send_and_sync::<StagedFunction<TestDomain, ArrayType, ArrayType>>();
        assert_send_and_sync::<LoweredFunction<TestDomain, ArrayType, ArrayType>>();
        assert_send_and_sync::<CompiledFunction<TestDomain, ArrayType, ArrayType>>();

        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );
        assert_dispatcher_send_and_sync(&function);
    }

    #[test]
    fn test_compiled_function_dispatcher_serves_concurrent_warm_calls() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );
        function.call(true, Array::scalar(1.0)).unwrap();

        std::thread::scope(|scope| {
            for index in 0..4 {
                let function = function.clone();
                scope.spawn(move || {
                    let value = f64::from(index);
                    assert_eq!(function.call(true, Array::scalar(value)).unwrap(), Array::scalar(-value));
                });
            }
        });

        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 4);
        assert_eq!(function.statistics().traces, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_concurrent_cold_misses_produce_duplicates_and_one_entry() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );

        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let function = function.clone();
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    assert_eq!(function.call(true, Array::scalar(3.0)).unwrap(), Array::scalar(-3.0));
                });
            }
        });

        // Cross-thread same-key cold misses deliberately race and may duplicate frontend production; both inserts are
        // idempotent and exactly one retained entry remains, while the shared compilation context deduplicates the
        // backend compilation itself.
        assert_eq!(function.specialization_count(), 1);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits + statistics.dispatch_misses, 2);
        assert!(statistics.traces >= 1 && statistics.traces <= 2);
        assert_eq!(statistics.traces, statistics.dispatch_misses);
        assert_eq!(domain.compilation_count(), 1);
    }
}
