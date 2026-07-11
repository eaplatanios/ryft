use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lru::LruCache;

use crate::captures::{CaptureReference, CapturingContext, ClosedProgram};
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::constants::Constant;
use crate::parameters::{ParameterError, ParameterPath, Parameterized, ParameterizedFamily};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{DomainTracingContext, Tracer};
use crate::types::Typed;

use super::contexts::CompilationDomain;

/// Flat source-program representation stored by nested compiled-call operations.
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
        D::Operation: Clone,
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
    fn lifted_program(&self) -> Result<Rc<FlatCompilationProgram<D>>, ProgramError>
    where
        D::Operation: Clone;

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
    executable: ExecutableProgram<D, Input, Output>,

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
    pub fn new(executable: &ExecutableProgram<D, Input, Output>, inputs: Input::To<D::Value>) -> Self {
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
    fn executable(&self) -> &ExecutableProgram<D, Self::Input, Self::Output>;

    /// Returns flat public runtime inputs, excluding captures.
    fn inputs(&self) -> &[D::Value];

    /// Consumes the request and prepends runtime captures to its flat inputs.
    fn into_arguments(self) -> Vec<D::Value>;

    /// Reconstructs structured runtime outputs.
    fn reconstruct(
        executable: &ExecutableProgram<D, Self::Input, Self::Output>,
        outputs: Vec<D::Value>,
    ) -> Result<Self::RuntimeOutput, D::Error>;
}

impl<D, F, Input, Output> StageRequest<D> for CompilationStagingRequest<D, F, Input, Output>
where
    D: CompilationDomain<Operation: Clone>,
    F: FnOnce(
        Vec<D::Constant>,
        Vec<CompilationTracer<D>>,
        Input::To<CompilationTracer<D>>,
    ) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
    Output: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
            To<CompilationTracer<D>>: Parameterized<
                CompilationTracer<D>,
                To<D::Type> = Output,
                To<D::Constant> = Output::To<D::Constant>,
            >,
        >,
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

/// Host value that participates in retained JIT trace specialization.
///
/// Static parameters are ordinary Rust values available to the traced closure. Equal values reuse one specialization;
/// unequal values trace independently. Runtime arrays and other backend values should remain dynamic inputs rather
/// than implementing this trait merely because they provide identity equality.
pub trait Specialization: Clone + Debug + Eq + Hash {}

impl<S: Clone + Debug + Eq + Hash> Specialization for S {}

/// Snapshot of one retained [`JittedFunction`]'s dispatch activity.
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

    /// Dispatch misses served by a retained trace without rerunning the Rust closure.
    pub trace_hits: u64,

    /// Dispatch misses served by a retained lowering without rerunning backend lowering.
    pub lowering_hits: u64,

    /// Total host nanoseconds spent flattening inputs and preparing their abstract types.
    pub input_abstractification_duration_ns: u64,

    /// Total host nanoseconds spent looking up retained specializations.
    pub dispatch_duration_ns: u64,

    /// Total host nanoseconds spent tracing specialization misses.
    pub tracing_duration_ns: u64,

    /// Total host nanoseconds spent lowering newly traced specializations.
    pub lowering_duration_ns: u64,
}

struct JitCacheStatisticsState {
    dispatch_hits: Cell<u64>,
    dispatch_misses: Cell<u64>,
    traces: Cell<u64>,
    lowerings: Cell<u64>,
    compilation_requests: Cell<u64>,
    trace_hits: Cell<u64>,
    lowering_hits: Cell<u64>,
    input_abstractification_duration_ns: Cell<u64>,
    dispatch_duration_ns: Cell<u64>,
    tracing_duration_ns: Cell<u64>,
    lowering_duration_ns: Cell<u64>,
}

impl JitCacheStatisticsState {
    fn new() -> Self {
        Self {
            dispatch_hits: Cell::new(0),
            dispatch_misses: Cell::new(0),
            traces: Cell::new(0),
            lowerings: Cell::new(0),
            compilation_requests: Cell::new(0),
            trace_hits: Cell::new(0),
            lowering_hits: Cell::new(0),
            input_abstractification_duration_ns: Cell::new(0),
            dispatch_duration_ns: Cell::new(0),
            tracing_duration_ns: Cell::new(0),
            lowering_duration_ns: Cell::new(0),
        }
    }

    fn snapshot(&self) -> JitCacheStatistics {
        JitCacheStatistics {
            dispatch_hits: self.dispatch_hits.get(),
            dispatch_misses: self.dispatch_misses.get(),
            traces: self.traces.get(),
            lowerings: self.lowerings.get(),
            compilation_requests: self.compilation_requests.get(),
            trace_hits: self.trace_hits.get(),
            lowering_hits: self.lowering_hits.get(),
            input_abstractification_duration_ns: self.input_abstractification_duration_ns.get(),
            dispatch_duration_ns: self.dispatch_duration_ns.get(),
            tracing_duration_ns: self.tracing_duration_ns.get(),
            lowering_duration_ns: self.lowering_duration_ns.get(),
        }
    }

    fn clear(&self) {
        self.dispatch_hits.set(0);
        self.dispatch_misses.set(0);
        self.traces.set(0);
        self.lowerings.set(0);
        self.compilation_requests.set(0);
        self.trace_hits.set(0);
        self.lowering_hits.set(0);
        self.input_abstractification_duration_ns.set(0);
        self.dispatch_duration_ns.set(0);
        self.tracing_duration_ns.set(0);
        self.lowering_duration_ns.set(0);
    }

    fn add_duration(counter: &Cell<u64>, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        counter.set(counter.get().saturating_add(nanoseconds));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct JitCacheKey<T, Static> {
    static_parameters: Static,
    input_paths: Vec<ParameterPath>,
    input_types: Vec<T>,
}

/// Default number of compiled specializations retained by one [`JittedFunction`].
const DEFAULT_JIT_CACHE_CAPACITY: usize = 256;

/// Independent retained-JIT cache capacities.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JitCacheCapacities {
    /// Number of traced specializations retained.
    pub traces: usize,
    /// Number of backend lowerings retained.
    pub lowerings: usize,
    /// Number of compiled direct-dispatch entries retained.
    pub dispatches: usize,
}

impl JitCacheCapacities {
    /// Uses `capacity` for every lifecycle cache.
    #[inline]
    pub const fn uniform(capacity: usize) -> Self {
        Self { traces: capacity, lowerings: capacity, dispatches: capacity }
    }
}

impl Default for JitCacheCapacities {
    fn default() -> Self {
        Self::uniform(DEFAULT_JIT_CACHE_CAPACITY)
    }
}

/// Operation-family capability for representing a call to a staged program.
///
/// The payload owns a flat program whose captures have been lifted into leading inputs. The concrete operation family
/// decides how that boundary lowers and how batching, differentiation, partial evaluation, and other transforms rewrite
/// it. This keeps higher-order call semantics with the operation that owns them while allowing the lifecycle and
/// capture plumbing to remain backend-neutral.
pub trait CompiledProgramOperation<Constant: Value>: Operation<Constant::Type> + Sized {
    /// Constructs a call operation for `program`.
    fn compiled_call(program: Rc<Program<Constant, Self, Vec<Constant>, Vec<Constant>>>) -> Self;
}

/// Staged, unlowered form of one compiled function.
///
/// A staged function owns the typed source [`ClosedProgram`], its runtime capture table, exact public input/output
/// structures, and compilation options. It can be inspected, lowered by its domain, or embedded as a compiled-call
/// boundary in an enclosing trace. No backend lowering or executable compilation has happened yet.
pub struct StagedFunction<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Shared immutable staging metadata. Staged-handle clones are therefore constant-time even for large programs.
    state: Rc<StagedFunctionState<D, Input, Output>>,
}

struct StagedFunctionState<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Source program and concrete runtime captures produced by tracing.
    source_program: ClosedProgram<D::Value, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>>,

    /// Flat declared public input types, excluding hidden captures.
    input_types: Vec<D::Type>,

    /// Flat declared output types in source-program order.
    output_types: Vec<D::Type>,

    /// Output parameter structure used to reconstruct structured values.
    output_structure: Output::ParameterStructure,

    /// Options applied before tracing and retained for lowering and compilation.
    options: Arc<D::Options>,

    /// Memoized source program with captures lifted into leading flat inputs.
    lifted_program: std::cell::OnceCell<Rc<FlatCompilationProgram<D>>>,
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> Clone for StagedFunction<D, Input, Output>
{
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> StagedFunction<D, Input, Output>
{
    /// Returns the source program and its runtime captures.
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<D::Value, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>> {
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
        D::Operation: Clone + CompiledProgramOperation<D::Constant>,
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
        D::Operation: Clone + CompiledProgramOperation<D::Constant>,
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
        let outputs = context.bind(D::Operation::compiled_call(self.lifted_program()?), flat_inputs.as_slice())?;
        Output::To::<V>::from_parameters(self.state.output_structure.clone(), outputs).map_err(Into::into)
    }

    /// Binds this staged function using capture references already registered in an enclosing capture table.
    pub fn call_with_flat_capture_references<V>(
        &self,
        capture_references: &[D::Constant],
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        D::Operation: Clone + CompiledProgramOperation<D::Constant>,
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
        D::Operation: Clone + CompiledProgramOperation<D::Constant>,
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
        context.bind(D::Operation::compiled_call(self.lifted_program()?), flat_inputs.as_slice())
    }

    /// Returns the source program with runtime captures lifted into leading flat inputs.
    #[doc(hidden)]
    pub fn lifted_program(&self) -> Result<Rc<FlatCompilationProgram<D>>, ProgramError>
    where
        D::Operation: Clone,
    {
        if let Some(program) = self.state.lifted_program.get() {
            return Ok(program.clone());
        }
        let program = Rc::new(self.state.source_program.to_program_with_lifted_captures()?);
        Ok(self.state.lifted_program.get_or_init(|| program).clone())
    }
}

impl<
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> LoweringRequest<D> for StagedFunction<D, Input, Output>
{
    type Input = Input;
    type Output = Output;

    fn staged(&self) -> &Self {
        self
    }

    fn lifted_program(&self) -> Result<Rc<FlatCompilationProgram<D>>, ProgramError>
    where
        D::Operation: Clone,
    {
        StagedFunction::lifted_program(self)
    }

    fn into_lowered(self, program: D::LoweredProgram, output_types: Vec<D::Type>) -> LoweredFunction<D, Input, Output> {
        LoweredFunction::from_parts(self, program, output_types)
    }
}

/// Backend lowering of a [`StagedFunction`], ready for executable compilation.
pub struct LoweredFunction<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
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

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> Clone for LoweredFunction<D, Input, Output>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), staged: self.staged.clone(), output_types: self.output_types.clone() }
    }
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> LoweredFunction<D, Input, Output>
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

impl<
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> CompileRequest<D> for LoweredFunction<D, Input, Output>
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
/// This handle retains only the state required to validate and execute calls: the backend executable,
/// concrete captures, flat signatures, and structured output shape. It deliberately does not retain the staged or
/// lowered programs, so it becomes [`Send`] and [`Sync`] automatically whenever those runtime fields are `Send + Sync`.
/// In contrast, [`CompiledFunction`] remains transformable and may retain `Rc`-backed source programs.
pub struct ExecutableProgram<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    state: Arc<ExecutableProgramState<D, Input, Output>>,
}

struct ExecutableProgramState<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    program: Arc<D::CompiledProgram>,
    captures: Vec<D::Value>,
    input_types: Vec<D::Type>,
    output_types: Vec<D::Type>,
    output_structure: Output::ParameterStructure,
    input: PhantomData<fn(Input)>,
}

impl<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> Clone
    for ExecutableProgram<D, Input, Output>
{
    #[inline]
    fn clone(&self) -> Self {
        Self { state: Arc::clone(&self.state) }
    }
}

impl<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>>
    ExecutableProgram<D, Input, Output>
{
    /// Replaces the backend payload after the backend has established call-boundary compatibility.
    #[doc(hidden)]
    pub fn with_compiled_program(&self, program: Arc<D::CompiledProgram>, output_types: Vec<D::Type>) -> Self {
        Self {
            state: Arc::new(ExecutableProgramState {
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

impl<D, Input, Output> ExecutableProgram<D, Input, Output>
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

    fn executable(&self) -> &ExecutableProgram<D, Input, Output> {
        &self.executable
    }

    fn inputs(&self) -> &[D::Value] {
        &self.inputs
    }

    fn into_arguments(self) -> Vec<D::Value> {
        self.executable.arguments_with_captures(self.inputs)
    }

    fn reconstruct(
        executable: &ExecutableProgram<D, Input, Output>,
        outputs: Vec<D::Value>,
    ) -> Result<Self::RuntimeOutput, D::Error> {
        executable.reconstruct_outputs(outputs).map_err(D::Error::from)
    }
}

/// Compiled executable plus the staged and lowered metadata required to invoke it safely.
pub struct CompiledFunction<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Runtime-only executable handle.
    executable_program: ExecutableProgram<D, Input, Output>,

    /// Lowered function retaining source metadata and options.
    lowered: LoweredFunction<D, Input, Output>,
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> Clone for CompiledFunction<D, Input, Output>
{
    fn clone(&self) -> Self {
        Self { executable_program: self.executable_program.clone(), lowered: self.lowered.clone() }
    }
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> CompiledFunction<D, Input, Output>
{
    /// Assembles a compiled function from backend-validated parts.
    #[doc(hidden)]
    pub fn from_parts(
        lowered: LoweredFunction<D, Input, Output>,
        program: Arc<D::CompiledProgram>,
        output_types: Vec<D::Type>,
    ) -> Self {
        let executable_program = ExecutableProgram {
            state: Arc::new(ExecutableProgramState {
                program,
                captures: lowered.staged.source_program().captures().to_vec(),
                input_types: lowered.staged.input_types().to_vec(),
                output_types,
                output_structure: lowered.staged.output_structure().clone(),
                input: PhantomData,
            }),
        };
        Self { executable_program, lowered }
    }

    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &D::CompiledProgram {
        self.executable_program.compiled_program()
    }

    /// Returns the runtime-only handle, which omits staged and lowered transform metadata.
    #[inline]
    pub fn executable_program(&self) -> &ExecutableProgram<D, Input, Output> {
        &self.executable_program
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable_program(self) -> ExecutableProgram<D, Input, Output> {
        self.executable_program
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
    ) -> &ClosedProgram<D::Value, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>> {
        self.staged().source_program()
    }

    /// Returns the effective flat output types produced by the executable.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        self.executable_program.output_types()
    }
}

/// Retained JIT dispatcher that caches compiled specializations of one Rust closure.
///
/// Unlike [`CompiledFunction`], which represents one already-specialized executable, a `JittedFunction` accepts
/// explicit host-side static parameters and runtime dynamic inputs. Its first call for a specialization traces,
/// lowers, and requests compilation; later calls with the same static values, parameter paths, and runtime-derived
/// abstract input types dispatch directly to the retained compiled function. Domain staging may normalize distinct
/// runtime signatures to the same staged signature, which can produce harmless duplicate dispatch specializations.
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
pub struct JittedFunction<
    D,
    F,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
{
    state: Rc<JittedFunctionState<D, F, Static, Input, Output>>,
}

struct JittedFunctionState<
    D,
    F,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
{
    domain: D,
    function: F,
    options: D::Options,
    traces: RefCell<LruCache<JitCacheKey<D::Type, Static>, StagedFunction<D, Input, Output>>>,
    lowerings: RefCell<LruCache<JitCacheKey<D::Type, Static>, LoweredFunction<D, Input, Output>>>,
    specializations: RefCell<LruCache<JitCacheKey<D::Type, Static>, CompiledFunction<D, Input, Output>>>,
    in_flight: RefCell<HashSet<JitCacheKey<D::Type, Static>>>,
    statistics: JitCacheStatisticsState,
}

impl<
    D,
    F,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> Clone for JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
{
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

struct JitProducerGuard<'a, Key: Eq + Hash> {
    in_flight: &'a RefCell<HashSet<Key>>,
    key: Option<Key>,
}

impl<'a, Key: Eq + Hash> JitProducerGuard<'a, Key> {
    fn new(in_flight: &'a RefCell<HashSet<Key>>, key: Key) -> Self {
        Self { in_flight, key: Some(key) }
    }
}

impl<Key: Eq + Hash> Drop for JitProducerGuard<'_, Key> {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            self.in_flight.borrow_mut().remove(&key);
        }
    }
}

impl<
    D,
    F,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
{
    fn new(domain: &D, function: F, options: D::Options, capacities: JitCacheCapacities) -> Self {
        let trace_capacity =
            NonZeroUsize::new(capacities.traces.max(1)).expect("JIT trace cache capacity is clamped to at least one");
        let lowering_capacity = NonZeroUsize::new(capacities.lowerings.max(1))
            .expect("JIT lowering cache capacity is clamped to at least one");
        let dispatch_capacity = NonZeroUsize::new(capacities.dispatches.max(1))
            .expect("JIT dispatch cache capacity is clamped to at least one");
        Self {
            state: Rc::new(JittedFunctionState {
                domain: domain.clone(),
                function,
                options,
                traces: RefCell::new(LruCache::new(trace_capacity)),
                lowerings: RefCell::new(LruCache::new(lowering_capacity)),
                specializations: RefCell::new(LruCache::new(dispatch_capacity)),
                in_flight: RefCell::new(HashSet::new()),
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
        self.state.specializations.borrow().len()
    }

    /// Returns the maximum number of compiled specializations retained by this dispatcher.
    #[inline]
    pub fn cache_capacity(&self) -> usize {
        self.state.specializations.borrow().cap().get()
    }

    /// Returns the independent trace, lowering, and dispatch cache capacities.
    pub fn cache_capacities(&self) -> JitCacheCapacities {
        JitCacheCapacities {
            traces: self.state.traces.borrow().cap().get(),
            lowerings: self.state.lowerings.borrow().cap().get(),
            dispatches: self.state.specializations.borrow().cap().get(),
        }
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
        self.state.traces.borrow_mut().clear();
        self.state.lowerings.borrow_mut().clear();
        self.state.specializations.borrow_mut().clear();
    }

    /// Invalidates every retained specialization for `static_parameters` and returns the number removed.
    pub fn invalidate_static(&self, static_parameters: &Static) -> usize {
        let mut keys = HashSet::new();
        for key in self.state.traces.borrow().iter().map(|(key, _)| key) {
            if &key.static_parameters == static_parameters {
                keys.insert(key.clone());
            }
        }
        for key in self.state.lowerings.borrow().iter().map(|(key, _)| key) {
            if &key.static_parameters == static_parameters {
                keys.insert(key.clone());
            }
        }
        for key in self.state.specializations.borrow().iter().map(|(key, _)| key) {
            if &key.static_parameters == static_parameters {
                keys.insert(key.clone());
            }
        }
        let mut traces = self.state.traces.borrow_mut();
        let mut lowerings = self.state.lowerings.borrow_mut();
        let mut specializations = self.state.specializations.borrow_mut();
        for key in &keys {
            traces.pop(key);
            lowerings.pop(key);
            specializations.pop(key);
        }
        keys.len()
    }

    /// Calls this dispatcher with explicit host-side `static_parameters` and dynamic runtime `inputs`.
    pub fn call(&self, static_parameters: Static, inputs: Input::To<D::Value>) -> Result<Output::To<D::Value>, D::Error>
    where
        D::Operation: Clone,
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
        let input_paths = inputs.parameter_paths().collect::<Vec<_>>();
        let input_structure = inputs.parameter_structure();
        let input_types = inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let input_types = Input::from_parameters(input_structure, input_types)
            .map_err(|error| D::Error::from(ProgramError::from(error)))?;
        let abstractification_duration = abstractification_start.elapsed();
        JitCacheStatisticsState::add_duration(
            &self.state.statistics.input_abstractification_duration_ns,
            abstractification_duration,
        );
        let key = JitCacheKey {
            static_parameters: static_parameters.clone(),
            input_paths,
            input_types: input_types.parameters().cloned().collect(),
        };

        let dispatch_start = Instant::now();
        if let Some(compiled) = self.state.specializations.borrow_mut().get(&key).cloned() {
            let dispatch_duration = dispatch_start.elapsed();
            JitCacheStatisticsState::add_duration(&self.state.statistics.dispatch_duration_ns, dispatch_duration);
            self.state.statistics.dispatch_hits.set(self.state.statistics.dispatch_hits.get().saturating_add(1));
            return call_function(&self.state.domain, compiled.executable_program(), inputs);
        }

        let dispatch_duration = dispatch_start.elapsed();
        JitCacheStatisticsState::add_duration(&self.state.statistics.dispatch_duration_ns, dispatch_duration);
        self.state
            .statistics
            .dispatch_misses
            .set(self.state.statistics.dispatch_misses.get().saturating_add(1));
        if !self.state.in_flight.borrow_mut().insert(key.clone()) {
            return Err(ProgramError::InvalidArgument {
                message: "recursive JIT dispatch requested a specialization that is already being produced".into(),
            }
            .into());
        }
        let _producer_guard = JitProducerGuard::new(&self.state.in_flight, key.clone());

        let lowered = if let Some(lowered) = self.state.lowerings.borrow_mut().get(&key).cloned() {
            self.state.statistics.lowering_hits.set(self.state.statistics.lowering_hits.get().saturating_add(1));
            lowered
        } else {
            let staged = if let Some(staged) = self.state.traces.borrow_mut().get(&key).cloned() {
                self.state.statistics.trace_hits.set(self.state.statistics.trace_hits.get().saturating_add(1));
                staged
            } else {
                self.state.statistics.traces.set(self.state.statistics.traces.get().saturating_add(1));
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
                self.state.traces.borrow_mut().put(key.clone(), staged.clone());
                staged
            };
            self.state.statistics.lowerings.set(self.state.statistics.lowerings.get().saturating_add(1));
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
            self.state.lowerings.borrow_mut().put(key.clone(), lowered.clone());
            lowered
        };
        self.state
            .statistics
            .compilation_requests
            .set(self.state.statistics.compilation_requests.get().saturating_add(1));
        let compiled = self.state.domain.compile(lowered)?;
        self.state.specializations.borrow_mut().put(key, compiled.clone());
        call_function(&self.state.domain, compiled.executable_program(), inputs)
    }
}

/// Constructs a retained dispatcher for a fallible closure using explicit options and cache capacity.
pub fn try_jit_with_options_and_capacity<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
    capacity: usize,
) -> JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    try_jit_with_options_and_capacities(domain, function, options, JitCacheCapacities::uniform(capacity))
}

/// Constructs a retained dispatcher for a fallible closure using explicit options and lifecycle capacities.
pub fn try_jit_with_options_and_capacities<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
    capacities: JitCacheCapacities,
) -> JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    JittedFunction::new(domain, function, options, capacities)
}

/// Constructs a retained dispatcher for a fallible closure using explicit options.
#[inline]
pub fn try_jit_with_options<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
) -> JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    try_jit_with_options_and_capacities(domain, function, options, JitCacheCapacities::default())
}

/// Constructs a retained dispatcher for a fallible closure using default options.
#[inline]
pub fn try_jit<D, F, Static, Input, Output>(domain: &D, function: F) -> JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Options: Default>,
    D::Type: Eq + Hash,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    try_jit_with_options(domain, function, D::Options::default())
}

/// Constructs a retained dispatcher for an infallible closure using explicit options.
pub fn jit_with_options<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: D::Options,
) -> JittedFunction<
    D,
    impl Fn(Static, Input::To<CompilationTracer<D>>) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Static,
    Input,
    Output,
>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
    F: Fn(Static, Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
    Output:
        Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
{
    try_jit_with_options(domain, move |static_parameters, inputs| Ok(function(static_parameters, inputs)), options)
}

/// Constructs a retained dispatcher for an infallible closure using default options.
#[inline]
pub fn jit<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
) -> JittedFunction<
    D,
    impl Fn(Static, Input::To<CompilationTracer<D>>) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Static,
    Input,
    Output,
>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Options: Default>,
    D::Type: Eq + Hash,
    F: Fn(Static, Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
    Output:
        Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
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
    D: CompilationDomain<Operation: Clone>,
    F: FnOnce(Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
    Output: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
            To<CompilationTracer<D>>: Parameterized<
                CompilationTracer<D>,
                To<D::Type> = Output,
                To<D::Constant> = Output::To<D::Constant>,
            >,
        >,
{
    domain.stage(CompilationStagingRequest::new(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, options))
}

/// Executes a structured runtime call through `domain`.
pub fn call_function<D, Input, Output>(
    domain: &D,
    executable: &ExecutableProgram<D, Input, Output>,
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
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
    F: FnOnce(
        Vec<D::Constant>,
        Vec<CompilationTracer<D>>,
        Input::To<CompilationTracer<D>>,
    ) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>>,
    Output: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<CompilationTracer<D>>,
            To<CompilationTracer<D>>: Parameterized<
                CompilationTracer<D>,
                To<D::Type> = Output,
                To<D::Constant> = Output::To<D::Constant>,
            >,
        >,
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
    let source_program = ClosedProgram::new(program, captures)?.without_unused_captures()?;
    Ok(StagedFunction {
        state: Rc::new(StagedFunctionState {
            source_program,
            input_types: input_type_values,
            output_types,
            output_structure,
            options: Arc::new(options),
            lifted_program: std::cell::OnceCell::new(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::compilation::{CompilationCacheDomain, CompilationContext};
    use crate::operations::Operation;
    use crate::types::{DataType, Type, TypeError};

    use super::*;

    #[derive(Clone, Debug)]
    struct NegateOperation;

    impl Operation<DataType> for NegateOperation {
        fn name(&self) -> &'static str {
            "test_negate"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError {
                    message: format!("test_negate expects 1 input but got {}", input_types.len()),
                });
            }
            Ok(input_types.to_vec())
        }
    }

    #[derive(Clone)]
    struct TestLoweredProgram {
        program: FlatCompilationProgram<TestDomain>,
        capture_count: usize,
        output_types: Vec<DataType>,
        options: TestOptions,
    }

    struct TestCompiledProgram {
        program: FlatCompilationProgram<TestDomain>,
        output_types: Vec<DataType>,
    }

    #[derive(Clone, Debug, Default)]
    struct TestOptions {
        staged_input_type: Option<DataType>,
        lowered_output_type: Option<DataType>,
        compiled_output_type: Option<DataType>,
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
        type Type = DataType;
        type Value = Scalar;
        type Constant = CaptureReference<DataType>;
        type Operation = NegateOperation;
    }

    impl CompilationDomain for TestDomain {
        type LoweredProgram = TestLoweredProgram;
        type CompiledProgram = TestCompiledProgram;
        type Options = TestOptions;
        type Error = ProgramError;

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
            let program = staged.lifted_program()?;
            let mut output_types: Vec<DataType> = program
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
                    Ok(vec![-inputs[0]])
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
    ) -> CompiledFunction<TestDomain, DataType, DataType> {
        let staged = stage_function(
            domain,
            |input| if negate { input.unary(NegateOperation) } else { input },
            DataType::F64,
            TestOptions::default(),
        )
        .unwrap();
        domain.compile(domain.lower(staged).unwrap()).unwrap()
    }

    #[test]
    fn test_compilation_key_distinguishes_computations_from_one_call_site() {
        let domain = TestDomain::new();

        let identity = compile_from_one_call_site(&domain, false);
        let negate = compile_from_one_call_site(&domain, true);

        assert_eq!(
            call_function(&domain, identity.executable_program(), Scalar::from(3.0)).unwrap(),
            Scalar::from(3.0)
        );
        assert_eq!(call_function(&domain, negate.executable_program(), Scalar::from(3.0)).unwrap(), Scalar::from(-3.0));
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_staged_and_lowered_handles_reuse_one_compilation() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, DataType, DataType> = stage_function(
            &domain,
            |input: CompilationTracer<TestDomain>| input.unary(NegateOperation),
            DataType::F64,
            TestOptions::default(),
        )
        .unwrap();
        let staged_clone = staged.clone();
        assert!(std::ptr::eq(staged.options(), staged_clone.options()));

        let first = domain.compile(domain.lower(staged_clone).unwrap()).unwrap();
        let second = domain.compile(domain.lower(staged).unwrap()).unwrap();

        assert_eq!(call_function(&domain, first.executable_program(), Scalar::from(2.0)).unwrap(), Scalar::from(-2.0));
        assert_eq!(call_function(&domain, second.executable_program(), Scalar::from(4.0)).unwrap(), Scalar::from(-4.0));
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(domain.cache.statistics().memory_hits, 1);
    }

    #[test]
    fn test_staging_options_apply_before_tracing() {
        let domain = TestDomain::new();
        let options = TestOptions { staged_input_type: Some(DataType::I64), ..TestOptions::default() };
        let staged: StagedFunction<TestDomain, DataType, DataType> =
            stage_function(&domain, |input: CompilationTracer<TestDomain>| input, DataType::F64, options).unwrap();

        assert_eq!(staged.input_types(), &[DataType::I64]);
        assert_eq!(staged.output_types(), &[DataType::I64]);
        assert_eq!(staged.options().staged_input_type, Some(DataType::I64));
    }

    #[test]
    fn test_compiled_function_rejects_runtime_input_type_mismatch() {
        let domain = TestDomain::new();
        let compiled = compile_from_one_call_site(&domain, false);

        assert!(matches!(
            call_function(&domain, compiled.executable_program(), Scalar::from(3_i64)),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type i64 does not refine declared type f64",
        ));
    }

    #[test]
    fn test_compiled_function_executes_explicit_capture() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), DataType> = domain
            .stage(CompilationStagingRequest::new(
                |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
                vec![Scalar::from(7.0)],
                (),
                TestOptions::default(),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();

        assert_eq!(call_function(&domain, compiled.executable_program(), ()).unwrap(), Scalar::from(7.0));
        assert_eq!(compiled.source_program().captures(), &[Scalar::from(7.0)]);
    }

    #[test]
    fn test_executable_program_outlives_transform_metadata_and_executes_captures() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), DataType> = domain
            .stage(CompilationStagingRequest::new(
                |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
                vec![Scalar::from(7.0)],
                (),
                TestOptions::default(),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
        let executable = compiled.into_executable_program();

        assert_eq!(executable.captures(), &[Scalar::from(7.0)]);
        assert!(executable.input_types().is_empty());
        assert_eq!(executable.output_types(), &[DataType::F64]);
        assert_eq!(call_function(&domain, &executable, ()).unwrap(), Scalar::from(7.0));
    }

    #[test]
    fn test_executable_program_is_send_and_sync_for_thread_safe_runtime_state() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        assert_send_and_sync::<ExecutableProgram<TestDomain, DataType, DataType>>();

        let domain = TestDomain::new();
        let executable = compile_from_one_call_site(&domain, true).into_executable_program();
        let second = executable.clone();
        let first_domain = domain.clone();
        let second_domain = domain.clone();
        let first_thread =
            std::thread::spawn(move || call_function(&first_domain, &executable, Scalar::from(3.0)).unwrap());
        let second_thread =
            std::thread::spawn(move || call_function(&second_domain, &second, Scalar::from(4.0)).unwrap());

        assert_eq!(first_thread.join().unwrap(), Scalar::from(-3.0));
        assert_eq!(second_thread.join().unwrap(), Scalar::from(-4.0));
    }

    #[test]
    fn test_fallible_staging_propagates_closure_error() {
        let domain = TestDomain::new();
        let result: Result<StagedFunction<TestDomain, DataType, DataType>, ProgramError> =
            domain.stage(CompilationStagingRequest::new(
                |_, _, _| Err(ProgramError::InvalidArgument { message: "staging failed".into() }),
                Vec::new(),
                DataType::F64,
                TestOptions::default(),
            ));

        assert!(matches!(result, Err(ProgramError::InvalidArgument { message }) if message == "staging failed"));
    }

    #[test]
    fn test_staging_prunes_unused_captures() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, DataType, DataType> = domain
            .stage(CompilationStagingRequest::new(
                |_, _, input: CompilationTracer<TestDomain>| Ok(input),
                vec![Scalar::from(7.0)],
                DataType::F64,
                TestOptions::default(),
            ))
            .unwrap();

        assert!(staged.source_program().captures().is_empty());
    }

    #[test]
    fn test_lower_rejects_incompatible_output_type() {
        let domain = TestDomain::new();
        let options = TestOptions { lowered_output_type: Some(DataType::I64), ..TestOptions::default() };

        let staged: StagedFunction<TestDomain, DataType, DataType> =
            stage_function(&domain, |input: CompilationTracer<TestDomain>| input, DataType::F64, options).unwrap();
        assert!(matches!(
            domain.lower(staged),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64 does not refine declared type f64",
        ));
    }

    #[test]
    fn test_compile_rejects_incompatible_output_type() {
        let domain = TestDomain::new();
        let options = TestOptions { compiled_output_type: Some(DataType::I64), ..TestOptions::default() };
        let staged: StagedFunction<TestDomain, DataType, DataType> =
            stage_function(&domain, |input: CompilationTracer<TestDomain>| input, DataType::F64, options).unwrap();
        let lowered = domain.lower(staged).unwrap();

        assert!(matches!(
            domain.compile(lowered),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64 does not refine declared type f64",
        ));
    }

    #[test]
    fn test_jitted_function_reuses_warm_specializations() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, bool, DataType, DataType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );

        assert_eq!(function.call(true, Scalar::from(2.0)).unwrap(), Scalar::from(-2.0));
        assert_eq!(function.call(true, Scalar::from(3.0)).unwrap(), Scalar::from(-3.0));
        assert_eq!(function.call(false, Scalar::from(4.0)).unwrap(), Scalar::from(4.0));
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
    fn test_jitted_function_applies_staging_options_before_tracing() {
        let domain = TestDomain::new();
        let options = TestOptions { staged_input_type: Some(DataType::I64), ..TestOptions::default() };
        let function: JittedFunction<TestDomain, _, (), DataType, DataType> =
            jit_with_options(&domain, |(), input: CompilationTracer<TestDomain>| input, options);

        assert!(matches!(
            function.call((), Scalar::from(2.0)),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type f64 does not refine declared type i64",
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
    fn test_jitted_function_distinguishes_hash_colliding_static_values() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, CollidingStatic, DataType, DataType> = jit(
            &domain,
            |static_parameters: CollidingStatic, input: CompilationTracer<TestDomain>| {
                if static_parameters.0 { input.unary(NegateOperation) } else { input }
            },
        );

        assert_eq!(function.call(CollidingStatic(false), Scalar::from(2.0)).unwrap(), Scalar::from(2.0));
        assert_eq!(function.call(CollidingStatic(true), Scalar::from(2.0)).unwrap(), Scalar::from(-2.0));
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_jitted_function_parameter_paths_partition_structures() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, (), Vec<DataType>, DataType> =
            try_jit(&domain, |(), mut inputs: Vec<CompilationTracer<TestDomain>>| {
                inputs.drain(..1).next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        assert_eq!(function.call((), vec![Scalar::from(2.0)]).unwrap(), Scalar::from(2.0));
        assert_eq!(function.call((), vec![Scalar::from(2.0), Scalar::from(3.0)]).unwrap(), Scalar::from(2.0),);
        assert_eq!(function.specialization_count(), 2);
    }

    #[test]
    fn test_jitted_function_invalidates_one_static_specialization() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, bool, DataType, DataType> = jit(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                if negate { input.unary(NegateOperation) } else { input }
            },
        );
        function.call(true, Scalar::from(2.0)).unwrap();
        function.call(false, Scalar::from(2.0)).unwrap();

        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.invalidate_static(&true), 1);
        assert_eq!(function.specialization_count(), 1);
        function.call(true, Scalar::from(2.0)).unwrap();
        assert_eq!(function.statistics().traces, 3);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_jitted_function_retries_failed_specialization() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, bool, DataType, DataType> =
            try_jit(&domain, |fail, input: CompilationTracer<TestDomain>| {
                if fail {
                    Err(ProgramError::InvalidArgument { message: "expected trace failure".into() })
                } else {
                    Ok(input)
                }
            });

        for _ in 0..2 {
            assert!(matches!(
                function.call(true, Scalar::from(2.0)),
                Err(ProgramError::InvalidArgument { message }) if message == "expected trace failure",
            ));
        }
        assert_eq!(function.call(false, Scalar::from(2.0)).unwrap(), Scalar::from(2.0));
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 3);
    }

    #[test]
    fn test_jitted_function_lru_capacity_retraces_evicted_specialization() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, bool, DataType, DataType> = try_jit_with_options_and_capacity(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegateOperation) } else { input })
            },
            TestOptions::default(),
            1,
        );

        function.call(false, Scalar::from(1.0)).unwrap();
        function.call(true, Scalar::from(1.0)).unwrap();
        function.call(false, Scalar::from(1.0)).unwrap();

        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_misses, 3);
        assert_eq!(function.statistics().traces, 3);
        assert_eq!(domain.compilation_count(), 2, "the global exact compilation cache should reuse the evicted entry");
    }

    #[test]
    fn test_jitted_function_independent_caches_reuse_lowering_after_dispatch_eviction() {
        let domain = TestDomain::new();
        let function: JittedFunction<TestDomain, _, bool, DataType, DataType> = try_jit_with_options_and_capacities(
            &domain,
            |negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegateOperation) } else { input })
            },
            TestOptions::default(),
            JitCacheCapacities { traces: 2, lowerings: 2, dispatches: 1 },
        );

        function.call(false, Scalar::from(1.0)).unwrap();
        function.call(true, Scalar::from(1.0)).unwrap();
        function.call(false, Scalar::from(1.0)).unwrap();

        assert_eq!(function.cache_capacities(), JitCacheCapacities { traces: 2, lowerings: 2, dispatches: 1 });
        assert_eq!(function.statistics().dispatch_misses, 3);
        assert_eq!(function.statistics().traces, 2);
        assert_eq!(function.statistics().lowerings, 2);
        assert_eq!(function.statistics().lowering_hits, 1);
        assert_eq!(domain.compilation_count(), 2);
    }
}
