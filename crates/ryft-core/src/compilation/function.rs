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

use crate::compilation::captures::CapturingContext;
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::check_builders;
use crate::operations::Operation;
use crate::operations::constants::Constant;
use crate::parameters::{ParameterError, ParameterPath, Parameterized, ParameterizedFamily};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{DomainTracingContext, Tracer};
use crate::types::Typed;

use super::captures::{CaptureReference, ClosedProgram};
use super::contexts::{
    AnalyzableCompilationDomain, CompilationCacheLevel, CompilationCacheOutcome, CompilationDomain, CompilationEvent,
    CompilationMissReason,
};
use super::options::CompilationOptions;

/// Flat source-program representation stored by nested compiled-call operations.
pub type FlatCompilationProgram<D> = Program<
    <D as Domain>::Constant,
    <D as Domain>::Operation,
    Vec<<D as Domain>::Constant>,
    Vec<<D as Domain>::Constant>,
>;

/// Tracer used while staging a compiled function in `D`'s operation universe.
pub type CompilationTracer<D> = Tracer<DomainTracingContext<D, <D as Domain>::Value>>;

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
/// The payload owns a flat program whose captures have been opened as leading inputs. The concrete operation family
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
/// structures, and the compilation domain. It can be inspected, lowered for a backend, or embedded as a compiled-call
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

    /// Domain that will lower, compile, and execute this function.
    domain: D,

    /// Memoized source program with captures opened as leading flat inputs.
    opened_program: std::cell::OnceCell<Rc<FlatCompilationProgram<D>>>,
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
    D: AnalyzableCompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> CompiledFunction<D, Input, Output>
{
    /// Returns this executable's backend-owned analysis without recompiling it.
    #[inline]
    pub fn analysis(&self) -> Result<D::Analysis, D::Error> {
        self.executable.analysis()
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

    /// Returns the domain that owns this compilation lifecycle.
    #[inline]
    pub fn domain(&self) -> &D {
        &self.state.domain
    }

    /// Lowers this staged function with `options` without compiling an executable.
    pub fn lower(self, options: CompilationOptions<D>) -> Result<LoweredFunction<D, Input, Output>, D::Error>
    where
        D::Operation: Clone,
    {
        let options = options.into_options();
        self.state.domain.validate_staged_input_types(self.state.input_types.as_slice(), &options)?;
        let program = self.opened_program()?;
        let lowered_program = self.state.domain.lower(program, self.state.source_program.captures().len(), &options)?;
        let lowered_output_types = self.state.domain.lowered_output_types(&lowered_program);
        let actual_output_count = lowered_output_types.len();
        let expected_output_count = self.state.output_types.len();
        if actual_output_count != expected_output_count {
            return Err(ProgramError::InvalidOutputCount {
                expected: expected_output_count,
                actual: actual_output_count,
            }
            .into());
        }
        for (declared, actual) in self.state.output_types.iter().zip(lowered_output_types) {
            self.state.domain.validate_output_type(declared, actual)?;
        }
        Ok(LoweredFunction { program: Arc::new(lowered_program), staged: self, options: Arc::new(options) })
    }

    /// Lowers and compiles this staged function with `options`.
    #[inline]
    pub fn compile(self, options: CompilationOptions<D>) -> Result<CompiledFunction<D, Input, Output>, D::Error>
    where
        D::Operation: Clone,
    {
        self.lower(options)?.compile()
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
            + CapturingContext<D::Value>
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
            + CapturingContext<D::Value>
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
        self.state.source_program.validate_capture_inputs(capture_references.as_slice())?;

        let mut flat_inputs = capture_references
            .into_iter()
            .map(|capture| context.constant(capture))
            .collect::<Result<Vec<_>, _>>()?;
        flat_inputs.extend(inputs.into_parameters());
        let outputs =
            context.bind(D::Operation::compiled_call(self.opened_program()?.clone()), flat_inputs.as_slice())?;
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
        self.state.source_program.validate_capture_inputs(capture_references)?;
        let mut flat_inputs = capture_references
            .iter()
            .cloned()
            .map(|capture| context.constant(capture))
            .collect::<Result<Vec<_>, _>>()?;
        flat_inputs.extend(inputs);
        context.bind(D::Operation::compiled_call(self.opened_program()?.clone()), flat_inputs.as_slice())
    }

    fn opened_program(&self) -> Result<&Rc<FlatCompilationProgram<D>>, ProgramError>
    where
        D::Operation: Clone,
    {
        if let Some(program) = self.state.opened_program.get() {
            return Ok(program);
        }
        let program = Rc::new(self.state.source_program.open_captures_as_inputs()?);
        Ok(self.state.opened_program.get_or_init(|| program))
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

    /// Options used for lowering and compilation.
    options: Arc<D::Options>,
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> Clone for LoweredFunction<D, Input, Output>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), staged: self.staged.clone(), options: self.options.clone() }
    }
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> LoweredFunction<D, Input, Output>
{
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
        self.options.as_ref()
    }

    /// Returns the effective flat output types after lowering-time rewrites.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        self.staged.domain().lowered_output_types(self.program.as_ref())
    }

    /// Compiles this lowering, reusing the domain cache when configured.
    pub fn compile(self) -> Result<CompiledFunction<D, Input, Output>, D::Error> {
        let cache_key = self.staged.domain().compilation_key(self.program.as_ref(), self.options.as_ref())?;
        let program = match self.staged.domain().cache() {
            Some(cache) => cache.get_or_compile(self.staged.domain(), cache_key, || {
                self.staged.domain().compile(self.program.as_ref(), self.options.as_ref())
            })?,
            None => Arc::new(self.staged.domain().compile(self.program.as_ref(), self.options.as_ref())?),
        };
        let compiled_output_types = self.staged.domain().compiled_output_types(&program);
        let lowered_output_types = self.output_types();
        if compiled_output_types.len() != lowered_output_types.len() {
            return Err(ProgramError::InvalidOutputCount {
                expected: lowered_output_types.len(),
                actual: compiled_output_types.len(),
            }
            .into());
        }
        for (declared, actual) in lowered_output_types.iter().zip(compiled_output_types) {
            self.staged.domain().validate_output_type(declared, actual)?;
        }
        let compiled_output_types = compiled_output_types.to_vec();
        let executable = ExecutableFunction {
            state: Arc::new(ExecutableFunctionState {
                program,
                domain: self.staged.domain().clone(),
                captures: self.staged.source_program().captures().to_vec(),
                input_types: self.staged.input_types().to_vec(),
                output_types: compiled_output_types,
                output_structure: self.staged.output_structure().clone(),
                input: PhantomData,
            }),
        };
        Ok(CompiledFunction { executable, lowered: self })
    }
}

/// Runtime-only handle for one compiled executable.
///
/// This handle retains only the state required to validate and execute calls: the backend executable, domain,
/// concrete captures, flat signatures, and structured output shape. It deliberately does not retain the staged or
/// lowered programs, so it becomes [`Send`] and [`Sync`] automatically whenever those runtime fields are `Send + Sync`.
/// In contrast, [`CompiledFunction`] remains transformable and may retain `Rc`-backed source programs.
pub struct ExecutableFunction<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    state: Arc<ExecutableFunctionState<D, Input, Output>>,
}

struct ExecutableFunctionState<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>> {
    program: Arc<D::CompiledProgram>,
    domain: D,
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

impl<D: CompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>>
    ExecutableFunction<D, Input, Output>
{
    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &D::CompiledProgram {
        &self.state.program
    }

    /// Returns the domain that owns this executable.
    #[inline]
    pub fn domain(&self) -> &D {
        &self.state.domain
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

    /// Returns a runtime handle that preserves this function's call signature and captures while dispatching to
    /// `program`.
    ///
    /// The replacement is accepted only when its output signature is compatible with the existing executable. This
    /// is the backend-neutral installation primitive used by adaptive recompilation: callers can prepare a new
    /// executable without holding dispatch locks, validate it here, and atomically publish the returned handle.
    pub fn with_compiled_program(&self, program: Arc<D::CompiledProgram>) -> Result<Self, D::Error> {
        self.state.domain.validate_replacement(&self.state.program, &program)?;
        let output_types = self.state.domain.compiled_output_types(&program);
        let output_types = output_types.to_vec();
        Ok(Self {
            state: Arc::new(ExecutableFunctionState {
                program,
                domain: self.state.domain.clone(),
                captures: self.state.captures.clone(),
                input_types: self.state.input_types.clone(),
                output_types,
                output_structure: self.state.output_structure.clone(),
                input: PhantomData,
            }),
        })
    }

    /// Executes this function after validating the runtime input and output signatures.
    pub fn call(&self, inputs: Input::To<D::Value>) -> Result<Output::To<D::Value>, D::Error>
    where
        Input::Family: ParameterizedFamily<D::Value>,
        Output::Family: ParameterizedFamily<D::Value>,
        Input::To<D::Value>: Parameterized<D::Value>,
        Output::To<D::Value>:
            Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let flat_inputs = inputs.into_parameters().collect::<Vec<_>>();
        if flat_inputs.len() != self.state.input_types.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.state.input_types.len(),
                actual: flat_inputs.len(),
            }
            .into());
        }
        for (expected, actual) in self.state.input_types.iter().zip(flat_inputs.iter().map(Typed::r#type)) {
            self.state.domain.validate_input_type(expected, actual.as_ref())?;
        }

        let mut arguments = self.state.captures.clone();
        arguments.extend(flat_inputs);
        let flat_outputs = self.state.domain.execute(&self.state.program, arguments)?;
        if flat_outputs.len() != self.state.output_types.len() {
            return Err(ProgramError::InvalidOutputCount {
                expected: self.state.output_types.len(),
                actual: flat_outputs.len(),
            }
            .into());
        }
        for (expected, actual) in self.state.output_types.iter().zip(flat_outputs.iter().map(Typed::r#type)) {
            self.state.domain.validate_output_type(expected, actual.as_ref())?;
        }
        Output::To::<D::Value>::from_parameters(self.state.output_structure.clone(), flat_outputs)
            .map_err(|error| D::Error::from(error.into()))
    }
}

impl<D: AnalyzableCompilationDomain, Input: Parameterized<D::Type>, Output: Parameterized<D::Type>>
    ExecutableFunction<D, Input, Output>
{
    /// Returns this executable's backend-owned analysis without recompiling it.
    #[inline]
    pub fn analysis(&self) -> Result<D::Analysis, D::Error> {
        self.state.domain.analyze(&self.state.program)
    }
}

/// Compiled executable plus the staged and lowered metadata required to invoke it safely.
pub struct CompiledFunction<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Runtime-only executable handle.
    executable: ExecutableFunction<D, Input, Output>,

    /// Lowered function retaining source metadata, domain, and options.
    lowered: LoweredFunction<D, Input, Output>,
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> Clone for CompiledFunction<D, Input, Output>
{
    fn clone(&self) -> Self {
        Self { executable: self.executable.clone(), lowered: self.lowered.clone() }
    }
}

impl<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> CompiledFunction<D, Input, Output>
{
    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &D::CompiledProgram {
        self.executable.compiled_program()
    }

    /// Returns the runtime-only handle, which omits staged and lowered transform metadata.
    #[inline]
    pub fn executable(&self) -> &ExecutableFunction<D, Input, Output> {
        &self.executable
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable(self) -> ExecutableFunction<D, Input, Output> {
        self.executable
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

    /// Returns the domain that owns this executable.
    #[inline]
    pub fn domain(&self) -> &D {
        self.staged().domain()
    }

    /// Returns the effective flat output types produced by the executable.
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        self.executable.output_types()
    }

    /// Executes this function after validating the runtime input and output signatures.
    pub fn call(&self, inputs: Input::To<D::Value>) -> Result<Output::To<D::Value>, D::Error>
    where
        Input::Family: ParameterizedFamily<D::Value>,
        Output::Family: ParameterizedFamily<D::Value>,
        Input::To<D::Value>: Parameterized<D::Value>,
        Output::To<D::Value>:
            Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        self.executable.call(inputs)
    }
}

/// Retained JIT dispatcher that caches compiled specializations of one Rust closure.
///
/// Unlike [`CompiledFunction`], which represents one already-specialized executable, a `JittedFunction` accepts
/// explicit host-side static parameters and runtime dynamic inputs. Its first call for a specialization traces,
/// lowers, and requests compilation; later calls with the same static values, parameter paths, and prepared abstract
/// input types dispatch directly to the retained compiled function.
///
/// Tracing executes Rust host code only on a specialization miss. Host side effects inside `function` therefore run
/// once per retained specialization, not once per runtime call; observable per-call work must be represented by staged
/// effectful operations. High-cardinality static values can cause repeated tracing and LRU eviction, so arrays and
/// frequently changing data should remain dynamic inputs.
///
/// The dispatcher identity only namespaces this process-local specialization cache. Executable correctness and reuse
/// still depend on [`CompilationDomain::compilation_key`], which receives the complete lowering.
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

    fn record_event(
        &self,
        level: CompilationCacheLevel,
        outcome: CompilationCacheOutcome,
        duration: Duration,
        miss_reason: Option<CompilationMissReason>,
    ) {
        if let Some(cache) = self.state.domain.cache() {
            cache.record_event(CompilationEvent { level, outcome, duration, miss_reason });
        }
    }

    fn classify_dispatch_miss(&self, requested: &JitCacheKey<D::Type, Static>) -> CompilationMissReason {
        let specializations = self.state.specializations.borrow();
        if specializations.iter().any(|(key, _)| {
            key.static_parameters == requested.static_parameters && key.input_types == requested.input_types
        }) {
            CompilationMissReason::InputStructure
        } else if specializations.iter().any(|(key, _)| {
            key.static_parameters == requested.static_parameters && key.input_paths == requested.input_paths
        }) {
            CompilationMissReason::InputType
        } else if specializations
            .iter()
            .any(|(key, _)| key.input_paths == requested.input_paths && key.input_types == requested.input_types)
        {
            CompilationMissReason::StaticParameter
        } else {
            CompilationMissReason::NotFound
        }
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
        let input_types = self.state.domain.prepare_input_types(input_types, &self.state.options)?;
        let abstractification_duration = abstractification_start.elapsed();
        JitCacheStatisticsState::add_duration(
            &self.state.statistics.input_abstractification_duration_ns,
            abstractification_duration,
        );
        self.record_event(
            CompilationCacheLevel::InputAbstractification,
            CompilationCacheOutcome::Succeeded,
            abstractification_duration,
            None,
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
            self.record_event(CompilationCacheLevel::Dispatch, CompilationCacheOutcome::Hit, dispatch_duration, None);
            return compiled.call(inputs);
        }

        let miss_reason = self.classify_dispatch_miss(&key);
        let dispatch_duration = dispatch_start.elapsed();
        JitCacheStatisticsState::add_duration(&self.state.statistics.dispatch_duration_ns, dispatch_duration);
        self.state
            .statistics
            .dispatch_misses
            .set(self.state.statistics.dispatch_misses.get().saturating_add(1));
        self.record_event(
            CompilationCacheLevel::Dispatch,
            CompilationCacheOutcome::Miss,
            dispatch_duration,
            Some(miss_reason),
        );
        if !self.state.in_flight.borrow_mut().insert(key.clone()) {
            return Err(ProgramError::InvalidArgument {
                message: "recursive JIT dispatch requested a specialization that is already being produced".into(),
            }
            .into());
        }
        let _producer_guard = JitProducerGuard::new(&self.state.in_flight, key.clone());

        let lowering_lookup_start = Instant::now();
        let lowered = if let Some(lowered) = self.state.lowerings.borrow_mut().get(&key).cloned() {
            let duration = lowering_lookup_start.elapsed();
            self.state.statistics.lowering_hits.set(self.state.statistics.lowering_hits.get().saturating_add(1));
            self.record_event(CompilationCacheLevel::Lowering, CompilationCacheOutcome::Hit, duration, None);
            lowered
        } else {
            let tracing_lookup_start = Instant::now();
            let staged = if let Some(staged) = self.state.traces.borrow_mut().get(&key).cloned() {
                let duration = tracing_lookup_start.elapsed();
                self.state.statistics.trace_hits.set(self.state.statistics.trace_hits.get().saturating_add(1));
                self.record_event(CompilationCacheLevel::Trace, CompilationCacheOutcome::Hit, duration, None);
                staged
            } else {
                self.state.statistics.traces.set(self.state.statistics.traces.get().saturating_add(1));
                let tracing_start = Instant::now();
                let staged = match try_stage(
                    &self.state.domain,
                    |traced_inputs| (self.state.function)(static_parameters, traced_inputs),
                    input_types,
                ) {
                    Ok(staged) => {
                        let duration = tracing_start.elapsed();
                        JitCacheStatisticsState::add_duration(&self.state.statistics.tracing_duration_ns, duration);
                        self.record_event(
                            CompilationCacheLevel::Trace,
                            CompilationCacheOutcome::Succeeded,
                            duration,
                            None,
                        );
                        staged
                    }
                    Err(error) => {
                        let duration = tracing_start.elapsed();
                        JitCacheStatisticsState::add_duration(&self.state.statistics.tracing_duration_ns, duration);
                        self.record_event(
                            CompilationCacheLevel::Trace,
                            CompilationCacheOutcome::Failed,
                            duration,
                            Some(CompilationMissReason::ProducerFailed),
                        );
                        return Err(error);
                    }
                };
                self.state.traces.borrow_mut().put(key.clone(), staged.clone());
                staged
            };
            self.state.statistics.lowerings.set(self.state.statistics.lowerings.get().saturating_add(1));
            let lowering_start = Instant::now();
            let lowered = match staged.lower(CompilationOptions::new(self.state.options.clone())) {
                Ok(lowered) => {
                    let duration = lowering_start.elapsed();
                    JitCacheStatisticsState::add_duration(&self.state.statistics.lowering_duration_ns, duration);
                    self.record_event(
                        CompilationCacheLevel::Lowering,
                        CompilationCacheOutcome::Succeeded,
                        duration,
                        None,
                    );
                    lowered
                }
                Err(error) => {
                    let duration = lowering_start.elapsed();
                    JitCacheStatisticsState::add_duration(&self.state.statistics.lowering_duration_ns, duration);
                    self.record_event(
                        CompilationCacheLevel::Lowering,
                        CompilationCacheOutcome::Failed,
                        duration,
                        Some(CompilationMissReason::ProducerFailed),
                    );
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
        let compiled = lowered.compile()?;
        self.state.specializations.borrow_mut().put(key, compiled.clone());
        compiled.call(inputs)
    }
}

/// Constructs a retained dispatcher for a fallible closure using explicit options and cache capacity.
pub fn try_jit_with_options_and_capacity<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: CompilationOptions<D>,
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
    options: CompilationOptions<D>,
    capacities: JitCacheCapacities,
) -> JittedFunction<D, F, Static, Input, Output>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    D::Type: Eq + Hash,
    Static: Specialization,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
{
    JittedFunction::new(domain, function, options.into_options(), capacities)
}

/// Constructs a retained dispatcher for a fallible closure using explicit options.
#[inline]
pub fn try_jit_with_options<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: CompilationOptions<D>,
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
    try_jit_with_options(domain, function, CompilationOptions::default())
}

/// Constructs a retained dispatcher for an infallible closure using explicit options.
pub fn jit_with_options<D, F, Static, Input, Output>(
    domain: &D,
    function: F,
    options: CompilationOptions<D>,
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
    jit_with_options(domain, function, CompilationOptions::default())
}

/// Traces `function` once and returns a staged function without lowering or compiling it.
#[track_caller]
pub fn stage<D, F, Input, Output>(
    domain: &D,
    function: F,
    input_types: Input,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
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
    stage_with_captures(domain, |_, inputs| function(inputs), Vec::new(), input_types)
}

/// Traces a fallible `function` once and returns a staged function without lowering or compiling it.
pub fn try_stage<D, F, Input, Output>(
    domain: &D,
    function: F,
    input_types: Input,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
    F: FnOnce(Input::To<CompilationTracer<D>>) -> Result<Output::To<CompilationTracer<D>>, D::Error>,
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
    try_stage_with_captures(domain, |_, inputs| function(inputs), Vec::new(), input_types)
}

/// Traces `function` with explicit runtime captures and returns a staged function.
#[track_caller]
pub fn stage_with_captures<D, F, Input, Output>(
    domain: &D,
    function: F,
    captures: Vec<D::Value>,
    input_types: Input,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
    F: FnOnce(Vec<CompilationTracer<D>>, Input::To<CompilationTracer<D>>) -> Output::To<CompilationTracer<D>>,
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
    stage_with_capture_references(
        domain,
        |_, capture_tracers, inputs| function(capture_tracers, inputs),
        captures,
        input_types,
    )
}

/// Traces a fallible `function` with explicit runtime captures and returns a staged function.
pub fn try_stage_with_captures<D, F, Input, Output>(
    domain: &D,
    function: F,
    captures: Vec<D::Value>,
    input_types: Input,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
    F: FnOnce(
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
    try_stage_with_capture_references(
        domain,
        |_, capture_tracers, inputs| function(capture_tracers, inputs),
        captures,
        input_types,
    )
}

/// Traces `function` with explicit runtime captures while also exposing their symbolic references.
///
/// Most callers should use [`stage_with_captures`]. This lower-level form is useful when a transform needs to register
/// captures once and pass their references into an operation-owned nested-call rule during the same trace.
#[track_caller]
pub fn stage_with_capture_references<D, F, Input, Output>(
    domain: &D,
    function: F,
    captures: Vec<D::Value>,
    input_types: Input,
) -> Result<StagedFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
    F: FnOnce(
        Vec<D::Constant>,
        Vec<CompilationTracer<D>>,
        Input::To<CompilationTracer<D>>,
    ) -> Output::To<CompilationTracer<D>>,
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
    try_stage_with_capture_references(
        domain,
        |capture_references, capture_tracers, inputs| Ok(function(capture_references, capture_tracers, inputs)),
        captures,
        input_types,
    )
}

/// Traces a fallible `function` with explicit runtime captures while exposing their symbolic references.
pub fn try_stage_with_capture_references<D, F, Input, Output>(
    domain: &D,
    function: F,
    captures: Vec<D::Value>,
    input_types: Input,
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
    drop(outputs);
    drop(context);
    let captures = Rc::try_unwrap(capture_table).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let program = builder.build(output_ids, input_structure, output_structure.clone())?.into_simplified()?;
    let source_program = ClosedProgram::new(program, captures)?.prune_unused_captures()?;
    Ok(StagedFunction {
        state: Rc::new(StagedFunctionState {
            source_program,
            input_types: input_type_values,
            output_types,
            output_structure,
            domain: domain.clone(),
            opened_program: std::cell::OnceCell::new(),
        }),
    })
}

/// Traces, lowers, and compiles `function` with explicit options.
#[track_caller]
pub fn compile_with_options<D, F, Input, Output>(
    domain: &D,
    function: F,
    input_types: Input,
    options: CompilationOptions<D>,
) -> Result<CompiledFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone>,
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
    let options = options.into_options();
    let input_types = domain.prepare_input_types(input_types, &options)?;
    stage(domain, function, input_types)?.compile(CompilationOptions::new(options))
}

/// Traces, lowers, and compiles `function` with default options.
#[track_caller]
pub fn compile<D, F, Input, Output>(
    domain: &D,
    function: F,
    input_types: Input,
) -> Result<CompiledFunction<D, Input, Output>, D::Error>
where
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>, Operation: Clone, Options: Default>,
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
    compile_with_options(domain, function, input_types, CompilationOptions::default())
}

#[cfg(test)]
mod tests {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::compilation::CompilationContext;
    use crate::operations::Operation;
    use crate::types::{DataType, TypeError};

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
    }

    struct TestCompiledProgram {
        program: FlatCompilationProgram<TestDomain>,
        output_types: Vec<DataType>,
    }

    #[derive(Clone, Debug, Default)]
    struct TestOptions {
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
        type CompilationKey = Vec<String>;

        fn lower(
            &self,
            program: &FlatCompilationProgram<Self>,
            capture_count: usize,
            options: &TestOptions,
        ) -> Result<TestLoweredProgram, ProgramError> {
            let mut output_types: Vec<DataType> = program
                .output_ids()
                .iter()
                .map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned())
                .collect();
            if let Some(output_type) = &options.lowered_output_type {
                output_types.fill(output_type.clone());
            }
            Ok(TestLoweredProgram { program: program.clone(), capture_count, output_types })
        }

        fn lowered_output_types<'a>(&self, program: &'a TestLoweredProgram) -> &'a [DataType] {
            program.output_types.as_slice()
        }

        fn compilation_key(
            &self,
            program: &TestLoweredProgram,
            options: &TestOptions,
        ) -> Result<Vec<String>, ProgramError> {
            let mut key = vec![
                format!("captures:{}", program.capture_count),
                format!("atoms:{:?}", program.program.atoms()),
                format!("inputs:{:?}", program.program.input_ids()),
                format!("outputs:{:?}", program.program.output_ids()),
                format!("options:{options:?}"),
            ];
            key.extend(program.program.instructions().iter().map(|instruction| {
                format!("{}:{:?}:{:?}", instruction.operation().name(), instruction.inputs(), instruction.outputs(),)
            }));
            Ok(key)
        }

        fn compile(
            &self,
            program: &TestLoweredProgram,
            options: &TestOptions,
        ) -> Result<TestCompiledProgram, ProgramError> {
            self.compilations.fetch_add(1, Ordering::Relaxed);
            let mut output_types = program.output_types.clone();
            if let Some(output_type) = &options.compiled_output_type {
                output_types.fill(output_type.clone());
            }
            Ok(TestCompiledProgram { program: program.program.clone(), output_types })
        }

        fn compiled_output_types<'a>(&self, program: &'a TestCompiledProgram) -> &'a [DataType] {
            program.output_types.as_slice()
        }

        fn execute(&self, program: &TestCompiledProgram, inputs: Vec<Scalar>) -> Result<Vec<Scalar>, ProgramError> {
            program.program.interpret_with(
                inputs,
                |_, capture| {
                    Err(ProgramError::MalformedProgram(format!(
                        "opened test program retained capture {}",
                        capture.index(),
                    )))
                },
                |_, inputs| {
                    if inputs.len() != 1 {
                        return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() });
                    }
                    Ok(vec![-inputs[0]])
                },
            )
        }

        fn cache(&self) -> Option<&CompilationContext<Self>> {
            Some(&self.cache)
        }
    }

    fn compile_from_one_call_site(
        domain: &TestDomain,
        negate: bool,
    ) -> CompiledFunction<TestDomain, DataType, DataType> {
        compile_with_options(
            domain,
            |input| if negate { input.unary(NegateOperation) } else { input },
            DataType::F64,
            CompilationOptions::new(TestOptions::default()),
        )
        .unwrap()
    }

    #[test]
    fn test_compilation_key_distinguishes_computations_from_one_call_site() {
        let domain = TestDomain::new();

        let identity = compile_from_one_call_site(&domain, false);
        let negate = compile_from_one_call_site(&domain, true);

        assert_eq!(identity.call(Scalar::from(3.0)).unwrap(), Scalar::from(3.0));
        assert_eq!(negate.call(Scalar::from(3.0)).unwrap(), Scalar::from(-3.0));
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_staged_and_lowered_handles_reuse_one_compilation() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, DataType, DataType> =
            stage(&domain, |input: CompilationTracer<TestDomain>| input.unary(NegateOperation), DataType::F64).unwrap();

        let first = staged.clone().lower(CompilationOptions::new(TestOptions::default())).unwrap().compile().unwrap();
        let second = staged.lower(CompilationOptions::new(TestOptions::default())).unwrap().compile().unwrap();

        assert_eq!(first.call(Scalar::from(2.0)).unwrap(), Scalar::from(-2.0));
        assert_eq!(second.call(Scalar::from(4.0)).unwrap(), Scalar::from(-4.0));
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(domain.cache.statistics().memory_hits, 1);
    }

    #[test]
    fn test_compiled_function_rejects_runtime_input_type_mismatch() {
        let domain = TestDomain::new();
        let compiled = compile_from_one_call_site(&domain, false);

        assert!(matches!(
            compiled.call(Scalar::from(3_i64)),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type i64 does not refine declared type f64",
        ));
    }

    #[test]
    fn test_compiled_function_executes_explicit_capture() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), DataType> =
            stage_with_captures(&domain, |mut captures, ()| captures.remove(0), vec![Scalar::from(7.0)], ()).unwrap();
        let compiled = staged.compile(CompilationOptions::new(TestOptions::default())).unwrap();

        assert_eq!(compiled.call(()).unwrap(), Scalar::from(7.0));
        assert_eq!(compiled.source_program().captures(), &[Scalar::from(7.0)]);
    }

    #[test]
    fn test_executable_function_outlives_transform_metadata_and_executes_captures() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), DataType> =
            stage_with_captures(&domain, |mut captures, ()| captures.remove(0), vec![Scalar::from(7.0)], ()).unwrap();
        let compiled = staged.compile(CompilationOptions::new(TestOptions::default())).unwrap();
        let executable = compiled.into_executable();

        assert_eq!(executable.captures(), &[Scalar::from(7.0)]);
        assert!(executable.input_types().is_empty());
        assert_eq!(executable.output_types(), &[DataType::F64]);
        assert_eq!(executable.call(()).unwrap(), Scalar::from(7.0));
    }

    #[test]
    fn test_executable_function_is_send_and_sync_for_thread_safe_runtime_state() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        assert_send_and_sync::<ExecutableFunction<TestDomain, DataType, DataType>>();

        let domain = TestDomain::new();
        let executable = compile_from_one_call_site(&domain, true).into_executable();
        let second = executable.clone();
        let first_thread = std::thread::spawn(move || executable.call(Scalar::from(3.0)).unwrap());
        let second_thread = std::thread::spawn(move || second.call(Scalar::from(4.0)).unwrap());

        assert_eq!(first_thread.join().unwrap(), Scalar::from(-3.0));
        assert_eq!(second_thread.join().unwrap(), Scalar::from(-4.0));
    }

    #[test]
    fn test_fallible_staging_propagates_closure_error() {
        let domain = TestDomain::new();
        let result: Result<StagedFunction<TestDomain, DataType, DataType>, ProgramError> = try_stage(
            &domain,
            |_| Err(ProgramError::InvalidArgument { message: "staging failed".into() }),
            DataType::F64,
        );

        assert!(matches!(result, Err(ProgramError::InvalidArgument { message }) if message == "staging failed"));
    }

    #[test]
    fn test_staging_prunes_unused_captures() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, DataType, DataType> = stage_with_captures(
            &domain,
            |_, input: CompilationTracer<TestDomain>| input,
            vec![Scalar::from(7.0)],
            DataType::F64,
        )
        .unwrap();

        assert!(staged.source_program().captures().is_empty());
    }

    #[test]
    fn test_lower_rejects_incompatible_output_type() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, DataType, DataType> =
            stage(&domain, |input: CompilationTracer<TestDomain>| input, DataType::F64).unwrap();
        let options = TestOptions { lowered_output_type: Some(DataType::I64), ..TestOptions::default() };

        assert!(matches!(
            staged.lower(CompilationOptions::new(options)),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64 does not refine declared type f64",
        ));
    }

    #[test]
    fn test_compile_rejects_incompatible_output_type() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, DataType, DataType> =
            stage(&domain, |input: CompilationTracer<TestDomain>| input, DataType::F64).unwrap();
        let options = TestOptions { compiled_output_type: Some(DataType::I64), ..TestOptions::default() };
        let lowered = staged.lower(CompilationOptions::new(options)).unwrap();

        assert!(matches!(
            lowered.compile(),
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
            CompilationOptions::default(),
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
            CompilationOptions::default(),
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
