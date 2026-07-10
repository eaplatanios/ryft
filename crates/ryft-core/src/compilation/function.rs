use std::rc::Rc;
use std::sync::Arc;

use crate::compilation::captures::CapturingContext;
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::check_builders;
use crate::operations::Operation;
use crate::operations::constants::Constant;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{DomainTracingContext, Tracer};
use crate::types::Typed;

use super::captures::{CaptureReference, ClosedProgram};
use super::domain::CompilationDomain;
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
        Ok(CompiledFunction { program, lowered: self })
    }
}

/// Compiled executable plus the staged and lowered metadata required to invoke it safely.
pub struct CompiledFunction<
    D: CompilationDomain<Constant = CaptureReference<<D as Domain>::Type>>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant>>,
> {
    /// Shared backend executable.
    program: Arc<D::CompiledProgram>,

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
        Self { program: self.program.clone(), lowered: self.lowered.clone() }
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
        &self.program
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
        self.domain().compiled_output_types(&self.program)
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
        if flat_inputs.len() != self.staged().input_types().len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.staged().input_types().len(),
                actual: flat_inputs.len(),
            }
            .into());
        }
        for (expected, actual) in self.staged().input_types().iter().zip(flat_inputs.iter().map(Typed::r#type)) {
            self.domain().validate_input_type(expected, actual.as_ref())?;
        }

        let mut arguments = self.source_program().captures().to_vec();
        arguments.extend(flat_inputs);
        let flat_outputs = self.domain().execute(&self.program, arguments)?;
        let output_types = self.output_types();
        if flat_outputs.len() != output_types.len() {
            return Err(
                ProgramError::InvalidOutputCount { expected: output_types.len(), actual: flat_outputs.len() }.into()
            );
        }
        for (expected, actual) in output_types.iter().zip(flat_outputs.iter().map(Typed::r#type)) {
            self.domain().validate_output_type(expected, actual.as_ref())?;
        }
        Output::To::<D::Value>::from_parameters(self.staged().output_structure().clone(), flat_outputs)
            .map_err(|error| D::Error::from(error.into()))
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use crate::compilation::CompilationContext;
    use crate::operations::Operation;
    use crate::scalars::Scalar;
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
}
