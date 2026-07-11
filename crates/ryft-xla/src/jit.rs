//! User-facing XLA compile-and-execute API.
//!
//! This module wraps the backend-neutral lifecycle from [`ryft_core::compilation`] with XLA-flavored handles and entry
//! points. [`jitted`] is the retained `jax.jit` analogue: it specializes on runtime-derived abstract signatures and
//! dispatches warm calls through cached compiled functions. [`compile`] instead traces and compiles exactly one explicit
//! abstract signature into a [`CompiledXlaFunction`]. [`stage`] is the trace-only counterpart: it returns a
//! [`StagedXlaFunction`] that can be embedded into outer traces via [`StagedXlaFunction::call`] and compiled later, so
//! functions that are only ever composed into larger programs never pay for their own executable. Staged functions
//! register runtime captures in their retained capture table instead of embedding runtime arrays in the IR, and every
//! compilation shares the domain's [`CompilationContext`](ryft_core::compilation::CompilationContext) cache.

use ryft_core::Batch;
use ryft_core::LinearizationTracer;
use ryft_core::Typed;
use ryft_core::batching::BatchAxis;
use ryft_core::compilation::{
    CapturingContext, ClosedProgram, CompilationDomain, CompilationStagingRequest, CompiledFunction, ExecutableProgram,
    JittedFunction as CoreJittedFunction, Specialization, StagedFunction, call_function,
    jit_with_options as core_jit_with_options, try_jit_with_options as core_try_jit_with_options,
};
use ryft_core::contexts::Context;
use ryft_core::differentiation::DifferentiationError;
use ryft_core::operations::constants::Constant;
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::programs::{ProgramError, Value};
use ryft_core::sharding::DeviceMesh;
use ryft_core::tracing::{DomainTracingContext, Tracer};
use ryft_core::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};
use ryft_core::types::ArrayType;
use ryft_pjrt::Execution;

use crate::Array;
use crate::experimental::domains::{XlaCompiledProgram, XlaDomain, XlaDomainError, XlaOptions};
use crate::experimental::ops::{XlaConstant, XlaOperation};
use crate::profile_guided::{AdaptiveProfileGuidedOptions, AdaptiveProfileGuidedXlaFunction};

/// Tracer leaf used while tracing an XLA compilation.
pub type XlaCompileTracer<'c> = Tracer<DomainTracingContext<XlaDomain<'c>, Array<'c>>>;

/// Retained XLA JIT dispatcher with explicit host-side static parameters.
///
/// A first call for each `(static parameters, dynamic parameter structure, dynamic abstract types)` specialization
/// traces, lowers, and requests compilation. Warm calls dispatch directly to the retained executable. Static values
/// should be low-cardinality configuration such as axes, shapes, or Boolean branch choices; arrays remain dynamic.
pub type JittedXlaFunction<'c, F, Static, In, Out> = CoreJittedFunction<XlaDomain<'c>, F, Static, In, Out>;

/// Constructs a retained dispatcher for a fallible XLA closure using explicit options.
pub fn try_jitted_with_options<'c, F, Static, In, Out>(
    function: F,
    domain: &XlaDomain<'c>,
    options: XlaOptions,
) -> JittedXlaFunction<'c, F, Static, In, Out>
where
    F: Fn(Static, In::To<XlaCompileTracer<'c>>) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    Static: Specialization,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaCompileTracer<'c>>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaCompileTracer<'c>>>,
{
    core_try_jit_with_options(domain, function, options)
}

/// Constructs a retained dispatcher for an infallible XLA closure using explicit options.
pub fn jitted_with_options<'c, F, Static, In, Out>(
    function: F,
    domain: &XlaDomain<'c>,
    options: XlaOptions,
) -> JittedXlaFunction<
    'c,
    impl Fn(Static, In::To<XlaCompileTracer<'c>>) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    Static: Specialization,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaCompileTracer<'c>>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaCompileTracer<'c>>>,
{
    core_jit_with_options(domain, function, options)
}

/// Constructs a retained dispatcher for an infallible XLA closure on `mesh`.
#[inline]
pub fn jitted<'c, F, Static, In, Out>(
    function: F,
    domain: &XlaDomain<'c>,
    mesh: DeviceMesh,
) -> JittedXlaFunction<
    'c,
    impl Fn(Static, In::To<XlaCompileTracer<'c>>) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    Static: Specialization,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaCompileTracer<'c>>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaCompileTracer<'c>>>,
{
    jitted_with_options(function, domain, XlaOptions::new(mesh))
}

/// Captured-constant output tree produced by tracing an XLA closure.
type XlaSourceProgramOutput<Out> = <Out as Parameterized<ArrayType>>::To<XlaConstant>;

/// Tracer leaf used while linearizing a compiled XLA function inside a compile trace.
/// Closure tracer type of the partial-evaluation-backed differentiation entry points (`gradient`, `vjp`, …) over the
/// jit compile trace.
type XlaCompileLinearizationTracer<'c> = LinearizationTracer<DomainTracingContext<XlaDomain<'c>, Array<'c>>>;

/// Staged-but-uncompiled XLA function handle. Returned by [`stage`] and [`stage_with_captures`].
///
/// Holds the traced source [`Program`](ryft_core::programs::Program) of one closure together with its captured
/// runtime [`Array`]s and input / output type metadata, **without** compiling a PJRT executable. This is the right
/// entry point for functions that are only ever composed into larger programs: [`Self::call`] embeds the staged
/// program into an active outer trace as a `jit_call` boundary, and [`XlaDomain::compile_staged_function`] produces a
/// [`CompiledXlaFunction`] when an executable is actually needed. Executable cache identity is derived from the
/// complete lowered computation and compile-relevant backend state, so equivalent lowerings can share an executable
/// even when they were produced at different Rust call sites.
pub struct StagedXlaFunction<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> {
    /// Backend-neutral staged function carrying the source program, captures, structures, and retained options.
    function: StagedFunction<XlaDomain<'c>, In, Out>,
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> Clone for StagedXlaFunction<'c, In, Out>
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> StagedXlaFunction<'c, In, Out>
{
    /// Returns the staged source [`Program`](ryft_core::Program) together with its captured runtime values. Useful for
    /// outer transforms (`grad` / `jvp` / `vjp` / `batch`), staged `jit_call` payloads, and diagnostics (printing the
    /// traced IR, instruction counts, graph rendering).
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<Array<'c>, XlaOperation, In::To<XlaConstant>, XlaSourceProgramOutput<Out>> {
        self.function.source_program()
    }

    /// Stages a call to this function into an active trace as a `jit_call` operation.
    ///
    /// This does not execute anything. It records a trace boundary carrying this function's retained source program
    /// so enclosing transforms can rewrite the boundary through the ordinary XLA operation rules. The call is
    /// value-generic: `V` is a plain [`Tracer`] under an ordinary trace, and a transform tracer (e.g. a
    /// forward-mode dual) when an enclosing transform differentiates or otherwise rewrites the boundary through the
    /// `jit_call` operation's own rules. Errors surface when capture registration or structured output reassembly
    /// fails in the active context.
    #[inline]
    pub fn call<V>(&self, inputs: In::To<V>) -> Result<Out::To<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
            + CapturingContext<Array<'c>>
            + Constant<V, XlaConstant>,
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<V>>,
        In::To<V>: Parameterized<V>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<V>>,
        Out::To<V>: Parameterized<V, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        self.function.call(inputs)
    }
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> From<StagedFunction<XlaDomain<'c>, In, Out>> for StagedXlaFunction<'c, In, Out>
{
    #[inline]
    fn from(function: StagedFunction<XlaDomain<'c>, In, Out>) -> Self {
        Self { function }
    }
}

/// Runtime-only XLA executable handle.
///
/// Unlike [`CompiledXlaFunction`], this type does not retain the `Rc`-backed staged program or lowering metadata used
/// by transforms. It can only execute and inspect its runtime signature. Its thread-safety is derived from its
/// backend state; no unsafe blanket implementation is used.
pub struct ExecutableXlaProgram<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> {
    function: ExecutableProgram<XlaDomain<'c>, In, Out>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for ExecutableXlaProgram<'c, In, Out> {
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> ExecutableXlaProgram<'c, In, Out> {
    /// Returns the flat output [`ArrayType`]s in executor order.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.function.output_types()
    }

    /// Returns the device mesh this executable runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.function.compiled_program().mesh()
    }
}

impl<'c> XlaDomain<'c> {
    /// Lowers and compiles an XLA function staged with options retained before tracing.
    pub fn compile_staged_function<In, Out>(
        &self,
        staged: StagedXlaFunction<'c, In, Out>,
    ) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    {
        let lowered = self.lower(staged.function)?;
        let function = self.compile(lowered)?;
        Ok(CompiledXlaFunction { function })
    }

    /// Executes an XLA runtime program on concrete [`Array`] inputs.
    #[inline]
    pub fn interpret<In, Out>(
        &self,
        executable: &ExecutableXlaProgram<'c, In, Out>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<Array<'c>>>,
        In::To<Array<'c>>: Parameterized<Array<'c>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<Array<'c>>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        call_function(self, &executable.function, inputs)
    }

    /// Enqueues an XLA runtime program and retains whole-execution completion, including for zero-output calls.
    pub fn interpret_async<In, Out>(
        &self,
        executable: &ExecutableXlaProgram<'c, In, Out>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Execution<Out::To<Array<'c>>>, XlaDomainError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<Array<'c>>>,
        In::To<Array<'c>>: Parameterized<Array<'c>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<Array<'c>>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let flat_inputs = inputs.into_parameters().collect::<Vec<_>>();
        executable.function.validate_flat_input_count(flat_inputs.as_slice())?;
        for (expected, actual) in executable.function.input_types().iter().zip(flat_inputs.iter().map(Typed::r#type)) {
            crate::experimental::domains::validate_xla_input_type(expected, actual.as_ref())?;
        }

        let arguments = executable.function.arguments_with_captures(flat_inputs);
        let execution = self.execute_compiled_async(executable.function.compiled_program(), arguments)?;
        let (flat_outputs, fence) = execution.into_parts();
        crate::experimental::domains::validate_runtime_outputs(
            executable.function.output_types(),
            flat_outputs.as_slice(),
        )?;
        let outputs = executable.function.reconstruct_outputs(flat_outputs).map_err(XlaDomainError::from)?;
        Ok(Execution::new(outputs, fence))
    }

    /// Replaces an executable program while preserving and validating its runtime signature.
    pub(crate) fn replace_executable_xla_program<In, Out>(
        &self,
        executable: &ExecutableXlaProgram<'c, In, Out>,
        program: std::sync::Arc<XlaCompiledProgram<'c>>,
    ) -> Result<ExecutableXlaProgram<'c, In, Out>, XlaDomainError>
    where
        In: Parameterized<ArrayType>,
        Out: Parameterized<ArrayType>,
    {
        self.validate_xla_replacement(executable.function.compiled_program(), &program)?;
        Ok(ExecutableXlaProgram {
            function: executable.function.with_compiled_program(program.clone(), program.output_types().to_vec()),
        })
    }

    /// Wraps a compiled function in an adaptive profile-guided dispatcher.
    ///
    /// The returned runtime-only handle samples a bounded number of baseline executions, recompiles the already
    /// lowered StableHLO through this domain's ordinary compilation cache with the aggregated XLA profile, and then
    /// atomically directs subsequent calls to the compatible optimized executable.
    pub fn adaptive_profile_guided_recompilation<In, Out>(
        &self,
        function: &CompiledXlaFunction<'c, In, Out>,
        options: AdaptiveProfileGuidedOptions,
    ) -> Result<AdaptiveProfileGuidedXlaFunction<'c, In, Out>, XlaDomainError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    {
        AdaptiveProfileGuidedXlaFunction::new(
            self.clone(),
            function.executable_program(),
            function.function.lowered().lowered_program().clone(),
            function.function.lowered().options().clone(),
            options,
        )
    }
}

/// Just-in-time compiled function handle. Returned by [`compile`], [`compile_with_options`], and
/// [`XlaDomain::compile_staged_function`].
///
/// Holds the cached PJRT-backed [`XlaCompiledProgram`] plus the [`StagedXlaFunction`] it was compiled from, whose
/// input / output type metadata marshals a [`Parameterized`] tree of [`Array`]s into the executable and reassembles
/// the outputs back into the user's expected output tree shape.
///
/// The retained staged function also keeps the **source [`Program`](ryft_core::Program)** that the execution domain
/// compiled into, exposed via [`Self::source_program`]. Useful for diagnostics (printing the traced IR, instruction
/// counts, graph rendering), for outer transforms, and for inner staging via [`Self::call`] with trace inputs, which
/// emits a `jit_call` boundary carrying the source program into the active outer trace context.
pub struct CompiledXlaFunction<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> {
    /// Backend-neutral compiled function backed by an XLA executable. Retains the staged source function,
    /// captured runtime buffers, output structure, and compilation options through its lowered metadata.
    function: CompiledFunction<XlaDomain<'c>, In, Out>,
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> Clone for CompiledXlaFunction<'c, In, Out>
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> CompiledXlaFunction<'c, In, Out>
{
    /// Returns the flat output [`ArrayType`]s in the order the executor produces them.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.function.output_types()
    }

    /// Returns a runtime-only handle that omits staged and lowered transform metadata.
    #[inline]
    pub fn executable_program(&self) -> ExecutableXlaProgram<'c, In, Out> {
        ExecutableXlaProgram { function: self.function.executable_program().clone() }
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable_program(self) -> ExecutableXlaProgram<'c, In, Out> {
        ExecutableXlaProgram { function: self.function.into_executable_program() }
    }

    /// Returns the staged function this executable was compiled from.
    #[inline]
    pub fn staged(&self) -> StagedXlaFunction<'c, In, Out> {
        self.function.staged().clone().into()
    }

    /// Returns the source [`Program`](ryft_core::Program) that produced the compiled artifact. Useful for outer
    /// transforms (`grad` / `jvp` / `vjp` / `batch`), staged `jit_call` payloads, and diagnostics (printing the traced
    /// IR, instruction counts, graph rendering).
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<Array<'c>, XlaOperation, In::To<XlaConstant>, XlaSourceProgramOutput<Out>> {
        self.function.source_program()
    }

    /// Returns the device mesh the compiled program runs against. Delegates to the cached
    /// [`XlaCompiledProgram::mesh`](crate::experimental::domains::XlaCompiledProgram::mesh).
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.function.compiled_program().mesh()
    }

    /// Stages a call to this compiled function into an active trace as a `jit_call` operation.
    ///
    /// Refer to the documentation of [`StagedXlaFunction::call`] for more information.
    #[inline]
    pub fn call<V>(&self, inputs: In::To<V>) -> Result<Out::To<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
            + CapturingContext<Array<'c>>
            + Constant<V, XlaConstant>,
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<V>>,
        In::To<V>: Parameterized<V>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<V>>,
        Out::To<V>: Parameterized<V, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        self.function.staged().call(inputs)
    }
}

/// Reverse-mode AD: compiles a new function that computes the gradient of a scalar-valued compiled function with
/// respect to its inputs. The original closure is never re-executed; [`Self::call`] emits a `jit_call` boundary, and
/// the active transform rewrites that operation through ordinary JVP and transpose rules.
impl<
    'c,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType, To = In> + ParameterizedFamily<XlaConstant>,
            To<ArrayType> = In,
            ParameterStructure: std::fmt::Debug + std::hash::Hash + PartialEq,
        >,
> CompiledXlaFunction<'c, In, ArrayType>
{
    /// Returns a new compiled function that computes the reverse-mode gradient of `self` with
    /// respect to its input. Mirrors `jax.grad(jax.jit(f))`.
    ///
    /// `self` must produce a single rank-0 scalar output (encoded by the `Out = ArrayType`
    /// impl-block constraint above). The returned compiled function has the same input shape
    /// and produces an output whose leaves carry the partial derivative at each input leaf.
    #[track_caller]
    pub fn gradient<'domain>(
        &'domain self,
        domain: &'domain XlaDomain<'c>,
    ) -> Result<CompiledXlaFunction<'c, In, In>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'c>> + ParameterizedFamily<XlaCompileLinearizationTracer<'c>>,
        In::To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<XlaCompileTracer<'c>> = In::To<XlaCompileTracer<'c>>,
                To<ArrayType> = In,
                To<XlaConstant> = In::To<XlaConstant>,
            >,
        In::To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<XlaCompileLinearizationTracer<'c>> = In::To<XlaCompileLinearizationTracer<'c>>,
            >,
        In::To<XlaCompileLinearizationTracer<'c>>: Parameterized<XlaCompileLinearizationTracer<'c>>,
    {
        let function = self;
        let staged = function.function.staged();
        let input_signature = staged.input_signature().map_err(|error| XlaDomainError::Array(error.into()))?;
        let mesh = function.mesh().clone();
        let captures = function.source_program().captures().to_vec();
        compile_with_flat_captures(
            move |capture_references, _, tracers| {
                let context = tracers
                    .parameters()
                    .next()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                Ok(context.gradient(
                    move |y| -> Result<XlaCompileLinearizationTracer<'c>, DifferentiationError> {
                        let outputs = staged.call_with_flat_capture_references(
                            capture_references.as_slice(),
                            y.into_parameters().collect(),
                        )?;
                        Ok(Parameterized::from_parameters(staged.output_structure().clone(), outputs)?)
                    },
                    tracers,
                )?)
            },
            captures,
            input_signature,
            domain,
            XlaOptions::new(mesh),
        )
    }
}

/// Forward-mode JVP packaged as a method. Mirrors `jax.jvp(jax.jit(f))`.
impl<
    'c,
    In: Clone
        + Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType, To = In> + ParameterizedFamily<XlaConstant>,
            To<ArrayType> = In,
            ParameterStructure: std::fmt::Debug + std::hash::Hash + PartialEq,
        >,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> CompiledXlaFunction<'c, In, Out>
{
    /// Returns a new compiled function that computes the forward-mode JVP of `self`. Mirrors
    /// `jax.jvp(f, primals, tangents)` packaged into one compiled function: the returned handle
    /// takes `(primals, tangents)` and returns `(primal_out, tangent_out)`.
    ///
    /// The implementation stages a `jit_call` operation and lets the ordinary JVP rule for that
    /// call operation build the tangent call boundary.
    #[track_caller]
    pub fn jvp<'domain>(
        &'domain self,
        domain: &'domain XlaDomain<'c>,
    ) -> Result<CompiledXlaFunction<'c, (In, In), (Out, Out)>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'c>, To = In::To<XlaCompileTracer<'c>>>,
        In::To<XlaCompileTracer<'c>>:
            Parameterized<XlaCompileTracer<'c>, Family = In::Family, ParameterStructure = In::ParameterStructure>,
        Out::Family: ParameterizedFamily<XlaCompileTracer<'c>, To = Out::To<XlaCompileTracer<'c>>>,
        Out::To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                Family = Out::Family,
                ParameterStructure = Out::ParameterStructure,
                To<ArrayType> = Out,
            >,
    {
        let function = self;
        let staged = function.function.staged();
        let input_signature = staged.input_signature().map_err(|error| XlaDomainError::Array(error.into()))?;
        let primals_and_tangents = (input_signature.clone(), input_signature);
        let mesh = function.mesh().clone();
        let captures = function.source_program().captures().to_vec();
        compile_with_flat_captures(
            move |capture_references, _, inputs| {
                let (primals, tangents) = inputs;
                let output_structure = staged.output_structure().clone();
                let primals = primals.into_parameters().collect::<Vec<_>>();
                let tangents = tangents.into_parameters().collect::<Vec<_>>();
                let context = primals
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                let (primal_outputs, tangent_outputs): (Vec<XlaCompileTracer<'c>>, Vec<XlaCompileTracer<'c>>) = context
                    .jvp(
                        move |inputs| staged.call_with_flat_capture_references(capture_references.as_slice(), inputs),
                        primals,
                        tangents,
                    )?;
                let primal_tree = Out::To::<_>::from_parameters(output_structure.clone(), primal_outputs)
                    .map_err(ProgramError::from)?;
                let tangent_tree =
                    Out::To::<_>::from_parameters(output_structure, tangent_outputs).map_err(ProgramError::from)?;
                Ok((primal_tree, tangent_tree))
            },
            captures,
            primals_and_tangents,
            domain,
            XlaOptions::new(mesh),
        )
    }

    /// Returns a new compiled function that runs `self` in parallel over `axis_size` batch items
    /// along a new leading axis of each input and output. Mirrors `jax.vmap(f)` with default
    /// `in_axes=0` / `out_axes=0`. Every input leaf gets a new leading axis of size `axis_size`;
    /// every output leaf is materialized with the batched axis at position 0. The batched
    /// leading axis is replicated for now.
    ///
    /// The implementation stages a `jit_call` operation and lets the ordinary batching rule for that
    /// call operation build the batched call boundary.
    ///
    /// # Limitation
    ///
    /// Programs that use `shard_map` or `linear_shard_map` will surface
    /// [`BatchingError::UnsupportedOperation`](ryft_core::batching::BatchingError) at batch time — the batching rules
    /// for those XLA-specific extension variants are not yet implemented. Non-shard-map ops (including the
    /// `reshard` and `sharding_constraint` sharding-control primitives) batch correctly through the per-op rules.
    #[track_caller]
    pub fn batch<'domain>(
        &'domain self,
        domain: &'domain XlaDomain<'c>,
        axis_size: usize,
    ) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'c>, To = In::To<XlaCompileTracer<'c>>>,
        In::To<XlaCompileTracer<'c>>:
            Parameterized<XlaCompileTracer<'c>, Family = In::Family, ParameterStructure = In::ParameterStructure>,
        Out::Family: ParameterizedFamily<XlaCompileTracer<'c>, To = Out::To<XlaCompileTracer<'c>>>,
        Out::To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                Family = Out::Family,
                ParameterStructure = Out::ParameterStructure,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
    {
        let function = self;
        let staged = function.function.staged();
        let unbatched_signature = staged.input_signature().map_err(|error| XlaDomainError::Array(error.into()))?;
        let input_structure = unbatched_signature.parameter_structure();
        let batched_leaves: Vec<ArrayType> = unbatched_signature
            .into_parameters()
            .map(|t| add_leading_batch_dim(t, axis_size))
            .collect::<Result<Vec<_>, _>>()?;
        let batched_input = In::from_parameters(input_structure, batched_leaves)
            .map_err(|error| XlaDomainError::Array(error.into()))?;
        let mesh = function.mesh().clone();
        let captures = function.source_program().captures().to_vec();
        compile_with_flat_captures(
            move |capture_references, _, batched_tracers| {
                let output_structure = staged.output_structure().clone();
                let output_count = function.output_types().len();
                let batched_tracers = batched_tracers.into_parameters().collect::<Vec<_>>();
                let input_count = batched_tracers.len();
                let context = batched_tracers
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                let flat_outputs: Vec<XlaCompileTracer<'c>> = context.batch(
                    move |inputs| staged.call_with_flat_capture_references(capture_references.as_slice(), inputs),
                    batched_tracers,
                    vec![BatchAxis::new(0); input_count],
                    vec![BatchAxis::new(0); output_count],
                    Some(axis_size),
                )?;
                Ok(Out::To::<_>::from_parameters(output_structure, flat_outputs).map_err(ProgramError::from)?)
            },
            captures,
            batched_input,
            domain,
            XlaOptions::new(mesh),
        )
    }
}

/// Adds a leading axis of `size` to `array_type`, replicated. Used by [`CompiledXlaFunction::batch`] to
/// construct the batched input signature. If `array_type` carried a sharding, the returned [`ArrayType`]
/// gets the same sharding with a leading [`ShardingDimension::replicated`] prepended so that
/// the sharding rank still matches the shape rank.
fn add_leading_batch_dim(array_type: ArrayType, size: usize) -> Result<ArrayType, XlaDomainError> {
    let mut dims: Vec<ryft_core::Size> = vec![ryft_core::Size::Static(size)];
    dims.extend(array_type.shape().dimensions().iter().copied());
    let sharding = match array_type.sharding() {
        Some(existing) => {
            let mut extended_dims = vec![ryft_core::sharding::ShardingDimension::replicated()];
            extended_dims.extend(existing.dimensions().iter().cloned());
            Some(
                ryft_core::sharding::Sharding::with_manual_axes(
                    existing.mesh().clone(),
                    extended_dims,
                    existing.unreduced_axes().clone(),
                    existing.reduced_axes().clone(),
                    existing.varying_manual_axes().clone(),
                )
                .map_err(|error| XlaDomainError::Array(error.into()))?,
            )
        }
        None => None,
    };
    ArrayType::new(array_type.data_type(), ryft_core::Shape::new(dims))
        .with_layout(array_type.layout().cloned())
        .with_sharding(sharding)
        .map_err(|error| XlaDomainError::Array(error.into()))
}

/// Compiles one explicit abstract signature of `function` and returns a [`CompiledXlaFunction`] that executes that
/// specialization on subsequent calls. Use [`jitted`] when runtime calls should select and retain specializations in
/// the style of `jax.jit`.
///
/// Equivalent to [`compile_with_options`] called with [`XlaOptions::new(mesh)`](XlaOptions::new).
///
/// The function is traced against the supplied `domain`, which lets nested compiled functions register runtime
/// captures in the same active trace. The resulting program is then compiled and executed against `domain`. Its
/// [`CompilationContext`](ryft_core::compilation::CompilationContext) structurally shares equivalent lowerings even
/// when they originate at different Rust call sites.
#[track_caller]
pub fn compile<
    'domain,
    'c: 'domain,
    F: FnOnce(In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    compile_with_options(function, input_types, domain, XlaOptions::new(mesh))
}

/// Same as [`compile`] but makes runtime arrays explicit captures of the compiled program.
///
/// The closure receives capture tracers first and ordinary input tracers second. Captures are compiled as hidden
/// executable arguments and are supplied from the returned [`CompiledXlaFunction`] at execution time, so callers of the
/// compiled function still pass only `In` inputs.
#[track_caller]
pub fn compile_with_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(Vec<XlaCompileTracer<'c>>, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    compile_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        domain,
        XlaOptions::new(mesh),
    )
}

/// Same as [`compile`] but accepts a full [`XlaOptions`] payload for XLA mesh placement, sharding overrides,
/// and per-input buffer donation flags.
#[track_caller]
pub fn compile_with_options<
    'domain,
    'c: 'domain,
    F: FnOnce(In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    compile_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, domain, options)
}

#[track_caller]
fn compile_with_flat_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(
        Vec<XlaConstant>,
        Vec<XlaCompileTracer<'c>>,
        In::To<XlaCompileTracer<'c>>,
    ) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    let staged = stage_with_flat_captures::<F, In, Out>(function, captures, input_types, domain, options)?;
    domain.compile_staged_function(staged)
}

/// Traces `function` into a staged program and returns a [`StagedXlaFunction`] handle, without compiling a PJRT
/// executable.
///
/// This is the trace-only counterpart of [`compile`]: use it for functions that are composed into larger programs
/// via [`StagedXlaFunction::call`] rather than executed directly, so no executable is built for them. Compile the
/// staged handle later with [`XlaDomain::compile_staged_function`] when direct execution is needed. `options` are
/// applied to the abstract input signature before tracing and retained for subsequent lowering and compilation.
#[track_caller]
pub fn stage<
    'domain,
    'c: 'domain,
    F: FnOnce(In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<StagedXlaFunction<'c, In, Out>, XlaDomainError> {
    stage_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, domain, options)
}

/// Same as [`stage`] but makes runtime arrays explicit captures of the staged program.
///
/// The closure receives capture tracers first and ordinary input tracers second, mirroring
/// [`compile_with_captures`]. Captures are retained on the staged handle and threaded through `jit_call` boundaries
/// when the handle is staged into outer traces via [`StagedXlaFunction::call`].
#[track_caller]
pub fn stage_with_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(Vec<XlaCompileTracer<'c>>, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<StagedXlaFunction<'c, In, Out>, XlaDomainError> {
    stage_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        domain,
        options,
    )
}

#[track_caller]
fn stage_with_flat_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(
        Vec<XlaConstant>,
        Vec<XlaCompileTracer<'c>>,
        In::To<XlaCompileTracer<'c>>,
    ) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'c>>,
            To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                To<ArrayType> = Out,
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<StagedXlaFunction<'c, In, Out>, XlaDomainError> {
    let function = domain.stage(CompilationStagingRequest::new(function, captures, input_types, options))?;
    Ok(StagedXlaFunction { function })
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
#[track_caller]
pub fn infer_output_types<
    F: FnOnce(In::To<XlaCompileTracer<'static>>) -> Out::To<XlaCompileTracer<'static>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'static>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'static>>,
            To<XlaCompileTracer<'static>>: Parameterized<XlaCompileTracer<'static>, To<ArrayType> = Out>,
        >,
>(
    function: F,
    input_types: In,
) -> Result<Out, ProgramError> {
    DomainTracingContext::<XlaDomain<'static>, Array<'static>>::infer_output_type(
        |tracers| Ok(function(tracers)),
        input_types,
    )
}

#[cfg(test)]
mod tests {
    use ryft_core::operations::differentiation::StopGradient;
    use ryft_core::operations::math::{Cos, Sin};
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use ryft_core::types::data_types::DataType;
    use ryft_core::types::{ArrayType, Shape, Size};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::experimental::domains::{XlaDomain, XlaDomainError, XlaOptions};
    use crate::experimental::ops::XlaOperation;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{
        AdaptiveProfileGuidedOptions, Array, CompiledXlaFunction, ExecutableXlaProgram, FromPjrt, JittedXlaFunction,
        StagedXlaFunction, XlaCompileTracer, compile, compile_with_captures, compile_with_options, infer_output_types,
        jitted, stage, stage_with_captures,
    };

    fn assert_send_sync<T: Send + Sync>() {}

    fn single_device_mesh(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap()
    }

    fn two_device_mesh(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let devices: Vec<Device> = client
            .addressable_devices()
            .unwrap()
            .iter()
            .take(2)
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap()
    }

    fn read_f32_array(client: &ryft_pjrt::Client<'_>, array: &Array<'_>) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<f32>(shard_bytes.as_slice())
    }

    #[test]
    fn test_jit_unary_function_runs_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_executable_xla_program_runs_without_transform_metadata() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableXlaProgram<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable_program();
        let input =
            Array::from_host_buffer(&client, input_type.clone(), mesh, values_to_bytes::<f32>(&[0.5]).as_slice())
                .unwrap();

        let output = domain.interpret(&executable, input).unwrap();

        assert_eq!(executable.output_types(), &[input_type]);
        assert!((read_f32_array(&client, &output)[0] - 0.5_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_executable_xla_program_rejects_foreign_client_before_enqueue() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let other_client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let other_mesh = single_device_mesh(&other_client);
        let domain = XlaDomain::new(&client);
        let other_domain = XlaDomain::new(&other_client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableXlaProgram<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable_program();
        let other_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(other_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let other_executable: ExecutableXlaProgram<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), other_input_type, &other_domain, other_mesh)
                .unwrap()
                .into_executable_program();
        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();

        assert!(matches!(
            other_domain.interpret(&executable, input.clone()),
            Err(XlaDomainError::ExecutableClientMismatch),
        ));
        assert!(matches!(
            other_domain.interpret_async(&executable, input),
            Err(XlaDomainError::ExecutableClientMismatch),
        ));
        assert!(matches!(
            domain.validate_xla_replacement(
                executable.function.compiled_program(),
                other_executable.function.compiled_program(),
            ),
            Err(XlaDomainError::ExecutableClientMismatch),
        ));
        assert!(matches!(
            other_domain.validate_xla_replacement(
                executable.function.compiled_program(),
                other_executable.function.compiled_program(),
            ),
            Err(XlaDomainError::ExecutableClientMismatch),
        ));
    }

    #[test]
    fn test_executable_xla_program_executes_across_threads() {
        assert_send_sync::<ExecutableXlaProgram<'static, ArrayType, ArrayType>>();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableXlaProgram<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable_program();
        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();

        let output =
            std::thread::scope(|scope| scope.spawn(move || domain.interpret(&executable, input)).join().unwrap())
                .unwrap();
        assert!((read_f32_array(&client, &output)[0] - 0.5_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_profile_guided_recompilation_rejects_cpu_at_construction() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type, &domain, mesh).unwrap();

        assert!(matches!(
            domain.adaptive_profile_guided_recompilation(&compiled, AdaptiveProfileGuidedOptions::default()),
            Err(XlaDomainError::InvalidCompilationOptions { reason }) if reason.contains("CUDA or ROCm"),
        ));
    }

    #[test]
    fn test_retained_jit_reuses_static_specializations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);
        let function: JittedXlaFunction<'_, _, bool, ArrayType, ArrayType> = jitted(
            |apply_sine, input: XlaCompileTracer<'_>| if apply_sine { input.sin().unwrap() } else { input },
            &domain,
            mesh.clone(),
        );

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();

        let sine = function.call(true, source.clone()).unwrap();
        let warm_sine = function.call(true, source.clone()).unwrap();
        let identity = function.call(false, source).unwrap();
        assert!((read_f32_array(&client, &sine)[0] - 0.5_f32.sin()).abs() < 1e-5);
        assert!((read_f32_array(&client, &warm_sine)[0] - 0.5_f32.sin()).abs() < 1e-5);
        assert!((read_f32_array(&client, &identity)[0] - 0.5).abs() < 1e-5);

        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
    }

    #[test]
    fn test_jit_stop_gradient_lowers_to_the_identity() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x.stop_gradient(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input * input;
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_jit_transfer_to_memory_round_trip_runs_end_to_end() {
        use ryft_core::tracing_v2::operations::TransferToMemory;
        use ryft_core::types::Memory;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();

        // Host offloading requires a pinned-host memory space on the target device: without one the lowered
        // `annotate_device_placement` annotations have nothing to legalize into, so skip on plugins that do not
        // expose it instead of failing.
        let devices = client.addressable_devices().unwrap();
        let has_pinned_host = devices[0]
            .addressable_memories()
            .unwrap()
            .iter()
            .any(|memory| memory.kind().map(|kind| kind == "pinned_host").unwrap_or(false));
        if !has_pinned_host {
            eprintln!("skipping transfer_to_memory smoke test: the plugin exposes no pinned_host memory space");
            return;
        }

        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            |x| x.transfer_to_memory(Memory::Host { pinned: true }).transfer_to_memory(Memory::Device),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input).abs() < 1e-6, "got {got}, expected {input}");
        }
    }

    /// Smoke test: after `compile` runs, the returned handle exposes the source
    /// [`Program`] that was traced. This is the foundation for diagnostics (printing the IR)
    /// and for transform composition / inner staging via
    /// [`CompiledXlaFunction::call`].
    #[test]
    fn test_compiled_function_retains_source_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // For `|x| x.sin()` with a single F32[4] input and a single F32[4] output, the program
        // should carry one input atom, one output atom, and at least one instruction (the sin).
        let source = compiled.source_program().program();
        assert_eq!(source.input_ids().len(), 1, "expected one program input for the unary closure");
        assert_eq!(source.output_ids().len(), 1, "expected one program output for the unary closure");
        assert!(
            !source.instructions().is_empty(),
            "traced program should carry at least one instruction (the body of x.sin())",
        );
    }

    #[test]
    fn test_interpret_async_retains_zero_output_completion() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ()> =
            compile(|_| (), input_type.clone(), &engine, mesh.clone()).unwrap();
        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[1.0]).as_slice()).unwrap();

        let execution = engine.interpret_async(&compiled.executable_program(), input).unwrap();
        assert_eq!(execution.output(), &());
        execution.fence().block_until_ready().unwrap();
        assert_eq!(execution.block_until_ready(), Ok(()));
    }

    /// Inner-composition smoke test: a compiled function can be staged into another
    /// `compile` closure as a sub-routine, producing the same result as if the
    /// whole computation were a single closure. Mirrors JAX's
    /// `jit(lambda x: jit(f)(x).cos())` pattern.
    ///
    /// Exercises [`CompiledXlaFunction::call`], which stages the retained source program behind a
    /// `jit_call` boundary in the active outer trace.
    #[test]
    fn test_compiled_function_staged_inside_compile() {
        use ryft_core::operations::math::Cos;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        // Inner: compile `f = |x| x.sin()`.
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // Outer: compile `g = |x| cos(inner(x))` by staging `inner` as one `jit_call` and applying `cos` to its
        // output.
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jit_call_count = outer
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .count();
        let inlined_sin_count = outer
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin(_)))
            .count();
        assert_eq!(jit_call_count, 1, "inner compiled function should stage as one jit_call");
        assert_eq!(inlined_sin_count, 0, "inner function body should not be inlined into the outer trace");

        // Execute and compare against the mathematical reference.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&outer.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input.sin().cos();
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_compile_with_captures_runs_with_hidden_capture_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();

        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(compiled.source_program().captures().len(), 1);
        assert_eq!(compiled.source_program().open_captures_as_inputs().unwrap().input_ids().len(), 2);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&compiled.executable_program(), input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_captured_compiled_function_stages_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        assert_eq!(outer.source_program().captures().len(), 1);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&outer.executable_program(), input).unwrap();
        let observed = read_f32_array(&client, &output);
        for (got, expected) in observed.iter().zip([3.0f32.sin(), 4.0f32.sin(), 5.0f32.sin(), 6.0f32.sin()]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_multiple_captured_compiled_functions_stage_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let left_bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let right_bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 10.0, 10.0, 10.0]).as_slice(),
        )
        .unwrap();
        let left: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![left_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let right: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![right_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| left.call(x.clone()).unwrap() + right.call(x).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(outer.source_program().captures().len(), 2);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&outer.executable_program(), input).unwrap();
        let observed = read_f32_array(&client, &output);
        for (got, expected) in observed.iter().zip([14.0, 16.0, 18.0, 20.0]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_jvp_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) =
            engine.interpret(&jvp_compiled.executable_program(), (primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_gradient_method_prunes_capture_unused_by_derivative() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let grad_compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();

        assert!(grad_compiled.source_program().captures().is_empty());

        let input =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&[3.0]).as_slice())
                .unwrap();
        let output = engine.interpret(&grad_compiled.executable_program(), input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![1.0]);
    }

    #[test]
    fn test_compile_can_stage_gradient_of_captured_compiled_function() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.25]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| (x + captures[0].clone()).sin().unwrap(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let context = x.context().clone();
                context.gradient(move |y| inner.call(y), x).expect("nested captured grad(jit) should stage")
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(gradient.source_program().captures().len(), 1);

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[point]).as_slice(),
            )
            .unwrap();
            let output = engine.interpret(&gradient.executable_program(), input).unwrap();
            let observed = read_f32_array(&client, &output);
            assert_eq!(observed.len(), 1);
            let expected = (point + 0.25).cos();
            assert!(
                (observed[0] - expected).abs() < 1e-5,
                "grad(sin(x + bias))({point}) expected ~{expected}, got {}",
                observed[0],
            );
        }
    }

    #[test]
    fn test_compile_can_stage_jvp_of_captured_compiled_function() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal, tangent)| {
                let context = primal.context().clone();
                context.jvp(move |x| inner.call(x), primal, tangent).expect("nested captured jvp(jit) should stage")
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) =
            engine.interpret(&jvp_compiled.executable_program(), (primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_batch_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(scalar_sharding).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            scalar_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            scalar_type,
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let batched: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.batch(&engine, 4).unwrap();

        assert_eq!(batched.source_program().captures().len(), 1);

        let batched_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let batched_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(batched_sharding)
            .unwrap();
        let input = Array::from_host_buffer(
            &client,
            batched_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&batched.executable_program(), input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    /// Outer-transform smoke test: applying `grad` to a `CompiledXlaFunction` produces a new compiled function that
    /// computes `d/dx f(x)`. This mirrors JAX's `grad(jit(f))` idiom.
    ///
    /// The mechanism: [`CompiledXlaFunction::call`] stages a `jit_call` operation inside a `grad` trace. The outer
    /// transform differentiates that operation through the same capture-free linearization machinery as primitive ops.
    #[test]
    fn test_gradient_of_compiled_function_round_trips() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        // Compile `f = |x| x.sin()`.
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let grad_compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();

        // d/dx sin(x) = cos(x). Verify at a few points.
        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let source = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = engine.interpret(&grad_compiled.executable_program(), source).unwrap();
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let observed = values_from_bytes::<f32>(shard_bytes.as_slice())[0];
            let expected = point.cos();
            assert!((observed - expected).abs() < 1e-5, "grad(sin)({point}) expected ~{expected}, got {observed}",);
        }
    }

    /// Equivalent to [`test_gradient_of_compiled_function_round_trips`] but through the
    /// [`CompiledXlaFunction::gradient`] method. Verifies that `inner.gradient()` produces a compiled function for
    /// which `engine.interpret(&function.executable_program(), point)` matches `cos(point)` for sample points.
    #[test]
    fn test_gradient_method_matches_in_trace_gradient() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let grad_compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let source = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = engine.interpret(&grad_compiled.executable_program(), source).unwrap();
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let observed = values_from_bytes::<f32>(shard_bytes.as_slice())[0];
            let expected = point.cos();
            assert!((observed - expected).abs() < 1e-5, "f.gradient()({point}) expected ~{expected}, got {observed}",);
        }
    }

    /// Equivalent `inner.gradient()` lowerings share one cache entry regardless of Rust call site.
    #[test]
    fn test_gradient_method_reuses_equivalent_lowering_across_call_sites() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let baseline = engine.cache_size();

        // Distinct Rust call sites produce the same transformed StableHLO and therefore share one entry.
        let _grad_a: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();
        let _grad_b: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();
        assert_eq!(engine.cache_size(), baseline + 1);

        // Repeating the same transformation remains a cache hit.
        let baseline = engine.cache_size();
        for _ in 0..3 {
            let _: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();
        }
        assert_eq!(engine.cache_size(), baseline);
    }

    /// For `f = |x| x.sin()`, `f.jvp()` produces a compiled function for which interpreting its executable program at
    /// `(primal, tangent)` returns `(sin(primal), cos(primal) * tangent)` within `1e-5`.
    #[test]
    fn test_jvp_method_returns_primal_and_tangent() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();
        let jit_call_count = jvp_compiled
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .count();
        let inlined_sin_count = jvp_compiled
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin(_)))
            .count();
        assert_eq!(jit_call_count, 2, "jvp(jit(f)) should stage separate primal and pushforward jit_call boundaries",);
        assert_eq!(inlined_sin_count, 0, "jvp(jit(f)) should not inline the callee body");

        for &(primal, tangent) in &[(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let primal_array = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([tangent].as_slice()).as_slice(),
            )
            .unwrap();
            let (primal_out, tangent_out) =
                engine.interpret(&jvp_compiled.executable_program(), (primal_array, tangent_array)).unwrap();
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let primal_observed = {
                let shard_bytes = primal_out
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                values_from_bytes::<f32>(shard_bytes.as_slice())[0]
            };
            let tangent_observed = {
                let shard_bytes = tangent_out
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                values_from_bytes::<f32>(shard_bytes.as_slice())[0]
            };
            let expected_primal = primal.sin();
            let expected_tangent = primal.cos() * tangent;
            assert!(
                (primal_observed - expected_primal).abs() < 1e-5,
                "jvp primal at (primal={primal}, tangent={tangent}): expected ~{expected_primal}, got {primal_observed}",
            );
            assert!(
                (tangent_observed - expected_tangent).abs() < 1e-5,
                "jvp tangent at (primal={primal}, tangent={tangent}): expected ~{expected_tangent}, got {tangent_observed}",
            );
        }
    }

    /// Driving a `jit_call` program through the capture-free forward path
    /// ([`Program::linearize`](ryft_core::Program::linearize), which dispatches
    /// [`JitCallOperation`]'s `jvp` rule) realizes `jvp(jit(f)) = jit(jvp f)`: the split keeps both compilation
    /// boundaries instead of inlining the callee.
    ///
    /// For `f = |x| x.sin()` wrapped in an outer `jit_call`, the split's primal half is itself a single
    /// `jit_call` producing the primal output plus one residual (and no inlined `Sin`/`Cos`), and its tangent half is a
    /// single `jit_call` consuming that residual (and no inlined `Sin`/`Cos`). Re-wrapping the two halves as a compiled
    /// `(x, dx) -> (primal, tangent)` function reproduces the legacy [`jvp`](CompiledXlaFunction::jvp) result —
    /// `(sin(x), cos(x) * dx)` — within `1e-5`.
    #[test]
    fn test_jvp_of_jit_call_preserves_boundary_and_matches_legacy_jvp() {
        use std::rc::Rc;

        use ryft_core::contexts::StagingContext;

        use crate::experimental::ops::{FlatXlaProgram, JitCallOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        // Inner `jit(sin)` staged as one `jit_call` inside an outer program, so the outer source program holds exactly
        // one `jit_call` instruction for the replay to differentiate.
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let inner_for_outer = inner.clone();
        let outer: StagedXlaFunction<'_, ArrayType, ArrayType> = stage(
            move |x| inner_for_outer.call(x).unwrap(),
            input_type.clone(),
            &engine,
            XlaOptions::new(mesh.clone()),
        )
        .unwrap();
        let outer_program = outer.source_program().program();
        assert_eq!(
            outer_program
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
                .count(),
            1,
            "outer program should stage the inner compiled function as exactly one jit_call",
        );

        // Linearize the outer `jit_call` program, flattened first because `Program::linearize` is defined on the
        // canonical flat form. This drives the replay through `JitCallOperation`'s `jvp` rule, which re-wraps the
        // split callee into a primal `jit_call` (outputs followed by residuals) and a tangent `jit_call` (input
        // tangents followed by residuals).
        let linearization = outer_program.to_flat_program().linearize().unwrap();
        assert!(linearization.residual_count() >= 1, "sin's pushforward needs at least one residual (cos(x))");

        // Both halves must keep the callee behind a `jit_call` boundary rather than inlining `sin`/`cos`.
        let count_operations = |program: &FlatXlaProgram| {
            let jit_calls = program
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
                .count();
            let inlined = program
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin(_) | XlaOperation::Cos(_)))
                .count();
            (jit_calls, inlined)
        };
        let (primal_jit_calls, primal_inlined) = count_operations(linearization.primal());
        let (tangent_jit_calls, tangent_inlined) = count_operations(linearization.tangent());
        assert_eq!(primal_jit_calls, 1, "primal half should keep the callee behind one jit_call");
        assert_eq!(primal_inlined, 0, "primal half should not inline the callee body");
        assert_eq!(tangent_jit_calls, 1, "tangent half should keep the pushforward behind one jit_call");
        assert_eq!(tangent_inlined, 0, "tangent half should not inline the pushforward body");
        assert_eq!(
            linearization.primal().output_types().len(),
            1 + linearization.residual_count(),
            "primal half should produce the primal output followed by the residuals",
        );
        assert_eq!(
            linearization.tangent().input_types().len(),
            1 + linearization.residual_count(),
            "tangent half should consume the input tangent followed by the residuals",
        );

        // Re-wrap the two halves as a compiled `(x, dx) -> (primal, tangent)` function. This mirrors the
        // structure the value-level reroute will stage and exercises the real XLA lowering and execution of both
        // jit_call boundaries.
        let (primal_half, tangent_half, residual_count) = linearization.into_parts();
        let (primal_half, tangent_half) = (Rc::new(primal_half), Rc::new(tangent_half));
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal_input, tangent_input)| {
                let context = primal_input.context().clone();
                let mut primal_outputs = context
                    .stage_operation(
                        XlaOperation::JitCall(Box::new(JitCallOperation::new(primal_half.clone()))),
                        &[primal_input],
                    )
                    .expect("primal jit_call should stage");
                let residuals = primal_outputs.split_off(1);
                assert_eq!(residuals.len(), residual_count, "primal half residual count should match linearization");
                let primal_output = primal_outputs.remove(0);
                let mut tangent_inputs = vec![tangent_input];
                tangent_inputs.extend(residuals);
                let tangent_output = context
                    .stage_operation(
                        XlaOperation::JitCall(Box::new(JitCallOperation::new(tangent_half.clone()))),
                        tangent_inputs.as_slice(),
                    )
                    .expect("tangent jit_call should stage")
                    .remove(0);
                (primal_output, tangent_output)
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let legacy_jvp: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();

        for &(primal, tangent) in &[(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let primal_array = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([tangent].as_slice()).as_slice(),
            )
            .unwrap();
            let (jvp_primal, jvp_tangent) = engine
                .interpret(&jvp_compiled.executable_program(), (primal_array.clone(), tangent_array.clone()))
                .unwrap();
            let (legacy_primal, legacy_tangent) =
                engine.interpret(&legacy_jvp.executable_program(), (primal_array, tangent_array)).unwrap();

            let jvp_primal_value = read_f32_array(&client, &jvp_primal)[0];
            let jvp_tangent_value = read_f32_array(&client, &jvp_tangent)[0];
            let legacy_primal_value = read_f32_array(&client, &legacy_primal)[0];
            let legacy_tangent_value = read_f32_array(&client, &legacy_tangent)[0];
            let expected_primal = primal.sin();
            let expected_tangent = primal.cos() * tangent;

            assert!(
                (jvp_primal_value - expected_primal).abs() < 1e-5,
                "primal at (primal={primal}, tangent={tangent}): expected ~{expected_primal}, \
                 got {jvp_primal_value}",
            );
            assert!(
                (jvp_tangent_value - expected_tangent).abs() < 1e-5,
                "tangent at (primal={primal}, tangent={tangent}): expected ~{expected_tangent}, \
                 got {jvp_tangent_value}",
            );
            assert!(
                (jvp_primal_value - legacy_primal_value).abs() < 1e-5,
                "primal {jvp_primal_value} should match legacy jvp primal {legacy_primal_value}",
            );
            assert!(
                (jvp_tangent_value - legacy_tangent_value).abs() < 1e-5,
                "tangent {jvp_tangent_value} should match legacy jvp tangent {legacy_tangent_value}",
            );
        }
    }

    /// For scalar `f = |x| x.sin()`, interpreting `f.batch(4)?.executable_program()` with an array of four inputs
    /// returns `[sin(x[0]), ..., sin(x[3])]` within `1e-5`.
    #[test]
    fn test_batch_method_batches_leading_axis() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_input_type =
            ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(scalar_sharding).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), scalar_input_type, &engine, mesh.clone()).unwrap();
        let batched: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.batch(&engine, 4).unwrap();
        let jit_call_count = batched
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .count();
        let inlined_sin_count = batched
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin(_)))
            .count();
        assert_eq!(jit_call_count, 1, "batch(jit(f)) should stage one batched jit_call boundary");
        assert_eq!(inlined_sin_count, 0, "batch(jit(f)) should not inline the callee body");

        let batched_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let batched_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(batched_sharding)
            .unwrap();
        let inputs = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            batched_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&inputs).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&batched.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed.len(), 4, "expected 4 batch item outputs");
        for (got, &input) in observed.iter().zip(inputs.iter()) {
            let expected = input.sin();
            assert!((got - expected).abs() < 1e-5, "batch(sin)({input}) expected ~{expected}, got {got}");
        }
    }

    #[test]
    fn test_jvp_and_batch_methods_compose_around_jit_call() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_input_type =
            ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(scalar_sharding).unwrap();
        let batched_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let batched_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(batched_sharding)
            .unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), scalar_input_type, &engine, mesh.clone()).unwrap();
        let jvp_inner: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();
        let batch_jvp: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            jvp_inner.batch(&engine, 4).unwrap();
        let batch_inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.batch(&engine, 4).unwrap();
        let jvp_batch: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            batch_inner.jvp(&engine).unwrap();

        let primals = [0.0f32, 0.25, 0.5, 1.0];
        let tangents = [1.0f32, 2.0, -0.5, 0.7];
        let make_array = |values: &[f32]| {
            Array::from_host_buffer(
                &client,
                batched_input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };

        for (label, compiled) in [("batch(jvp(jit(f)))", batch_jvp), ("jvp(batch(jit(f)))", jvp_batch)] {
            let (primal_output, tangent_output) = engine
                .interpret(&compiled.executable_program(), (make_array(&primals), make_array(&tangents)))
                .unwrap();
            let primal_observed = read_f32_array(&client, &primal_output);
            let tangent_observed = read_f32_array(&client, &tangent_output);
            for (index, (&primal, &tangent)) in primals.iter().zip(tangents.iter()).enumerate() {
                let expected_primal = primal.sin();
                let expected_tangent = primal.cos() * tangent;
                assert!(
                    (primal_observed[index] - expected_primal).abs() < 1e-5,
                    "{label} primal batch item {index}: expected ~{expected_primal}, got {}",
                    primal_observed[index],
                );
                assert!(
                    (tangent_observed[index] - expected_tangent).abs() < 1e-5,
                    "{label} tangent batch item {index}: expected ~{expected_tangent}, got {}",
                    tangent_observed[index],
                );
            }
        }
    }

    #[test]
    fn test_jit_binary_function_with_tuple_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(3)]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharding.clone()).unwrap();
        let compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile(|(a, b)| a + b, (input_type.clone(), input_type.clone()), &engine, mesh.clone()).unwrap();

        let a_values = [10.0f32, 20.0, 30.0];
        let b_values = [1.0f32, 2.0, 3.0];
        let a = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&a_values).as_slice(),
        )
        .unwrap();
        let b =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&b_values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), (a, b)).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_jit_cache_hit_on_repeated_call_site() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        assert_eq!(engine.cache_size(), 0);
        // Equivalent lowerings share one executable, so the second invocation is a cache hit.
        for _ in 0..2 {
            let _: CompiledXlaFunction<'_, ArrayType, ArrayType> =
                compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cache_size(), 1, "an equivalent repeated lowering should hit the cache");
    }

    #[test]
    fn test_jit_equivalent_lowerings_share_cache_across_call_sites() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        assert_eq!(engine.cache_size(), 0);
        // Cache identity is the complete lowering plus options and target, not the Rust source location.
        let _: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let _: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type, &engine, mesh).unwrap();
        assert_eq!(engine.cache_size(), 1);
    }

    #[test]
    fn test_compile_with_options_donates_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = XlaOptions::new(mesh.clone()).with_donate(true);
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        // The output remains independently readable after the donating call returns. Donation
        // is opaque from the host side — PJRT may reuse the input's device buffer for the
        // output, but the public API only observes the resulting `Array`.
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_donation_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        // Build a `donation_flags` vec whose length does not match the function's flat input
        // arity. `with_donate` enforces matching shape via the `Parameterized<bool>` bound on
        // its argument, so producing an arity mismatch requires setting `donation_flags`
        // directly on `XlaOptions`.
        let mut xla_options = XlaOptions::new(mesh.clone());
        xla_options.donation_flags = Some(vec![true, false, false]); // 3 entries, 1 input
        let options = xla_options;
        let result: Result<CompiledXlaFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    /// Unused non-parameter input metadata does not partition compilation when the complete lowering is identical.
    #[test]
    fn test_compile_reuses_lowering_when_unused_input_metadata_differs() {
        use ryft_core::parameters::Parameter;
        use ryft_macros::Parameterized;

        #[derive(Parameterized, Debug, Clone, PartialEq, Eq, Hash)]
        #[ryft(crate = "ryft_core")]
        struct HyperparamInput<P: Parameter> {
            array: P,
            batch_size: usize,
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        for batch_size in [32usize, 64usize] {
            let input = HyperparamInput { array: array_type.clone(), batch_size };
            let _: CompiledXlaFunction<'_, HyperparamInput<ArrayType>, ArrayType> =
                compile(|input| input.array.sin().unwrap(), input, &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cache_size(), 1);
    }

    #[test]
    fn test_compile_with_options_in_shardings_override_replaces_input_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        // input_type carries the abstract shape & dtype but a "wrong" sharding (replicated). The
        // `in_shardings` override replaces it with a 2-way shard along "x" before tracing, so
        // the compiled program shards the input across the 2-device mesh.
        let shape = Shape::new(vec![Size::Static(4)]);
        let abstract_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let abstract_input_type =
            ArrayType::new(DataType::F32, shape.clone()).with_sharding(abstract_sharding).unwrap();
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let xla_options = XlaOptions::new(mesh.clone()).with_in_shardings(vec![sharded.clone()]);
        let options = xla_options;
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), abstract_input_type, &engine, options).unwrap();

        // Build the input array under the overridden sharding so it matches the executable's
        // expected layout.
        let input_type = ArrayType::new(DataType::F32, shape).with_sharding(sharded).unwrap();
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        // Reassemble values from both shards in device order.
        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_in_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // Two shardings for one flat input — should fail.
        let xla_options = XlaOptions::new(mesh).with_in_shardings(vec![sharding.clone(), sharding]);
        let options = xla_options;
        let result: Result<CompiledXlaFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    #[test]
    fn test_compile_with_options_out_shardings_override_propagates_to_output_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharded.clone()).unwrap();
        // Override the output sharding to the same 2-way shard along "x" so the partitioner
        // emits a fully-sharded output and `Array`'s sharding metadata matches.
        let xla_options = XlaOptions::new(mesh.clone()).with_out_shardings(vec![sharded.clone()]);
        let options = xla_options;
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        // The returned Array should carry the overridden sharding.
        assert_eq!(compiled.output_types()[0].sharding(), Some(&sharded));

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_jit_implicitly_reshards_mismatched_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        // The executable expects a 2-way shard along "x", but the caller will pass a fully
        // replicated array. `CompiledXlaFunction::interpret` should silently reshard before executing.
        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharded.clone()).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let replicated_input_type = ArrayType::new(DataType::F32, shape).with_sharding(replicated).unwrap();
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            replicated_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        // Calling with a replicated source against a sharded-expecting executable would error
        // without implicit reshard. With reshard it should succeed and produce correct output.
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_out_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let xla_options = XlaOptions::new(mesh).with_out_shardings(vec![sharding.clone(), sharding]);
        let options = xla_options;
        let result: Result<CompiledXlaFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    #[test]
    fn test_infer_output_types_returns_output_types_without_compiling() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(7)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        let output_type: ArrayType = infer_output_types(|x| x.sin().unwrap(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `infer_output_types` must not have populated the compile cache.
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_jit_gradient_compiles_and_runs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();

        let primal: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = primal.gradient(&engine).unwrap();

        let input_value = 0.75f32;
        let source = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed.len(), 1);
        let expected = input_value.cos();
        assert!(
            (observed[0] - expected).abs() < 1e-5,
            "expected d/dx sin({input_value}) ~= {expected}, got {}",
            observed[0],
        );
    }

    /// Verifies that `sharding_constraint` works inside a `compile`-compiled function over an auto mesh axis: the
    /// propagation hint is staged into the trace and lowers to `sdy.sharding_constraint`, and the output array carries
    /// the (input-derived) sharding on each device.
    #[test]
    fn test_jit_with_sharding_constraint_constrains_output_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharded.clone()).unwrap();
        let target_sharding = sharded.clone();

        // The user invokes `sharding_constraint` directly inside the staged closure — it's compiled into the
        // same MLIR program as the rest of the function body.
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let constrained = crate::experimental::shard_map::sharding_constraint(x, target_sharding.clone())
                    .expect("staged sharding constraint should succeed");
                constrained.sin().unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    /// Multiple staged reshards inside one `compile` body compile into a single MLIR program with chained
    /// `sdy.sharding_constraint` ops — exactly one cache entry, exactly one PJRT execute per call. This is the
    /// async-pipelined regime: PJRT runs the whole compiled program in one shot without per-reshard host sync. The
    /// mesh axis is explicit so each reshard is a tracked transition and the final (replicated) sharding governs the
    /// output buffer.
    #[test]
    fn test_jit_with_chained_reshards_compiles_to_one_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let devices: Vec<Device> = client
            .addressable_devices()
            .unwrap()
            .iter()
            .take(2)
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharded.clone()).unwrap();
        let constraint_a = replicated.clone();
        let constraint_b = sharded.clone();
        let constraint_c = replicated;

        // Three staged reshards compose inside one closure. Each emits a `sdy.sharding_constraint` op into
        // the same MLIR program. After trace+compile, the executable runs all three in one PJRT dispatch.
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let a = crate::experimental::shard_map::reshard(x, constraint_a.clone()).unwrap();
                let b = crate::experimental::shard_map::reshard(a.sin().unwrap(), constraint_b.clone()).unwrap();
                crate::experimental::shard_map::reshard(b.sin().unwrap(), constraint_c.clone()).unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // One compile means one cache entry for the whole pipeline.
        assert_eq!(engine.cache_size(), 1, "three staged reshards should compile into one program");

        let values = [0.1f32, 0.2, 0.3, 0.4];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        // Final output is replicated (last constraint) — every device sees the full vector.
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input.sin().sin();
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    /// Composing `grad` with a staged `sharding_constraint`: the gradient flows through the hint because the
    /// constraint is self-adjoint under transposition (its linear transpose re-applies the same hint), mirroring JAX's
    /// `jax.grad(jax.jit(... with_sharding_constraint ...))` behavior.
    #[test]
    fn test_jit_with_grad_through_sharding_constraint_runs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();

        // d/dx sin(sharding_constraint(x, S)) = cos(x), because the constraint is the identity at the value
        // level and self-adjoint under transposition, so the gradient passes through.
        let target_sharding = sharding.clone();
        let primal: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                crate::experimental::shard_map::sharding_constraint(x, target_sharding.clone())
                    .unwrap()
                    .sin()
                    .unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = primal.gradient(&engine).unwrap();

        let input_value = 0.5f32;
        let source = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&compiled.executable_program(), source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed.len(), 1);
        let expected = input_value.cos();
        assert!(
            (observed[0] - expected).abs() < 1e-5,
            "expected d/dx sin(sharding_constraint(x, S)) ~= cos({input_value}) = {expected}, got {}",
            observed[0],
        );
    }

    /// Staging alone must not build a PJRT executable; compiling the staged handle afterwards must match a direct
    /// [`compile`] of the same closure numerically.
    #[test]
    fn test_stage_then_compile_matches_direct_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let staged: StagedXlaFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), &engine, XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(engine.cache_size(), 0, "staging must not compile a PJRT executable");

        let staged_compiled = engine.compile_staged_function(staged).unwrap();
        assert_eq!(engine.cache_size(), 1, "compiling the staged handle should build exactly one executable");

        let direct: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let input_value = 0.5f32;
        let make_input = || {
            Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
            )
            .unwrap()
        };
        let staged_output =
            read_f32_array(&client, &engine.interpret(&staged_compiled.executable_program(), make_input()).unwrap());
        let direct_output =
            read_f32_array(&client, &engine.interpret(&direct.executable_program(), make_input()).unwrap());
        assert_eq!(staged_output, direct_output);
        assert!((staged_output[0] - input_value.sin()).abs() < 1e-6);
    }

    /// A staged-but-never-compiled function with captures can be called inside an outer [`compile`], threading its
    /// captures into the outer compiled function. Only the outer program is compiled.
    #[test]
    fn test_staged_function_call_stages_into_outer_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.25]).as_slice(),
        )
        .unwrap();
        let inner: StagedXlaFunction<'_, ArrayType, ArrayType> = stage_with_captures(
            |captures, x| (x + captures[0].clone()).sin().unwrap(),
            vec![bias],
            input_type.clone(),
            &engine,
            XlaOptions::new(mesh.clone()),
        )
        .unwrap();
        assert_eq!(engine.cache_size(), 0, "staging the inner function must not compile it");

        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(engine.cache_size(), 1, "only the outer program should be compiled");
        assert_eq!(outer.source_program().captures().len(), 1);

        let input_value = 0.5f32;
        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let observed = read_f32_array(&client, &engine.interpret(&outer.executable_program(), input).unwrap());
        assert_eq!(observed.len(), 1);
        let expected = (input_value + 0.25).sin().cos();
        assert!((observed[0] - expected).abs() < 1e-5, "expected cos(sin(x + bias)) = {expected}, got {}", observed[0]);
    }

    /// Input sharding overrides change the input types before tracing begins.
    #[test]
    fn test_staging_applies_in_shardings() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();

        let options = XlaOptions::new(mesh).with_in_shardings(vec![sharding]);
        let staged: StagedXlaFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type, &engine, options).unwrap();
        assert!(staged.source_program().program().input_types()[0].sharding().is_some());
        assert_eq!(engine.cache_size(), 0, "staging must not build an executable");
    }

    /// End-to-end check that a function calling the same staged block twice — whose two `jit_call`s deduplicate into
    /// one shared `func.func` at lowering — still compiles and computes the correct result through PJRT. Here
    /// `g(x) = sin(x) + sin(x) = 2 sin(x)`.
    #[test]
    fn test_repeated_staged_call_round_trips() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: StagedXlaFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), &engine, XlaOptions::new(mesh.clone())).unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| inner.call(x.clone()).unwrap() + inner.call(x).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let observed = read_f32_array(&client, &engine.interpret(&outer.executable_program(), input).unwrap());
            assert_eq!(observed.len(), 1);
            let expected = 2.0 * point.sin();
            assert!((observed[0] - expected).abs() < 1e-5, "expected 2*sin({point}) = {expected}, got {}", observed[0]);
        }
    }

    /// End-to-end check that reverse-mode AD through repeated staged calls — which produces structurally-identical
    /// `jit_call`s in both the primal and pullback that deduplicate at lowering — compiles and computes the correct
    /// gradient through PJRT. With `g(x) = 2 sin(x)`, `g'(x) = 2 cos(x)`.
    #[test]
    fn test_grad_over_repeated_staged_calls_round_trips() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: StagedXlaFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), &engine, XlaOptions::new(mesh.clone())).unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| inner.call(x.clone()).unwrap() + inner.call(x).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledXlaFunction<'_, ArrayType, ArrayType> = outer.gradient(&engine).unwrap();

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let observed = read_f32_array(&client, &engine.interpret(&gradient.executable_program(), input).unwrap());
            assert_eq!(observed.len(), 1);
            let expected = 2.0 * point.cos();
            assert!(
                (observed[0] - expected).abs() < 1e-5,
                "expected d/dx[2*sin(x)] = 2*cos({point}) = {expected}, got {}",
                observed[0],
            );
        }
    }
}
