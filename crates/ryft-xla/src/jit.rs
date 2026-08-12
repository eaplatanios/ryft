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

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use ryft_core::{
    ArrayIrType, ArrayIrValue, ArrayType, CapturingContext, ClosedProgram, CompilationDomain,
    CompilationStagingRequest, CompiledFunction, CompiledFunctionDispatcher as CoreCompiledFunctionDispatcher,
    Constant, Context, DeviceMesh, DifferentiableType, DomainTracingContext, ExecutableFunction,
    ForwardModeDifferentiate, JitCacheStatistics, Parameterized, ParameterizedFamily, ProgramError, ProjectedContext,
    ProjectedValue, ReverseModeDifferentiate, StagedFunction, Tracer, Typed, Value, ValueProjection, call_function,
    try_jit_with_options as core_try_jit_with_options,
};
use ryft_pjrt::Execution;

use crate::experimental::XlaDomainError;
use crate::experimental::domains::XlaCompiledProgram;
use crate::experimental::ops::{XlaConstant, XlaOperation};
use crate::{AdaptiveProfileGuidedOptions, AdaptiveProfileGuidedXlaFunction, Array, XlaDomain, XlaOptions};

/// Composite tracer retained by the production XLA program.
type XlaProgramTracer<'c> = Tracer<DomainTracingContext<XlaDomain<'c>, ArrayIrValue<Array<'c>>>>;

/// Tracer leaf exposed by the public array-only XLA compilation facade.
pub type XlaCompileTracer<'c> = ProjectedValue<ArrayType, XlaProgramTracer<'c>>;

/// Internal reparameterization of one public array parameter tree into the production composite type family.
type XlaProgramParameters<P> = <P as Parameterized<ArrayType>>::To<ArrayIrType>;

/// Internal value tree corresponding to one public array parameter tree.
type XlaProgramParameterValues<P, V> = <XlaProgramParameters<P> as Parameterized<ArrayIrType>>::To<V>;

/// Captured-constant tree corresponding to one public array parameter tree.
type XlaProgramConstants<P> = XlaProgramParameterValues<P, XlaConstant>;

/// Concrete runtime value retained by the production composite domain.
type XlaProgramValue<'c> = ArrayIrValue<Array<'c>>;

/// Projects one internal composite tracer tree to the public array-only tracer tree.
fn project_tracers<'c, P>(
    values: XlaProgramParameterValues<P, XlaProgramTracer<'c>>,
) -> Result<P::To<XlaCompileTracer<'c>>, XlaDomainError>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<ArrayIrType>,
{
    let structure = values.parameter_structure();
    let parameters = values
        .into_parameters()
        .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    P::To::<XlaCompileTracer<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Lifts one public array-only tracer tree back into the internal composite tracer tree.
fn lift_tracers<'c, P>(
    values: P::To<XlaCompileTracer<'c>>,
) -> Result<XlaProgramParameterValues<P, XlaProgramTracer<'c>>, XlaDomainError>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
{
    let structure = values.parameter_structure();
    let parameters = values.into_parameters().map(ProjectedValue::into_value);
    XlaProgramParameterValues::<P, XlaProgramTracer<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Lifts one public runtime array tree into the production composite value family.
fn lift_arrays<'c, P>(
    values: P::To<Array<'c>>,
) -> Result<XlaProgramParameterValues<P, XlaProgramValue<'c>>, XlaDomainError>
where
    P: Parameterized<ArrayType>,
    P::Family:
        ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Array<'c>> + ParameterizedFamily<XlaProgramValue<'c>>,
{
    let structure = values.parameter_structure();
    let parameters = values.into_parameters().map(ArrayIrValue::Array);
    XlaProgramParameterValues::<P, XlaProgramValue<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Projects one production composite output tree back to public runtime arrays.
fn project_arrays<'c, P>(
    values: XlaProgramParameterValues<P, XlaProgramValue<'c>>,
) -> Result<P::To<Array<'c>>, XlaDomainError>
where
    P: Parameterized<ArrayType>,
    P::Family:
        ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaProgramValue<'c>> + ParameterizedFamily<Array<'c>>,
{
    let structure = values.parameter_structure();
    let parameters = values
        .into_parameters()
        .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    P::To::<Array<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Retained array-only XLA JIT dispatcher with explicit host-side static parameters.
///
/// A first call for each `(static parameters, dynamic parameter structure, dynamic abstract types)` specialization
/// traces, lowers, and requests compilation. Warm calls dispatch directly to the retained executable. Static values
/// should be low-cardinality configuration such as axes, shapes, or Boolean branch choices; arrays remain dynamic.
pub struct JittedXlaFunction<
    'c,
    F,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
> where
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    /// Composite-domain dispatcher hidden behind the public array projection boundary.
    function:
        CoreCompiledFunctionDispatcher<XlaDomain<'c>, F, Static, XlaProgramParameters<In>, XlaProgramParameters<Out>>,
}

impl<'c, F, Static, In, Out> Clone for JittedXlaFunction<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    CoreCompiledFunctionDispatcher<XlaDomain<'c>, F, Static, XlaProgramParameters<In>, XlaProgramParameters<Out>>:
        Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, F, Static, In, Out> JittedXlaFunction<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<Array<'c>>
        + ParameterizedFamily<XlaProgramValue<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<Array<'c>>
        + ParameterizedFamily<XlaProgramValue<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
{
    /// Calls this dispatcher with public array inputs and projects its composite runtime outputs back to arrays.
    pub fn call(
        &self,
        static_parameters: Static,
        inputs: In::To<Array<'c>>,
    ) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        XlaOptions: Clone,
        F: Fn(
            Static,
            XlaProgramParameterValues<In, XlaProgramTracer<'c>>,
        ) -> Result<XlaProgramParameterValues<Out, XlaProgramTracer<'c>>, XlaDomainError>,
        XlaProgramParameters<In>: Parameterized<ArrayIrType, To<ArrayIrType> = XlaProgramParameters<In>>,
        XlaProgramParameters<Out>: Parameterized<ArrayIrType, To<ArrayIrType> = XlaProgramParameters<Out>>,
        XlaProgramParameterValues<In, XlaProgramValue<'c>>:
            Parameterized<XlaProgramValue<'c>, To<ArrayIrType> = XlaProgramParameters<In>>,
        XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
                XlaProgramTracer<'c>,
                To<ArrayIrType> = XlaProgramParameters<Out>,
                To<XlaConstant> = XlaProgramConstants<Out>,
            >,
    {
        let inputs = lift_arrays::<In>(inputs)?;
        project_arrays::<Out>(self.function.call(static_parameters, inputs)?)
    }

    /// Returns a snapshot of this dispatcher's cache activity.
    #[inline]
    pub fn statistics(&self) -> JitCacheStatistics {
        self.function.statistics()
    }
}

/// Constructs a retained dispatcher for a fallible XLA closure using explicit options.
pub fn try_jitted_with_options<'c, F, Static, In, Out>(
    function: F,
    domain: &XlaDomain<'c>,
    options: XlaOptions,
) -> JittedXlaFunction<
    'c,
    impl Fn(
        Static,
        XlaProgramParameterValues<In, XlaProgramTracer<'c>>,
    ) -> Result<XlaProgramParameterValues<Out, XlaProgramTracer<'c>>, XlaDomainError>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<XlaCompileTracer<'c>>) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>,
{
    let function = move |static_parameters, inputs| {
        let outputs = function(static_parameters, project_tracers::<In>(inputs)?)?;
        lift_tracers::<Out>(outputs)
    };
    JittedXlaFunction { function: core_try_jit_with_options(domain, function, options) }
}

/// Constructs a retained dispatcher for an infallible XLA closure using explicit options.
pub fn jitted_with_options<'c, F, Static, In, Out>(
    function: F,
    domain: &XlaDomain<'c>,
    options: XlaOptions,
) -> JittedXlaFunction<
    'c,
    impl Fn(
        Static,
        XlaProgramParameterValues<In, XlaProgramTracer<'c>>,
    ) -> Result<XlaProgramParameterValues<Out, XlaProgramTracer<'c>>, XlaDomainError>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>,
{
    try_jitted_with_options(move |static_parameters, inputs| Ok(function(static_parameters, inputs)), domain, options)
}

/// Constructs a retained dispatcher for an infallible XLA closure on `mesh`.
#[inline]
pub fn jitted<'c, F, Static, In, Out>(
    function: F,
    domain: &XlaDomain<'c>,
    mesh: DeviceMesh,
) -> JittedXlaFunction<
    'c,
    impl Fn(
        Static,
        XlaProgramParameterValues<In, XlaProgramTracer<'c>>,
    ) -> Result<XlaProgramParameterValues<Out, XlaProgramTracer<'c>>, XlaDomainError>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>
        + ParameterizedFamily<XlaCompileTracer<'c>>,
{
    jitted_with_options(function, domain, XlaOptions::new(mesh))
}

/// Captured-constant output tree produced by tracing an XLA closure.
type XlaSourceProgramOutput<Out> = XlaProgramConstants<Out>;

/// Staged-but-uncompiled XLA function handle. Returned by [`stage`] and [`stage_with_captures`].
///
/// Holds the traced source [`Program`](ryft_core::programs::Program) of one closure together with its captured
/// runtime [`Array`]s and input / output type metadata, **without** compiling a PJRT executable. This is the right
/// entry point for functions that are only ever composed into larger programs: [`Self::call`] embeds the staged
/// program into an active outer trace as a `jit_call` boundary, and [`XlaDomain::compile_staged_function`] produces a
/// [`CompiledXlaFunction`] when an executable is actually needed. Executable cache identity is derived from the
/// complete lowered computation and compile-relevant backend state, so equivalent lowerings can share an executable
/// even when they were produced at different Rust call sites.
pub struct StagedXlaFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    /// Backend-neutral staged function carrying the source program, captures, structures, and retained options.
    function: StagedFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for StagedXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> StagedXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    /// Returns the staged source [`Program`](ryft_core::Program) together with its captured runtime values. Useful for
    /// outer transforms (`grad` / `jvp` / `vjp` / `batch`), staged `jit_call` payloads, and diagnostics (printing the
    /// traced IR, instruction counts, graph rendering).
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<
        XlaProgramValue<'c>,
        XlaConstant,
        XlaOperation,
        XlaProgramConstants<In>,
        XlaSourceProgramOutput<Out>,
    > {
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
    pub fn call<V>(
        &self,
        inputs: In::To<ProjectedValue<ArrayType, V>>,
    ) -> Result<Out::To<ProjectedValue<ArrayType, V>>, ProgramError>
    where
        V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
        V::DispatchDomain: Context<Type = ArrayIrType, Constant = XlaConstant, Operation = XlaOperation>
            + CapturingContext<Capture = XlaProgramValue<'c>>
            + Constant<V, XlaConstant>,
        In::Family: ParameterizedFamily<V> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
        Out::Family: ParameterizedFamily<V> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
    {
        let input_structure = inputs.parameter_structure();
        let inputs = XlaProgramParameterValues::<In, V>::from_parameters(
            input_structure,
            inputs.into_parameters().map(ProjectedValue::into_value),
        )?;
        let outputs = self.function.call(inputs)?;
        let output_structure = outputs.parameter_structure();
        let outputs = outputs
            .into_parameters()
            .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Out::To::<ProjectedValue<ArrayType, V>>::from_parameters(output_structure, outputs).map_err(ProgramError::from)
    }

    /// Consumes this facade and returns its composite staged artifact for crate-internal verification.
    #[cfg(test)]
    pub(crate) fn into_inner(
        self,
    ) -> StagedFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<Out>> {
        self.function
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
    From<StagedFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<Out>>>
    for StagedXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    #[inline]
    fn from(function: StagedFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<Out>>) -> Self {
        Self { function }
    }
}

/// Runtime-only XLA executable handle.
///
/// Unlike [`CompiledXlaFunction`], this type does not retain the `Arc`-backed staged program or lowering metadata used
/// by transforms. It can only execute and inspect its runtime signature. Its thread-safety is derived from its
/// backend state; no unsafe blanket implementation is used.
pub struct ExecutableXlaFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    function: ExecutableFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for ExecutableXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> ExecutableXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    /// Returns the flat output [`ArrayType`]s in executor order.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.function.compiled_program().output_types()
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
        In: Parameterized<ArrayType>,
        In::Family:
            ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
        Out: Parameterized<ArrayType>,
        Out::Family:
            ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    {
        let lowered = self.lower(staged.function)?;
        let function = self.compile(lowered)?;
        Ok(CompiledXlaFunction { function, derived: Arc::new(DerivedFunctionSlots::new()) })
    }

    /// Executes an XLA runtime program on concrete [`Array`] inputs.
    #[inline]
    pub fn interpret<In, Out>(
        &self,
        executable: &ExecutableXlaFunction<'c, In, Out>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<XlaConstant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<XlaProgramValue<'c>>,
            >,
        Out: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<XlaConstant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<XlaProgramValue<'c>>,
            >,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs = lift_arrays::<In>(inputs)?;
        project_arrays::<Out>(call_function(self, &executable.function, inputs)?)
    }

    /// Enqueues an XLA runtime program and retains whole-execution completion, including for zero-output calls.
    pub fn interpret_async<In, Out>(
        &self,
        executable: &ExecutableXlaFunction<'c, In, Out>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Execution<Out::To<Array<'c>>>, XlaDomainError>
    where
        In: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<XlaConstant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<XlaProgramValue<'c>>,
            >,
        Out: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<XlaConstant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<XlaProgramValue<'c>>,
            >,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let flat_inputs = lift_arrays::<In>(inputs)?.into_parameters().collect::<Vec<_>>();
        executable.function.validate_flat_input_count(flat_inputs.as_slice())?;
        for (expected, actual) in executable.function.input_types().iter().zip(flat_inputs.iter().map(Typed::r#type)) {
            crate::experimental::domains::validate_xla_input_type(
                <&ArrayType>::try_from(expected).map_err(ProgramError::from)?,
                <&ArrayType>::try_from(actual.as_ref()).map_err(ProgramError::from)?,
            )?;
        }

        let arguments = executable
            .function
            .arguments_with_captures(flat_inputs)
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let execution = self.execute_compiled_async(executable.function.compiled_program(), arguments)?;
        let (flat_outputs, fence) = execution.into_parts();
        crate::experimental::domains::validate_runtime_outputs(
            executable.function.compiled_program().output_types(),
            flat_outputs.as_slice(),
        )?;
        let outputs = executable
            .function
            .reconstruct_outputs(flat_outputs.into_iter().map(ArrayIrValue::Array).collect())
            .map_err(XlaDomainError::from)?;
        Ok(Execution::new(project_arrays::<Out>(outputs)?, fence))
    }

    /// Replaces an executable program while preserving and validating its runtime signature.
    pub(crate) fn replace_executable_xla_program<In, Out>(
        &self,
        executable: &ExecutableXlaFunction<'c, In, Out>,
        program: std::sync::Arc<XlaCompiledProgram<'c>>,
    ) -> Result<ExecutableXlaFunction<'c, In, Out>, XlaDomainError>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    {
        self.validate_xla_replacement(executable.function.compiled_program(), &program)?;
        Ok(ExecutableXlaFunction {
            function: executable.function.with_compiled_program(
                program.clone(),
                program.output_types().iter().cloned().map(Into::into).collect(),
            ),
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
        In: Parameterized<ArrayType>,
        In::Family:
            ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
        Out: Parameterized<ArrayType>,
        Out::Family:
            ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    {
        AdaptiveProfileGuidedXlaFunction::new(
            self.clone(),
            function.executable_function(),
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
pub struct CompiledXlaFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    /// Backend-neutral compiled function backed by an XLA executable. Retains the staged source function,
    /// captured runtime buffers, output structure, and compilation options through its lowered metadata.
    function: CompiledFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<Out>>,

    /// Derived transformed functions retained by [`Self::gradient`] and [`Self::jvp`]. Shared across clones so a
    /// wrapper's clones reuse one retained derivative per transform.
    derived: Arc<DerivedXlaFunctions<'c, In, Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for CompiledXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone(), derived: Arc::clone(&self.derived) }
    }
}

/// Retention slots for the derived functions of one [`CompiledXlaFunction`]: the reverse-mode gradient and the
/// forward-mode JVP compiled from its retained source program.
type DerivedXlaFunctions<'c, In, Out> = DerivedFunctionSlots<
    'c,
    CompiledFunction<XlaDomain<'c>, XlaProgramParameters<In>, XlaProgramParameters<In>>,
    CompiledFunction<XlaDomain<'c>, XlaProgramParameters<(In, In)>, XlaProgramParameters<(Out, Out)>>,
>;

/// One-entry retention slots for functions derived from a [`CompiledXlaFunction`].
///
/// [`CompiledXlaFunction::gradient`] and [`CompiledXlaFunction::jvp`] take no per-call transform configuration, so
/// each derived function has exactly one valid form per domain identity: the slot retains the most recently produced
/// artifact together with a witness of the domain that produced it. A later call with a matching domain reuses the
/// retained artifact and skips source reconstruction, structural differentiation, staging, and lowering entirely; a
/// call with a different domain identity produces a fresh artifact and replaces the slot. Failed productions are
/// never stored, so they retry naturally.
///
/// A slot is a retention slot and not a single-flight barrier: its lock is released before the artifact is derived, so
/// concurrent cold calls may each derive their own copy and the last insert wins. That is sound because the copies are
/// interchangeable derivations of the same source program, and it is deliberate because waiting single-flight belongs
/// exclusively to the shared [`CompilationContext`](ryft_core::compilation::CompilationContext), which still
/// deduplicates the backend compilation those duplicate derivations request.
struct DerivedFunctionSlots<'c, Gradient, Jvp> {
    /// Retained reverse-mode gradient function, if any.
    gradient: Mutex<Option<(XlaDomainWitness<'c>, Gradient)>>,

    /// Retained forward-mode JVP function, if any.
    jvp: Mutex<Option<(XlaDomainWitness<'c>, Jvp)>>,
}

impl<'c, Gradient, Jvp> DerivedFunctionSlots<'c, Gradient, Jvp> {
    /// Creates empty retention slots.
    fn new() -> Self {
        Self { gradient: Mutex::new(None), jvp: Mutex::new(None) }
    }
}

/// Identity of the [`XlaDomain`] that produced a retained derived function.
///
/// [`XlaDomain`] handles are cheap clones of shared state, so two handles denote interchangeable compilation
/// pipelines exactly when they share one compilation context, one client, equal compilation-option templates, and an
/// equal mesh. The witness stores a full domain clone, which both provides those comparisons and keeps the shared
/// compilation context alive so its address can never be reused by a later allocation.
struct XlaDomainWitness<'c> {
    /// Domain handle captured when the retained artifact was produced.
    domain: XlaDomain<'c>,
}

impl<'c> XlaDomainWitness<'c> {
    /// Captures the identity of `domain`.
    fn new(domain: &XlaDomain<'c>) -> Self {
        Self { domain: domain.clone() }
    }

    /// Returns whether `domain` is interchangeable with the domain that produced the retained artifact.
    fn matches(&self, domain: &XlaDomain<'c>) -> bool {
        std::ptr::eq(self.domain.compilation_context(), domain.compilation_context())
            && match (self.domain.client().ok(), domain.client().ok()) {
                (Some(retained), Some(current)) => std::ptr::eq(retained, current),
                (None, None) => true,
                _ => false,
            }
            && self.domain.compilation_options() == domain.compilation_options()
            && self.domain.mesh().ok() == domain.mesh().ok()
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> CompiledXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
{
    /// Returns the flat output [`ArrayType`]s in the order the executor produces them.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.function.compiled_program().output_types()
    }

    /// Returns a runtime-only handle that omits staged and lowered transform metadata.
    #[inline]
    pub fn executable_function(&self) -> ExecutableXlaFunction<'c, In, Out> {
        ExecutableXlaFunction { function: self.function.executable_function().clone() }
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable_function(self) -> ExecutableXlaFunction<'c, In, Out> {
        ExecutableXlaFunction { function: self.function.into_executable_function() }
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
    ) -> &ClosedProgram<
        XlaProgramValue<'c>,
        XlaConstant,
        XlaOperation,
        XlaProgramConstants<In>,
        XlaSourceProgramOutput<Out>,
    > {
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
    pub fn call<V>(
        &self,
        inputs: In::To<ProjectedValue<ArrayType, V>>,
    ) -> Result<Out::To<ProjectedValue<ArrayType, V>>, ProgramError>
    where
        V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
        V::DispatchDomain: Context<Type = ArrayIrType, Constant = XlaConstant, Operation = XlaOperation>
            + CapturingContext<Capture = XlaProgramValue<'c>>
            + Constant<V, XlaConstant>,
        In::Family: ParameterizedFamily<V> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
        Out::Family: ParameterizedFamily<V> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
    {
        self.staged().call(inputs)
    }
}

/// Reverse-mode AD: compiles a new function that computes the gradient of a scalar-valued compiled function with
/// respect to its inputs. The original closure is never re-executed; [`Self::call`] emits a `jit_call` boundary, and
/// the active transform rewrites that operation through ordinary JVP and transpose rules.
impl<'c, In: Parameterized<ArrayType, To<ArrayType> = In>> CompiledXlaFunction<'c, In, ArrayType>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: std::fmt::Debug + Hash + PartialEq,
{
    /// Returns a new compiled function that computes the reverse-mode gradient of `self` with
    /// respect to its input. Mirrors `jax.grad(jax.jit(f))`.
    ///
    /// `self` must produce a single rank-0 scalar output (encoded by the `Out = ArrayType`
    /// impl-block constraint above). The returned compiled function has the same input shape
    /// and produces an output whose leaves carry the partial derivative at each input leaf.
    ///
    /// The derived function is retained on this wrapper (shared across its clones): repeated calls with an
    /// interchangeable `domain` return the retained compiled function without repeating source reconstruction,
    /// structural differentiation, staging, or lowering, while a call with a different domain identity produces a
    /// fresh derivative and replaces the retained one. The retained artifact is a compiled program whose residuals
    /// are runtime values, so reuse across calls with different runtime inputs is sound by construction. Retention is
    /// not single-flight: concurrent cold calls may each derive a gradient, and because those derivations are
    /// interchangeable the inserts are idempotent and the last one wins, while the shared
    /// [`CompilationContext`](ryft_core::compilation::CompilationContext) still deduplicates their backend compilation.
    #[track_caller]
    pub fn gradient<'domain>(
        &'domain self,
        domain: &'domain XlaDomain<'c>,
    ) -> Result<CompiledXlaFunction<'c, In, In>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'c>, To = In::To<XlaCompileTracer<'c>>>,
        In::To<XlaCompileTracer<'c>>: Parameterized<
                XlaCompileTracer<'c>,
                Family = In::Family,
                ParameterStructure = In::ParameterStructure,
                To<ArrayType> = In,
                To<XlaConstant> = In::To<XlaConstant>,
            >,
        XlaProgramParameterValues<In, XlaProgramTracer<'c>>: Parameterized<
                XlaProgramTracer<'c>,
                To<ArrayIrType> = XlaProgramParameters<In>,
                To<XlaConstant> = XlaProgramConstants<In>,
            >,
    {
        {
            let slot = self.derived.gradient.lock().expect("derived-gradient retention slot mutex poisoned");
            if let Some((witness, artifact)) = slot.as_ref() {
                if witness.matches(domain) {
                    return Ok(CompiledXlaFunction {
                        function: artifact.clone(),
                        derived: Arc::new(DerivedFunctionSlots::new()),
                    });
                }
            }
        }
        let staged = self.function.staged();
        let input_structure = staged.source_program().program().input_structure().clone();
        let input_signature = In::from_parameters(
            input_structure.clone(),
            staged
                .input_types()
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned().map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(ProgramError::from)?;
        let mesh = self.mesh().clone();
        let captures = self
            .source_program()
            .captures()
            .iter()
            .cloned()
            .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let compiled = compile_with_flat_captures(
            move |capture_references, _, primals| {
                let primals = primals.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let context = primals
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                let (output, pullback) = context.vjp(
                    move |inputs| {
                        let mut outputs =
                            staged.call_with_flat_capture_references(capture_references.as_slice(), inputs)?;
                        if outputs.len() != 1 {
                            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() });
                        }
                        Ok(outputs.remove(0))
                    },
                    primals,
                )?;
                let output = ValueProjection::<ArrayType>::into_projected(output).map_err(ProgramError::from)?;
                let seed = ProjectedContext::<_, ArrayType>::new(context).gradient_seed(&output, false)?.into_value();
                let gradients = pullback.apply(seed)?;
                let gradients = gradients
                    .into_iter()
                    .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                In::To::<XlaCompileTracer<'c>>::from_parameters(input_structure, gradients)
                    .map_err(ProgramError::from)
                    .map_err(Into::into)
            },
            captures,
            input_signature,
            domain,
            XlaOptions::new(mesh),
        )?;
        *self.derived.gradient.lock().expect("derived-gradient retention slot mutex poisoned") =
            Some((XlaDomainWitness::new(domain), compiled.function.clone()));
        Ok(compiled)
    }
}

/// Forward-mode JVP packaged as a method. Mirrors `jax.jvp(jax.jit(f))`.
impl<'c, In: Clone + Parameterized<ArrayType, To<ArrayType> = In>, Out: Parameterized<ArrayType>>
    CompiledXlaFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: std::fmt::Debug + Hash + PartialEq,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
{
    /// Returns a new compiled function that computes the forward-mode JVP of `self`. Mirrors
    /// `jax.jvp(f, primals, tangents)` packaged into one compiled function: the returned handle
    /// takes `(primals, tangents)` and returns `(primal_out, tangent_out)`.
    ///
    /// The implementation stages a composite `jit_call` operation and lets its ordinary JVP rule build the tangent
    /// call boundary. Public array tracers are projected only at this facade boundary; the transform itself runs over
    /// the production composite program.
    ///
    /// The derived function is retained on this wrapper (shared across its clones): repeated calls with an
    /// interchangeable `domain` return the retained compiled function without repeating source reconstruction,
    /// structural differentiation, staging, or lowering, while a call with a different domain identity produces a
    /// fresh derivative and replaces the retained one. Retention is not single-flight: concurrent cold calls may each
    /// derive a JVP, and because those derivations are interchangeable the inserts are idempotent and the last one
    /// wins, while the shared [`CompilationContext`](ryft_core::compilation::CompilationContext) still deduplicates
    /// their backend compilation.
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
                To<XlaConstant> = Out::To<XlaConstant>,
            >,
        XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
                XlaProgramTracer<'c>,
                To<ArrayIrType> = XlaProgramParameters<Out>,
                To<XlaConstant> = XlaProgramConstants<Out>,
            >,
    {
        {
            let slot = self.derived.jvp.lock().expect("derived-jvp retention slot mutex poisoned");
            if let Some((witness, artifact)) = slot.as_ref() {
                if witness.matches(domain) {
                    return Ok(CompiledXlaFunction {
                        function: artifact.clone(),
                        derived: Arc::new(DerivedFunctionSlots::new()),
                    });
                }
            }
        }
        let staged = self.function.staged();
        let input_signature = In::from_parameters(
            staged.source_program().program().input_structure().clone(),
            staged
                .input_types()
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned().map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(ProgramError::from)?;
        let tangent_signature =
            input_signature.clone().map_parameters(|r#type| r#type.tangent()).map_err(ProgramError::from)?;
        let mesh = self.mesh().clone();
        let captures = self
            .source_program()
            .captures()
            .iter()
            .cloned()
            .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let compiled = compile_with_flat_captures(
            move |capture_references, _, (primals, tangents)| {
                let output_structure = staged.output_structure().clone();
                let primals = primals.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let tangents = tangents.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let context = primals
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                let (primal_outputs, tangent_outputs) = context.jvp(
                    move |inputs| staged.call_with_flat_capture_references(capture_references.as_slice(), inputs),
                    primals,
                    tangents,
                )?;
                let primal_outputs = primal_outputs
                    .into_iter()
                    .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                let tangent_outputs = tangent_outputs
                    .into_iter()
                    .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                let primal_tree = Out::To::<_>::from_parameters(output_structure.clone(), primal_outputs)
                    .map_err(ProgramError::from)?;
                let tangent_tree =
                    Out::To::<_>::from_parameters(output_structure, tangent_outputs).map_err(ProgramError::from)?;
                Ok((primal_tree, tangent_tree))
            },
            captures,
            (input_signature, tangent_signature),
            domain,
            XlaOptions::new(mesh),
        )?;
        *self.derived.jvp.lock().expect("derived-jvp retention slot mutex poisoned") =
            Some((XlaDomainWitness::new(domain), compiled.function.clone()));
        Ok(compiled)
    }

    /// Returns a new compiled function that runs `self` in parallel over `axis_size` batch items
    /// along a new leading axis of each input and output. Mirrors `jax.vmap(f)` with default
    /// `in_axes=0` / `out_axes=0`. Every input leaf gets a new leading axis of size `axis_size`;
    /// every output leaf is materialized with the batched axis at position 0. The batched
    /// leading axis is replicated for now.
    ///
    /// Composite-region batching is assigned to Phase 5. Until that support lands, this method returns a precise
    /// unsupported-operation diagnostic instead of reinterpreting projected batching values.
    ///
    /// Homogeneous static-extent array regions already retain structurally batched programs through
    /// `ryft_core::programs::transforms::Transform`. Composite XLA batching must additionally define how its live
    /// first-class extent crosses the transformed boundary: only the extent's static type/identity contract belongs in
    /// the transform key, while the current extent value remains an explicit runtime operand. The axis name, mapped
    /// sharding, normalized input axes, and output policy likewise complete that composite marker's structural key.
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
    {
        let _ = (self, domain, axis_size);
        Err(ProgramError::UnsupportedOperation {
            message: "compiled XLA batching requires Phase 5 composite batching support".to_string(),
        }
        .into())
    }
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
pub fn compile<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
    compile_with_options(function, input_types, domain, XlaOptions::new(mesh))
}

/// Same as [`compile`] but makes runtime arrays explicit captures of the compiled program.
///
/// The closure receives capture tracers first and ordinary input tracers second. Captures are compiled as hidden
/// executable arguments and are supplied from the returned [`CompiledXlaFunction`] at execution time, so callers of the
/// compiled function still pass only `In` inputs.
#[track_caller]
pub fn compile_with_captures<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(Vec<XlaCompileTracer<'c>>, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
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
pub fn compile_with_options<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
    compile_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, domain, options)
}

#[track_caller]
fn compile_with_flat_captures<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(
        Vec<XlaConstant>,
        Vec<XlaCompileTracer<'c>>,
        In::To<XlaCompileTracer<'c>>,
    ) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
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
pub fn stage<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<StagedXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
    stage_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, domain, options)
}

/// Same as [`stage`] but makes runtime arrays explicit captures of the staged program.
///
/// The closure receives capture tracers first and ordinary input tracers second, mirroring
/// [`compile_with_captures`]. Captures are retained on the staged handle and threaded through `jit_call` boundaries
/// when the handle is staged into outer traces via [`StagedXlaFunction::call`].
#[track_caller]
pub fn stage_with_captures<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<StagedXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(Vec<XlaCompileTracer<'c>>, In::To<XlaCompileTracer<'c>>) -> Out::To<XlaCompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
    stage_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        domain,
        options,
    )
}

#[track_caller]
fn stage_with_flat_captures<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: XlaOptions,
) -> Result<StagedXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(
        Vec<XlaConstant>,
        Vec<XlaCompileTracer<'c>>,
        In::To<XlaCompileTracer<'c>>,
    ) -> Result<Out::To<XlaCompileTracer<'c>>, XlaDomainError>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'c>>
        + ParameterizedFamily<XlaProgramTracer<'c>>,
    Out::To<XlaCompileTracer<'c>>:
        Parameterized<XlaCompileTracer<'c>, To<ArrayType> = Out, To<XlaConstant> = Out::To<XlaConstant>>,
    XlaProgramParameterValues<Out, XlaProgramTracer<'c>>: Parameterized<
            XlaProgramTracer<'c>,
            To<ArrayIrType> = XlaProgramParameters<Out>,
            To<XlaConstant> = XlaProgramConstants<Out>,
        >,
{
    let captures = captures.into_iter().map(ArrayIrValue::Array).collect();
    let input_structure = input_types.parameter_structure();
    let input_types = XlaProgramParameters::<In>::from_parameters(
        input_structure,
        input_types.into_parameters().map(ArrayIrType::from),
    )
    .map_err(ProgramError::from)?;
    let function = move |capture_references,
                         capture_tracers: Vec<XlaProgramTracer<'c>>,
                         inputs: XlaProgramParameterValues<In, XlaProgramTracer<'c>>| {
        let capture_tracers = capture_tracers
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let inputs = project_tracers::<In>(inputs)?;
        lift_tracers::<Out>(function(capture_references, capture_tracers, inputs)?)
    };
    let function = domain.stage(CompilationStagingRequest::new(function, captures, input_types, options))?;
    Ok(StagedXlaFunction { function })
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
#[track_caller]
pub fn infer_output_types<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
) -> Result<Out, ProgramError>
where
    F: FnOnce(In::To<XlaCompileTracer<'static>>) -> Out::To<XlaCompileTracer<'static>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'static>>
        + ParameterizedFamily<XlaProgramTracer<'static>>,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<XlaCompileTracer<'static>>
        + ParameterizedFamily<XlaProgramTracer<'static>>,
    Out::To<XlaCompileTracer<'static>>: Parameterized<XlaCompileTracer<'static>, To<ArrayType> = Out>,
{
    let input_structure = input_types.parameter_structure();
    let input_types = XlaProgramParameters::<In>::from_parameters(
        input_structure,
        input_types.into_parameters().map(ArrayIrType::from),
    )?;
    let output_types = DomainTracingContext::<XlaDomain<'static>, XlaProgramValue<'static>>::infer_output_type(
        |tracers| {
            let input_structure = tracers.parameter_structure();
            let input_parameters = tracers
                .into_parameters()
                .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let tracers = In::To::<XlaCompileTracer<'static>>::from_parameters(input_structure, input_parameters)?;
            let outputs = function(tracers);
            let output_structure = outputs.parameter_structure();
            XlaProgramParameterValues::<Out, XlaProgramTracer<'static>>::from_parameters(
                output_structure,
                outputs.into_parameters().map(ProjectedValue::into_value),
            )
            .map_err(ProgramError::from)
        },
        input_types,
    )?;
    let output_structure = output_types.parameter_structure();
    let output_parameters = output_types
        .into_parameters()
        .map(|r#type| <&ArrayType>::try_from(&r#type).cloned().map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    Out::from_parameters(output_structure, output_parameters).map_err(ProgramError::from)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ryft_core::operations::custom_call::{CustomCall, CustomCallOperation};
    use ryft_core::operations::random::Random;
    use ryft_core::operations::sort::{ArgMax, TopK};
    use ryft_core::{
        Add, Array as CpuArray, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, Atan2, Broadcast,
        CalleeRegionDriver, Compare, ComparisonDirection, Context, Cos, DataType, Device, DeviceMesh,
        DifferentiableType, Dimension, Div, DomainTracingContext, Dot, DotDimensionNumbers, DynamicSlice,
        DynamicUpdateSlice, EagerContext, Exp, Fill, ForwardModeDifferentiate, Hessian, HessianDifferentiate, Iota,
        Jacobian, JacobianDifferentiate, LogicalMesh, Logistic, MeshAxis, MeshAxisType, Mul, OneLike, ProgramError,
        ProjectedValue, Reduce, ReductionKind, Reshape, ReverseModeDifferentiate, Select, Shape, Sharding,
        ShardingDimension, Sin, StopGradient, Sub, Tanh, Typed, Value, ValueProjection, WhileOperation, ZeroLike,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::experimental::XlaDomainError;
    use crate::experimental::ops::XlaOperation;
    use crate::jit::{
        CompiledXlaFunction, ExecutableXlaFunction, JittedXlaFunction, StagedXlaFunction, XlaCompileTracer, compile,
        compile_with_captures, compile_with_options, infer_output_types, jitted, stage, stage_with_captures,
    };
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{AdaptiveProfileGuidedOptions, Array, FromPjrt, XlaDomain, XlaOptions};

    use super::XlaProgramTracer;

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

    fn read_f64_array(client: &ryft_pjrt::Client<'_>, array: &Array<'_>) -> Vec<f64> {
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
        values_from_bytes::<f64>(shard_bytes.as_slice())
    }

    #[test]
    fn test_jit_unary_function_runs_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();

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
    fn test_dense_differentiation_compiles_as_part_of_the_enclosing_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);
        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        let forward: CompiledXlaFunction<'_, ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> =
            compile(
                |input| {
                    input
                        .dispatch_domain()
                        .jacobian_forward(|value| Mul::mul(&value, &value), input)
                        .expect("forward Jacobian should stage")
                },
                input_type.clone(),
                &domain,
                mesh.clone(),
            )
            .unwrap();
        let reverse: CompiledXlaFunction<'_, ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> =
            compile(
                |input| {
                    input
                        .dispatch_domain()
                        .jacobian_reverse(|value| Mul::mul(&value, &value), input)
                        .expect("reverse Jacobian should stage")
                },
                input_type.clone(),
                &domain,
                mesh.clone(),
            )
            .unwrap();
        let second: CompiledXlaFunction<'_, ArrayType, Hessian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| {
                input
                    .dispatch_domain()
                    .hessian(|value| Mul::mul(&value, &value), input)
                    .expect("Hessian should stage")
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(domain.cache_size(), 2);
        let _: CompiledXlaFunction<'_, ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| {
                input
                    .dispatch_domain()
                    .jacobian_forward(|value| Mul::mul(&value, &value), input)
                    .expect("forward Jacobian should stage")
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(domain.cache_size(), 2, "an equivalent dense transform should reuse its enclosing compilation");

        assert!(forward.source_program().program().regions().iter().any(|region| {
            region.instructions().iter().any(|instruction| {
                matches!(instruction.operation(), XlaOperation::Array(ArrayOperation::CoordinateBasis(_)))
            })
        }));
        assert!(reverse.source_program().program().regions().iter().any(|region| {
            region.instructions().iter().any(|instruction| {
                matches!(instruction.operation(), XlaOperation::Array(ArrayOperation::CoordinateBasis(_)))
            })
        }));
        assert!(second.source_program().program().regions().iter().any(|region| {
            region.instructions().iter().any(|instruction| {
                matches!(instruction.operation(), XlaOperation::Array(ArrayOperation::CoordinateBasis(_)))
            })
        }));

        for (compiled, expected) in [(forward, 6.0), (reverse, 6.0)] {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[3.0]).as_slice(),
            )
            .unwrap();
            let jacobian: Jacobian<ArrayType, Array<'_>, ArrayType, ArrayType> =
                domain.interpret(&compiled.executable_function(), input).unwrap();
            assert_eq!(read_f32_array(&client, jacobian.iter_blocks().next().unwrap().value()), vec![expected]);
        }

        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[3.0]).as_slice()).unwrap();
        let hessian: Hessian<ArrayType, Array<'_>, ArrayType, ArrayType> =
            domain.interpret(&second.executable_function(), input).unwrap();
        assert_eq!(read_f32_array(&client, hessian.iter_blocks().next().unwrap().value()), vec![2.0]);
    }

    #[test]
    fn test_promoted_broadcast_dense_differentiation_compiles_as_part_of_the_enclosing_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        let forward: CompiledXlaFunction<
            '_,
            (ArrayType, ArrayType),
            Jacobian<ArrayType, ArrayType, (ArrayType, ArrayType), ArrayType>,
        > = compile(
            |inputs| {
                inputs
                    .0
                    .dispatch_domain()
                    .jacobian_forward(|(scalar, vector)| scalar.atan2(&vector), inputs)
                    .expect("forward Jacobian should stage")
            },
            (scalar_type.clone(), vector_type.clone()),
            &domain,
            mesh.clone(),
        )
        .unwrap();

        let scalar =
            Array::from_host_buffer(&client, scalar_type, mesh.clone(), values_to_bytes::<f32>(&[2.0]).as_slice())
                .unwrap();
        let vector =
            Array::from_host_buffer(&client, vector_type, mesh, values_to_bytes::<f64>(&[1.0, 4.0]).as_slice())
                .unwrap();
        let jacobian: Jacobian<ArrayType, Array<'_>, (ArrayType, ArrayType), ArrayType> =
            domain.interpret(&forward.executable_function(), (scalar, vector)).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        for (actual, expected) in read_f64_array(&client, blocks[0].value()).into_iter().zip([0.2, 0.2]) {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }
        for (actual, expected) in read_f64_array(&client, blocks[1].value()).into_iter().zip([-0.4, 0.0, 0.0, -0.1]) {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }
    }

    #[test]
    fn test_executable_xla_program_runs_without_transform_metadata() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableXlaFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable_function();
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableXlaFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable_function();
        let other_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(other_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let other_executable: ExecutableXlaFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), other_input_type, &other_domain, other_mesh)
                .unwrap()
                .into_executable_function();
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
    fn test_compiled_and_staged_xla_functions_are_send_and_sync() {
        assert_send_sync::<CompiledXlaFunction<'static, ArrayType, ArrayType>>();
        assert_send_sync::<StagedXlaFunction<'static, ArrayType, ArrayType>>();
    }

    #[test]
    fn test_executable_xla_program_executes_across_threads() {
        assert_send_sync::<ExecutableXlaFunction<'static, ArrayType, ArrayType>>();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableXlaFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable_function();
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x.stop_gradient(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();

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
        use ryft_core::{Memory, TransferToMemory};

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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();

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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ()> =
            compile(|_| (), input_type.clone(), &engine, mesh.clone()).unwrap();
        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[1.0]).as_slice()).unwrap();

        let execution = engine.interpret_async(&compiled.executable_function(), input).unwrap();
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
        use ryft_core::Cos;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Array(ArrayOperation::Sin(_))))
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
        let output = engine.interpret(&outer.executable_function(), source).unwrap();

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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        assert_eq!(compiled.source_program().to_program_with_lifted_captures().unwrap().input_ids().len(), 2);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = engine.interpret(&compiled.executable_function(), input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_compile_with_captures_erases_zero_space_capture_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let capture = Array::from_host_buffer(&client, zero_type.clone(), mesh.clone(), []).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, _| captures[0].clone(),
            vec![capture],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[1.0])).unwrap();

        let output = engine.interpret(&compiled.executable_function(), input).unwrap();

        assert_eq!(output.r#type().as_ref(), &zero_type);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
        output.block_until_ready().unwrap();
    }

    #[test]
    fn test_captured_compiled_function_stages_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        let output = engine.interpret(&outer.executable_function(), input).unwrap();
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

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        let output = engine.interpret(&outer.executable_function(), input).unwrap();
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
            engine.interpret(&jvp_compiled.executable_function(), (primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_gradient_method_preserves_compiled_function_captures() {
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
            |captures, x| x * captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();

        assert_eq!(gradient.source_program().captures().len(), 1);

        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[3.0]).as_slice()).unwrap();
        let output = engine.interpret(&gradient.executable_function(), input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![2.0]);
    }

    #[test]
    fn test_deep_jit_call_callee_chain_transforms_on_default_stack() {
        // Transforming and compiling a function whose `jit_call` callee contains a long elementwise chain used to
        // overflow the default libtest thread stack in debug builds (around 150 chained operations aborted), because
        // linearization and transposition rebuilt programs by recursing along use-def chains. The rebuilds are
        // worklist-driven now, so deriving `jvp` and `gradient` functions across a callee chain well past that
        // threshold must succeed on a default-size test thread.
        const CALLEE_OPERATIONS: usize = 600;
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let callee: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            |x| {
                let mut value = x;
                for index in 0..CALLEE_OPERATIONS {
                    value = if index % 2 == 0 { value.sin().unwrap() } else { value.cos().unwrap() };
                }
                value
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| callee.call(x).unwrap().sin().unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();

        let jvp: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = outer.jvp(&engine).unwrap();
        assert_eq!(jvp.output_types().len(), 2);
        let gradient: CompiledXlaFunction<'_, ArrayType, ArrayType> = outer.gradient(&engine).unwrap();
        assert_eq!(gradient.output_types().len(), 1);

        // A reference chain evaluated in `f64` pins the gradient's value: each step contributes its local derivative
        // at the pre-step value, and the appended outer `sin` contributes the final `cos` factor.
        let (mut expected_value, mut expected_gradient) = (0.5f64, 1.0f64);
        for index in 0..CALLEE_OPERATIONS {
            let (next, derivative) = if index % 2 == 0 {
                (expected_value.sin(), expected_value.cos())
            } else {
                (expected_value.cos(), -expected_value.sin())
            };
            expected_value = next;
            expected_gradient *= derivative;
        }
        expected_gradient *= expected_value.cos();
        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();
        let output = engine.interpret(&gradient.executable_function(), input).unwrap();
        assert!((read_f32_array(&client, &output)[0] as f64 - expected_gradient).abs() < 1e-3);
    }

    /// For `f = |x| x * x`, repeated `f.gradient(&domain)` calls with one domain must reuse the retained derived
    /// function (no frontend re-derivation and no compilation-context traffic), while different runtime inputs to the
    /// shared derived function must still produce their own correct value-dependent gradients.
    #[test]
    fn test_gradient_method_retains_derived_function_per_domain() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x, input_type.clone(), &engine, mesh.clone()).unwrap();

        let first: CompiledXlaFunction<'_, ArrayType, ArrayType> = compiled.gradient(&engine).unwrap();
        let statistics = engine.compilation_context().statistics();
        let second = compiled.gradient(&engine).unwrap();

        // The warm call reuses the retained executable and never reaches the shared compilation context.
        assert!(std::ptr::eq(first.function.compiled_program(), second.function.compiled_program()));
        assert_eq!(engine.compilation_context().statistics(), statistics);

        // The retained derived function keeps residuals as runtime values: different primals through the shared
        // executable produce different, correct value-dependent gradients (`d(x*x)/dx = 2x`).
        for point in [3.0f32, -5.0f32] {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = engine.interpret(&second.executable_function(), input).unwrap();
            assert_eq!(read_f32_array(&client, &output), vec![2.0 * point]);
        }

        // A different domain identity misses, produces a fresh derivative, and replaces the retained slot.
        let other_engine = XlaDomain::new(&client);
        let third = compiled.gradient(&other_engine).unwrap();
        assert!(!std::ptr::eq(first.function.compiled_program(), third.function.compiled_program()));
        let other_statistics = other_engine.compilation_context().statistics();
        let fourth = compiled.gradient(&other_engine).unwrap();
        assert!(std::ptr::eq(third.function.compiled_program(), fourth.function.compiled_program()));
        assert_eq!(other_engine.compilation_context().statistics(), other_statistics);
    }

    /// Mirrors [`test_gradient_method_retains_derived_function_per_domain`] for the forward-mode
    /// [`CompiledXlaFunction::jvp`] path.
    #[test]
    fn test_jvp_method_retains_derived_function_per_domain() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x, input_type.clone(), &engine, mesh.clone()).unwrap();

        let first: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compiled.jvp(&engine).unwrap();
        let statistics = engine.compilation_context().statistics();
        let second = compiled.jvp(&engine).unwrap();

        assert!(std::ptr::eq(first.function.compiled_program(), second.function.compiled_program()));
        assert_eq!(engine.compilation_context().statistics(), statistics);

        // The retained JVP computes `(x * x, 2 * x * t)` for runtime `(x, t)` supplied per call.
        let primal = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[4.0]).as_slice()).unwrap();
        let (primal_output, tangent_output) =
            engine.interpret(&second.executable_function(), (primal, tangent)).unwrap();
        assert_eq!(read_f32_array(&client, &primal_output), vec![9.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![24.0]);
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
                let primal = primal.into_value();
                let tangent = tangent.into_value();
                let context = primal.context().clone();
                let (primal_output, tangent_output) = context
                    .jvp(
                        move |x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x).map_err(ProgramError::from)?;
                            Ok(inner.call(x)?.into_value())
                        },
                        primal,
                        tangent,
                    )
                    .expect("nested captured jvp(jit) should stage");
                (
                    ValueProjection::<ArrayType>::into_projected(primal_output).expect("primal should remain an array"),
                    ValueProjection::<ArrayType>::into_projected(tangent_output)
                        .expect("tangent should remain an array"),
                )
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
            engine.interpret(&jvp_compiled.executable_function(), (primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    /// Outer-transform smoke test: applying `grad` to a `CompiledXlaFunction` produces a new compiled function that
    /// For `f = |x| x.sin()`, `f.jvp()` produces a compiled function for which interpreting its executable function at
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
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Array(ArrayOperation::Sin(_))))
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
                engine.interpret(&jvp_compiled.executable_function(), (primal_array, tangent_array)).unwrap();
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

    #[test]
    fn test_jvp_method_executes_zero_space_boundaries() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let primal_type = ArrayType::new(DataType::Boolean, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let tangent_type = primal_type.tangent();
        let primal: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|value| value, primal_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = primal.jvp(&engine).unwrap();
        let zero_identity: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|value| value, tangent_type.clone(), &engine, mesh.clone()).unwrap();

        let primal_input = Array::from_host_buffer(&client, primal_type.clone(), mesh.clone(), [1u8]).unwrap();
        let tangent_input = Array::from_host_buffer(&client, tangent_type.clone(), mesh, []).unwrap();
        let (primal_output, tangent_output) =
            engine.interpret(&jvp.executable_function(), (primal_input, tangent_input)).unwrap();

        assert_eq!(primal_output.r#type().as_ref(), &primal_type);
        assert_eq!(tangent_output.r#type().as_ref(), &tangent_type);
        assert!(tangent_output.addressable_shards().next().unwrap().buffer().is_none());

        let chained_tangent = engine.interpret(&zero_identity.executable_function(), tangent_output).unwrap();
        assert_eq!(chained_tangent.r#type().as_ref(), &tangent_type);
        assert!(chained_tangent.addressable_shards().next().unwrap().buffer().is_none());
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
        use std::sync::Arc;

        use ryft_core::StagingContext;

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
                .filter(|instruction| {
                    matches!(
                        instruction.operation(),
                        XlaOperation::Array(ArrayOperation::Sin(_) | ArrayOperation::Cos(_))
                    )
                })
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
        let (primal_half, tangent_half) = (Arc::new(primal_half), Arc::new(tangent_half));
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal_input, tangent_input)| {
                let context = primal_input.value().context().clone();
                let mut primal_outputs = context
                    .stage_operation(
                        XlaOperation::JitCall(JitCallOperation::new()),
                        CalleeRegionDriver::new(&[primal_half.clone()]),
                        &[primal_input.into_value()],
                    )
                    .expect("primal jit_call should stage");
                let residuals = primal_outputs.split_off(1);
                assert_eq!(residuals.len(), residual_count, "primal half residual count should match linearization");
                let primal_output = primal_outputs.remove(0);
                let mut tangent_inputs = vec![tangent_input.into_value()];
                tangent_inputs.extend(residuals);
                let tangent_output = context
                    .stage_operation(
                        XlaOperation::JitCall(JitCallOperation::new()),
                        CalleeRegionDriver::new(&[tangent_half.clone()]),
                        tangent_inputs.as_slice(),
                    )
                    .expect("tangent jit_call should stage")
                    .remove(0);
                (
                    ValueProjection::<ArrayType>::into_projected(primal_output)
                        .expect("primal jit_call output should remain an array"),
                    ValueProjection::<ArrayType>::into_projected(tangent_output)
                        .expect("tangent jit_call output should remain an array"),
                )
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

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
                .interpret(&jvp_compiled.executable_function(), (primal_array.clone(), tangent_array.clone()))
                .unwrap();
            let jvp_primal_value = read_f32_array(&client, &jvp_primal)[0];
            let jvp_tangent_value = read_f32_array(&client, &jvp_tangent)[0];
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
        }
    }

    /// Compiled-function batching remains an explicit Phase 5 boundary until composite region batching lands.
    #[test]
    fn test_batch_method_reports_composite_region_deferral() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_input_type =
            ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(scalar_sharding).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), scalar_input_type, &engine, mesh.clone()).unwrap();
        let error = match inner.batch(&engine, 4) {
            Ok(_) => panic!("compiled batching should remain deferred until Phase 5"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), "compiled XLA batching requires Phase 5 composite batching support");
    }

    #[test]
    fn test_jit_binary_function_with_tuple_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Dimension::Static(3)]);
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
        let output = engine.interpret(&compiled.executable_function(), (a, b)).unwrap();

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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = XlaOptions::new(mesh.clone()).with_donate(true);
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();

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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
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
        use ryft_core::Parameter;
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
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
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
        let shape = Shape::new(vec![Dimension::Static(4)]);
        let abstract_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let abstract_input_type =
            ArrayType::new(DataType::F32, shape.clone()).with_sharding(abstract_sharding).unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
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
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();

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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
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

        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
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
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();
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
        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
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
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(7)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        let output_type: ArrayType = infer_output_types(|x| x.sin().unwrap(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `infer_output_types` must not have populated the compile cache.
        assert_eq!(engine.cache_size(), 0);
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

        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
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
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();
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

        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
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
        let output = engine.interpret(&compiled.executable_function(), source).unwrap();

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
            read_f32_array(&client, &engine.interpret(&staged_compiled.executable_function(), make_input()).unwrap());
        let direct_output =
            read_f32_array(&client, &engine.interpret(&direct.executable_function(), make_input()).unwrap());
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
        let observed = read_f32_array(&client, &engine.interpret(&outer.executable_function(), input).unwrap());
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
        let ArrayIrType::Array(staged_input_type) = &staged.source_program().program().input_types()[0] else {
            panic!("public staged input should remain an array");
        };
        assert!(staged_input_type.sharding().is_some());
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
            let observed = read_f32_array(&client, &engine.interpret(&outer.executable_function(), input).unwrap());
            assert_eq!(observed.len(), 1);
            let expected = 2.0 * point.sin();
            assert!((observed[0] - expected).abs() < 1e-5, "expected 2*sin({point}) = {expected}, got {}", observed[0]);
        }
    }

    /// Number of repeated derived-function constructions and warm dispatches performed by the Phase 0 transform
    /// caching baselines below.
    const BASELINE_REPETITIONS: usize = 5;

    /// Prints one Phase 0 baseline timing table: the cold (first) call, every warm call, and the warm mean.
    fn print_baseline_durations(label: &str, durations: &[Duration]) {
        println!("phase 0 baseline: {label} ({} calls)", durations.len());
        for (index, duration) in durations.iter().enumerate() {
            let tier = if index == 0 { "cold" } else { "warm" };
            println!("  call {index} ({tier}): {:.3} ms", duration.as_secs_f64() * 1e3);
        }
        let warm = &durations[1..];
        let warm_mean = warm.iter().map(Duration::as_secs_f64).sum::<f64>() / warm.len().max(1) as f64 * 1e3;
        println!("  warm mean: {warm_mean:.3} ms");
    }

    /// Prints the domain compilation-cache counters observed across a Phase 0 baseline workload.
    fn print_baseline_compilation_statistics(domain: &XlaDomain<'_>) {
        let statistics = domain.compilation_context().statistics();
        println!(
            "  compilation cache: memory_hits={} persistent_hits={} misses={} compilations={} waits={}",
            statistics.memory_hits,
            statistics.persistent_hits,
            statistics.misses,
            statistics.compilations,
            statistics.waits,
        );
        println!(
            "  compilation cache: retained_entries={} lookup_ms={:.3} compilation_ms={:.3}",
            domain.cache_size(),
            statistics.memory_lookup_duration_ns as f64 / 1e6,
            statistics.compilation_duration_ns as f64 / 1e6,
        );
    }

    /// Caching-plan measurement for repeated [`CompiledXlaFunction::jvp`] construction.
    ///
    /// Originally captured as the Phase 0 baseline, where every call reconstructed, re-traced, and re-lowered the
    /// derived forward-mode function with only backend compilation deduplicated through the shared
    /// [`CompilationContext`](ryft_core::compilation::CompilationContext). Since the Phase 4 retention landed, warm
    /// calls reuse the derived function retained on the wrapper, so this now measures the improvement against that
    /// recorded baseline. The printed table records the cold call, the warm calls, and the compilation-cache
    /// counters.
    #[test]
    #[ignore = "phase 0 baseline measurement"]
    fn test_baseline_repeated_compiled_function_jvp() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap().cos().unwrap(), input_type, &domain, mesh).unwrap();

        domain.compilation_context().clear_statistics();
        let mut durations = Vec::with_capacity(BASELINE_REPETITIONS);
        for _ in 0..BASELINE_REPETITIONS {
            let start = Instant::now();
            let derived: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
                compiled.jvp(&domain).unwrap();
            durations.push(start.elapsed());
            assert_eq!(derived.output_types().len(), 2);
        }

        print_baseline_durations("repeated CompiledXlaFunction::jvp", &durations);
        print_baseline_compilation_statistics(&domain);
    }

    /// Caching-plan measurement for repeated [`CompiledXlaFunction::gradient`] construction. Mirrors
    /// [`test_baseline_repeated_compiled_function_jvp`] for the reverse-mode path.
    #[test]
    #[ignore = "phase 0 baseline measurement"]
    fn test_baseline_repeated_compiled_function_gradient() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap().reduce(&[0, 1], ReductionKind::Sum), input_type, &domain, mesh).unwrap();

        domain.compilation_context().clear_statistics();
        let mut durations = Vec::with_capacity(BASELINE_REPETITIONS);
        for _ in 0..BASELINE_REPETITIONS {
            let start = Instant::now();
            let derived: CompiledXlaFunction<'_, ArrayType, ArrayType> = compiled.gradient(&domain).unwrap();
            durations.push(start.elapsed());
            assert_eq!(derived.output_types().len(), 1);
        }

        print_baseline_durations("repeated CompiledXlaFunction::gradient", &durations);
        print_baseline_compilation_statistics(&domain);
    }

    /// Phase 0 caching-plan composition baseline: a jitted closure that internally runs an eager reverse-mode
    /// gradient. The JIT boundary stages the eager transform exactly once per specialization, so repeated calls with
    /// same-shaped inputs are served entirely by retained dispatch. The recorded counters are the reference point the
    /// caching redesign must preserve, and they also show whether the current trace and lowering fallback tiers ever
    /// hit under this workload.
    #[test]
    #[ignore = "phase 0 baseline measurement"]
    fn test_baseline_jit_composition_caches_eager_transforms() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);

        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let function: JittedXlaFunction<'_, _, (), ArrayType, ArrayType> = jitted(
            |_, input: XlaCompileTracer<'_>| {
                input
                    .dispatch_domain()
                    .gradient(|value| Mul::mul(&value, &value), input)
                    .expect("the eager gradient should stage inside the jit boundary")
            },
            &domain,
            mesh.clone(),
        );

        domain.compilation_context().clear_statistics();
        let mut durations = Vec::with_capacity(BASELINE_REPETITIONS);
        for index in 0..BASELINE_REPETITIONS {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[index as f32]).as_slice(),
            )
            .unwrap();
            let start = Instant::now();
            let output = function.call((), input).unwrap();
            durations.push(start.elapsed());
            assert_eq!(read_f32_array(&client, &output), vec![2.0 * index as f32]);
        }

        // The JIT boundary is the whole-function cache: one trace, one lowering, one compilation request, and every
        // later same-shaped call served by a retained specialization without touching the fallback tiers.
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, (BASELINE_REPETITIONS - 1) as u64);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);

        print_baseline_durations("jit(eager gradient) repeated dispatch", &durations);
        println!(
            "  jit cache: dispatch_hits={} dispatch_misses={} traces={} lowerings={} compilation_requests={}",
            statistics.dispatch_hits,
            statistics.dispatch_misses,
            statistics.traces,
            statistics.lowerings,
            statistics.compilation_requests,
        );
        println!(
            "  jit cache: abstractification_ms={:.3} dispatch_ms={:.3} tracing_ms={:.3} lowering_ms={:.3}",
            statistics.input_abstractification_duration_ns as f64 / 1e6,
            statistics.dispatch_duration_ns as f64 / 1e6,
            statistics.tracing_duration_ns as f64 / 1e6,
            statistics.lowering_duration_ns as f64 / 1e6,
        );
        print_baseline_compilation_statistics(&domain);
    }

    /// Number of distinct outer specializations that stage a `jit_call` of one shared callee in the Phase 5 gate
    /// measurement below.
    const CALLEE_OUTER_SPECIALIZATIONS: usize = 4;

    /// Callee chain lengths compared by the Phase 5 gate measurement below.
    const CALLEE_OPERATION_COUNTS: [usize; 2] = [2, 200];

    /// Prints one Phase 5 gate row group: per-outer total construction time, the backend compilation time charged to
    /// the shared [`CompilationContext`](ryft_core::compilation::CompilationContext) during it, and the frontend
    /// remainder that callee transformation lives in.
    fn print_callee_transformation_rows(transform: &str, totals: &[Duration], backends: &[Duration]) {
        for (index, (total, backend)) in totals.iter().zip(backends.iter()).enumerate() {
            println!(
                "    {transform:<9} | {index:>5} | {:>8.3} | {:>10.3} | {:>11.3}",
                total.as_secs_f64() * 1e3,
                backend.as_secs_f64() * 1e3,
                (*total - *backend).as_secs_f64() * 1e3,
            );
        }
    }

    /// Returns the mean of `totals - backends` in milliseconds: the frontend-only cost per derived-function
    /// construction, excluding backend compilation served by the shared compilation context.
    fn mean_frontend_milliseconds(totals: &[Duration], backends: &[Duration]) -> f64 {
        let sum: f64 =
            totals.iter().zip(backends.iter()).map(|(total, backend)| (*total - *backend).as_secs_f64()).sum();
        sum / totals.len() as f64 * 1e3
    }

    /// Caching-plan measurement for transforming one *shared* `jit_call` callee across distinct outer
    /// specializations.
    ///
    /// Every outer function stages a `jit_call` of the same inner compiled function and appends a trivially
    /// different `sin` epilogue, so the outers are genuinely distinct specializations that cannot share a Phase 4
    /// retention slot. Each `jvp`/`gradient` on an outer runs the `JitCallOperation` rules over the attached callee
    /// program — `jvp` exercises the forward-mode rule, `gradient` exercises both the forward-mode and the transpose
    /// rule. Scaling the callee from a two-operation chain to a two-hundred-operation chain isolates the
    /// callee-proportional part of that cost.
    ///
    /// Originally captured as the Phase 5 gate baseline, where every outer re-derived the shared callee's transforms
    /// from scratch, so that the per-call delta between the two callee sizes was exactly the repeated work a
    /// callee-keyed transform cache could remove and that delta times the number of outers was the total repeated
    /// work in this workload. Since the per-`Region` transform cache landed, the callee's linearizations and
    /// transpositions are derived once and served warm to every later outer, so this now measures the improvement
    /// against that recorded baseline: what remains callee-proportional is the work outside the retained artifacts,
    /// such as replaying the derived programs into each outer.
    ///
    /// Backend compilation is timed separately and subtracted, because the derived programs differ per outer and are
    /// therefore genuinely distinct compilations that no frontend cache can remove.
    #[test]
    #[ignore = "phase 5 gate measurement"]
    fn test_baseline_repeated_jit_call_callee_transformation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let domain = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        println!(
            "phase 5 gate: shared jit_call callee transformed for {CALLEE_OUTER_SPECIALIZATIONS} outer \
             specializations, warm transforms served by the region transform cache",
        );
        println!("  compilation cache before:");
        print_baseline_compilation_statistics(&domain);
        domain.compilation_context().clear_statistics();

        // Per callee size: every outer's cold forward-mode and reverse-mode construction, split into the
        // total elapsed time and the backend compilation time charged inside it.
        let mut measurements: Vec<(usize, [Vec<Duration>; 2], [Vec<Duration>; 2])> = Vec::new();
        for callee_operations in CALLEE_OPERATION_COUNTS {
            let callee: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
                move |x| {
                    let mut value = x;
                    for index in 0..callee_operations {
                        value = if index % 2 == 0 { value.sin().unwrap() } else { value.cos().unwrap() };
                    }
                    value
                },
                input_type.clone(),
                &domain,
                mesh.clone(),
            )
            .unwrap();

            let mut outers = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            for index in 0..CALLEE_OUTER_SPECIALIZATIONS {
                let callee = callee.clone();
                let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
                    move |x| {
                        let mut value = callee.call(x).unwrap();
                        for _ in 0..=index {
                            value = value.sin().unwrap();
                        }
                        value
                    },
                    input_type.clone(),
                    &domain,
                    mesh.clone(),
                )
                .unwrap();

                // The callee stays behind a `jit_call` boundary: the outer program holds exactly one
                // `jit_call` and none of the callee's `cos` operations, which the `sin`-only epilogue could
                // never have produced itself.
                let instructions = outer.source_program().program().instructions();
                let jit_call_count = instructions
                    .iter()
                    .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
                    .count();
                let inlined_cosine_count = instructions
                    .iter()
                    .filter(|instruction| {
                        matches!(instruction.operation(), XlaOperation::Array(ArrayOperation::Cos(_)))
                    })
                    .count();
                assert_eq!(jit_call_count, 1, "each outer specialization should stage one jit_call boundary");
                assert_eq!(inlined_cosine_count, 0, "the outer program should not inline the callee body");
                outers.push(outer);
            }

            let mut jvp_totals = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            let mut jvp_backends = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            for outer in &outers {
                let before = domain.compilation_context().statistics().compilation_duration_ns;
                let start = Instant::now();
                let derived: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
                    outer.jvp(&domain).unwrap();
                jvp_totals.push(start.elapsed());
                let after = domain.compilation_context().statistics().compilation_duration_ns;
                jvp_backends.push(Duration::from_nanos(after - before));
                assert_eq!(derived.output_types().len(), 2);
            }

            let mut gradient_totals = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            let mut gradient_backends = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            for outer in &outers {
                let before = domain.compilation_context().statistics().compilation_duration_ns;
                let start = Instant::now();
                let derived: CompiledXlaFunction<'_, ArrayType, ArrayType> = outer.gradient(&domain).unwrap();
                gradient_totals.push(start.elapsed());
                let after = domain.compilation_context().statistics().compilation_duration_ns;
                gradient_backends.push(Duration::from_nanos(after - before));
                assert_eq!(derived.output_types().len(), 1);
            }

            measurements.push((callee_operations, [jvp_totals, jvp_backends], [gradient_totals, gradient_backends]));
        }

        for (callee_operations, jvp, gradient) in &measurements {
            println!("  callee with {callee_operations} operations (all times in milliseconds):");
            println!("    transform | outer |    total |    backend |    frontend");
            print_callee_transformation_rows("jvp", &jvp[0], &jvp[1]);
            print_callee_transformation_rows("gradient", &gradient[0], &gradient[1]);
        }

        let (small_operations, small_jvp, small_gradient) = &measurements[0];
        let (large_operations, large_jvp, large_gradient) = &measurements[1];
        let small_jvp_mean = mean_frontend_milliseconds(&small_jvp[0], &small_jvp[1]);
        let large_jvp_mean = mean_frontend_milliseconds(&large_jvp[0], &large_jvp[1]);
        let small_gradient_mean = mean_frontend_milliseconds(&small_gradient[0], &small_gradient[1]);
        let large_gradient_mean = mean_frontend_milliseconds(&large_gradient[0], &large_gradient[1]);
        let added_operations = (large_operations - small_operations) as f64;
        println!(
            "  frontend-only summary (mean over {CALLEE_OUTER_SPECIALIZATIONS} distinct outer \
             specializations, milliseconds):",
        );
        println!(
            "    transform | {small_operations:>3}-op callee | {large_operations:>3}-op callee | \
             delta/call | delta*outers | per callee op",
        );
        println!(
            "    jvp       | {small_jvp_mean:>13.3} | {large_jvp_mean:>13.3} | {:>10.3} | {:>12.3} | \
             {:>13.4}",
            large_jvp_mean - small_jvp_mean,
            (large_jvp_mean - small_jvp_mean) * CALLEE_OUTER_SPECIALIZATIONS as f64,
            (large_jvp_mean - small_jvp_mean) / added_operations,
        );
        println!(
            "    gradient  | {small_gradient_mean:>13.3} | {large_gradient_mean:>13.3} | {:>10.3} | \
             {:>12.3} | {:>13.4}",
            large_gradient_mean - small_gradient_mean,
            (large_gradient_mean - small_gradient_mean) * CALLEE_OUTER_SPECIALIZATIONS as f64,
            (large_gradient_mean - small_gradient_mean) / added_operations,
        );
        println!("  compilation cache after:");
        print_baseline_compilation_statistics(&domain);
    }

    /// Tracing context that stages the decode-loop demo's `While` condition and body region programs in the XLA
    /// domain universe: its tracers are exactly [`XlaCompileTracer`]s, so the shared [`decode_step`] model runs
    /// unchanged inside the staged regions and in the eager reference loop.
    type DecodeTraceContext<'c> = DomainTracingContext<XlaDomain<'c>, ArrayIrValue<Array<'c>>>;

    /// Shape and sampling hyperparameters of the tiny decode-loop demo model.
    #[derive(Copy, Clone)]
    struct DecodeConfiguration {
        /// Vocabulary size (the number of token embeddings and output logits).
        vocabulary: usize,

        /// Model dimension shared by the embeddings, the attention cache rows, and the gated MLP.
        dimension: usize,

        /// Number of decode steps performed by the loop (also the key/value cache capacity).
        steps: usize,

        /// Number of highest-probability logits retained by [`DecodeSampling::TopK`] sampling.
        top_k: usize,
    }

    /// Token-selection strategy of one decode step.
    #[derive(Copy, Clone, PartialEq)]
    enum DecodeSampling {
        /// Deterministically selects the highest-scoring logit.
        Greedy,

        /// Restricts the logits to their top-k entries and draws one categorically using the threaded ThreeFry
        /// generator state.
        TopK,
    }

    /// Attention implementation used by one decode step.
    #[derive(Copy, Clone, PartialEq)]
    enum DecodeAttention {
        /// Masked scaled dot-product attention composed from `ryft` operations.
        Composed,

        /// A registered [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] XLA FFI kernel computing the same attention.
        CustomCall,
    }

    /// Returns a static [`Shape`] with the provided dimensions.
    fn static_shape(dimensions: &[usize]) -> Shape {
        Shape::new(dimensions.iter().map(|&dimension| Dimension::Static(dimension)).collect())
    }

    /// Returns `count` deterministic pseudo-random weight values in `[-0.5, 0.5)` derived from `seed` with an
    /// xorshift generator, so the demo model is reproducible without carrying literal weight tables in the test.
    fn decode_weight_values(seed: u32, count: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// One decode step of the tiny gated-attention language model shared by the compiled decode-loop demo and its
    /// eager reference loop. The step is written once against `ryft`'s value-level capability traits, so the same
    /// function executes eagerly over reference [`CpuArray`]s and stages symbolically over [`XlaCompileTracer`]s
    /// inside the compiled `While` body.
    ///
    /// The decode state is a thirteen-value vector: the current position and token (`i32` scalars), the `[steps,
    /// dimension]` key and value caches, the `[steps]` decoded-token record, the `ui64[2]` ThreeFry generator
    /// state, and the seven loop-invariant weight matrices (embeddings, query/key/value projections, the hidden
    /// and gate MLP projections, and the output projection). One step embeds the current token, appends its
    /// projected key/value row to the caches at `position`, attends over the visible cache prefix, applies the
    /// `tanh`/`logistic` gated MLP and output projection, selects the next token per `sampling`, records it, and
    /// advances the position.
    fn decode_step<C, V>(
        context: &C,
        state: &[V],
        configuration: &DecodeConfiguration,
        sampling: DecodeSampling,
        attention: DecodeAttention,
    ) -> Result<Vec<V>, ProgramError>
    where
        V: Clone
            + Value<Type = ArrayType>
            + Add
            + Sub
            + Mul
            + Div
            + Exp
            + Tanh
            + Logistic
            + Dot
            + Reduce
            + Compare<V>
            + Select
            + TopK
            + ArgMax
            + DynamicSlice
            + DynamicUpdateSlice
            + Reshape
            + Broadcast
            + Random
            + CustomCall
            + OneLike
            + ZeroLike,
        C: Fill<f32, V> + Fill<i32, V> + Iota<V>,
    {
        let dimension = configuration.dimension;
        let position = state[0].clone();
        let token = state[1].clone();
        let tokens = state[4].clone();
        let generator = state[5].clone();
        let embeddings = &state[6];
        let query_weights = &state[7];
        let key_weights = &state[8];
        let value_weights = &state[9];
        let hidden_weights = &state[10];
        let gate_weights = &state[11];
        let output_weights = &state[12];

        // Embed the current token and append its projected key/value row to the caches at `position`.
        let zero_index = position.zero_like();
        let embedding = embeddings
            .dynamic_slice(&[token, zero_index.clone()], &[1, dimension])?
            .reshape(static_shape(&[dimension]))?;
        let vector_times_matrix = DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new());
        let query = embedding.dot(query_weights, &vector_times_matrix);
        let key = embedding.dot(key_weights, &vector_times_matrix);
        let value = embedding.dot(value_weights, &vector_times_matrix);
        let cache_keys = state[2].dynamic_update_slice(
            &key.reshape(static_shape(&[1, dimension]))?,
            &[position.clone(), zero_index.clone()],
        )?;
        let cache_values = state[3]
            .dynamic_update_slice(&value.reshape(static_shape(&[1, dimension]))?, &[position.clone(), zero_index])?;

        // Masked scaled dot-product attention over the visible cache prefix `[0, position]`.
        let attended = match attention {
            DecodeAttention::Composed => {
                let scores =
                    cache_keys.dot(&query, &DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()));
                let scores_type = scores.r#type().into_owned();
                let scale = context.fill(&scores_type, 1.0 / (dimension as f32).sqrt())?;
                let scores = scores.mul(&scale)?;
                let positions_type = tokens.r#type().into_owned();
                let positions = context.iota(&positions_type, 0)?;
                let visible = positions
                    .compare(&position.broadcast(positions_type, &[])?, ComparisonDirection::LessThanOrEqual)?;
                let masked = V::select(&visible, &scores, &context.fill(&scores_type, -1.0e30f32)?)?;
                let stabilized =
                    masked.sub(&masked.reduce(&[0], ReductionKind::Max).broadcast(scores_type.clone(), &[])?)?;
                let exponentials = stabilized.exp()?;
                let weights =
                    exponentials.div(&exponentials.reduce(&[0], ReductionKind::Sum).broadcast(scores_type, &[])?)?;
                weights.dot(&cache_values, &DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new()))
            }
            DecodeAttention::CustomCall => {
                let operation =
                    CustomCallOperation::new(DECODE_ATTENTION_CUSTOM_CALL_TARGET, vec![query.r#type().into_owned()]);
                let inputs = [cache_keys.clone(), cache_values.clone(), query, position.clone()];
                V::custom_call(&operation, &inputs)?.remove(0)
            }
        };

        // Gated multilayer perceptron, output projection, and next-token selection.
        let hidden = attended.dot(hidden_weights, &vector_times_matrix).tanh()?;
        let gate = attended.dot(gate_weights, &vector_times_matrix).logistic()?;
        let logits = hidden.mul(&gate)?.dot(output_weights, &vector_times_matrix);
        let (generator, next_token) = match sampling {
            DecodeSampling::Greedy => (generator, logits.argmax(0)?),
            DecodeSampling::TopK => {
                let (top_logits, top_indices) = logits.top_k(configuration.top_k, 0)?;
                let (generator, choice) = generator.categorical(&top_logits, 0)?;
                let next_token = top_indices.dynamic_slice(&[choice], &[1])?.reshape(static_shape(&[]))?;
                (generator, next_token)
            }
        };
        let tokens = tokens.dynamic_update_slice(&next_token.reshape(static_shape(&[1]))?, &[position.clone()])?;
        let position = position.add(&position.one_like())?;
        Ok(vec![
            position,
            next_token,
            cache_keys,
            cache_values,
            tokens,
            generator,
            embeddings.clone(),
            query_weights.clone(),
            key_weights.clone(),
            value_weights.clone(),
            hidden_weights.clone(),
            gate_weights.clone(),
            output_weights.clone(),
        ])
    }

    /// Stages the decode-loop demo as a `While` operation over the thirteen-value decode state: both regions are
    /// traced with [`decode_step`] and a `position < steps` condition through the public tracing API, and the
    /// staged loop returns the decoded tokens together with the final key/value caches.
    fn build_decode_loop<'c>(
        inputs: Vec<XlaCompileTracer<'c>>,
        configuration: &DecodeConfiguration,
        sampling: DecodeSampling,
        attention: DecodeAttention,
    ) -> Vec<XlaCompileTracer<'c>> {
        let context = inputs[0].value().context().clone();
        let carry_types = inputs.iter().map(|input| input.value().r#type().into_owned()).collect::<Vec<ArrayIrType>>();
        let steps = configuration.steps;
        let (_, condition) = <DecodeTraceContext<'c>>::trace(
            |state: Vec<XlaProgramTracer<'c>>| {
                let state = state
                    .into_iter()
                    .map(|value| {
                        ValueProjection::<ArrayType>::into_projected(value)
                            .expect("decode condition state should remain array-valued")
                    })
                    .collect::<Vec<_>>();
                let position = &state[0];
                let limit = position.dispatch_domain().fill(&position.r#type().into_owned(), steps as i32)?;
                Ok(vec![position.compare(&limit, ComparisonDirection::LessThan)?.into_value()])
            },
            carry_types.clone(),
        )
        .unwrap();
        let (_, body) = <DecodeTraceContext<'c>>::trace(
            |state: Vec<XlaProgramTracer<'c>>| {
                let state = state
                    .into_iter()
                    .map(|value| {
                        ValueProjection::<ArrayType>::into_projected(value)
                            .expect("decode body state should remain array-valued")
                    })
                    .collect::<Vec<_>>();
                let context = state[0].dispatch_domain();
                Ok(decode_step(&context, &state, configuration, sampling, attention)?
                    .into_iter()
                    .map(ProjectedValue::into_value)
                    .collect::<Vec<_>>())
            },
            carry_types,
        )
        .unwrap();
        let inputs = inputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>();
        let outputs = context
            .bind(XlaOperation::While(WhileOperation::new()), vec![condition, body], inputs.as_slice())
            .unwrap();
        let outputs = outputs
            .into_iter()
            .map(|value| {
                ValueProjection::<ArrayType>::into_projected(value)
                    .expect("decode loop output should remain array-valued")
            })
            .collect::<Vec<_>>();
        vec![outputs[4].clone(), outputs[2].clone(), outputs[3].clone()]
    }

    /// Runs the decode-loop demo's eager reference: the same [`decode_step`] executed over reference [`CpuArray`]
    /// values in a plain Rust loop, with no `While` staging, compilation, or XLA involvement.
    fn reference_decode_loop(
        initial_state: Vec<CpuArray>,
        configuration: &DecodeConfiguration,
        sampling: DecodeSampling,
    ) -> Vec<CpuArray> {
        let context = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new();
        let mut state = initial_state;
        for _ in 0..configuration.steps {
            state = decode_step(&context, &state, configuration, sampling, DecodeAttention::Composed).unwrap();
        }
        state
    }

    /// Name of the XLA custom call target registered by [`ensure_decode_attention_handler_registered`]: a masked
    /// scaled dot-product attention kernel over `(keys [steps, dimension], values [steps, dimension],
    /// query [dimension], position i32[])` producing the attended `[dimension]` vector, used to exercise a foreign
    /// FFI kernel inside a compiled decode loop.
    const DECODE_ATTENTION_CUSTOM_CALL_TARGET: &str = "ryft.test.decode_attention";

    /// Registers the [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] XLA FFI handler with the plugin backing the provided
    /// client. Registration is idempotent and process-global (re-registering the same target name is rejected by
    /// the XLA runtime), so the outcome is cached in a [`OnceLock`](std::sync::OnceLock).
    fn ensure_decode_attention_handler_registered(client: &ryft_pjrt::Client<'_>) -> Result<(), ryft_pjrt::Error> {
        use std::sync::OnceLock;

        use ryft_pjrt::extensions::ffi::{FfiHandler, FfiHandlerTraits, XLA_FFI_Handler};

        static DECODE_ATTENTION_HANDLER_REGISTRATION: OnceLock<Result<(), ryft_pjrt::Error>> = OnceLock::new();
        DECODE_ATTENTION_HANDLER_REGISTRATION
            .get_or_init(|| {
                let platform_name = client.platform_name()?.into_owned();
                client.register_ffi_handler(
                    DECODE_ATTENTION_CUSTOM_CALL_TARGET,
                    platform_name,
                    FfiHandler::from(decode_attention_handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )
            })
            .clone()
    }

    /// XLA FFI handler for [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] custom calls: computes masked scaled
    /// dot-product attention over the cache prefix `[0, position]`, matching the composed attention staged by
    /// [`decode_step`].
    unsafe extern "C" fn decode_attention_handler(
        call_frame: *mut ryft_pjrt::extensions::ffi::XLA_FFI_CallFrame,
    ) -> *mut ryft_pjrt::extensions::ffi::XLA_FFI_Error {
        use ryft_pjrt::extensions::ffi::{FfiCallFrame, FfiExecutionStage, FfiTypeId};

        // SAFETY: The XLA runtime passes a call frame that is valid for the duration of this invocation, and all
        // further unsafe access to it is localized in the safe `FfiCallFrame` wrapper and
        // `handle_decode_attention_call_frame`.
        unsafe {
            match FfiCallFrame::from_c_api(call_frame) {
                Err(_) => std::ptr::null_mut(),
                Ok(call_frame) if call_frame.register_metadata(FfiTypeId::default()) => std::ptr::null_mut(),
                Ok(call_frame) if call_frame.stage() != FfiExecutionStage::Execution => std::ptr::null_mut(),
                Ok(call_frame) => match call_frame.api() {
                    Err(_) => std::ptr::null_mut(),
                    Ok(api) => match handle_decode_attention_call_frame(&call_frame) {
                        Ok(()) => std::ptr::null_mut(),
                        Err(error) => error.to_c_api(api),
                    },
                },
            }
        }
    }

    /// Decodes a [`DECODE_ATTENTION_CUSTOM_CALL_TARGET`] call frame and fills its `[dimension]` output buffer with
    /// the masked scaled dot-product attention of the query over the `[0, position]` cache prefix, using the same
    /// `-1e30` mask value and max-stabilized softmax as the composed attention in [`decode_step`].
    fn handle_decode_attention_call_frame(
        call_frame: &ryft_pjrt::extensions::ffi::FfiCallFrame<'_>,
    ) -> Result<(), ryft_pjrt::extensions::ffi::FfiError> {
        use ryft_pjrt::extensions::ffi::{FfiBufferType, FfiError, FfiInput, FfiOutput};

        let mut inputs = call_frame.inputs();
        let mut next_buffer = |name: &str| match inputs.next() {
            Some(Ok(FfiInput::Buffer { buffer })) => Ok(buffer),
            _ => Err(FfiError::invalid_argument(format!(
                "expected the '{name}' input buffer of the '{DECODE_ATTENTION_CUSTOM_CALL_TARGET}' custom call"
            ))),
        };
        let keys = next_buffer("keys")?;
        let values = next_buffer("values")?;
        let query = next_buffer("query")?;
        let position = next_buffer("position")?;
        let mut outputs = call_frame.outputs();
        let Some(Ok(FfiOutput::Buffer { buffer: output })) = outputs.next() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the '{DECODE_ATTENTION_CUSTOM_CALL_TARGET}' custom call to have one output buffer"
            )));
        };
        let &[steps, dimension] = keys.dimensions() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the 'keys' input of the '{DECODE_ATTENTION_CUSTOM_CALL_TARGET}' custom call to have rank 2"
            )));
        };
        if keys.element_type() != FfiBufferType::F32
            || values.element_type() != FfiBufferType::F32
            || query.element_type() != FfiBufferType::F32
            || position.element_type() != FfiBufferType::I32
            || values.dimensions() != [steps, dimension]
            || query.dimensions() != [dimension]
            || !position.dimensions().is_empty()
            || output.dimensions() != [dimension]
        {
            return Err(FfiError::invalid_argument(format!(
                "unexpected '{DECODE_ATTENTION_CUSTOM_CALL_TARGET}' custom call buffer types or shapes"
            )));
        }
        let steps = steps.max(0) as usize;
        let dimension = dimension.max(0) as usize;
        // SAFETY: All data pointers are provided by the XLA runtime, are valid for the duration of the handler
        // invocation, and (per the element type and shape checks above) are backed by allocations of the checked
        // sizes. The runtime allocates inputs and outputs separately so they do not overlap.
        unsafe {
            let keys = std::slice::from_raw_parts(keys.data() as *const f32, steps * dimension);
            let values = std::slice::from_raw_parts(values.data() as *const f32, steps * dimension);
            let query = std::slice::from_raw_parts(query.data() as *const f32, dimension);
            let position = *(position.data() as *const i32);
            let output = std::slice::from_raw_parts_mut(output.data() as *mut f32, dimension);

            let scale = 1.0f32 / (dimension as f32).sqrt();
            let mut scores = vec![0.0f32; steps];
            for step in 0..steps {
                let row = &keys[step * dimension..(step + 1) * dimension];
                let score = row.iter().zip(query).map(|(key, query)| key * query).sum::<f32>() * scale;
                scores[step] = if step as i32 <= position { score } else { -1.0e30 };
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials: Vec<f32> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let denominator: f32 = exponentials.iter().sum();
            output.fill(0.0);
            for step in 0..steps {
                let weight = exponentials[step] / denominator;
                let row = &values[step * dimension..(step + 1) * dimension];
                for (accumulator, value) in output.iter_mut().zip(row) {
                    *accumulator += weight * value;
                }
            }
        }
        Ok(())
    }

    /// Reads a replicated `i32` array back from the single test device.
    fn read_i32_array(client: &ryft_pjrt::Client<'_>, array: &Array<'_>) -> Vec<i32> {
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
        values_from_bytes::<i32>(shard_bytes.as_slice())
    }

    /// Shared driver of the decode-loop demo tests: traces the decode loop through the public API, compiles and
    /// executes it on the CPU plugin, runs the same model eagerly over the reference backend, and cross-checks the
    /// decoded token sequence exactly (the ThreeFry draws are bit-exact across backends) and the final key/value
    /// caches within floating-point tolerance.
    fn run_decode_loop_demo(sampling: DecodeSampling, attention: DecodeAttention) {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        if attention == DecodeAttention::CustomCall {
            ensure_decode_attention_handler_registered(&client).unwrap();
        }

        let configuration = DecodeConfiguration { vocabulary: 16, dimension: 8, steps: 6, top_k: 4 };
        let DecodeConfiguration { vocabulary, dimension, steps, .. } = configuration;
        let weight_dimensions: [&[usize]; 7] = [
            &[vocabulary, dimension],
            &[dimension, dimension],
            &[dimension, dimension],
            &[dimension, dimension],
            &[dimension, dimension],
            &[dimension, dimension],
            &[dimension, vocabulary],
        ];
        let weights: Vec<Vec<f32>> = weight_dimensions
            .iter()
            .enumerate()
            .map(|(index, dimensions)| decode_weight_values(index as u32 + 1, dimensions.iter().product()))
            .collect();

        let replicated = |data_type: DataType, dimensions: &[usize]| {
            ArrayType::new(data_type, static_shape(dimensions))
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))
                .unwrap()
        };
        let mut input_types = vec![
            replicated(DataType::I32, &[]),
            replicated(DataType::I32, &[]),
            replicated(DataType::F32, &[steps, dimension]),
            replicated(DataType::F32, &[steps, dimension]),
            replicated(DataType::I32, &[steps]),
            replicated(DataType::U64, &[2]),
        ];
        input_types.extend(weight_dimensions.iter().map(|dimensions| replicated(DataType::F32, dimensions)));

        let compiled: CompiledXlaFunction<'_, Vec<ArrayType>, Vec<ArrayType>> = compile(
            |inputs| build_decode_loop(inputs, &configuration, sampling, attention),
            input_types.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let device_input = |index: usize, bytes: Vec<u8>| {
            Array::from_host_buffer(&client, input_types[index].clone(), mesh.clone(), bytes.as_slice()).unwrap()
        };
        let mut device_inputs = vec![
            device_input(0, values_to_bytes::<i32>(&[0])),
            device_input(1, values_to_bytes::<i32>(&[3])),
            device_input(2, values_to_bytes::<f32>(&vec![0.0; steps * dimension])),
            device_input(3, values_to_bytes::<f32>(&vec![0.0; steps * dimension])),
            device_input(4, values_to_bytes::<i32>(&vec![0; steps])),
            device_input(5, values_to_bytes::<u64>(&[42, 0])),
        ];
        device_inputs
            .extend(weights.iter().enumerate().map(|(index, values)| device_input(index + 6, values_to_bytes(values))));
        let outputs = engine.interpret(&compiled.executable_function(), device_inputs).unwrap();
        assert_eq!(outputs.len(), 3);
        let device_tokens = read_i32_array(&client, &outputs[0]);
        let device_cache_keys = read_f32_array(&client, &outputs[1]);
        let device_cache_values = read_f32_array(&client, &outputs[2]);

        let unsharded = |data_type: DataType, dimensions: &[usize]| ArrayType::new(data_type, static_shape(dimensions));
        let mut reference_state = vec![
            CpuArray::from_elements(unsharded(DataType::I32, &[]), &[0_i32]).unwrap(),
            CpuArray::from_elements(unsharded(DataType::I32, &[]), &[3_i32]).unwrap(),
            CpuArray::from_elements(unsharded(DataType::F32, &[steps, dimension]), &vec![0.0_f32; steps * dimension])
                .unwrap(),
            CpuArray::from_elements(unsharded(DataType::F32, &[steps, dimension]), &vec![0.0_f32; steps * dimension])
                .unwrap(),
            CpuArray::from_elements(unsharded(DataType::I32, &[steps]), &vec![0_i32; steps]).unwrap(),
            CpuArray::from_elements(unsharded(DataType::U64, &[2]), &[42_u64, 0]).unwrap(),
        ];
        reference_state.extend(weight_dimensions.iter().zip(&weights).map(|(dimensions, values)| {
            CpuArray::from_elements(unsharded(DataType::F32, dimensions), values).unwrap()
        }));
        let reference_state = reference_decode_loop(reference_state, &configuration, sampling);
        let reference_tokens = reference_state[4].elements::<i32>().unwrap();

        // The decoded token sequences must agree exactly: greedy selection and the categorical draws (over
        // bit-identical ThreeFry bits) are only sensitive to floating-point differences at exact logit ties,
        // which the deterministic pseudo-random weights avoid.
        assert_eq!(device_tokens, reference_tokens);
        assert!(reference_tokens.iter().all(|&token| (0..vocabulary as i32).contains(&token)));
        // The decoded sequences are pinned as fixtures: they are deterministic functions of the pseudo-random
        // weights and the ThreeFry seed, and pinning them guards against both backends drifting together. The
        // greedy and sampled sequences differ, demonstrating that the categorical draws actually steer decoding.
        let expected_tokens: &[i32] = match sampling {
            DecodeSampling::Greedy => &[12, 8, 12, 8, 12, 8],
            DecodeSampling::TopK => &[10, 15, 10, 10, 15, 15],
        };
        assert_eq!(device_tokens, expected_tokens);
        for (state_index, device_cache) in [(2, &device_cache_keys), (3, &device_cache_values)] {
            let reference_cache = reference_state[state_index].to_f64s();
            assert_eq!(device_cache.len(), reference_cache.len());
            for (index, (&device_value, &reference_value)) in
                device_cache.iter().zip(reference_cache.iter()).enumerate()
            {
                assert!(
                    (device_value as f64 - reference_value).abs() < 1e-4,
                    "cache element {index}: device {device_value} vs reference {reference_value}",
                );
            }
        }
    }

    /// End-to-end greedy decode-loop demo: a compiled `While` loop maintaining a `DynamicUpdateSlice` key/value
    /// cache with composed masked attention, a gated `tanh`/`logistic` MLP, and greedy `argmax` sampling, traced
    /// through the public API and cross-checked against the eager reference backend.
    #[test]
    fn test_jit_decode_loop_greedy_matches_eager_reference() {
        run_decode_loop_demo(DecodeSampling::Greedy, DecodeAttention::Composed);
    }

    /// End-to-end sampling decode-loop demo: like the greedy demo, but selecting tokens with `top_k` plus a
    /// categorical draw from the ThreeFry generator state threaded through the loop carry, exercising
    /// `RngBitGenerator` inside a compiled `While` region with bit-exact cross-backend token parity.
    #[test]
    fn test_jit_decode_loop_top_k_sampling_matches_eager_reference() {
        run_decode_loop_demo(DecodeSampling::TopK, DecodeAttention::Composed);
    }

    /// End-to-end decode-loop demo with the attention body swapped for a registered XLA FFI custom-call kernel,
    /// exercising `custom_call` inside a compiled `While` region against the composed eager reference.
    #[test]
    fn test_jit_decode_loop_custom_call_attention_matches_eager_reference() {
        run_decode_loop_demo(DecodeSampling::Greedy, DecodeAttention::CustomCall);
    }
}
