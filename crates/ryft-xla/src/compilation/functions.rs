//! XLA function staging, retained JIT dispatch, compilation, and execution.
//!
//! This module wraps the lifecycle from [`ryft_core::compilation`] with XLA array projections and options. [`jit`]
//! creates a [`CompiledFunctionDispatcher`] that selects specializations from runtime inputs. Its first execution of a
//! specialization stages, lowers, and requests compilation, while warm executions reuse the retained executable.
//! [`compile`] stages and obtains a [`CompiledFunction`] for one explicit abstract signature immediately. Both paths
//! share the session's [`CompilationCache`](ryft_core::CompilationCache) cache.
//!
//! [`stage`] returns a [`StagedFunction`] for functions that are composed into larger programs before compilation.
//! Staged functions register runtime captures in their retained capture table instead of embedding runtime arrays in
//! the IR. Runtime invocation uses [`CompiledFunctionDispatcher::execute`] or an [`ExecutableFunction`] obtained
//! through [`CompiledFunction::executable`]. [`ExecutableFunction::execute`] returns structured array outputs, and
//! [`ExecutableFunction::execute_async`] also retains the completion of the whole invocation. Staged functions,
//! compiled functions, and dispatchers can also be called from any context, which is how they compose with transforms
//! and with other compilations.
//!
//! [`jit_statefully`] and [`compile_statefully`] expose the heterogeneous boundary used by external array
//! references. Asynchronous stateful calls atomically publish read leases and pending mutation generations immediately
//! after submission, commit hidden final states without holding host mutexes through device execution, and return a
//! [`StatefulExecution`] that reports completion failures. [`StatefulCompiledFunctionDispatcher::execute_statefully`]
//! returns that completion, and [`StatefulCompiledFunctionDispatcher::execute_statefully_and_wait`] waits for it before
//! returning public outputs. [`StatefulCompiledFunction`] supplies the corresponding entry points with a fixed
//! signature.
//!
//! # Calling Compiled Functions
//!
//! [`StagedFunction::call`], [`CompiledFunction::call`], and [`CompiledFunctionDispatcher::call`] bind a `jit_call`
//! operation, which carries the retained source program of the called function, in the context that their inputs belong
//! to. These inputs and outputs are the `ArrayType` leaves of a [`ProjectedContext`] over an array IR context, which
//! the call recovers from the first input. They are [`Array`]s under a [`Session`] and [`ProjectedValue`]s of
//! tracers under a trace or a transform (e.g., the [`CompileTracer`]s of an enclosing compilation). The
//! `call_in_context` variants take the array IR context explicitly, which also serves nullary functions. Every parent
//! context is supported:
//!
//!   - An eager parent executes the call immediately through eager dispatch, which lowers it on its first use rather
//!     than reusing the retained executable of a [`CompiledFunction`], but which runs it on the same devices, accepts
//!     the same inputs, and places and commits its results as that executable does (refer to the contract of
//!     [`JitCallOperation`](crate::JitCallOperation)). [`ExecutableFunction::execute`] and
//!     [`CompiledFunctionDispatcher::execute`] run a retained executable directly on concrete arrays instead.
//!   - A staging parent (e.g., the trace of an enclosing [`compile`] or [`stage`]) records the call, and the enclosing
//!     program lowers it as part of its own computation.
//!   - Batching, differentiation, and partial evaluation parents rewrite the call through the ordinary rules of the
//!     `jit_call` operation, so a compilation boundary does not change the meaning of these transforms. Transforms
//!     nest in any order, and a transformed call can itself be staged and compiled again.
//!
//! A call places values at its boundary as executions do. Each input is first brought to the placement of the
//! corresponding input of the called function in the calling context (i.e., to the sharding that
//! [`Options::with_input_shardings`] requests for it, if any), through a `reshard` over explicit mesh axes and a
//! `constrain_sharding` over auto mesh axes, so that the call composes with transforms like any other placement. The
//! output sharding overrides of [`Options::with_output_shardings`] are staged on the outputs of the traced function in
//! the same way. Types record placement over explicit and manual mesh axes only, so calls and executions return the
//! same output types, while the runtime placement of executed arrays (i.e., [`Array::sharding`]) also covers the auto
//! mesh axes that an override names.
//!
//! The captured runtime arrays of the called function enter the calling context through [`ImportingContext`] and
//! become leading inputs of the call. An eager parent uses the arrays directly. A root trace registers them in its own
//! capture table, so they become captures of the enclosing program rather than literals. Transform contexts import
//! them through their parent and then treat them like constants (i.e., as replicated under batching, with a
//! structurally zero tangent under differentiation, and as known under partial evaluation).
//!
//! Some bodies are traced over constants alone (i.e., in a `TracingContext<Constant, Operation>`), such as the bodies
//! of [`rematerialize`](ryft_core::rematerialize) and [`composite`](ryft_core::composite), the custom rules of
//! composites, and the regions nested in them. Their capture tables cannot hold runtime arrays, so calls in them
//! import captures through [`ThroughSession`](crate::ThroughSession) instead. The body stages a
//! [`RetainedArray`](crate::RetainedArray) handle to each such array, which retains the array for as long as the
//! handle exists and keeps the callee closed. An eager parent resolves the handle when it lifts it, and lowering passes
//! every distinct handle of a program as a hidden argument of the entry function after its captures, so the executable
//! depends only on the types of these arrays.
//!
//! A dispatcher retains only executables for its runtime specializations, so [`CompiledFunctionDispatcher::call`]
//! traces its closure again on demand into a staged specialization, which it retains in a separate cache (refer to
//! [`CoreCompiledFunctionDispatcher::call_in_context`]). The closure is traced with the named axes and default mesh of
//! the calling context, which also key the staged specialization, so that it resolves named batch and mesh axes
//! (e.g., through [`AxisIndex`](ryft_core::AxisIndex)) as it would if it were inlined. A call neither lowers
//! nor compiles anything itself.
//!
//! # Transforming Compiled Functions
//!
//! [`CompiledFunction::batch`], [`CompiledFunction::gradient`], and [`CompiledFunction::jvp`] are thin conveniences.
//! Each stages the corresponding [`ryft_core`] transform around a `jit_call` of the function and compiles the result
//! through the ordinary lifecycle, so the derived function is itself transformable and the captures of the original
//! function become its captures. [`CompiledFunction::batch`] requires an explicit static batch extent. Runtime extents
//! are available by calling [`batch`](ryft_core::batch) inside a function that is staged with the extent as an
//! input.
//!
//! General transforms call compiled functions from their closures. Over array IR [`Value`]s, the ordinary
//! [`batch`](ryft_core::batch) and [`differentiate_at`](ryft_core::differentiate_at) entry points apply directly, and
//! their closures pass the `ArrayType` projections of their leaves (refer to [`ValueProjection`]) to the call. Over
//! [`Array`] or [`CompileTracer`] leaves, those entry points transform the [`ProjectedContext`] itself, whose
//! operation family has no call operation, so their closures cannot call compiled functions.
//! [`batch_in_parent`](ryft_core::batch_in_parent) and
//! [`differentiate_in_parent_at`](ryft_core::differentiate_in_parent_at) instead run the transform in the array IR
//! parent of those leaves, hand the closure projected leaves, and project the results back. The reusable
//! [`ParentPushforward`](ryft_core::ParentPushforward) and [`ParentPullback`](ryft_core::ParentPullback) results
//! project at every application as well. Jacobians and Hessians are available on both routes. Reusable derivative
//! results also apply in descendant contexts of the context that built them through their `apply_in_context`
//! functions (e.g., under [`batch`](ryft_core::batch) inside the same compilation), which import their residuals the
//! way calls import captures.
//!
//! # Limitations
//!
//!   - A compiled function called inside a [`shard_map`](fn@ryft_core::shard_map) body cannot capture runtime arrays,
//!     because the body is traced as a fresh program whose captures `shard_map` rejects with
//!     [`ProgramError::DiscardedCaptures`]. Session constants cannot appear in a `shard_map` body either, because
//!     `sdy.manual_computation` bodies are isolated from the arguments of the entry function. The `shard_map`
//!     operation also rejects batching with a dynamic batch extent, with or without a compilation boundary.
//!   - Batching that stores batched values into a shared (i.e., unbatched) external reference is rejected, exactly as
//!     it is without a compilation boundary. The external reference section below describes the rest of the stateful
//!     surface.
//!
//! # External Reference ABI
//!
//! The stateful surface is experimental. Its names and serialized executable metadata may change while external
//! automatic differentiation and the semantics of preserved kernels remain under development. Ordinary staging,
//! compilation, and execution over arrays alone are unaffected.
//!
//! Reference lowering opens each external reference as an ordinary array input. A read-only reference has no hidden
//! result. A mutated reference gains one hidden final state output after the public output prefix, and the runtime
//! commits that value atomically into the same reference after execution. The entry argument also receives a
//! `tf.aliasing_output` relation to the hidden result. That relation is a non-semantic may-alias hint. It permits, but
//! never requires, backend buffer reuse, and it never promises physical in-place mutation. Reference state inputs are
//! themselves never donated, so each reference retains the snapshot it passed in, and correctness always comes from
//! committing the returned final value. Read snapshots and retained initializers consequently remain immutable even
//! when a backend chooses to reuse storage.
//!
//! Calls reject unsupported state before launch. The admitted boundary is device memory state on a mesh in which the
//! calling process participates (i.e., every device is visible to its client under the process that owns it, and at
//! least one is addressable by this process). Static references may be replicated or sharded with identical input and
//! final state sharding. Finite bounded dynamic references must be read-only and replicated, and they are admitted only
//! on a fully addressable, single-process mesh. Over a mesh that spans processes (i.e., multi-host execution), every
//! process calls with its own reference over the same mesh, binds and commits only its addressable shards, and so each
//! process commits its own part of the final state. As for any multi-controller output, a failure on one process does
//! not roll back the commits of the others. Zero space, host memory, unbounded dynamic, dynamic nonreplicated, input
//! bucketed, foreign or unknown device, nonparticipating mesh, and bounded dynamic mutation cases are rejected.
//! Reference arguments must be distinct root handles. Internal accesses carry transform paths and dynamic indices,
//! which discharge rewrites as array operations before compilation. Views are not separate program values and cannot
//! cross this boundary.
//!
//! # Stateful Call Shape
//!
//! ```text
//! compile_statefully(|(state, update)| {
//!     old = state.swap(update)
//!     return old                       // public result
//! })
//!
//! execute_statefully_and_wait((reference, update))
//!     -> execute(array(reference), update)
//!     -> receive(old, hidden_final_state)
//!     -> commit hidden_final_state into reference
//!     -> return old
//!
//! compile_statefully_with_captures(
//!     |captures, input| captures[0].add_update(input),
//!     captures = [reference],
//! )
//! execute_statefully_and_wait(input)
//!     -> lift reference as the leading executable input
//!     -> commit its hidden final state through the same transaction
//! ```
//!
//! Captured references use the same protocol after capture lifting and cannot also appear as public arguments. Calls
//! that use the same mutable reference chain their generations through dependencies, while read-only calls publish
//! leases and may overlap. Dropping [`StatefulExecution`] does not cancel work. Awaiting it reports a failure of the
//! whole invocation. A failure between the submission handoff and the commit of the hidden state replacements poisons
//! the complete affected mutation group, so later reference access cannot silently observe stale state. Poisoning is
//! terminal there because no buffer can be proven current after that boundary, and recovery requires constructing a
//! new reference from independently trusted state. Once the hidden final state replacements are committed, the pending
//! generations are disarmed, so a later failure (e.g., in public output reconstruction) surfaces as an ordinary error
//! while the committed reference state stays valid.
//!
//! Local references, including local views and references inside supported conditions, bounded loops, scans, and
//! nested calls, use the ordinary [`jit`] or [`compile`] surface. They are discharged to array SSA before StableHLO
//! lowering and do not change the public ABI. Batching and automatic differentiation of local references likewise
//! discharge first. Rematerialization recomputes local reference lifecycles inside its body, saves reads of external
//! references, and never recomputes their mutations. Automatic differentiation of external references (e.g.,
//! `value_and_gradient`, `linearize`, or `jvp` with reference tangents) and batching over mapped references, or over
//! shared references that are only read, behave inside [`compile_statefully`] and [`jit_statefully`] exactly as they
//! do without a compilation boundary, including around nested calls of functions that take references. Batching that
//! stores batched values into a shared reference is rejected on both paths. References in custom derivative rules and
//! preserved references outside the experimental kernel boundary are rejected explicitly.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use ryft_core::{
    ArrayIrBatchingPolicy, ArrayType, Batch, BatchAxis, BatchAxisSpecification, BatchingError, BatchingTracer,
    ClosedProgram, Compile, CompiledFunction as CoreCompiledFunction,
    CompiledFunctionDispatcher as CoreCompiledFunctionDispatcher, ConstrainSharding, Context, ContextProjection,
    DeviceMesh, DifferentiableType, Dimension, DimensionValue, DispatcherStatistics, DomainTracingContext,
    ExecutableFunction as CoreExecutableFunction, ForwardModeDifferentiate, ImportingContext, MeshAxisType,
    MeshContext, NamedAxes, Parameter, Parameterized, ParameterizedFamily, ProgramError, ProjectedContext,
    ProjectedValue, Reshard, ReverseModeDifferentiate, Sharding, ShardingDimension,
    StagedFunction as CoreStagedFunction, StatefulExecution, Tracer, Typed, Value as CoreValue, ValueProjection,
};
use ryft_pjrt::Execution;

use crate::compilation::CompilationError;
use crate::compilation::domains::with_input_shardings;
use crate::compilation::options::Options;
use crate::compilation::profile_guided::{AdaptiveProfileGuidedFunction, AdaptiveProfileGuidedOptions};
use crate::compilation::programs::CompiledProgram;
use crate::execution::{validate_input_type, validate_runtime_outputs};
use crate::{Array, Constant, Error, ExecutionError, Operation, Session, Type, Value};

/// Array IR tracer retained by the experimental stateful XLA program.
///
/// # Stability
///
/// This alias is experimental while the external reference ABI is evolving.
pub type StatefulCompileTracer = Tracer<DomainTracingContext<Session, Value>>;

/// Tracer leaf exposed by the public XLA compilation facade over arrays.
pub type CompileTracer = ProjectedValue<ArrayType, StatefulCompileTracer>;

/// Heterogeneous XLA runtime value used by the stateful compilation facade.
///
/// # Stability
///
/// This alias is experimental while the external reference ABI is evolving.
pub type StatefulValue = Value;

/// Retained XLA JIT dispatcher over arrays with explicit static parameters on the host.
///
/// [`Self::execute`] runs the dispatcher on concrete arrays. Its first execution for each `(static parameters, dynamic
/// parameter structure, dynamic abstract types)` specialization stages, lowers, and requests compilation, and warm
/// executions dispatch directly to the retained executable. Static values should be configuration with a low
/// cardinality, such as axes, shapes, or Boolean branch choices, while arrays remain dynamic.
///
/// [`Self::call`] instead calls the closure as a `jit_call` operation in any context, including eager, staging, and
/// transform contexts. Executables retain no source program, so a call traces the closure again on demand into a source
/// specialization, which is additionally keyed by the named axes and default mesh of the calling context and retained
/// in a cache separate from the executables. Transformed calls therefore never evict executables, and warm executions
/// never trace. Host side effects of the closure run once per executable specialization and once per source
/// specialization. Both kinds of specialization stage the output sharding overrides of the dispatcher's [`Options`],
/// and calls bring their inputs to the input sharding overrides first, so that calls and executions agree on the
/// placement of values at the boundary (refer to the [module documentation](crate::compilation::functions)).
pub struct CompiledFunctionDispatcher<
    F,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
> where
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Array IR domain dispatcher hidden behind the public array projection boundary.
    function: CoreCompiledFunctionDispatcher<Session, Static, ProgramParameters<In>, ProgramParameters<Out>, F>,
}

impl<F, Static, In, Out> Clone for CompiledFunctionDispatcher<F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    CoreCompiledFunctionDispatcher<Session, Static, ProgramParameters<In>, ProgramParameters<Out>, F>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<F, Static, In, Out> CompiledFunctionDispatcher<F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<Array>
        + ParameterizedFamily<StatefulValue>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<Array>
        + ParameterizedFamily<StatefulValue>
        + ParameterizedFamily<StatefulCompileTracer>,
{
    /// Executes a specialization with public array inputs, compiling it on a miss and reusing it on warm invocations.
    /// The array IR runtime outputs are projected back to the array representation of the output container family.
    pub fn execute(&self, static_parameters: Static, inputs: In::To<Array>) -> Result<Out::To<Array>, Error>
    where
        F: Fn(
            Static,
            ProgramParameterValues<In, StatefulCompileTracer>,
        ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer>, Error>,
        ProgramParameters<In>: Parameterized<Type, To<Type> = ProgramParameters<In>>,
        ProgramParameters<Out>: Parameterized<Type, To<Type> = ProgramParameters<Out>>,
        ProgramParameterValues<In, StatefulValue>: Parameterized<StatefulValue, To<Type> = ProgramParameters<In>>,
        ProgramParameterValues<Out, StatefulCompileTracer>: Parameterized<
                StatefulCompileTracer,
                To<Type> = ProgramParameters<Out>,
                To<Constant> = ProgramConstants<Out>,
            >,
    {
        let inputs = lift_values::<In, StatefulValue>(inputs)?;
        Ok(project_values::<Out, StatefulValue>(self.function.execute(static_parameters, inputs)?)?)
    }

    /// Calls the staged specialization of this dispatcher for `static_parameters` and the abstract types of `inputs` as
    /// a `jit_call` operation in the array IR context recovered from `inputs`. Unlike [`Self::execute`], it works under
    /// any context, including eager, staging, and transform contexts, so this dispatcher's closure composes with
    /// enclosing transforms and compilations. Inputs and outputs are `ArrayType` values of a [`ProjectedContext`] over
    /// that context (e.g., [`Array`]s under a [`Session`] and [`ProjectedValue`]s under traces and transforms),
    /// as for [`StagedFunction::call`]. The closure is traced with the named axes and default mesh of that context, so
    /// it can read named batch and mesh axes of its caller. Refer to
    /// [`CoreCompiledFunctionDispatcher::call_in_context`] for how staged specializations are traced and retained.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `inputs` has no leaves, and the errors of
    /// [`Self::call_in_context`] otherwise.
    #[inline]
    pub fn call<L, C, Path>(&self, static_parameters: Static, inputs: In::To<L>) -> Result<Out::To<L>, Error>
    where
        F: Fn(
            Static,
            ProgramParameterValues<In, StatefulCompileTracer>,
        ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer>, Error>,
        L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
        C: Context<Type = Type, Constant = Constant, Operation = Operation>
            + ContextProjection<ArrayType>
            + ImportingContext<Session, Path>
            + NamedAxes
            + MeshContext,
        C::Value: ValueProjection<ArrayType, Projected = L> + Reshard + ConstrainSharding,
        In::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<L>,
        Out::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<L>,
    {
        let context = inputs
            .parameters()
            .next()
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
            .domain();
        self.call_in_context(context.parent(), static_parameters, inputs)
    }

    /// Calls like [`Self::call`] in the provided array IR `context`, whose `ArrayType` projection the inputs and
    /// outputs use. An explicit context also serves nullary calls.
    ///
    /// # Errors
    ///
    /// Returns an error when the session rejects the staged input types, when tracing the closure fails, when the same
    /// thread recursively requests a staged specialization that it is still producing, or when importing a capture,
    /// binding the call, or reassembling the structured outputs fails.
    #[inline]
    pub fn call_in_context<C, Path>(
        &self,
        context: &C,
        static_parameters: Static,
        inputs: In::To<<C::Value as ValueProjection<ArrayType>>::Projected>,
    ) -> Result<Out::To<<C::Value as ValueProjection<ArrayType>>::Projected>, Error>
    where
        F: Fn(
            Static,
            ProgramParameterValues<In, StatefulCompileTracer>,
        ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer>, Error>,
        C: Context<Type = Type, Constant = Constant, Operation = Operation>
            + ImportingContext<Session, Path>
            + NamedAxes
            + MeshContext,
        C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
        In::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
        Out::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    {
        // The staged specialization is staged at the input types with the input sharding overrides applied, so inputs
        // that differ from those only in their sharding are brought to them first, as `StagedFunction::call` does.
        let input_structure = inputs.parameter_structure();
        let inputs = inputs
            .into_parameters()
            .map(<C::Value as ValueProjection<ArrayType>>::from_projected)
            .collect::<Vec<_>>();
        let inputs = match self.function.options().input_shardings() {
            None => inputs,
            Some(overrides) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                let input_types = with_input_shardings(input_types, self.function.options())?;
                place_inputs(inputs, &input_types, Some(overrides))?
            }
        };
        let inputs = ProgramParameterValues::<In, C::Value>::from_parameters(input_structure, inputs)
            .map_err(ProgramError::from)?;
        let outputs = self.function.call_in_context(context, static_parameters, inputs)?;
        Ok(project_values::<Out, C::Value>(outputs)?)
    }
}

impl<F, Static, In, Out> CompiledFunctionDispatcher<F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Returns a snapshot of this dispatcher's frontend activity (refer to [`DispatcherStatistics`]).
    #[inline]
    pub fn statistics(&self) -> DispatcherStatistics {
        self.function.statistics()
    }
}

/// Constructs a retained dispatcher for `function` with explicit options. Execute it on concrete arrays through
/// [`CompiledFunctionDispatcher::execute`], and call it from traces and transforms through
/// [`CompiledFunctionDispatcher::call`].
///
/// `function` is fallible so that it can reject its inputs while it is traced, and an infallible function returns its
/// outputs wrapped in [`Ok`].
pub fn jit_with_options<F, Static, In, Out>(
    function: F,
    session: &Session,
    options: Options,
) -> CompiledFunctionDispatcher<
    impl Fn(
        Static,
        ProgramParameterValues<In, StatefulCompileTracer>,
    ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<CompileTracer>) -> Result<Out::To<CompileTracer>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>
        + ParameterizedFamily<CompileTracer>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>
        + ParameterizedFamily<CompileTracer>,
{
    // Executable and staged specializations both trace this closure, so both stage the output sharding overrides.
    let output_shardings = options.output_shardings().map(<[Sharding]>::to_vec);
    let function = move |static_parameters, inputs| {
        let outputs = function(static_parameters, project_values::<In, StatefulCompileTracer>(inputs)?)?;
        place_outputs(lift_values::<Out, StatefulCompileTracer>(outputs)?, output_shardings.as_deref())
    };
    CompiledFunctionDispatcher { function: session.jit_with_options(function, options) }
}

/// Constructs a retained dispatcher for `function` on `mesh` with otherwise default options. Refer to
/// [`jit_with_options`] for more information.
///
/// `function` is fallible so that it can reject its inputs while it is traced, and an infallible function returns its
/// outputs wrapped in [`Ok`].
#[inline]
pub fn jit<F, Static, In, Out>(
    function: F,
    session: &Session,
    mesh: DeviceMesh,
) -> CompiledFunctionDispatcher<
    impl Fn(
        Static,
        ProgramParameterValues<In, StatefulCompileTracer>,
    ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<CompileTracer>) -> Result<Out::To<CompileTracer>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>
        + ParameterizedFamily<CompileTracer>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>
        + ParameterizedFamily<CompileTracer>,
{
    jit_with_options(function, session, Options::new(mesh))
}

/// Retained heterogeneous XLA JIT dispatcher whose calls may bind external references.
///
/// The dispatcher retains one compiled specialization per combination of static parameters and input types, exactly
/// like [`CompiledFunctionDispatcher`], but its boundary is the heterogeneous [`StatefulValue`] family, whose leaves of
/// reference type bind live [`ryft_core::ArrayReference`] values at call time instead of immutable arrays. Every
/// reference leaf must be an unrenamed root handle whose type refines the effective compiled boundary, including its
/// declared sharding and memory. A read-only bounded dynamic reference may refine its finite replicated declaration.
/// The runtime snapshots every reference, releases its state locks at the backend submission handoff, and publishes
/// read leases that carry the completion of the call, or hidden final states, before public outputs are exposed.
///
/// # Stability
///
/// This external state surface is experimental. Ordinary [`CompiledFunctionDispatcher`] calls remain the stable
/// boundary over arrays.
pub struct StatefulCompiledFunctionDispatcher<
    F,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<Type>,
    Out: Parameterized<Type>,
> where
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Constant>,
{
    /// Backend-neutral dispatcher retaining heterogeneous runtime values and compiled specializations.
    function: CoreCompiledFunctionDispatcher<Session, Static, In, Out, F>,
}

impl<F, Static, In, Out> Clone for StatefulCompiledFunctionDispatcher<F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<Type>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>,
    Out: Parameterized<Type>,
    Out::Family: ParameterizedFamily<Constant>,
    CoreCompiledFunctionDispatcher<Session, Static, In, Out, F>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<F, Static, In, Out> StatefulCompiledFunctionDispatcher<F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<Type>,
    In::ParameterStructure: Eq + Hash,
    In::Family:
        ParameterizedFamily<Constant> + ParameterizedFamily<StatefulValue> + ParameterizedFamily<StatefulCompileTracer>,
    In::To<StatefulValue>:
        Parameterized<StatefulValue, Family = In::Family, ParameterStructure = In::ParameterStructure, To<Type> = In>,
    Out: Parameterized<Type>,
    Out::Family:
        ParameterizedFamily<Constant> + ParameterizedFamily<StatefulValue> + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<StatefulValue>:
        Parameterized<StatefulValue, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    Out::To<StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = Out, To<Constant> = Out::To<Constant>>,
{
    /// Executes one specialization statefully, compiling it on the first call and reusing it on warm calls.
    ///
    /// The call snapshots every bound reference in ascending identity order, completes fallible preparation without
    /// retaining reference state locks, and revalidates reference generations before submission. It waits for the
    /// completion returned by [`Self::execute_statefully`], so completion errors are reported and read-only leases
    /// have completed before it returns. Mutated references reconcile their ready or poisoned state lazily on their
    /// next access.
    pub fn execute_statefully_and_wait(
        &self,
        static_parameters: Static,
        inputs: In::To<StatefulValue>,
    ) -> Result<Out::To<StatefulValue>, Error>
    where
        F: Fn(Static, In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    {
        self.function.execute_statefully_and_wait(static_parameters, inputs)
    }

    /// Executes one retained specialization statefully and returns its [`StatefulExecution`], which carries the
    /// completion of the call.
    ///
    /// The returned [`StatefulExecution`] may already carry reconstructed public outputs backed by pending device
    /// values. Awaiting it observes the completion of the whole invocation, including asynchronous execution errors
    /// that would otherwise surface only when the affected reference is next accessed. Dropping the result never
    /// cancels submitted work.
    pub fn execute_statefully(
        &self,
        static_parameters: Static,
        inputs: In::To<StatefulValue>,
    ) -> StatefulExecution<Out::To<StatefulValue>, Error>
    where
        F: Fn(Static, In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    {
        self.function.execute_statefully(static_parameters, inputs)
    }
}

impl<F, Static, In, Out> StatefulCompiledFunctionDispatcher<F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<Type>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>,
    Out: Parameterized<Type>,
    Out::Family: ParameterizedFamily<Constant>,
{
    /// Returns a snapshot of this dispatcher's frontend activity (refer to [`DispatcherStatistics`]).
    #[inline]
    pub fn statistics(&self) -> DispatcherStatistics {
        self.function.statistics()
    }
}

/// Constructs a retained stateful dispatcher for a heterogeneous XLA `function` with explicit options.
///
/// `function` is fallible so that it can reject its inputs while it is traced, and an infallible function returns its
/// outputs wrapped in [`Ok`].
///
/// # Stability
///
/// This external state entry point is experimental.
pub fn jit_statefully_with_options<F, Static, In, Out>(
    function: F,
    session: &Session,
    options: Options,
) -> StatefulCompiledFunctionDispatcher<
    impl Fn(Static, In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<Type, Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>>,
    In::ParameterStructure: Eq + Hash,
    Out: Parameterized<Type, Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>>,
{
    let output_shardings = options.output_shardings().map(<[Sharding]>::to_vec);
    let function = move |static_parameters, inputs| {
        place_outputs(function(static_parameters, inputs)?, output_shardings.as_deref())
    };
    StatefulCompiledFunctionDispatcher { function: session.jit_with_options(function, options) }
}

/// Constructs a retained stateful dispatcher for a heterogeneous XLA `function` on `mesh` with otherwise default
/// options.
///
/// `function` is fallible so that it can reject its inputs while it is traced, and an infallible function returns its
/// outputs wrapped in [`Ok`].
///
/// # Stability
///
/// This external state entry point is experimental.
#[inline]
pub fn jit_statefully<F, Static, In, Out>(
    function: F,
    session: &Session,
    mesh: DeviceMesh,
) -> StatefulCompiledFunctionDispatcher<
    impl Fn(Static, In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<Type, Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>>,
    In::ParameterStructure: Eq + Hash,
    Out: Parameterized<Type, Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>>,
{
    jit_statefully_with_options(function, session, Options::new(mesh))
}

/// Compiled heterogeneous XLA function whose invocation may bind external references.
///
/// This is the counterpart of [`StatefulCompiledFunctionDispatcher`] for a fixed signature. Its input leaves of
/// reference type bind unrenamed root [`ryft_core::ArrayReference`] handles whose types refine the effective compiled
/// boundary, including its declared sharding and memory. Read-only bounded dynamic references may refine finite
/// replicated declarations. Each invocation uses the same snapshot, generation revalidation, asynchronous publication,
/// and completion protocol as the retained dispatcher.
///
/// # Stability
///
/// This external state surface is experimental. Ordinary [`CompiledFunction`] calls remain the stable boundary over
/// arrays.
pub struct StatefulCompiledFunction<In: Parameterized<Type>, Out: Parameterized<Type>>
where
    In::Family: ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Constant>,
{
    /// Backend-neutral compiled function retaining the array IR runtime boundary.
    function: CoreCompiledFunction<Session, In, Out>,
}

impl<In, Out> StatefulCompiledFunction<In, Out>
where
    In: Parameterized<Type>,
    Out: Parameterized<Type>,
    In::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulValue>,
    Out::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulValue>,
    In::To<StatefulValue>: Parameterized<StatefulValue>,
    Out::To<StatefulValue>:
        Parameterized<StatefulValue, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
{
    /// Returns the executable handle, which retains only the runtime state of this function.
    #[inline]
    pub fn executable(&self) -> &CoreExecutableFunction<Session, In, Out> {
        self.function.executable()
    }

    /// Returns the public flat output types, excluding the hidden final state output suffix that reference
    /// discharge appends for mutated external references.
    #[inline]
    pub fn output_types(&self) -> Cow<'_, [Type]> {
        self.function.executable().output_types()
    }

    /// Executes synchronously and returns public outputs after the reference transaction completes.
    ///
    /// The call waits for the completion of the call after pending reference state and read leases are published.
    /// A failed completion therefore cannot leave silently stale reference state behind. Refer to the documentation of
    /// [`StatefulCompiledFunctionDispatcher::execute_statefully_and_wait`] for the complete transaction contract.
    pub fn execute_statefully_and_wait(
        &self,
        session: &Session,
        inputs: In::To<StatefulValue>,
    ) -> Result<Out::To<StatefulValue>, Error> {
        self.function.executable().execute_statefully_and_wait(session, inputs)
    }

    /// Submits a stateful call and returns its result, which carries the completion of the call and never cancels it.
    ///
    /// Refer to the documentation of [`StatefulCompiledFunctionDispatcher::execute_statefully`] for the contract that
    /// governs completion and the observation of errors.
    pub fn execute_statefully(
        &self,
        session: &Session,
        inputs: In::To<StatefulValue>,
    ) -> StatefulExecution<Out::To<StatefulValue>, Error> {
        self.function.executable().execute_statefully(session, inputs)
    }
}

/// Compiles one heterogeneous signature for stateful invocation.
///
/// # Stability
///
/// This external state entry point is experimental.
pub fn compile_statefully<F, In, Out>(
    function: F,
    input_types: In,
    session: &Session,
    options: Options,
) -> Result<StatefulCompiledFunction<In, Out>, Error>
where
    F: FnOnce(In::To<StatefulCompileTracer>) -> Result<Out::To<StatefulCompileTracer>, Error>,
    In: Parameterized<Type>,
    In::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out: Parameterized<Type>,
    Out::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = Out, To<Constant> = Out::To<Constant>>,
{
    compile_statefully_with_captures(move |_, inputs| function(inputs), Vec::new(), input_types, session, options)
}

/// Compiles one heterogeneous signature with explicit array or reference captures for stateful invocation.
///
/// # Stability
///
/// This external state entry point is experimental. A captured reference participates in the same reference
/// transaction as a public reference input and cannot alias any other capture or public argument in the invocation.
pub fn compile_statefully_with_captures<F, In, Out>(
    function: F,
    captures: Vec<StatefulValue>,
    input_types: In,
    session: &Session,
    options: Options,
) -> Result<StatefulCompiledFunction<In, Out>, Error>
where
    F: FnOnce(
        Vec<StatefulCompileTracer>,
        In::To<StatefulCompileTracer>,
    ) -> Result<Out::To<StatefulCompileTracer>, Error>,
    In: Parameterized<Type>,
    In::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out: Parameterized<Type>,
    Out::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = Out, To<Constant> = Out::To<Constant>>,
{
    let input_types = with_input_shardings(input_types, &options)?;
    let output_shardings = options.output_shardings().map(<[Sharding]>::to_vec);
    let staged = CoreStagedFunction::<Session, In, Out>::stage(
        |_, capture_tracers, inputs| place_outputs(function(capture_tracers, inputs)?, output_shardings.as_deref()),
        captures,
        input_types,
        options,
    )?;
    let lowered = staged.lower(session)?;
    Ok(StatefulCompiledFunction { function: lowered.compile(session)? })
}

/// Staged XLA function that has not been compiled. Returned by [`stage`] and [`stage_with_captures`].
///
/// Holds the traced source [`Program`](ryft_core::programs::Program) of one closure together with its captured
/// runtime [`Array`]s and input and output type metadata, without compiling a PJRT executable. This is the right
/// handle for functions that are only composed into larger programs. [`Self::call`] calls it in any context as a
/// `jit_call` boundary that carries its source program (refer to the
/// [module documentation](crate::compilation::functions)), and [`Session::compile_staged_function`] produces a
/// [`CompiledFunction`] when an executable is needed. Executable cache identity is derived from the complete lowered
/// computation and the backend state that affects compilation, so equivalent lowerings can share an executable even
/// when they were produced at different Rust call sites.
pub struct StagedFunction<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Backend-neutral staged function carrying the source program, captures, structures, and retained options.
    function: CoreStagedFunction<Session, ProgramParameters<In>, ProgramParameters<Out>>,
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for StagedFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> StagedFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Returns the flat staged source [`Program`](ryft_core::Program) together with its captured runtime values.
    /// Use it for program transforms (e.g., transposition or partial evaluation of the program itself) and for
    /// diagnostics such as printed IR, instruction counts, and graph rendering. Transforms of values instead call this
    /// function through [`Self::call`].
    #[inline]
    pub fn source_program(&self) -> &ClosedProgram<StatefulValue, Constant, Operation, Vec<Constant>, Vec<Constant>> {
        self.function.source_program()
    }

    /// Calls this function as a `jit_call` operation in the array IR context `C` recovered from `inputs`.
    ///
    /// The call binds a boundary that carries this function's retained source program, so enclosing transforms
    /// rewrite it through the ordinary rules of the `jit_call` operation. Inputs and outputs are `ArrayType` values of
    /// a [`ProjectedContext`] over `C`, which the call recovers from the first input. Under a [`Session`], they
    /// are [`Array`]s and the call executes immediately through eager dispatch. Under a trace or a transform (e.g.,
    /// forward mode differentiation), they are [`ProjectedValue`]s of its tracers and the call is recorded or
    /// rewritten there. The captured runtime arrays of this function enter `C` through [`ImportingContext`], which
    /// treats them as values that do not vary across a batch, have structurally zero tangents, and are known to partial
    /// evaluation. Traces over constants alone (e.g., the bodies of [`rematerialize`](ryft_core::rematerialize) and
    /// [`composite`](ryft_core::composite)) import them as session constants instead (refer to
    /// [`ThroughSession`](crate::ThroughSession)). An input whose type differs from the corresponding input type of
    /// this function only in its sharding is first brought to that type in `C` (refer to the
    /// [module documentation](crate::compilation::functions)). Use [`Self::call_in_context`] for nullary functions,
    /// because no input exists from which to recover the context.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `inputs` has no leaves, and an error when importing a capture,
    /// binding the call, or reassembling the structured outputs fails.
    #[inline]
    pub fn call<L, C, Path>(&self, inputs: In::To<L>) -> Result<Out::To<L>, ProgramError>
    where
        L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
        C: Context<Type = Type, Constant = Constant, Operation = Operation>
            + ContextProjection<ArrayType>
            + ImportingContext<Session, Path>,
        C::Value: ValueProjection<ArrayType, Projected = L> + Reshard + ConstrainSharding,
        In::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<L>,
        Out::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<L>,
    {
        let context = inputs
            .parameters()
            .next()
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
            .domain();
        self.call_in_context(context.parent(), inputs)
    }

    /// Calls this function like [`Self::call`] in the provided array IR `context`, whose `ArrayType` projection the
    /// inputs and outputs use. An explicit context also serves nullary functions.
    ///
    /// # Errors
    ///
    /// Returns an error when importing a capture, binding the call, or reassembling the structured outputs fails.
    pub fn call_in_context<C, Path>(
        &self,
        context: &C,
        inputs: In::To<<C::Value as ValueProjection<ArrayType>>::Projected>,
    ) -> Result<Out::To<<C::Value as ValueProjection<ArrayType>>::Projected>, ProgramError>
    where
        C: Context<Type = Type, Constant = Constant, Operation = Operation> + ImportingContext<Session, Path>,
        C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
        In::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
        Out::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    {
        call_staged_in_context::<In, Out, C, Path>(&self.function, context, inputs)
    }

    /// Consumes this facade and returns its array IR staged artifact for verification within the crate.
    #[cfg(test)]
    pub(crate) fn into_inner(self) -> CoreStagedFunction<Session, ProgramParameters<In>, ProgramParameters<Out>> {
        self.function
    }
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
    From<CoreStagedFunction<Session, ProgramParameters<In>, ProgramParameters<Out>>> for StagedFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    #[inline]
    fn from(function: CoreStagedFunction<Session, ProgramParameters<In>, ProgramParameters<Out>>) -> Self {
        Self { function }
    }
}

/// XLA executable handle that retains only runtime state.
///
/// Unlike [`CompiledFunction`], this type does not retain the shared staged program or the lowering metadata that
/// transforms use. It can only execute and inspect its runtime signature, so it cannot be called inside traces or
/// transforms, which require [`CompiledFunction::call`] and the source program that it carries. Its thread safety is
/// derived from its backend state, and no unsafe blanket implementation is used.
pub struct ExecutableFunction<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Backend-neutral executable function retaining the compiled program, the captured runtime values, and the
    /// runtime signature.
    function: CoreExecutableFunction<Session, ProgramParameters<In>, ProgramParameters<Out>>,
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for ExecutableFunction<In, Out>
where
    In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> ExecutableFunction<In, Out>
where
    In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
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

    /// Executes this function through `session` on concrete [`Array`] inputs and reconstructs its structured outputs.
    /// Returned arrays can still have pending device work. Use [`Self::execute_async`] when a completion handle for
    /// the whole invocation is needed, including for a function with no array outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the executable belongs to another client, input validation or output reconstruction fails,
    /// or execution reports a failure. Deferred device failures can also surface when an output is awaited.
    #[inline]
    pub fn execute(&self, session: &Session, inputs: In::To<Array>) -> Result<Out::To<Array>, Error>
    where
        In: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<Type>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array>
                            + ParameterizedFamily<StatefulValue>,
            >,
        Out: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<Type>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array>
                            + ParameterizedFamily<StatefulValue>,
            >,
        Out::To<Array>: Parameterized<Array, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs = lift_values::<In, StatefulValue>(inputs)?;
        Ok(project_values::<Out, StatefulValue>(self.function.execute(session, inputs)?)?)
    }

    /// Submits this function through `session` and returns its structured outputs together with the completion of the
    /// whole invocation, including for a function with no array outputs. Await the returned [`Execution`] to observe
    /// asynchronous execution failures.
    ///
    /// Erased inputs of zero size await producer completion before submission because no physical input buffer can
    /// carry their dependency. Input preparation and output reconstruction for bounded dynamic shapes can also wait
    /// for transfers.
    ///
    /// # Errors
    ///
    /// Returns an error if the executable requires stateful invocation, belongs to another client, or fails input
    /// validation, submission, or output reconstruction.
    pub fn execute_async(&self, session: &Session, inputs: In::To<Array>) -> Result<Execution<Out::To<Array>>, Error>
    where
        In: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<Type>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array>
                            + ParameterizedFamily<StatefulValue>,
            >,
        Out: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<Type>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array>
                            + ParameterizedFamily<StatefulValue>,
            >,
        Out::To<Array>: Parameterized<Array, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        if self.function.compiled_program().requires_stateful_call() {
            return Err(ExecutionError::StatefulInvocationRequired.into());
        }
        let flat_inputs = lift_values::<In, StatefulValue>(inputs)?.into_parameters().collect::<Vec<_>>();
        self.function.validate_input_count(flat_inputs.len())?;
        for (expected, actual) in self.function.input_types().iter().zip(flat_inputs.iter().map(Typed::r#type)) {
            validate_input_type(
                <&ArrayType>::try_from(expected).map_err(ProgramError::from)?,
                <&ArrayType>::try_from(actual.as_ref()).map_err(ProgramError::from)?,
            )?;
        }

        let inputs = self
            .function
            .inputs_with_captures(flat_inputs)
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let execution = session.execute_compiled_async(self.function.compiled_program(), inputs)?;
        let (flat_outputs, fence) = execution.into_parts();
        validate_runtime_outputs(self.function.compiled_program().output_types(), flat_outputs.as_slice())?;
        let outputs = ProgramParameterValues::<Out, Value>::from_parameters(
            self.function.output_structure().clone(),
            flat_outputs.into_iter().map(Value::Array),
        )
        .map_err(ProgramError::from)?;
        Ok(Execution::new(project_values::<Out, StatefulValue>(outputs)?, fence))
    }
}

impl Session {
    /// Lowers and compiles a staged XLA function using the options that it retained when it was staged.
    pub fn compile_staged_function<In, Out>(
        &self,
        staged: StagedFunction<In, Out>,
    ) -> Result<CompiledFunction<In, Out>, Error>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    {
        let function = staged.function.lower(self)?.compile(self)?;
        Ok(CompiledFunction { function, derived: Arc::new(DerivedFunctionSlots::new()) })
    }

    /// Replaces an executable program while preserving and validating its runtime signature.
    pub(crate) fn replace_executable_program<In, Out>(
        &self,
        executable: &ExecutableFunction<In, Out>,
        program: Arc<CompiledProgram>,
    ) -> Result<ExecutableFunction<In, Out>, Error>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    {
        self.validate_replacement(executable.function.compiled_program(), &program)?;
        Ok(ExecutableFunction { function: executable.function.with_compiled_program(program) })
    }

    /// Wraps a compiled function in an adaptive profile-guided dispatcher.
    ///
    /// The returned handle retains only runtime state. It samples a bounded number of baseline executions, recompiles
    /// the already lowered StableHLO through this session's ordinary compilation cache with the aggregated XLA profile,
    /// and then atomically directs subsequent calls to the compatible optimized executable.
    pub fn adaptive_profile_guided_recompilation<In, Out>(
        &self,
        function: &CompiledFunction<In, Out>,
        options: AdaptiveProfileGuidedOptions,
    ) -> Result<AdaptiveProfileGuidedFunction<In, Out>, Error>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    {
        AdaptiveProfileGuidedFunction::new(
            self.clone(),
            function.executable(),
            function.function.lowered().lowered_program().clone(),
            function.function.lowered().options().clone(),
            options,
        )
    }
}

/// Compiled XLA function for one explicit abstract signature. Returned by [`compile`], [`compile_with_captures`],
/// [`compile_with_options`], and [`Session::compile_staged_function`].
///
/// Holds the cached [`CompiledProgram`], which a PJRT executable backs, together with the [`StagedFunction`] it was
/// compiled from, whose input and output type metadata marshals a [`Parameterized`] tree of [`Array`]s into the
/// executable and reassembles the outputs into the expected output structure. [`Self::executable`] extracts the
/// [`ExecutableFunction`] that retains only runtime state and executes this signature directly.
///
/// The retained staged function also keeps the source [`Program`](ryft_core::Program) that was compiled, which makes
/// this handle transformable. [`Self::call`] calls the function in any context as a `jit_call` boundary that carries
/// this source (refer to the [module documentation](crate::compilation::functions)), so it composes with enclosing
/// transforms and compilations. [`Self::batch`], [`Self::gradient`], and [`Self::jvp`] compile transformed functions
/// around such a call, and [`Self::source_program`] exposes the source for program transforms and diagnostics (e.g.,
/// printed IR, instruction counts, and graph rendering).
pub struct CompiledFunction<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Backend-neutral compiled function backed by an XLA executable. Retains the staged source function,
    /// captured runtime buffers, output structure, and compilation options through its lowered metadata.
    function: CoreCompiledFunction<Session, ProgramParameters<In>, ProgramParameters<Out>>,

    /// Derived transformed functions retained by [`Self::gradient`] and [`Self::jvp`]. Shared across clones so a
    /// wrapper's clones reuse one retained derivative per transform.
    derived: Arc<DerivedFunctions<In, Out>>,
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for CompiledFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone(), derived: Arc::clone(&self.derived) }
    }
}

impl<In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> CompiledFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Type> + ParameterizedFamily<Constant>,
{
    /// Returns the flat output [`ArrayType`]s in the order the executor produces them.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.function.compiled_program().output_types()
    }

    /// Returns a handle that retains only runtime state and omits staged and lowered transform metadata.
    #[inline]
    pub fn executable(&self) -> ExecutableFunction<In, Out> {
        ExecutableFunction { function: self.function.executable().clone() }
    }

    /// Returns the staged function this executable was compiled from.
    #[inline]
    pub fn staged(&self) -> StagedFunction<In, Out> {
        self.function.staged().clone().into()
    }

    /// Returns the flat source [`Program`](ryft_core::Program) that produced the compiled artifact. Use it for
    /// program transforms (e.g., transposition or partial evaluation of the program itself) and for diagnostics such
    /// as printed IR, instruction counts, and graph rendering. Transforms of values instead call this function through
    /// [`Self::call`].
    #[inline]
    pub fn source_program(&self) -> &ClosedProgram<StatefulValue, Constant, Operation, Vec<Constant>, Vec<Constant>> {
        self.function.source_program()
    }

    /// Returns the device mesh the compiled program runs against. Delegates to the cached
    /// [`CompiledProgram::mesh`](crate::compilation::CompiledProgram::mesh).
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.function.compiled_program().mesh()
    }

    /// Calls this compiled function as a `jit_call` operation in the array IR context recovered from `inputs`, exactly
    /// as [`StagedFunction::call`] calls its retained staged function. Under a [`Session`], the call executes
    /// through eager dispatch rather than through the retained executable, so use [`ExecutableFunction::execute`] to
    /// run that executable directly on concrete arrays.
    ///
    /// Refer to the documentation of [`StagedFunction::call`] for more information.
    #[inline]
    pub fn call<L, C, Path>(&self, inputs: In::To<L>) -> Result<Out::To<L>, ProgramError>
    where
        L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
        C: Context<Type = Type, Constant = Constant, Operation = Operation>
            + ContextProjection<ArrayType>
            + ImportingContext<Session, Path>,
        C::Value: ValueProjection<ArrayType, Projected = L> + Reshard + ConstrainSharding,
        In::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<L>,
        Out::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<L>,
    {
        let context = inputs
            .parameters()
            .next()
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
            .domain();
        self.call_in_context(context.parent(), inputs)
    }

    /// Calls this compiled function like [`Self::call`] in the provided array IR `context`. Refer to the
    /// documentation of [`StagedFunction::call_in_context`] for more information.
    #[inline]
    pub fn call_in_context<C, Path>(
        &self,
        context: &C,
        inputs: In::To<<C::Value as ValueProjection<ArrayType>>::Projected>,
    ) -> Result<Out::To<<C::Value as ValueProjection<ArrayType>>::Projected>, ProgramError>
    where
        C: Context<Type = Type, Constant = Constant, Operation = Operation> + ImportingContext<Session, Path>,
        C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
        In::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
        Out::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    {
        call_staged_in_context::<In, Out, C, Path>(self.function.staged(), context, inputs)
    }

    /// Consumes this transformable handle and returns its executable, which retains only runtime state.
    #[inline]
    pub fn into_executable(self) -> ExecutableFunction<In, Out> {
        ExecutableFunction { function: self.function.into_executable() }
    }

    /// Returns the [`Options`] with which functions derived from this one (e.g., its gradient) are compiled. They keep
    /// the options that select how programs lower (i.e., the mesh, the kernel compiler, and the ragged dot lowering
    /// strategy) and drop the ones that describe this function's own signature or executable (i.e., its input and
    /// output shardings, donation flags, input bound bucketing, and feedback-directed profile), which do not apply to a
    /// derived function.
    fn derived_options(&self) -> Options {
        let options = self.function.staged().options();
        let derived =
            Options::new(self.mesh().clone()).with_ragged_dot_lowering_strategy(options.ragged_dot_lowering_strategy());
        match options.kernel_compiler() {
            Some(compiler) => derived.with_kernel_compiler(compiler.clone()),
            None => derived,
        }
    }

    /// Returns the input types of this function in its input structure, from which functions derived from this one
    /// (e.g., its gradient) are traced.
    fn input_signature(&self) -> Result<In, Error> {
        let staged = self.function.staged();
        let input_types = staged
            .input_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(In::from_parameters(staged.input_structure().clone(), input_types).map_err(ProgramError::from)?)
    }
}

// Reverse mode differentiation packaged as a function. Mirrors `jax.grad(jax.jit(f))`. The original closure is never
// traced again. The derived trace calls the retained staged function through `call_flat_in_context`, which binds a
// `jit_call` boundary that carries the sharding overrides of the function (refer to `Session::call_operation`) and that
// reverse mode differentiation rewrites through its ordinary JVP and transpose rules.
impl<In: Parameterized<ArrayType, To<ArrayType> = In>> CompiledFunction<In, ArrayType>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Debug + Hash + PartialEq,
{
    /// Returns a new compiled function that computes the reverse mode gradient of `self` with respect to its inputs.
    /// Mirrors `jax.grad(jax.jit(f))`.
    ///
    /// `self` must produce a single rank-0 output. The `Out = ArrayType` bound of this function requires a single
    /// output, and seeding the pullback rejects outputs of any other rank. The returned compiled function has the input
    /// structure of `self`, and each of its output leaves holds the partial derivative with respect to the
    /// corresponding input leaf.
    ///
    /// This function is a thin adapter over the [`ryft_core`] differentiation builder. It stages a compilation that
    /// applies [`vjp`](ryft_core::ReverseModeDifferentiate::vjp) to a call of this function, seeds and applies the
    /// resulting pullback, and compiles the result through the ordinary lifecycle. The original closure is therefore
    /// never traced again, the `jit_call` boundary is differentiated by its ordinary rules, and the captures of `self`
    /// enter the new trace through [`ImportingContext`] and become captures of the derived function.
    ///
    /// The derived function is retained on this wrapper and shared across its clones. Repeated calls with an
    /// interchangeable `session` return the retained compiled function without repeating differentiation, staging, or
    /// lowering, while a call with a different session identity produces a fresh derivative and replaces the retained
    /// one. The retained artifact is a compiled program whose residuals are runtime values, so reuse across calls with
    /// different runtime inputs is sound by construction. Retention is not single-flight. Concurrent cold calls may
    /// each derive a gradient, and because those derivations are interchangeable, the inserts are idempotent and the
    /// last one wins, while the shared [`CompilationCache`](ryft_core::compilation::CompilationCache) still
    /// deduplicates their backend compilation.
    #[track_caller]
    pub fn gradient(&self, session: &Session) -> Result<CompiledFunction<In, In>, Error>
    where
        In::Family: ParameterizedFamily<CompileTracer, To = In::To<CompileTracer>>,
        In::To<CompileTracer>: Parameterized<
                CompileTracer,
                Family = In::Family,
                ParameterStructure = In::ParameterStructure,
                To<ArrayType> = In,
                To<Constant> = In::To<Constant>,
            >,
        ProgramParameterValues<In, StatefulCompileTracer>:
            Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<In>, To<Constant> = ProgramConstants<In>>,
    {
        {
            let slot = self.derived.gradient.lock().expect("derived-gradient retention slot mutex poisoned");
            if let Some((witness, artifact)) = slot.as_ref()
                && witness.matches(session)
            {
                return Ok(CompiledFunction {
                    function: artifact.clone(),
                    derived: Arc::new(DerivedFunctionSlots::new()),
                });
            }
        }
        let staged = self.function.staged();
        let input_structure = staged.input_structure().clone();
        let input_signature = self.input_signature()?;
        let compiled = compile_with_flat_captures(
            move |_, _, primals| {
                let primals = primals.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let context = primals
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                let (output, pullback) = context.vjp(
                    move |inputs, ()| {
                        // The call imports the captures of this function into the trace, where they become captures of
                        // the derived function.
                        let context = inputs
                            .first()
                            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                            .context()
                            .clone();
                        let mut outputs = staged.call_flat_in_context(&context, inputs)?;
                        if outputs.len() != 1 {
                            return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() });
                        }
                        Ok(outputs.remove(0))
                    },
                    primals,
                    (),
                )?;
                let output = ValueProjection::<ArrayType>::into_projected(output).map_err(ProgramError::from)?;
                let seed = ProjectedContext::<_, ArrayType>::new(context).gradient_seed(&output, false)?.into_value();
                let gradients = pullback.apply_materialized(seed)?;
                let gradients = gradients
                    .into_iter()
                    .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                In::To::<CompileTracer>::from_parameters(input_structure, gradients)
                    .map_err(ProgramError::from)
                    .map_err(Into::into)
            },
            Vec::new(),
            input_signature,
            session,
            self.derived_options(),
        )?;
        *self.derived.gradient.lock().expect("derived-gradient retention slot mutex poisoned") =
            Some((SessionWitness::new(session), compiled.function.clone()));
        Ok(compiled)
    }
}

// Forward mode differentiation packaged as a function. Mirrors `jax.jvp(jax.jit(f))`. The derived trace calls the
// retained staged function through `call_flat_in_context`, which binds a `jit_call` boundary that carries the sharding
// overrides of the function (refer to `Session::call_operation`) and that forward mode differentiation rewrites through
// its ordinary JVP rule.
impl<In: Clone + Parameterized<ArrayType, To<ArrayType> = In>, Out: Parameterized<ArrayType>> CompiledFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Debug + Hash + PartialEq,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer>,
{
    /// Returns a new compiled function that computes the forward mode Jacobian-Vector Product (JVP) of `self`. Mirrors
    /// `jax.jvp(f, primals, tangents)` packaged into one compiled function. The returned function takes
    /// `(primals, tangents)` and returns `(primal_outputs, tangent_outputs)`.
    ///
    /// This function is a thin adapter over the [`ryft_core`] differentiation builder. It stages a compilation that
    /// applies [`jvp`](ryft_core::ForwardModeDifferentiate::jvp) to a call of this function and compiles the result
    /// through the ordinary lifecycle. The ordinary JVP rule of the `jit_call` operation therefore builds the tangent
    /// call boundary over the array IR program, and array tracers are projected only at the boundary of the returned
    /// function. The captures of `self` enter the new trace through [`ImportingContext`] and become captures of the
    /// derived function.
    ///
    /// The derived function is retained on this wrapper and shared across its clones. Repeated calls with an
    /// interchangeable `session` return the retained compiled function without repeating differentiation, staging, or
    /// lowering, while a call with a different session identity produces a fresh derivative and replaces the retained
    /// one. Retention is not single-flight. Concurrent cold calls may each derive a JVP, and because those derivations
    /// are interchangeable, the inserts are idempotent and the last one wins, while the shared
    /// [`CompilationCache`](ryft_core::compilation::CompilationCache) still deduplicates their backend compilation.
    #[track_caller]
    pub fn jvp(&self, session: &Session) -> Result<CompiledFunction<(In, In), (Out, Out)>, Error>
    where
        In::Family: ParameterizedFamily<CompileTracer, To = In::To<CompileTracer>>,
        In::To<CompileTracer>:
            Parameterized<CompileTracer, Family = In::Family, ParameterStructure = In::ParameterStructure>,
        Out::Family: ParameterizedFamily<CompileTracer, To = Out::To<CompileTracer>>,
        Out::To<CompileTracer>: Parameterized<
                CompileTracer,
                Family = Out::Family,
                ParameterStructure = Out::ParameterStructure,
                To<ArrayType> = Out,
                To<Constant> = Out::To<Constant>,
            >,
        ProgramParameterValues<Out, StatefulCompileTracer>: Parameterized<
                StatefulCompileTracer,
                To<Type> = ProgramParameters<Out>,
                To<Constant> = ProgramConstants<Out>,
            >,
    {
        {
            let slot = self.derived.jvp.lock().expect("derived-jvp retention slot mutex poisoned");
            if let Some((witness, artifact)) = slot.as_ref()
                && witness.matches(session)
            {
                return Ok(CompiledFunction {
                    function: artifact.clone(),
                    derived: Arc::new(DerivedFunctionSlots::new()),
                });
            }
        }
        let staged = self.function.staged();
        let input_signature = self.input_signature()?;
        let tangent_signature = input_signature.clone().try_map_parameters(|r#type| r#type.tangent())?;
        let compiled = compile_with_flat_captures(
            move |_, _, (primals, tangents)| {
                let output_structure = staged.output_structure().clone();
                let primals = primals.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let tangents = tangents.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let context = primals
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                let (primal_outputs, tangent_outputs) = context.jvp(
                    move |inputs, ()| {
                        // The call imports the captures of this function into the trace, where they become captures of
                        // the derived function.
                        let context = inputs
                            .first()
                            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                            .context()
                            .clone();
                        staged.call_flat_in_context(&context, inputs)
                    },
                    primals,
                    tangents,
                    (),
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
            Vec::new(),
            (input_signature, tangent_signature),
            session,
            self.derived_options(),
        )?;
        *self.derived.jvp.lock().expect("derived-jvp retention slot mutex poisoned") =
            Some((SessionWitness::new(session), compiled.function.clone()));
        Ok(compiled)
    }
}

// Batching packaged as a function. Mirrors `jax.jit(jax.vmap(f))` for a static batch extent. The derived trace calls
// the retained staged function through `call_in_context` under `batch`, which rewrites the resulting `jit_call`
// boundary, which carries the sharding overrides of the function (refer to `Session::call_operation`), through its
// ordinary batching rule.
impl<In: Parameterized<ArrayType, To<ArrayType> = In>, Out: Parameterized<ArrayType>> CompiledFunction<In, Out>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<BatchAxis>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>
        + ParameterizedFamily<CompileBatchTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<BatchAxis>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>
        + ParameterizedFamily<CompileBatchTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    /// Returns a new compiled function that runs `self` once over a whole batch of `batch_axis.extent()` items instead
    /// of once per item. Mirrors `jax.jit(jax.vmap(f))`. The returned function stages [`batch`](ryft_core::batch)
    /// around a call of this function and compiles the result, so the batched function is ordinary staged source that
    /// composes with further transforms and compilations, and this function's captures become its captures, which
    /// do not vary across the batch.
    ///
    /// The arguments follow [`batch`](ryft_core::batch). `input_batch_axes` and `output_batch_axes` are broadcast into
    /// the input and output structures of this function. A mapped input leaf gains a batch axis of the requested extent
    /// at its position, while a replicated input leaf keeps its type. `batch_axis` must carry an explicit static
    /// extent, which also drives batches whose inputs are all replicated. Its placement, if any, places the new axis of
    /// every mapped input and output leaf (e.g., `ShardingDimension::sharded(["data"])` splits the batch items across
    /// the `data` mesh axis), and the new axis is replicated across the mesh otherwise. Its axis name, if any, is
    /// visible to the function, so that collectives and [`AxisIndex`](ryft_core::AxisIndex) reads inside it can
    /// address the batch. Runtime batch extents are available by calling [`batch`](ryft_core::batch) directly inside a
    /// function that is staged with an explicit extent input, and [`batch_in_parent`](ryft_core::batch_in_parent)
    /// batches calls of this function over [`Array`] leaves without compiling a batched function.
    ///
    /// # Errors
    ///
    /// Returns an error when `batch_axis` has no extent, when this function has no inputs, when the batch axes do not
    /// fit the input or output structures, or when batching or compiling the function fails.
    #[track_caller]
    pub fn batch<InputBatchAxes: Parameterized<BatchAxis>, OutputBatchAxes: Parameterized<BatchAxis>>(
        &self,
        session: &Session,
        input_batch_axes: InputBatchAxes,
        output_batch_axes: OutputBatchAxes,
        batch_axis: BatchAxisSpecification<usize>,
    ) -> Result<CompiledFunction<In, Out>, Error> {
        let extent = *batch_axis.extent().ok_or_else(|| ProgramError::InvalidArgument {
            message: "compiled function batching requires an explicit static batch extent".to_string(),
        })?;
        let axis_name = batch_axis.name().map(str::to_string);
        let sharding = batch_axis.sharding().cloned();
        let staged = self.function.staged().clone();
        let input_structure = staged.input_structure().clone();
        let output_structure = staged.output_structure().clone();
        let input_types = self.input_signature()?;
        let input_batch_axes = input_batch_axes
            .broadcast_to_parameter_structure::<In::To<BatchAxis>>(input_structure.clone())
            .map_err(ProgramError::from)?
            .into_parameters()
            .collect::<Vec<_>>();
        let output_batch_axes = output_batch_axes
            .broadcast_to_parameter_structure::<Out::To<BatchAxis>>(output_structure.clone())
            .map_err(ProgramError::from)?
            .into_parameters()
            .collect::<Vec<_>>();

        // Insert the batch axis into the type of every mapped input leaf with the requested placement, normalizing it
        // against the batched rank as `batch` does for its inputs.
        let batch_sharding = sharding.clone().unwrap_or_else(ShardingDimension::replicated);
        let batched_input_types = input_types
            .into_parameters()
            .zip(&input_batch_axes)
            .map(|(r#type, batch_axis)| match batch_axis.axis() {
                None => Ok(r#type),
                Some(axis) => {
                    let position = axis.normalize(r#type.rank() + 1).map_err(ProgramError::from)?;
                    Ok(r#type.batched(position, Dimension::Static(extent), batch_sharding.clone())?)
                }
            })
            .collect::<Result<Vec<_>, BatchingError>>()
            .map_err(ProgramError::from)?;
        let batched_input_types =
            In::from_parameters(input_structure.clone(), batched_input_types).map_err(ProgramError::from)?;

        compile_with_flat_captures::<_, In, Out>(
            move |_, _, inputs| {
                let inputs = inputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
                let context =
                    inputs.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context().clone();

                // An explicit first-class extent also defines the batch when every input is replicated.
                let extent = context.lift(Constant::Dimension(DimensionValue::constant(extent)?))?;
                let batch_axis = match axis_name {
                    Some(axis_name) => BatchAxisSpecification::named(axis_name).with_extent(extent),
                    None => BatchAxisSpecification::default().with_extent(extent),
                };
                let batch_axis = match sharding {
                    Some(sharding) => batch_axis.with_sharding(sharding),
                    None => batch_axis,
                };
                let outputs = context
                    .batch(
                        |inputs: Vec<CompileBatchTracer>| {
                            let context = inputs
                                .first()
                                .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                                .context()
                                .clone();
                            let inputs = ProgramParameterValues::<In, CompileBatchTracer>::from_parameters(
                                input_structure,
                                inputs,
                            )?;
                            let outputs = staged.call_in_context::<CompileBatchTracer, _, _>(&context, inputs)?;
                            Ok(outputs.into_parameters().collect::<Vec<CompileBatchTracer>>())
                        },
                        inputs,
                        input_batch_axes,
                        output_batch_axes,
                        batch_axis,
                    )
                    .map_err(ProgramError::from)?;
                let outputs = outputs
                    .into_iter()
                    .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Out::To::<CompileTracer>::from_parameters(output_structure, outputs).map_err(ProgramError::from)?)
            },
            Vec::new(),
            batched_input_types,
            session,
            self.derived_options(),
        )
    }
}

/// Compiles one explicit abstract signature of `function` and returns a [`CompiledFunction`] that executes that
/// specialization on subsequent calls. Use [`jit`] when runtime calls should select and retain specializations in
/// the style of `jax.jit`.
///
/// Equivalent to [`compile_with_options`] called with [`Options::new(mesh)`](Options::new).
///
/// The function is traced into a fresh XLA program, in which compiled functions that it calls register their runtime
/// captures as captures of the new program (refer to [`StagedFunction::call`]). The resulting program is then
/// compiled and executed against `session`. Its [`CompilationCache`](ryft_core::compilation::CompilationCache)
/// structurally shares equivalent lowerings even when they originate at different Rust call sites.
#[track_caller]
pub fn compile<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    session: &Session,
    mesh: DeviceMesh,
) -> Result<CompiledFunction<In, Out>, Error>
where
    F: FnOnce(In::To<CompileTracer>) -> Out::To<CompileTracer>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    compile_with_options(function, input_types, session, Options::new(mesh))
}

/// Same as [`compile`] but makes runtime arrays explicit captures of the compiled program.
///
/// The closure receives capture tracers first and ordinary input tracers second. Captures are compiled as hidden
/// executable arguments and are supplied from the returned [`CompiledFunction`] at execution time, so callers of the
/// compiled function still pass only `In` inputs.
#[track_caller]
pub fn compile_with_captures<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array>,
    input_types: In,
    session: &Session,
    mesh: DeviceMesh,
) -> Result<CompiledFunction<In, Out>, Error>
where
    F: FnOnce(Vec<CompileTracer>, In::To<CompileTracer>) -> Out::To<CompileTracer>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    compile_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        session,
        Options::new(mesh),
    )
}

/// Same as [`compile`] but accepts an explicit [`Options`] payload.
#[track_caller]
pub fn compile_with_options<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    session: &Session,
    options: Options,
) -> Result<CompiledFunction<In, Out>, Error>
where
    F: FnOnce(In::To<CompileTracer>) -> Out::To<CompileTracer>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    compile_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, session, options)
}

/// Traces `function` into a staged program and returns a [`StagedFunction`] handle, without compiling a PJRT
/// executable.
///
/// This is the counterpart of [`compile`] that only stages. Use it for functions that are composed into larger programs
/// or transforms through [`StagedFunction::call`] rather than executed directly, so that no executable is built for
/// them. Compile the staged handle later with [`Session::compile_staged_function`] when direct execution is needed.
/// `options` are applied to the abstract input signature before tracing and retained for subsequent lowering and
/// compilation.
#[track_caller]
pub fn stage<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    options: Options,
) -> Result<StagedFunction<In, Out>, Error>
where
    F: FnOnce(In::To<CompileTracer>) -> Out::To<CompileTracer>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    stage_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, options)
}

/// Same as [`stage`] but makes runtime arrays explicit captures of the staged program.
///
/// The closure receives capture tracers first and ordinary input tracers second, mirroring
/// [`compile_with_captures`]. Captures are retained on the staged handle, and every [`StagedFunction::call`] imports
/// them into the calling context as leading inputs of its `jit_call` boundary.
#[track_caller]
pub fn stage_with_captures<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array>,
    input_types: In,
    options: Options,
) -> Result<StagedFunction<In, Out>, Error>
where
    F: FnOnce(Vec<CompileTracer>, In::To<CompileTracer>) -> Out::To<CompileTracer>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    stage_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        options,
    )
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without lowering or compiling.
/// Mirrors `jax.eval_shape`.
#[track_caller]
pub fn infer_output_types<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
) -> Result<Out, Error>
where
    F: FnOnce(In::To<CompileTracer>) -> Out::To<CompileTracer>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out>,
{
    let input_structure = input_types.parameter_structure();
    let input_types =
        ProgramParameters::<In>::from_parameters(input_structure, input_types.into_parameters().map(Type::from))?;
    let output_types = DomainTracingContext::<Session, StatefulValue>::infer_output_type(
        |tracers| {
            let input_structure = tracers.parameter_structure();
            let input_parameters = tracers
                .into_parameters()
                .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let tracers = In::To::<CompileTracer>::from_parameters(input_structure, input_parameters)?;
            let outputs = function(tracers);
            let output_structure = outputs.parameter_structure();
            ProgramParameterValues::<Out, StatefulCompileTracer>::from_parameters(
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
    Out::from_parameters(output_structure, output_parameters).map_err(Error::from)
}

/// Internal reparameterization of one public array parameter tree into the production array IR type family.
type ProgramParameters<P> = <P as Parameterized<ArrayType>>::To<Type>;

/// Internal value tree corresponding to one public array parameter tree.
type ProgramParameterValues<P, V> = <ProgramParameters<P> as Parameterized<Type>>::To<V>;

/// Tree of captured constants corresponding to one public array parameter tree.
type ProgramConstants<P> = ProgramParameterValues<P, Constant>;

/// Retention slots for the derived functions of one [`CompiledFunction`] (i.e., the reverse mode gradient and the
/// forward mode JVP compiled around a call of its retained source program).
type DerivedFunctions<In, Out> = DerivedFunctionSlots<
    CoreCompiledFunction<Session, ProgramParameters<In>, ProgramParameters<In>>,
    CoreCompiledFunction<Session, ProgramParameters<(In, In)>, ProgramParameters<(Out, Out)>>,
>;

/// Retention slots, each holding one entry, for functions derived from a [`CompiledFunction`].
///
/// [`CompiledFunction::gradient`] and [`CompiledFunction::jvp`] take no transform configuration per call, so each
/// derived function has exactly one valid form per session identity. The slot retains the most recently produced
/// artifact together with a witness of the session that produced it. A later call with a matching session reuses the
/// retained artifact and skips differentiation, staging, and lowering entirely, while a call with a different session
/// identity produces a fresh artifact and replaces the slot. Failed productions are never stored, so they retry
/// naturally.
///
/// A slot is a retention slot and not a single-flight barrier. Its lock is released before the artifact is derived, so
/// concurrent cold calls may each derive their own copy and the last insert wins. That is sound because the copies are
/// interchangeable derivations of the same source program, and it is deliberate because waiting single-flight belongs
/// exclusively to the shared [`CompilationCache`](ryft_core::compilation::CompilationCache), which still
/// deduplicates the backend compilation those duplicate derivations request.
struct DerivedFunctionSlots<Gradient, Jvp> {
    /// Retained reverse mode gradient function, if any.
    gradient: Mutex<Option<(SessionWitness, Gradient)>>,

    /// Retained forward mode JVP function, if any.
    jvp: Mutex<Option<(SessionWitness, Jvp)>>,
}

impl<Gradient, Jvp> DerivedFunctionSlots<Gradient, Jvp> {
    /// Creates empty retention slots.
    fn new() -> Self {
        Self { gradient: Mutex::new(None), jvp: Mutex::new(None) }
    }
}

/// Identity of the [`Session`] that produced a retained derived function.
///
/// [`Session`] handles are cheap clones of shared state, so two handles denote interchangeable compilation
/// pipelines exactly when they are handles of the same session, and therefore share one client, one compilation
/// context, and one template of compilation options, and have an equal mesh. The witness stores a full session clone,
/// which both provides those comparisons and keeps the shared state alive so its address can never be reused by a
/// later allocation.
struct SessionWitness {
    /// Session handle captured when the retained artifact was produced.
    session: Session,
}

impl SessionWitness {
    /// Captures the identity of `session`.
    fn new(session: &Session) -> Self {
        Self { session: session.clone() }
    }

    /// Returns whether `session` is interchangeable with the session that produced the retained artifact.
    fn matches(&self, session: &Session) -> bool {
        self.session.is_same_session(session) && self.session.device_mesh().ok() == session.device_mesh().ok()
    }
}

/// Batched tracer of one array IR batching level directly inside an XLA compilation trace.
type CompileBatchTracer = BatchingTracer<DomainTracingContext<Session, Value>, ArrayIrBatchingPolicy>;

/// Projects one tree of array IR values of any context (e.g., runtime [`Value`]s or [`StatefulCompileTracer`]s) to the
/// tree of their `ArrayType` projections.
fn project_values<P, V>(
    values: ProgramParameterValues<P, V>,
) -> Result<P::To<<V as ValueProjection<ArrayType>>::Projected>, ProgramError>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<V>
        + ParameterizedFamily<<V as ValueProjection<ArrayType>>::Projected>,
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: Parameter>,
{
    let structure = values.parameter_structure();
    let parameters = values
        .into_parameters()
        .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(P::To::<<V as ValueProjection<ArrayType>>::Projected>::from_parameters(structure, parameters)?)
}

/// Lifts one tree of `ArrayType` projections (e.g., public [`Array`]s or [`CompileTracer`]s) back into the tree of the
/// array IR values that they project.
fn lift_values<P, V>(
    values: P::To<<V as ValueProjection<ArrayType>>::Projected>,
) -> Result<ProgramParameterValues<P, V>, ProgramError>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<V>
        + ParameterizedFamily<<V as ValueProjection<ArrayType>>::Projected>,
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: Parameter>,
{
    let structure = values.parameter_structure();
    let parameters = values.into_parameters().map(<V as ValueProjection<ArrayType>>::from_projected);
    Ok(ProgramParameterValues::<P, V>::from_parameters(structure, parameters)?)
}

/// Stages the transition of `value` to the placement `sharding` that a sharding override of a compiled function
/// boundary requests. A `reshard` brings the placement that types record (i.e., the placement over explicit mesh axes)
/// to that of `sharding`, and a `constrain_sharding` then constrains the placement over the auto mesh axes that
/// `sharding` names, which types never record. Afterwards, the type of `value` carries `sharding` without its auto
/// mesh axes, which is also the type that the corresponding input or output of the compiled function has.
fn place<V>(value: V, sharding: &Sharding) -> Result<V, ProgramError>
where
    V: CoreValue<Type = Type> + Reshard + ConstrainSharding,
{
    let current = <&ArrayType>::try_from(value.r#type().as_ref())?.sharding().cloned();
    let tracked = sharding.without_auto_axes();
    let value = if current.as_ref() == Some(&tracked) { value } else { value.reshard(&tracked)? };
    if !sharding.references_auto_axis() {
        return Ok(value);
    }

    // The constraint names only the auto mesh axes of `sharding`, because lowering merges it with the explicit
    // placement that the type now tracks (refer to `ConstrainShardingOperation::lowered_sharding`).
    let is_auto = |axis: &&String| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Auto);
    let dimensions = sharding
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            ShardingDimension::Sharded(axes) => {
                let axes = axes.iter().filter(is_auto).cloned().collect::<Vec<_>>();
                if axes.is_empty() { ShardingDimension::Replicated } else { ShardingDimension::Sharded(axes) }
            }
            dimension => dimension.clone(),
        })
        .collect();
    let constraint = Sharding::new(sharding.mesh().clone(), dimensions)
        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
    value.constrain_sharding(&constraint)
}

/// Stages the output sharding overrides `shardings` (i.e., [`Options::output_shardings`]) on the outputs of a traced
/// function (refer to `place`), so that the staged function, and every call of it, has the output types that its
/// executions return. Outputs that are not arrays are left for lowering to reject.
fn place_outputs<V, P>(outputs: P, shardings: Option<&[Sharding]>) -> Result<P, Error>
where
    V: CoreValue<Type = Type> + Reshard + ConstrainSharding,
    P: Parameterized<V>,
{
    let Some(shardings) = shardings else {
        return Ok(outputs);
    };
    let structure = outputs.parameter_structure();
    let outputs = outputs.into_parameters().collect::<Vec<_>>();
    if shardings.len() != outputs.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "output",
            expected: outputs.len(),
            actual: shardings.len(),
        }));
    }
    let outputs = outputs
        .into_iter()
        .zip(shardings)
        .map(|(output, sharding)| match output.r#type().as_ref() {
            Type::Array(_) => place(output, sharding),
            Type::Dimension(_) | Type::Reference(_) => Ok(output),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(P::from_parameters(structure, outputs).map_err(ProgramError::from)?)
}

/// Brings each of the flat array `inputs` of a call to the placement of the corresponding input of the called function
/// (refer to `place`). That placement is the sharding override in `overrides`, if any, which may also place the input
/// over auto mesh axes, and otherwise the sharding of the callee input type if the input type differs from it only in
/// its sharding. The call thereby constrains its inputs to the placement of the called function in the context of the
/// caller, which composes with transforms, much like an execution reshards concrete inputs. A callee input type without
/// a sharding requests no placement, so an input that differs from it only in its sharding stays where it is, and the
/// call accepts it as an execution of the callee does (refer to `validate_call_input_types`). Other inputs are returned
/// unchanged, so that binding the call reports any other mismatch.
fn place_inputs<V>(inputs: Vec<V>, input_types: &[Type], overrides: Option<&[Sharding]>) -> Result<Vec<V>, ProgramError>
where
    V: CoreValue<Type = Type> + Reshard + ConstrainSharding,
{
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            if let Some(sharding) = overrides.and_then(|overrides| overrides.get(index))
                && matches!(input.r#type().as_ref(), Type::Array(_))
            {
                return place(input, sharding);
            }
            let sharding = match (input.r#type().as_ref(), input_types.get(index)) {
                (Type::Array(actual), Some(Type::Array(expected))) if actual != expected => expected
                    .sharding()
                    .filter(|_| {
                        actual
                            .clone()
                            .with_sharding(expected.sharding().cloned())
                            .is_ok_and(|actual| &actual == expected)
                    })
                    .cloned(),
                _ => None,
            };
            match sharding {
                Some(sharding) => place(input, &sharding),
                None => Ok(input),
            }
        })
        .collect()
}

/// Calls `function` as a `jit_call` operation in `context` on behalf of [`StagedFunction::call_in_context`] and
/// [`CompiledFunction::call_in_context`]. Inputs whose type differs from the input type of `function` only in its
/// sharding are first brought to its placement in `context` (refer to `place_inputs`), and the call carries the
/// boundary of `function` (refer to `Session::call_operation`).
fn call_staged_in_context<In, Out, C, Path>(
    function: &CoreStagedFunction<Session, ProgramParameters<In>, ProgramParameters<Out>>,
    context: &C,
    inputs: In::To<<C::Value as ValueProjection<ArrayType>>::Projected>,
) -> Result<Out::To<<C::Value as ValueProjection<ArrayType>>::Projected>, ProgramError>
where
    In: Parameterized<ArrayType>,
    In::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<C::Value>
        + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<C::Value>
        + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    C: Context<Type = Type, Constant = Constant, Operation = Operation> + ImportingContext<Session, Path>,
    C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
{
    let input_structure = inputs.parameter_structure();
    let inputs = inputs.into_parameters().map(<C::Value as ValueProjection<ArrayType>>::from_projected).collect();
    let inputs = ProgramParameterValues::<In, C::Value>::from_parameters(
        input_structure,
        place_inputs(inputs, function.input_types(), function.options().input_shardings())?,
    )?;
    project_values::<Out, C::Value>(function.call_in_context(context, inputs)?)
}

/// Stages `function` through [`stage_with_flat_captures`] and compiles the result against `session`. This is the shared
/// implementation of [`compile`], [`compile_with_captures`], [`compile_with_options`], and the transforms of
/// [`CompiledFunction`].
///
/// # Parameters
///
///   - `function`: Fallible closure that receives the symbolic capture constants, the capture tracers, and the
///     structured input tracers.
///   - `captures`: Runtime arrays to register in the capture table of the source program.
///   - `input_types`: Structured abstract input signature, to which the input sharding overrides of `options` apply.
///   - `session`: Domain that lowers and compiles the staged program.
///   - `options`: Options retained for lowering and compilation.
#[track_caller]
fn compile_with_flat_captures<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array>,
    input_types: In,
    session: &Session,
    options: Options,
) -> Result<CompiledFunction<In, Out>, Error>
where
    F: FnOnce(Vec<Constant>, Vec<CompileTracer>, In::To<CompileTracer>) -> Result<Out::To<CompileTracer>, Error>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    let staged = stage_with_flat_captures::<F, In, Out>(function, captures, input_types, options)?;
    session.compile_staged_function(staged)
}

/// Traces `function` into a [`StagedFunction`] over the internal array IR type family. The closure sees public
/// [`CompileTracer`]s, and its outputs receive the output sharding overrides of `options` (refer to `place_outputs`).
/// This is the shared implementation of [`stage`], [`stage_with_captures`], and [`compile_with_flat_captures`].
///
/// # Parameters
///
///   - `function`: Fallible closure that receives the symbolic capture constants, the capture tracers, and the
///     structured input tracers.
///   - `captures`: Runtime arrays to register in the capture table of the source program.
///   - `input_types`: Structured abstract input signature, to which the input sharding overrides of `options` apply.
///   - `options`: Options retained for lowering and compilation.
#[track_caller]
fn stage_with_flat_captures<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array>,
    input_types: In,
    options: Options,
) -> Result<StagedFunction<In, Out>, Error>
where
    F: FnOnce(Vec<Constant>, Vec<CompileTracer>, In::To<CompileTracer>) -> Result<Out::To<CompileTracer>, Error>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Type>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer>
        + ParameterizedFamily<StatefulCompileTracer>,
    Out::To<CompileTracer>: Parameterized<CompileTracer, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer>:
        Parameterized<StatefulCompileTracer, To<Type> = ProgramParameters<Out>, To<Constant> = ProgramConstants<Out>>,
{
    let captures = captures.into_iter().map(Value::Array).collect();
    let input_structure = input_types.parameter_structure();
    let input_types =
        ProgramParameters::<In>::from_parameters(input_structure, input_types.into_parameters().map(Type::from))
            .map_err(ProgramError::from)?;
    let output_shardings = options.output_shardings().map(<[Sharding]>::to_vec);
    let function = move |capture_references,
                         capture_tracers: Vec<StatefulCompileTracer>,
                         inputs: ProgramParameterValues<In, StatefulCompileTracer>| {
        let capture_tracers = capture_tracers
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let inputs = project_values::<In, StatefulCompileTracer>(inputs)?;
        let outputs =
            lift_values::<Out, StatefulCompileTracer>(function(capture_references, capture_tracers, inputs)?)?;
        place_outputs(outputs, output_shardings.as_deref())
    };
    let input_types = with_input_shardings(input_types, &options)?;
    let function = CoreStagedFunction::<Session, ProgramParameters<In>, ProgramParameters<Out>>::stage(
        function,
        captures,
        input_types,
        options,
    )?;
    Ok(StagedFunction { function })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, AddOperation, Array as CoreArray, ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex,
        ArraySliceAxis, ArrayType, Atan2, AttachedRegions, AxisIndex, BatchAxis, BatchAxisSpecification, Broadcast,
        CaptureReference, Compare, ComparisonDirection, Condition, Context, ConvertElementType, Cos,
        CotangentAccumulator, CotangentDestinationKind, DataType, Device, DifferentiableType, Differentiate,
        DifferentiationError, Dimension, DimensionBounds, DimensionSize, DimensionSizeOperation, DimensionVariable,
        DomainTracer, Dot, DotDimensionNumbers, DotsSavable, EagerInterpretationDriver, ErasedOperation,
        Error as CoreError, ForwardModeDifferentiate, Hessian, ImportingContext, InterpretableOperation, Jacobian,
        LogicalMesh, MaybeZero, Memory, MeshAxis, MeshAxisType, Mul, Neg, NothingSavable, Parameter,
        PartialEvaluationOutput, PartitionedProgram, Place, Placeholder, Placement, Program, ProgramBuilder,
        ProgramError, ProjectedContext, Reduce, ReductionKind, ReferenceAddUpdate, ReferenceAddUpdateOperation,
        ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceType, ReshapeOperation, ResidualCandidate, ResidualDecision,
        ResidualInputSource, ResidualPolicy, ResidualPolicyError, ResidualPolicyReference, ResidualRejection,
        ResidualStorage, SaveAndOffloadOnlyTheseNames, ScanOperation, ScanOptions, Shape, ShardMapOptions, Sharding,
        ShardingDimension, Sin, Slice, Softmax, StagedFunction as CoreStagedFunction, StopGradient, Tag, TagOperation, Trace, Tracer,
        TracingContext, TranspositionContext, Typed, TypedProgram, Value as CoreValue, ValueProjection, WhileOptions,
        batch, batch_in_parent, composite, differentiate_at, differentiate_in_parent_at, rematerialize, scan,
        shard_map, softmax_cross_entropy_with_target_logits, r#while,
    };
    use ryft_macros::Parameterized;

    use crate::compilation::CompilationError;
    use crate::compilation::options::Options;
    use crate::compilation::profile_guided::AdaptiveProfileGuidedOptions;
    use crate::sessions::tests::{
        domain_mesh, read_first_shard, read_sharded_f32_array, replicated_scalar_type, replicated_vector_type,
        two_device_mesh,
    };
    use crate::tests::{
        assert_send_sync, cpu_client, execution_client, other_cpu_client, values_from_bytes, values_to_bytes,
    };
    use crate::{
        Array, ArrayShard, Constant, Error, ExecutionError, FromPjrt, JitCallOperation, Operation,
        RaggedDotLoweringStrategy, Reference, Session, ThroughSession, Type, Value,
    };

    use super::*;

    /// Traces `f(reference, update) = { reference += update; read(reference) }` over a `ref<f32[]>` and an `f32[]` into
    /// the XLA program universe, so that program transforms of a program with references can be replayed into a
    /// stateful compilation.
    fn accumulate_and_read_program(
        reference_type: &Type,
        array_type: &ArrayType,
    ) -> Program<Constant, Operation, Vec<Constant>, Vec<Constant>> {
        let (_, program) = <Session>::trace(
            |(reference, update): (DomainTracer<Session>, DomainTracer<Session>)| {
                reference.add_update(&update)?;
                reference.read()
            },
            (reference_type.clone(), Type::Array(array_type.clone())),
        )
        .unwrap();
        program.into_flat_program()
    }

    #[test]
    fn test_compiled_function_dispatcher_execute() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let function: CompiledFunctionDispatcher<_, bool, ArrayType, ArrayType> = jit(
            |apply_sine, input: CompileTracer| Ok(if apply_sine { input.sin()? } else { input }),
            &session,
            mesh.clone(),
        );

        let input_type = replicated_vector_type(&mesh, 1);
        let source =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes::<f32>(&[0.5]).as_slice())
                .unwrap();

        let sine = function.execute(true, source.clone()).unwrap();
        let warm_sine = function.execute(true, source.clone()).unwrap();
        let identity = function.execute(false, source).unwrap();
        assert!((read_first_shard::<f32>(client, &sine)[0] - 0.5f32.sin()).abs() < 1e-5);
        assert!((read_first_shard::<f32>(client, &warm_sine)[0] - 0.5f32.sin()).abs() < 1e-5);
        assert!((read_first_shard::<f32>(client, &identity)[0] - 0.5).abs() < 1e-5);

        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_specializes_on_input_placement_over_auto_axes() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let bytes = values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5]);
        let sharded_input =
            Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), Some(sharded.clone()), bytes.as_slice())
                .unwrap();
        let replicated_input =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), Some(replicated.clone()), bytes.as_slice())
                .unwrap();
        assert_eq!(sharded_input.r#type(), replicated_input.r#type());

        // Inputs whose types agree but whose placements over auto mesh axes differ get their own executables, which
        // keep each placement instead of resharding to the other, and the compiler chooses the output placements.
        let dispatcher: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit_with_options(|(), x: CompileTracer| Ok::<_, Error>(x.sin()?), &engine, Options::new(mesh.clone()));
        let sharded_output = dispatcher.execute((), sharded_input.clone()).unwrap();
        let replicated_output = dispatcher.execute((), replicated_input).unwrap();
        let _ = dispatcher.execute((), sharded_input).unwrap();
        let statistics = dispatcher.statistics();
        assert_eq!((statistics.dispatch_misses, statistics.dispatch_hits), (2, 1));
        assert_eq!(sharded_output.sharding(), &sharded);
        assert_eq!(replicated_output.sharding(), &replicated);
        assert_eq!(sharded_output.r#type(), replicated_output.r#type());
    }

    #[test]
    fn test_compiled_function_dispatcher_call() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 2);
        let function: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit(|(), x: CompileTracer| Ok::<_, Error>(x.sin()?), &engine, mesh.clone());

        // A call traces the closure into the calling trace, and neither lowers, compiles, nor executes it.
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| function.call((), x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (0, 1, 1));
        assert_eq!((statistics.dispatch_misses, statistics.lowerings, statistics.compilation_requests), (0, 0, 0));
        let input = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.5, -1.0]).as_slice(),
        )
        .unwrap();
        let observed = read_first_shard::<f32>(client, &outer.executable().execute(&engine, input).unwrap());
        assert!(observed.iter().zip([0.5f32, -1.0]).all(|(observed, x)| (observed - x.sin().cos()).abs() < 1e-6));

        // Another call with the same types and scope reuses the staged specialization.
        compile::<_, ArrayType, ArrayType>(|x| function.call((), x).unwrap(), input_type, &engine, mesh).unwrap();
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (1, 1, 1));
    }

    #[test]
    fn test_compiled_function_dispatcher_call_reads_axis_index_inside_jvp() {
        // The index of an eager batch axis lives on the mesh with which the session places values that name no mesh
        // (i.e., the canonical mesh of one device), so the batched values live there too and combine with it.
        let client = execution_client();
        let mesh = domain_mesh(client, "device", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let xs = Array::from_host_buffer(
            &engine,
            vector_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0]).as_slice(),
        )
        .unwrap();

        // `x ↦ x + i` reads the index `i` of the enclosing batch axis from inside a `jvp` trace. Calling it as a
        // compiled function, whose staged specialization is traced with the named axes of the caller, matches calling
        // it inline.
        macro_rules! plus_axis_index {
            ($x:expr) => {{
                let x = $x;
                let index = x.domain().parent().axis_index("i")?;
                let index = ValueProjection::<ArrayType>::into_projected(index)
                    .map_err(ProgramError::from)?
                    .convert_element_type(DataType::F32)?;
                x.add(&index)?
            }};
        }
        let dispatcher: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit(|(), x: CompileTracer| Ok::<_, Error>(plus_axis_index!(x)), &engine, mesh.clone());
        let inline = batch_in_parent(
            |x| Ok(differentiate_in_parent_at(x.clone()).jvp(x, |x| Ok(plus_axis_index!(x)))?),
            xs.clone(),
            BatchAxis::new(0),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxisSpecification::named("i"),
        )
        .unwrap();
        let compiled = batch_in_parent(
            |x| {
                Ok(differentiate_in_parent_at(x.clone()).jvp(x, |x| {
                    dispatcher.call((), x).map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
                })?)
            },
            xs,
            BatchAxis::new(0),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxisSpecification::named("i"),
        )
        .unwrap();
        for (output, tangent) in [inline, compiled] {
            assert_eq!(read_first_shard::<f32>(client, &output), vec![10.0, 21.0, 32.0]);
            assert_eq!(read_first_shard::<f32>(client, &tangent), vec![10.0, 20.0, 30.0]);
        }
    }

    #[test]
    fn test_compiled_function_dispatcher_call_with_named_axes() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();

        // The closure reads the index along the named batch axis `i` of the level that calls it. The call traces
        // it in that level's scope, and batching the resulting call resolves the axis once per batch item.
        let function: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> = jit(
            |(), x: CompileTracer| {
                let index = x.into_value().context().axis_index("i")?;
                Ok::<_, Error>(
                    ValueProjection::<ArrayType>::into_projected(index)
                        .map_err(ProgramError::from)?
                        .convert_element_type(DataType::F32)?,
                )
            },
            &engine,
            mesh.clone(),
        );
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            |x| {
                let output = batch(
                    |x| Ok(function.call((), ValueProjection::<ArrayType>::into_projected(x)?).unwrap().into_value()),
                    x.into_value(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            matrix_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &engine,
            matrix_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0; 6]).as_slice(),
        )
        .unwrap();
        assert_eq!(
            read_first_shard::<f32>(client, &outer.executable().execute(&engine, input).unwrap()),
            vec![0.0, 1.0, 2.0]
        );

        // With only replicated inputs and an explicit extent, the call must still be rebatched at the named level,
        // because its callee reads that level by name. Binding the callee unchanged would let `axis_index` escape it.
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            |x| {
                let x = x.into_value();
                let extent = x.dimension_size(0).unwrap();
                let output = batch(
                    |x| Ok(function.call((), ValueProjection::<ArrayType>::into_projected(x)?).unwrap().into_value()),
                    x,
                    BatchAxis::replicated(),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("i").with_extent(extent),
                )
                .unwrap();
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            matrix_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let input =
            Array::from_host_buffer(&engine, matrix_type, mesh, None, values_to_bytes::<f32>(&[0.0; 6]).as_slice())
                .unwrap();
        assert_eq!(
            read_first_shard::<f32>(client, &outer.executable().execute(&engine, input).unwrap()),
            vec![0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_compiled_function_dispatcher_call_keeps_replicated_output_override_inside_compile() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values =
            |array: &Array| array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();

        // The closure splits its value along `x` but requests a replicated output, which stages nothing in its body.
        // Executed directly or called inside a compiled function, it returns two replicas.
        let vector_type = ArrayType::new_static(DataType::I32, [4]);
        let vector = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let placing: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> = jit_with_options(
            move |(), x: CompileTracer| Ok::<_, Error>(x.place(&target, None)?),
            &engine,
            Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()]),
        );
        let wrapped: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| placing.call((), x).unwrap(), vector_type.clone(), &engine, mesh.clone()).unwrap();
        for output in
            [placing.execute((), vector.clone()).unwrap(), wrapped.executable().execute(&engine, vector).unwrap()]
        {
            output.block_until_ready().unwrap();
            assert_eq!(output.sharding(), &replicated);
            assert!(output.is_committed());
            assert_eq!(shapes(&output), vec!["[4]", "[4]"]);
            assert_eq!(values(&output), vec![values_to_bytes(&[1i32, 2, 3, 4]), values_to_bytes(&[1i32, 2, 3, 4])]);
        }
    }

    #[test]
    fn test_stateful_compiled_function_dispatcher_execute_statefully_and_wait() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let function = jit_statefully::<_, (), (Type, Type), Type>(
            |(), (reference, update): (StatefulCompileTracer, StatefulCompileTracer)| {
                reference.add_update(&update)?;
                Ok(reference.read()?)
            },
            &session,
            mesh.clone(),
        );
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );

        for expected in [3.0f32, 5.0] {
            let update = Array::from_host_buffer(
                &session,
                array_type.clone(),
                mesh.clone(),
                None,
                2.0f32.to_ne_bytes().as_slice(),
            )
            .unwrap();
            let output = function
                .execute_statefully_and_wait(
                    (),
                    (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
                )
                .unwrap();
            let Value::Array(output) = output else { panic!("stateful public output must be an array") };
            assert_eq!(read_first_shard::<f32>(client, &output), vec![expected]);
        }
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
    }

    #[test]
    fn test_stateful_compiled_function_dispatcher_execute_statefully() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let function = jit_statefully_with_options::<_, (), (Type, Type), Type>(
            |(), (reference, update): (StatefulCompileTracer, StatefulCompileTracer)| {
                reference.add_update(&update)?;
                Ok(reference.read()?)
            },
            &session,
            Options::new(mesh.clone()),
        );
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        let output = function
            .execute_statefully(
                (),
                (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
            )
            .r#await()
            .unwrap();
        let Value::Array(output) = output else { panic!("stateful public output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &output), vec![3.0]);
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_stateful_compiled_function_output_types() {
        // The public output types exclude the hidden final state output that the mutated reference appends.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh),
        )
        .unwrap();
        assert_eq!(compiled.output_types().as_ref(), &[Type::Array(array_type.clone())]);
        assert_eq!(compiled.executable().output_types().as_ref(), &[Type::Array(array_type)]);
    }

    #[test]
    fn test_stateful_compiled_function_execute_statefully_and_wait() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let initial =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap();
        assert_eq!(initial.r#type().as_ref(), &replicated_scalar_type(&mesh, DataType::F32));
        let reference = ArrayReference::new(initial);
        let retained_snapshot = reference.read().unwrap();
        let rejected_update =
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        assert!(matches!(
            compiled.executable().execute(
                &session,
                (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(rejected_update),)
            ),
            Err(Error::Execution(ExecutionError::StatefulInvocationRequired)),
        ));
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![1.0]);

        for expected in [3.0f32, 5.0] {
            let update = Array::from_host_buffer(
                &session,
                array_type.clone(),
                mesh.clone(),
                None,
                2.0f32.to_ne_bytes().as_slice(),
            )
            .unwrap();
            let output = compiled
                .execute_statefully_and_wait(
                    &session,
                    (Value::Reference(Reference::new(reference.clone(), session.clone())), Value::Array(update)),
                )
                .unwrap();
            let Value::Array(output) = output else { panic!("stateful public output must be an array") };
            assert_eq!(read_first_shard::<f32>(client, &output), vec![expected]);
            assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![expected]);
        }
        assert_eq!(read_first_shard::<f32>(client, &retained_snapshot), vec![1.0]);
    }

    #[test]
    fn test_stateful_compiled_function_execute_statefully_and_wait_commits_tangent_references() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));

        // The program transform derives `[r, x, ṫr, ẋ] ↦ [y, ẏ]`, and replaying it into the stateful trace makes
        // both the reference and its tangent reference public holders of the compiled function.
        let jvp = accumulate_and_read_program(&reference_type, &array_type).jvp().unwrap();
        assert_eq!(
            jvp.input_types(),
            vec![
                reference_type.clone(),
                Type::Array(array_type.clone()),
                reference_type.clone(),
                Type::Array(array_type.clone()),
            ],
        );
        let compiled = compile_statefully::<_, (Type, Type, Type, Type), (Type, Type)>(
            |(reference, update, tangent_reference, tangent_update)| {
                let context = reference.context().clone();
                let mut outputs =
                    jvp.interpret_in_context(&context, vec![reference, update, tangent_reference, tangent_update])?;
                let tangent = outputs.remove(1);
                Ok((outputs.remove(0), tangent))
            },
            (reference_type.clone(), Type::Array(array_type.clone()), reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();

        // Both references are committed after the call. The primal holder accumulates the update, the tangent holder
        // accumulates the tangent update, and the outputs read the committed states.
        let scalar = |value: f32| {
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, value.to_ne_bytes().as_slice())
                .unwrap()
        };
        let reference = ArrayReference::new(scalar(1.0));
        let tangent_reference = ArrayReference::new(scalar(10.0));
        let (primal, tangent) = compiled
            .execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(reference.clone(), session.clone())),
                    Value::Array(scalar(3.0)),
                    Value::Reference(Reference::new(tangent_reference.clone(), session.clone())),
                    Value::Array(scalar(100.0)),
                ),
            )
            .unwrap();
        let Value::Array(primal) = primal else { panic!("stateful primal output must be an array") };
        let Value::Array(tangent) = tangent else { panic!("stateful tangent output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &primal), vec![4.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent), vec![110.0]);
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![4.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent_reference.read().unwrap()), vec![110.0]);
    }

    #[test]
    fn test_stateful_compiled_function_execute_statefully_and_wait_commits_cotangent_references() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));

        // The program pullback of `y = { r += x; read(r) }` under a `Reference` destination for `r` consumes
        // `[ȳ, r̄]` and returns `[r̄, x̄]`, where the destination is returned by identity. The stateful ABI cannot
        // return a forwarded reference, so the compiled closure keeps `x̄` and lets the destination commit as a public
        // holder.
        let (_, tangent, residual_count, _) =
            accumulate_and_read_program(&reference_type, &array_type).linearize().unwrap().into_parts();
        assert_eq!(residual_count, 0);
        let pullback = tangent
            .transpose_with_respect_to(
                &[0, 1],
                &[CotangentDestinationKind::Reference, CotangentDestinationKind::Return],
            )
            .unwrap();
        assert_eq!(pullback.input_types(), vec![Type::Array(array_type.clone()), reference_type.clone()]);
        assert_eq!(pullback.output_types(), vec![reference_type.clone(), Type::Array(array_type.clone())]);
        let compiled = compile_statefully::<_, (Type, Type), Type>(
            |(cotangent_reference, output_cotangent)| {
                let context = cotangent_reference.context().clone();
                let mut outputs =
                    pullback.interpret_in_context(&context, vec![output_cotangent, cotangent_reference])?;
                Ok(outputs.remove(1))
            },
            (reference_type, Type::Array(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();

        // The read accumulates `ȳ` into the destination and the accumulation hands the accumulated cotangent to `x̄`
        // while leaving the destination unchanged, so both end at `5 + 2`.
        let scalar = |value: f32| {
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, value.to_ne_bytes().as_slice())
                .unwrap()
        };
        let cotangent_reference = ArrayReference::new(scalar(5.0));
        let input_cotangent = compiled
            .execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(cotangent_reference.clone(), session.clone())),
                    Value::Array(scalar(2.0)),
                ),
            )
            .unwrap();
        let Value::Array(input_cotangent) = input_cotangent else {
            panic!("stateful cotangent output must be an array")
        };
        assert_eq!(read_first_shard::<f32>(client, &input_cotangent), vec![7.0]);
        assert_eq!(read_first_shard::<f32>(client, &cotangent_reference.read().unwrap()), vec![7.0]);
    }

    #[test]
    fn test_stateful_compiled_function_execute_statefully_and_wait_without_references() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, Type, Type>(
            |input| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                Ok((input.clone() + input).into_value())
            },
            Type::Array(array_type.clone()),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let input = Array::from_host_buffer(&session, array_type, mesh, None, 3.0f32.to_ne_bytes().as_slice()).unwrap();

        let output = compiled.execute_statefully_and_wait(&session, Value::Array(input)).unwrap();
        let Value::Array(output) = output else { panic!("stateful all-array output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &output), vec![6.0]);
    }

    #[test]
    fn test_stateful_compiled_function_execute_statefully_and_wait_keeps_read_only_holder_ready() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, Type, Type>(
            |reference| reference.read().map_err(Into::into),
            Type::Reference(ReferenceType::new(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type, mesh, None, 7.0f32.to_ne_bytes().as_slice()).unwrap(),
        );
        for _ in 0..2 {
            let output = compiled
                .execute_statefully_and_wait(
                    &session,
                    Value::Reference(Reference::new(reference.clone(), session.clone())),
                )
                .unwrap();
            let Value::Array(output) = output else { panic!("stateful public output must be an array") };
            assert_eq!(read_first_shard::<f32>(client, &output), vec![7.0]);
            assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![7.0]);
        }
    }

    #[test]
    fn test_stateful_compiled_function_execute_statefully() {
        // An invocation without references binds no holders, but the asynchronous surface still promises that awaiting
        // the returned execution observes the completion of the whole invocation, so it returns a pending execution
        // carrying the execution fence rather than an already completed result.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, Type, Type>(
            |input| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                Ok((input.clone() + input).into_value())
            },
            Type::Array(array_type.clone()),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let input = Array::from_host_buffer(&session, array_type, mesh, None, 3.0f32.to_ne_bytes().as_slice()).unwrap();

        let output = compiled.execute_statefully(&session, Value::Array(input)).r#await().unwrap();
        let Value::Array(output) = output else { panic!("stateful all-array output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &output), vec![6.0]);
    }

    #[test]
    fn test_compile_statefully_reference_transforms_match_uncompiled_transforms() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);
        let reference_type = Type::Reference(ReferenceType::new(scalar_type.clone()));
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(
                &session,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let bind = |reference: &ArrayReference<_>| Value::Reference(Reference::new(reference.clone(), session.clone()));
        let read = |value: Value| {
            let Value::Array(array) = value else { panic!("expected an array output") };
            read_first_shard::<f32>(client, &array)
        };

        // `r ↦ { r += read(r); read(r)² }` mutates its reference, and its gradient is taken with respect to the
        // contents of the reference at entry. Differentiating it inside a stateful compilation matches doing so
        // eagerly.
        macro_rules! square_after_doubling {
            ($reference:expr) => {{
                let reference = $reference;
                let current = reference.read()?;
                reference.add_update(&current)?;
                let current = ValueProjection::<ArrayType>::into_projected(reference.read()?)?;
                current.mul(&current).map(|value| value.into_value())
            }};
        }
        let eager_reference = ArrayReference::new(array(&scalar_type, &[3.0]));
        let (value, gradient) = differentiate_at(bind(&eager_reference))
            .value_and_gradient(|reference| square_after_doubling!(reference))
            .unwrap();
        let compiled = compile_statefully::<_, Type, (Type, Type)>(
            |reference| {
                Ok(differentiate_at(reference)
                    .value_and_gradient(|reference| square_after_doubling!(reference))
                    .map_err(ProgramError::from)?)
            },
            reference_type.clone(),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let compiled_reference = ArrayReference::new(array(&scalar_type, &[3.0]));
        let (compiled_value, compiled_gradient) =
            compiled.execute_statefully_and_wait(&session, bind(&compiled_reference)).unwrap();
        assert_eq!((read(value), read(gradient)), (vec![36.0], vec![24.0]));
        assert_eq!((read(compiled_value), read(compiled_gradient)), (vec![36.0], vec![24.0]));
        assert_eq!(read_first_shard::<f32>(client, &eager_reference.read().unwrap()), vec![6.0]);
        assert_eq!(read_first_shard::<f32>(client, &compiled_reference.read().unwrap()), vec![6.0]);

        // Storing batched values into a shared reference is rejected with the same error on both paths.
        let shared_reference = ArrayReference::new(array(&scalar_type, &[1.0]));
        let eager_error = batch(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read()
            },
            (bind(&shared_reference), Value::Array(array(&vector_type, &[1.0, 2.0, 3.0]))),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap_err();
        let compiled_error = compile_statefully::<_, (Type, Type), Type>(
            |(reference, updates)| {
                Ok(batch(
                    |(reference, update)| {
                        reference.add_update(&update)?;
                        reference.read()
                    },
                    (reference, updates),
                    (BatchAxis::replicated(), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    BatchAxisSpecification::default(),
                )
                .map_err(ProgramError::from)?)
            },
            (reference_type, Type::Array(vector_type)),
            &session,
            Options::new(mesh),
        )
        .err()
        .unwrap();
        let message = "`reference_add_update` cannot store a batched value into an unbatched reference; pass the \
                       reference as a batched input instead";
        assert_eq!(eager_error.to_string(), message);
        assert_eq!(compiled_error.to_string(), message);
        assert_eq!(read_first_shard::<f32>(client, &shared_reference.read().unwrap()), vec![1.0]);
    }

    #[test]
    fn test_compile_statefully_commits_accumulating_custom_backward_buffers() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::new_static(DataType::F32, [4]);

        // `x ↦ x[1]`, whose accumulating backward rule adds its seed to only the affected entry of the caller buffer.
        // The compiled pullback therefore updates the buffer in place, without a cotangent temporary of full size.
        type ProgramTracer = DomainTracer<Session>;
        let element = |x: &ProgramTracer| -> Result<ProgramTracer, ProgramError> {
            Ok(ValueProjection::<ArrayType>::into_projected(x.clone())?.slice(&[1], &[2usize], &[1])?.into_value())
        };
        let function = composite(move |x: ProgramTracer| element(&x)).with_accumulating_vjp(
            move |x: ProgramTracer| Ok((element(&x)?, ())),
            |context: &mut TranspositionContext<Constant, Operation>,
             (): (),
             seed: MaybeZero<ProgramTracer>,
             accumulator: CotangentAccumulator| {
                let (MaybeZero::Value(seed), Some(buffer)) = (seed, accumulator.reference(context)?) else {
                    return Err(ProgramError::InvalidArgument { message: "expected a caller buffer".to_string() });
                };
                let slice = ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 1, 1)] };
                buffer.add_update_through(&seed, &[slice], &[])?;
                Ok(())
            },
        );
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes::<f32>(values))
                .unwrap()
        };
        let input = Value::Array(array(&array_type, &[1.0, 2.0, 3.0, 4.0]));
        let (_, pullback) = session.differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert!(pullback.residuals().is_empty());
        let pullback = pullback.transposed_program(&[CotangentDestinationKind::Reference]).unwrap().into_program();
        let [seed_type, buffer_type] = <[Type; 2]>::try_from(pullback.input_types().into_owned()).unwrap();
        assert!(matches!(&buffer_type, Type::Reference(_)));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}], \
                %1:ref<f32[4][sharding={mesh<['x'=1:auto]>, [{}]}]> .
                let () = reference_add_update [transforms=[slice(axes=[1:2])]] %1 %0
                in ()
            "}
            .trim_end(),
        );
        let compiled = compile_statefully::<_, (Type, Type), ()>(
            |(buffer, seed)| {
                let context = buffer.context().clone();
                pullback.interpret_in_context(&context, vec![seed, buffer])?;
                Ok(())
            },
            (buffer_type, seed_type),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let buffer = ArrayReference::new(array(&array_type, &[1.0, 1.0, 1.0, 1.0]));
        compiled
            .execute_statefully_and_wait(
                &session,
                (
                    Value::Reference(Reference::new(buffer.clone(), session.clone())),
                    Value::Array(array(&ArrayType::new_static(DataType::F32, [1]), &[5.0])),
                ),
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &buffer.read().unwrap()), vec![1.0, 6.0, 1.0, 1.0]);
    }

    #[test]
    fn test_compile_statefully_rejects_escaping_reference_outputs() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::scalar(DataType::F32);

        // A reference allocated inside the program has no host identity the ABI could hand back.
        let error = compile_statefully::<_, Type, Type>(
            |input| input.reference_new().map_err(Into::into),
            Type::Array(array_type.clone()),
            &session,
            Options::new(mesh.clone()),
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason })
                if reason == "output 0 is a reference allocated inside the program and cannot be returned through the \
                              stateful XLA ABI, which has no protocol to turn a backend array into a host reference",
        ));

        // A forwarded external reference is rejected as well, because the ABI commits holders in place and returns no
        // reference outputs.
        let error = compile_statefully::<_, Type, Type>(
            Ok,
            Type::Reference(ReferenceType::new(array_type)),
            &session,
            Options::new(mesh),
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason })
                if reason == "output 0 forwards the external reference bound at input 0, and the stateful XLA ABI \
                              does not return forwarded references; read the reference after the call instead",
        ));
    }

    #[test]
    fn test_compile_statefully_forwards_references_through_nested_calls() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = Type::Reference(ReferenceType::new(array_type.clone()));

        fn forward(
            _: Vec<Constant>,
            _: Vec<StatefulCompileTracer>,
            input: StatefulCompileTracer,
        ) -> Result<StatefulCompileTracer, Error> {
            Ok(input)
        }

        // A nested call may return the external reference that it receives, because that reference never crosses the
        // ABI as an output of the enclosing program.
        let inner = CoreStagedFunction::<Session, Type, Type>::stage(
            forward,
            Vec::new(),
            reference_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let outer = compile_statefully::<_, Type, Type>(
            move |reference| inner.call(reference)?.read().map_err(Into::into),
            reference_type,
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type, mesh, None, 7.0f32.to_ne_bytes().as_slice()).unwrap(),
        );
        let output = outer
            .execute_statefully_and_wait(&session, Value::Reference(Reference::new(reference.clone(), session.clone())))
            .unwrap();
        let Value::Array(output) = output else { panic!("stateful public output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &output), vec![7.0]);
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![7.0]);
    }

    #[test]
    fn test_compile_statefully_rejects_nested_calls_that_return_allocated_references() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);

        fn allocate_reference(
            _: Vec<Constant>,
            _: Vec<StatefulCompileTracer>,
            input: StatefulCompileTracer,
        ) -> Result<StatefulCompileTracer, Error> {
            input.reference_new().map_err(Into::into)
        }

        // A reference that a nested call allocates has no caller allocation that could receive it, so embedding the
        // call is rejected with a diagnostic that names the offending output.
        let inner = CoreStagedFunction::<Session, Type, Type>::stage(
            allocate_reference,
            Vec::new(),
            Type::Array(array_type.clone()),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let error = compile_statefully::<_, Type, Type>(
            move |input| inner.call(input)?.freeze().map_err(Into::into),
            Type::Array(array_type),
            &session,
            Options::new(mesh),
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            Error::Core(core_error) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "output 0 of the called function is a reference that the function allocates itself, \
                        which a nested call cannot return; allocate the reference in the caller and pass it to \
                        the function instead",
            ),
        ));
    }

    #[test]
    fn test_compile_statefully_reads_bounded_dynamic_state_and_rejects_mutation() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let extent = DimensionVariable::new("state_extent", DimensionBounds::new(1, Some(5)).unwrap());
        let declared_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let read = compile_statefully::<_, Type, Type>(
            |reference| reference.read().map_err(Into::into),
            Type::Reference(ReferenceType::new(declared_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let mutation = compile_statefully::<_, (Type, Type), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (Type::Reference(ReferenceType::new(declared_type.clone())), Type::Array(declared_type)),
            &session,
            Options::new(mesh.clone()),
        );
        assert!(matches!(
            mutation,
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "mutated bounded-dynamic external state input 0 is unsupported because backend alias \
                              compatibility has not been verified",
        ));

        for expected in [vec![1.0f32, 2.0], vec![1.0f32, 2.0, 3.0, 4.0]] {
            let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(expected.len())]))
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap();
            let reference = ArrayReference::new(
                Array::from_host_buffer(
                    &session,
                    actual_type,
                    mesh.clone(),
                    None,
                    values_to_bytes(expected.as_slice()).as_slice(),
                )
                .unwrap(),
            );
            let Value::Array(output) = read
                .execute_statefully_and_wait(
                    &session,
                    Value::Reference(Reference::new(reference.clone(), session.clone())),
                )
                .unwrap()
            else {
                panic!("stateful bounded-dynamic output must be an array")
            };
            assert_eq!(read_first_shard::<f32>(client, &output), expected);
            assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), expected);
        }
    }

    #[test]
    fn test_compile_statefully_with_captures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let compiled = compile_statefully_with_captures::<_, Type, Type>(
            |captures, update| {
                captures[0].add_update(&update)?;
                captures[0].read().map_err(Into::into)
            },
            vec![Value::Reference(Reference::new(reference.clone(), session.clone()))],
            Type::Array(array_type.clone()),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let update =
            Array::from_host_buffer(&session, array_type, mesh, None, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        let output = compiled.execute_statefully_and_wait(&session, Value::Array(update)).unwrap();
        let Value::Array(output) = output else { panic!("stateful public output must be an array") };
        assert_eq!(read_first_shard::<f32>(client, &output), vec![3.0]);
        assert_eq!(read_first_shard::<f32>(client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_compile_statefully_with_captures_rejects_duplicate_reference_captures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let result = compile_statefully_with_captures::<_, Type, Type>(
            |captures, _| captures[0].read().map_err(Into::into),
            vec![
                Value::Reference(Reference::new(reference.clone(), session.clone())),
                Value::Reference(Reference::new(reference, session.clone())),
            ],
            Type::Array(array_type),
            &session,
            Options::new(mesh),
        );
        assert!(matches!(
            result,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if *message == "capture 1 and capture 0 bind the same reference allocation",
            ),
        ));
    }

    #[test]
    fn test_staged_function_call() {
        // A staged function with captures that is never compiled itself can be called inside an outer `compile`,
        // which threads its captures into the outer compiled function. Only the outer program is compiled.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.25]).as_slice(),
        )
        .unwrap();
        let inner: StagedFunction<ArrayType, ArrayType> = stage_with_captures(
            |captures, x| (x + captures[0].clone()).sin().unwrap(),
            vec![bias],
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        assert_eq!(engine.cached_program_count(), 0, "staging the inner function must not compile it");

        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(engine.cached_program_count(), 1, "only the outer program should be compiled");
        assert_eq!(outer.source_program().captures().len(), 1);

        let input_value = 0.5f32;
        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let observed = read_first_shard::<f32>(client, &outer.executable().execute(&engine, input).unwrap());
        assert_eq!(observed.len(), 1);
        let expected = (input_value + 0.25).sin().cos();
        assert!((observed[0] - expected).abs() < 1e-5, "expected cos(sin(x + bias)) = {expected}, got {}", observed[0]);
    }

    #[test]
    fn test_staged_function_call_repeated() {
        // A function that calls the same staged block twice, whose two `jit_call`s deduplicate into one shared
        // `func.func` at lowering, still compiles and computes the correct result through PJRT. Here
        // `g(x) = sin(x) + sin(x) = 2 sin(x)`.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: StagedFunction<ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            move |x| inner.call(x.clone()).unwrap() + inner.call(x).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let input = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let observed = read_first_shard::<f32>(client, &outer.executable().execute(&engine, input).unwrap());
            assert_eq!(observed.len(), 1);
            let expected = 2.0 * point.sin();
            assert!((observed[0] - expected).abs() < 1e-5, "expected 2*sin({point}) = {expected}, got {}", observed[0]);
        }
    }

    #[test]
    fn test_staged_function_call_in_context_without_input_leaves() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = ArrayType::scalar(DataType::F32);
        let capture = Value::Array(
            Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, 7f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let inner = CoreStagedFunction::<Session, (), Type>::stage(
            |_, mut captures, ()| Ok(captures.remove(0)),
            vec![capture],
            (),
            Options::new(mesh.clone()),
        )
        .unwrap();
        assert_eq!(
            inner.lifted_program().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                in (%0)"},
        );

        let outer = CoreStagedFunction::<Session, Type, (Type, Type)>::stage(
            |_, _, input| {
                // No input leaf can recover the enclosing trace when the callee has no public inputs.
                assert!(matches!(
                    inner.call::<StatefulCompileTracer, _>(()),
                    Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
                ));
                let context = input.context();
                let typed = inner.call_in_context(context, ())?;
                let mut flat = inner.call_flat_in_context(context, Vec::new())?;
                Ok((typed, flat.remove(0)))
            },
            Vec::new(),
            Type::Array(array_type.clone()),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let compiled = outer.lower(&session).unwrap().compile(&session).unwrap();
        let input = Value::Array(
            Array::from_host_buffer(&session, array_type, mesh, None, 1f32.to_ne_bytes().as_slice()).unwrap(),
        );
        let (Value::Array(typed), Value::Array(flat)) = compiled.executable().execute(&session, input).unwrap() else {
            panic!("capture-only calls must return arrays");
        };
        assert_eq!(read_first_shard::<f32>(client, &typed), vec![7.0]);
        assert_eq!(read_first_shard::<f32>(client, &flat), vec![7.0]);
    }

    #[test]
    fn test_executable_function_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type, &session, mesh.clone()).unwrap();
        assert_eq!(compiled.executable().mesh(), &mesh);
    }

    #[test]
    fn test_executable_function_execute() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 1);
        let executable: ExecutableFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &session, mesh.clone())
                .unwrap()
                .into_executable();
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh,
            None,
            values_to_bytes::<f32>(&[0.5]).as_slice(),
        )
        .unwrap();

        let output = executable.execute(&session, input).unwrap();

        assert_eq!(executable.output_types(), &[input_type]);
        assert!((read_first_shard::<f32>(client, &output)[0] - 0.5f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_executable_function_execute_rejects_foreign_client() {
        let client = execution_client();
        let other_client = other_cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let other_mesh = domain_mesh(other_client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 1);
        let executable: ExecutableFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &session, mesh.clone())
                .unwrap()
                .into_executable();
        let other_input_type = replicated_vector_type(&other_mesh, 1);
        let other_executable: ExecutableFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), other_input_type, &other_session, other_mesh)
                .unwrap()
                .into_executable();
        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes::<f32>(&[0.5]).as_slice())
                .unwrap();

        assert!(matches!(
            executable.execute(&other_session, input.clone()),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
        assert!(matches!(
            executable.execute_async(&other_session, input),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
        assert!(matches!(
            session.validate_replacement(
                executable.function.compiled_program(),
                other_executable.function.compiled_program(),
            ),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
        assert!(matches!(
            other_session.validate_replacement(
                executable.function.compiled_program(),
                other_executable.function.compiled_program(),
            ),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
    }

    #[test]
    fn test_executable_function_execute_across_threads() {
        assert_send_sync::<ExecutableFunction<ArrayType, ArrayType>>();

        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 1);
        let executable: ExecutableFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &session, mesh.clone())
                .unwrap()
                .into_executable();
        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes::<f32>(&[0.5]).as_slice())
                .unwrap();

        let output =
            std::thread::scope(|scope| scope.spawn(move || executable.execute(&session, input)).join().unwrap())
                .unwrap();
        assert!((read_first_shard::<f32>(client, &output)[0] - 0.5f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_executable_function_execute_reshards_mismatched_inputs() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();

        // The executable expects an input sharded along `x` across both devices, but the caller passes a fully
        // replicated array, which `ExecutableFunction::execute` silently reshards before executing.
        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let replicated_input_type = ArrayType::new(DataType::F32, shape).with_sharding(replicated).unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), replicated_input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            replicated_input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        // Calling an executable that expects a sharded input with a replicated source would fail without the implicit
        // reshard. With the reshard, it succeeds and produces the correct output.
        let output = compiled.executable().execute(&engine, source).unwrap();
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
    fn test_executable_function_execute_async_rejects_stateful_executables() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled = compile_statefully::<_, Type, Type>(
            |reference| reference.read().map_err(Into::into),
            Type::Reference(ReferenceType::new(array_type.clone())),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();

        // A stateful compilation produces an ordinary runtime executable over the same flat signature, so the
        // asynchronous entry point for arrays is reachable and must refuse the holder transaction up front rather
        // than enqueueing an execution whose hidden final state outputs nothing would publish.
        let executable: ExecutableFunction<ArrayType, ArrayType> =
            ExecutableFunction { function: compiled.executable().clone() };
        let input = Array::from_host_buffer(&session, array_type, mesh, None, 7.0f32.to_ne_bytes().as_slice()).unwrap();
        assert!(matches!(
            executable.execute_async(&session, input),
            Err(Error::Execution(ExecutionError::StatefulInvocationRequired)),
        ));
    }

    #[test]
    fn test_executable_function_execute_async_retains_zero_output_completion() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 1);
        let compiled: CompiledFunction<ArrayType, ()> =
            compile(|_| (), input_type.clone(), &engine, mesh.clone()).unwrap();
        let input = Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes::<f32>(&[1.0]).as_slice())
            .unwrap();

        let execution = compiled.executable().execute_async(&engine, input).unwrap();
        assert_eq!(execution.output(), &());
        execution.fence().block_until_ready().unwrap();
        execution.block_until_ready().unwrap();
    }

    #[test]
    fn test_compiled_and_staged_functions_are_send_and_sync() {
        assert_send_sync::<CompiledFunction<ArrayType, ArrayType>>();
        assert_send_sync::<StagedFunction<ArrayType, ArrayType>>();
    }

    #[test]
    fn test_session_compile_staged_function() {
        // Staging alone must not build a PJRT executable, and compiling the staged handle afterwards must match a
        // direct `compile` of the same closure numerically.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let staged: StagedFunction<ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        assert_eq!(engine.cached_program_count(), 0, "staging must not compile a PJRT executable");

        let staged_compiled = engine.compile_staged_function(staged).unwrap();
        assert_eq!(engine.cached_program_count(), 1, "compiling the staged handle should build exactly one executable");

        let direct: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let input_value = 0.5f32;
        let make_input = || {
            Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
            )
            .unwrap()
        };
        let staged_output =
            read_first_shard::<f32>(client, &staged_compiled.executable().execute(&engine, make_input()).unwrap());
        let direct_output =
            read_first_shard::<f32>(client, &direct.executable().execute(&engine, make_input()).unwrap());
        assert_eq!(staged_output, direct_output);
        assert!((staged_output[0] - input_value.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_session_adaptive_profile_guided_recompilation_rejects_cpu() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 1);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type, &session, mesh).unwrap();

        assert!(matches!(
            session.adaptive_profile_guided_recompilation(&compiled, AdaptiveProfileGuidedOptions::default()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "adaptive profile-guided recompilation requires a CUDA or ROCm PJRT platform, \
                              but found `cpu`",
        ));
    }

    #[test]
    fn test_session_differentiate_at_with_captures() {
        // Array IR tracers dispatch `add` and `dot` through the array IR capability universe without projecting to
        // arrays first, and the captured gradient of `f(x; c) = (x + c) · c` is `c`.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_vector_type(&mesh, 2);
        let vector = |values: &[f32]| {
            Value::Array(
                Array::from_host_buffer(
                    &session,
                    array_type.clone(),
                    mesh.clone(),
                    None,
                    values_to_bytes::<f32>(values).as_slice(),
                )
                .unwrap(),
            )
        };
        let read = |value: Value| {
            read_first_shard::<f32>(client, &ValueProjection::<ArrayType>::into_projected(value).unwrap())
        };
        let (value, gradient) = session
            .differentiate_at(vector(&[1.0, 2.0]))
            .with_captures(vector(&[3.0, 4.0]))
            .value_and_gradient(|x, c| x.add(&c)?.dot(&c, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])))
            .unwrap();
        assert_eq!(read(value), vec![36.0]);
        assert_eq!(read(gradient), vec![3.0, 4.0]);
    }

    #[test]
    fn test_session_differentiate_at_rematerialized_function() {
        // `x ↦ sin(tag(x · x, "dot"))` executes its value and gradient on the XLA session under a policy that saves the
        // dot product and under one that offloads it to pinned host memory, and both match `sin(x · x)` and `2 cos(x ·
        // x) x`, which the residual placement must not change.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = replicated_vector_type(&mesh, 3);
        let values = [0.1f32, 0.2, 0.3];
        let dot = values.iter().map(|value| value * value).sum::<f32>();
        type ProgramTracer = DomainTracer<Session>;
        fn sine_of_dot(x: ProgramTracer) -> Result<ProgramTracer, ProgramError> {
            let x = ValueProjection::<ArrayType>::into_projected(x)?;
            let dot = x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.tag("dot")?;
            Ok(dot.sin()?.into_value())
        }
        fn value_and_gradient<P: Clone + ResidualPolicy<Type>>(
            session: &Session,
            input: Array,
            policy: P,
        ) -> (Value, Value) {
            let function = rematerialize(sine_of_dot).with_policy(policy);
            session.differentiate_at(Value::Array(input)).value_and_gradient(|x| function.call(x)).unwrap()
        }
        let input = || {
            Array::from_host_buffer(
                &session,
                array_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&values).as_slice(),
            )
            .unwrap()
        };
        let offload = SaveAndOffloadOnlyTheseNames::new(Vec::<String>::new(), ["dot"], Memory::Host { pinned: true });
        for (value, gradient) in [
            value_and_gradient(&session, input(), DotsSavable),
            value_and_gradient(&session, input(), offload.unwrap()),
        ] {
            // XLA's elementary functions may differ from the host ones in the last bit.
            let assert_close = |actual: Value, expected: Vec<f32>| {
                let actual =
                    read_first_shard::<f32>(client, &ValueProjection::<ArrayType>::into_projected(actual).unwrap());
                assert!(actual.iter().zip(&expected).all(|(actual, expected)| (actual - expected).abs() < 1e-6));
                assert_eq!(actual.len(), expected.len());
            };
            assert_close(value, vec![dot.sin()]);
            assert_close(gradient, values.iter().map(|value| 2.0 * dot.cos() * value).collect());
        }
    }

    #[test]
    fn test_compiled_function_source_program() {
        // After `compile` runs, the returned handle exposes the traced source program, which diagnostics (e.g.,
        // printing the IR) and transform composition through `CompiledFunction::call` build on.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] = sin %0
                in (%1)"},
        );
    }

    #[test]
    fn test_compiled_function_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type, &session, mesh.clone()).unwrap();
        assert_eq!(compiled.mesh(), &mesh);
    }

    #[test]
    fn test_compiled_function_call_inside_compile() {
        // A compiled function can be staged into another `compile` closure as a subroutine, which produces the same
        // result as if the whole computation were a single closure. This mirrors the JAX pattern
        // `jit(lambda x: jit(f)(x).cos())`. `CompiledFunction::call` stages the retained source program behind a
        // `jit_call` boundary in the active outer trace.

        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);

        // Compile the inner function `f = |x| x.sin()`.
        let inner: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // Compile the outer function `g = |x| cos(inner(x))` by staging `inner` as one `jit_call` and applying `cos` to
        // its output.
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(
            outer.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] = jit_call %0 [
                    callee={
                        lambda %0:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] .
                        let %1:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] = sin %0
                        in (%1)
                    },
                ]
                    %2:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] = cos %1
                in (%2)"},
        );

        // Execute and compare against the mathematical reference.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, source).unwrap();

        let observed = read_first_shard::<f32>(client, &output);
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input.sin().cos();
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_compiled_function_call_imports_captures_into_compile() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        assert_eq!(outer.source_program().captures().len(), 1);

        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, input).unwrap();
        let observed = read_first_shard::<f32>(client, &output);
        for (got, expected) in observed.iter().zip([3.0f32.sin(), 4.0f32.sin(), 5.0f32.sin(), 6.0f32.sin()]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_compiled_function_call_imports_captures_of_multiple_functions_into_compile() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);
        let left_bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let right_bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[10.0, 10.0, 10.0, 10.0]).as_slice(),
        )
        .unwrap();
        let left: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![left_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let right: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![right_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            move |x| left.call(x.clone()).unwrap() + right.call(x).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(outer.source_program().captures().len(), 2);

        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, input).unwrap();
        let observed = read_first_shard::<f32>(client, &output);
        for (got, expected) in observed.iter().zip([14.0, 16.0, 18.0, 20.0]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_compiled_function_call_constrains_inputs_to_input_sharding_overrides() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let engine = Session::builder().client(client).build().unwrap();
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated(1)).unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(replicated(2)).unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let assert_close = |actual: Vec<f32>, expected: Vec<f32>| {
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().zip(&expected).all(|(actual, expected)| (actual - expected).abs() < 1e-6));
        };
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let input = array(&vector_type, &values);
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);

        // Executions reshard the replicated input to the override, and calls constrain it to the override in the
        // calling context, eagerly, inside another compilation, and under transforms.
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), vector_type.clone(), &engine, options.clone()).unwrap();
        let executed = function.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(executed.sharding(), &sharded);
        let called = function.call(input.clone()).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| function.call(x).unwrap(), vector_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(outer.output_types(), function.output_types());
        let outer_output = outer.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(read_sharded_f32_array(&outer_output), read_sharded_f32_array(&executed));

        // Batching reshards the batched input, whose batch dimension stays replicated. Each device holds the columns
        // that it owns of both rows.
        let batched = batch(
            |x| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()),
            Value::Array(array(&matrix_type, &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        let batched = ValueProjection::<ArrayType>::into_projected(batched).unwrap();
        assert_eq!(
            batched.sharding(),
            &Sharding::new(
                mesh.logical_mesh().clone(),
                vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]
            )
            .unwrap(),
        );
        assert_close(
            read_sharded_f32_array(&batched),
            [0.0f32, 0.5, 2.0, 2.5, 1.0, 1.5, 3.0, 3.5].into_iter().map(f32::sin).collect(),
        );

        // Forward mode differentiation reshards the tangent along with the primal.
        let (primal, tangent) = engine
            .differentiate_at(Value::Array(input.clone()))
            .jvp(Value::Array(array(&vector_type, &[1.0, 1.0, 2.0, 2.0])), |x| {
                Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
            })
            .unwrap();
        let primal = ValueProjection::<ArrayType>::into_projected(primal).unwrap();
        let tangent = ValueProjection::<ArrayType>::into_projected(tangent).unwrap();
        assert_eq!(primal.r#type(), executed.r#type());
        assert_eq!(tangent.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&primal), read_sharded_f32_array(&executed));
        assert_close(
            read_sharded_f32_array(&tangent),
            values
                .iter()
                .zip([1.0f32, 1.0, 2.0, 2.0])
                .map(|(value, tangent)| f32::cos(*value) * tangent)
                .collect(),
        );

        // Dispatcher calls constrain their inputs to the input types of their staged specializations alike.
        let dispatcher: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit_with_options(|(), x: CompileTracer| Ok::<_, Error>(x.sin()?), &engine, options);
        let executed = dispatcher.execute((), input.clone()).unwrap();
        let called = dispatcher.call((), input).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
    }

    #[test]
    fn test_compiled_function_call_accepts_placed_inputs_without_input_shardings() {
        let client = cpu_client(2);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>()
        };
        let input_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let mesh = two_device_mesh(client, axis_type);
            let engine = Session::builder().client(client).build().unwrap();
            let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let vector_type = ArrayType::new_static(DataType::F32, [6]);
            let input = Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                Some(sharded.clone()),
                values_to_bytes(&input_values),
            )
            .unwrap();

            // A function compiled over a type without a sharding accepts the same type with any sharding, both when it
            // is executed and when another compiled function calls it with its own placement of that type. The call
            // keeps its input where it is, and the caller computes the values that an execution of the callee does.
            let identity: CompiledFunction<ArrayType, ArrayType> =
                compile(|x| x, vector_type.clone(), &engine, mesh.clone()).unwrap();
            let doubled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
                |x| identity.call(x.clone()).unwrap() + x,
                vector_type.clone(),
                &engine,
                Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]),
            )
            .unwrap();
            let executed = identity.executable().execute(&engine, input.clone()).unwrap();
            let called = identity.call(input.clone()).unwrap();
            let doubled = doubled.executable().execute(&engine, input).unwrap();
            assert_eq!(shapes(&executed), vec!["[6]", "[6]"]);
            assert_eq!(values(&executed), vec![input_values.to_vec(); 2]);

            // An eager call accepts the input that the execution accepts and returns what the execution returns.
            assert_eq!(called.sharding(), executed.sharding());
            assert!(called.is_committed());
            assert_eq!(shapes(&called), vec!["[6]", "[6]"]);
            assert_eq!(values(&called), vec![input_values.to_vec(); 2]);
            assert_eq!(doubled.sharding(), &sharded);
            assert!(doubled.is_committed());
            assert_eq!(shapes(&doubled), vec!["[3]", "[3]"]);
            assert_eq!(values(&doubled), vec![vec![2.0f32, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
        }
    }

    #[test]
    fn test_compiled_function_call_returns_output_sharding_overrides() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = replicated_vector_type(&mesh, 4);
        let input = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5]).as_slice(),
        )
        .unwrap();
        let options = Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]);

        // The override is staged on the output of the traced function, so calls return the output types that
        // executions return, both eagerly and inside another compilation.
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options.clone()).unwrap();
        let executed = function.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(executed.sharding(), &sharded);
        let called = function.call(input.clone()).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| function.call(x).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(outer.output_types(), function.output_types());

        // Executable and staged specializations of a dispatcher stage the override alike.
        let dispatcher: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit_with_options(|(), x: CompileTracer| Ok::<_, Error>(x.sin()?), &engine, options);
        let executed = dispatcher.execute((), input.clone()).unwrap();
        assert_eq!(executed.sharding(), &sharded);
        let called = dispatcher.call((), input).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| dispatcher.call((), x).unwrap(), input_type, &engine, mesh).unwrap();
        assert_eq!(outer.output_types(), std::slice::from_ref(executed.r#type().as_ref()));
    }

    #[test]
    fn test_compiled_function_call_agrees_with_execution_for_auto_axis_overrides() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let read = |array: &Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();
        let assert_close = |actual: Vec<f32>, expected: Vec<f32>| {
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().zip(&expected).all(|(actual, expected)| (actual - expected).abs() < 1e-5));
        };
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let input = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![sharded.clone()])
            .with_output_shardings(vec![sharded.clone()]);
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), vector_type.clone(), &engine, options).unwrap();

        // Types record no placement over auto mesh axes, so calls and executions return the same replicated output
        // type, and the call no longer rejects the replicated input. The runtime placement of the executed and eagerly
        // called outputs is the override.
        let executed = function.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(executed.r#type().sharding(), Some(&replicated));
        assert_eq!(executed.sharding(), &sharded);
        let called = function.call(input.clone()).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(called.sharding(), &sharded);
        assert_close(read(&called), values.iter().copied().map(f32::sin).collect());
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| function.call(x).unwrap(), vector_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(outer.output_types(), function.output_types());

        // Forward and reverse mode differentiation agree with the uncompiled derivatives.
        let (primal, tangent) = engine
            .differentiate_at(Value::Array(input.clone()))
            .jvp(Value::Array(input.clone()), |x| {
                Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
            })
            .unwrap();
        let primal = ValueProjection::<ArrayType>::into_projected(primal).unwrap();
        let tangent = ValueProjection::<ArrayType>::into_projected(tangent).unwrap();
        assert_eq!(primal.r#type(), executed.r#type());
        assert_eq!(tangent.r#type(), executed.r#type());
        assert_close(read(&tangent), values.iter().map(|value| f32::cos(*value) * value).collect());
        let total: CompiledFunction<ArrayType, ArrayType> = compile(
            |x| function.call(x).unwrap().reduce(&[0], ReductionKind::Sum).unwrap(),
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<ArrayType, ArrayType> = total.gradient(&engine).unwrap();
        let gradient = gradient.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(gradient.r#type().as_ref(), &vector_type);
        assert_close(read(&gradient), values.iter().copied().map(f32::cos).collect());

        // Batching places the batched input over the override, with a replicated batch dimension.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let matrix_values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let matrix = Array::from_host_buffer(
            &engine,
            matrix_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&matrix_values),
        )
        .unwrap();
        let batched = batch(
            |x| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()),
            Value::Array(matrix),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        let batched = ValueProjection::<ArrayType>::into_projected(batched).unwrap();
        assert_eq!(batched.r#type().as_ref(), &matrix_type);
        assert_close(read(&batched), matrix_values.iter().copied().map(f32::sin).collect());
    }

    #[test]
    fn test_compiled_function_call_agrees_with_execution_at_every_entry_point() {
        // What an entry point observes of the result of a callee.
        #[derive(Debug, PartialEq)]
        struct Observation {
            devices: Vec<Device>,
            shapes: Vec<String>,
            sharding: String,
            values: Vec<Vec<f32>>,
            committed: bool,
        }

        // Callee of a row, which is a function compiled ahead of time or a `jit` dispatcher.
        enum Callee<F> {
            Compiled(CompiledFunction<ArrayType, ArrayType>),
            Dispatcher(CompiledFunctionDispatcher<F, (), ArrayType, ArrayType>),
        }

        impl<F> Callee<F>
        where
            F: Fn(
                (),
                ProgramParameterValues<ArrayType, StatefulCompileTracer>,
            ) -> Result<ProgramParameterValues<ArrayType, StatefulCompileTracer>, Error>,
        {
            fn execute(&self, session: &Session, input: Array) -> Result<Array, Error> {
                match self {
                    Self::Compiled(function) => function.executable().execute(session, input),
                    Self::Dispatcher(dispatcher) => dispatcher.execute((), input),
                }
            }

            fn call<L, C, Path>(&self, input: L) -> Result<L, ProgramError>
            where
                L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
                C: Context<Type = Type, Constant = Constant, Operation = Operation>
                    + ContextProjection<ArrayType>
                    + ImportingContext<Session, Path>
                    + NamedAxes
                    + MeshContext,
                C::Value: ValueProjection<ArrayType, Projected = L> + Reshard + ConstrainSharding,
            {
                match self {
                    Self::Compiled(function) => function.call(input),
                    Self::Dispatcher(dispatcher) => dispatcher
                        .call((), input)
                        .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() }),
                }
            }
        }

        // Callee with one kind of boundary, which every entry point calls on `input`, inside callers that are compiled
        // for `caller_mesh`.
        struct Row<F> {
            name: &'static str,
            input_type: ArrayType,
            callee: Callee<F>,
            input: Array,
            caller_mesh: DeviceMesh,
        }

        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let single_device_mesh = |device: Device, axis_type: MeshAxisType| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, axis_type).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, vec![device]).unwrap()
        };
        let first_mesh = single_device_mesh(devices[0], MeshAxisType::Explicit);
        let second_mesh = single_device_mesh(devices[1], MeshAxisType::Explicit);
        let local_mesh = single_device_mesh(devices[0], MeshAxisType::Auto);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let unconstrained = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        let second_replicated = Sharding::replicated(second_mesh.logical_mesh().clone(), 1);
        let second_placement = Placement::new(second_mesh.clone(), second_replicated.clone()).unwrap();
        let second_device = devices[1];
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let short_vector_type = ArrayType::new_static(DataType::F32, [3]);
        let zero_type = ArrayType::new_static(DataType::Zero, [4]);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]));
        let upload = |r#type: &ArrayType, mesh: Option<&DeviceMesh>, sharding: Option<&Sharding>, values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.cloned(), sharding.cloned(), values_to_bytes(values))
                .unwrap()
        };
        let compiled = |input_type: &ArrayType, options: Options| {
            Callee::Compiled(compile_with_options(|x| x, input_type.clone(), &session, options).unwrap())
        };
        let inner: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x,
            vector_type.clone(),
            &session,
            Options::new(second_mesh.clone()).with_output_shardings(vec![second_replicated.clone()]),
        )
        .unwrap();
        let middle: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| inner.call(x).unwrap(), vector_type.clone(), &session, second_mesh.clone()).unwrap();
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let rows = vec![
            Row {
                name: "no overrides",
                input_type: vector_type.clone(),
                callee: compiled(&vector_type, Options::new(mesh.clone())),
                input: upload(&vector_type, Some(&mesh), None, &values),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "replicated output override",
                input_type: vector_type.clone(),
                callee: compiled(
                    &vector_type,
                    Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()]),
                ),
                input: upload(&vector_type, Some(&mesh), Some(&split), &values),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "split output override",
                input_type: vector_type.clone(),
                callee: compiled(&vector_type, Options::new(mesh.clone()).with_output_shardings(vec![split.clone()])),
                input: upload(&vector_type, Some(&mesh), None, &values),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "unconstrained output override",
                input_type: vector_type.clone(),
                callee: compiled(&vector_type, Options::new(mesh.clone()).with_output_shardings(vec![unconstrained])),
                input: upload(&vector_type, Some(&mesh), Some(&split), &values),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "input override",
                input_type: vector_type.clone(),
                callee: compiled(
                    &vector_type,
                    Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]),
                ),
                input: upload(&vector_type, Some(&mesh), Some(&split), &values),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "other single device",
                input_type: vector_type.clone(),
                callee: compiled(
                    &vector_type,
                    Options::new(second_mesh.clone()).with_output_shardings(vec![second_replicated.clone()]),
                ),
                input: upload(&vector_type, None, None, &values),
                caller_mesh: first_mesh.clone(),
            },
            Row {
                name: "single device replicated output override",
                input_type: vector_type.clone(),
                callee: compiled(
                    &vector_type,
                    Options::new(local_mesh.clone())
                        .with_output_shardings(vec![Sharding::replicated(local_mesh.logical_mesh().clone(), 1)]),
                ),
                input: upload(&vector_type, None, None, &values),
                caller_mesh: local_mesh,
            },
            Row {
                name: "bounded dynamic input",
                input_type: dynamic_type.clone(),
                callee: compiled(
                    &dynamic_type,
                    Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()]),
                ),
                input: upload(&short_vector_type, Some(&mesh), None, &values[..3]),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "bufferless result",
                input_type: zero_type.clone(),
                callee: compiled(&zero_type, Options::new(mesh.clone()).with_output_shardings(vec![split.clone()])),
                input: upload(&zero_type, Some(&mesh), None, &[]),
                caller_mesh: mesh.clone(),
            },
            Row {
                name: "placement in the body",
                input_type: vector_type.clone(),
                callee: Callee::Compiled(
                    compile(
                        |x: CompileTracer| x.place(&second_placement, None).unwrap(),
                        vector_type.clone(),
                        &session,
                        second_mesh.clone(),
                    )
                    .unwrap(),
                ),
                input: upload(&vector_type, None, None, &values),
                caller_mesh: second_mesh.clone(),
            },
            Row {
                name: "device in the body",
                input_type: vector_type.clone(),
                callee: Callee::Compiled(
                    compile(
                        |x: CompileTracer| x.place(&second_device, None).unwrap(),
                        vector_type.clone(),
                        &session,
                        second_mesh.clone(),
                    )
                    .unwrap(),
                ),
                input: upload(&vector_type, None, None, &values),
                caller_mesh: second_mesh.clone(),
            },
            Row {
                name: "committed capture",
                input_type: vector_type.clone(),
                callee: Callee::Compiled(
                    compile_with_captures(
                        |captures, x| x + captures[0].clone(),
                        vec![upload(&vector_type, Some(&second_mesh), None, &[0.0; 4])],
                        vector_type.clone(),
                        &session,
                        second_mesh.clone(),
                    )
                    .unwrap(),
                ),
                input: upload(&vector_type, None, None, &values),
                caller_mesh: second_mesh.clone(),
            },
            Row {
                name: "three levels of nesting",
                input_type: vector_type.clone(),
                callee: Callee::Compiled(
                    compile(|x| middle.call(x).unwrap(), vector_type.clone(), &session, second_mesh.clone()).unwrap(),
                ),
                input: upload(&vector_type, None, None, &values),
                caller_mesh: second_mesh.clone(),
            },
            Row {
                name: "dispatcher over auto axes",
                input_type: vector_type.clone(),
                callee: Callee::Dispatcher(jit(|(), x: CompileTracer| Ok::<_, Error>(x), &session, mesh.clone())),
                input: upload(&vector_type, Some(&mesh), Some(&split), &values),
                caller_mesh: mesh.clone(),
            },
        ];

        let observe = |array: &Array, negated: bool| {
            array.block_until_ready().unwrap();
            Observation {
                devices: array.shards().iter().map(ArrayShard::device).collect(),
                shapes: array.shards().iter().map(|shard| shard.shape().to_string()).collect(),
                sharding: array.sharding().to_string(),
                values: array
                    .shards()
                    .iter()
                    .map(|shard| match shard.buffer() {
                        Some(_) => values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap())
                            .into_iter()
                            .map(|value| if negated { -value } else { value })
                            .collect(),
                        None => Vec::new(),
                    })
                    .collect(),
                committed: array.is_committed(),
            }
        };
        let observe_value = |value: Value| match value {
            Value::Array(array) => Ok(observe(&array, false)),
            value => Err(format!("expected an array but got `{}`", value.r#type())),
        };
        let evaluate_partition = |partition: PartitionedProgram<Constant, Operation>, inputs: &[Value]| {
            let known_inputs = partition.known_input_indices().iter().map(|&index| inputs[index].clone()).collect();
            let known_outputs = partition.known_program().interpret_in_context(&session, known_inputs)?;
            let known_output_count = partition.outputs().iter().filter(|output| output.is_known()).count();
            let residual_inputs = partition
                .residual_inputs()
                .iter()
                .map(|source| match *source {
                    ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                        inputs[index].clone()
                    }
                    ResidualInputSource::KnownOutput(index) => known_outputs[index].clone(),
                    ResidualInputSource::ResidualEdge(edge) => known_outputs[known_output_count + edge].clone(),
                })
                .collect();
            let residual_outputs = partition.residual_program().interpret_in_context(&session, residual_inputs)?;
            Ok::<_, ProgramError>(match partition.outputs()[0] {
                PartialEvaluationOutput::Known(index) => known_outputs[index].clone(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[index].clone(),
            })
        };
        let columns = [
            "eager call",
            "nested call",
            "nested call consumed by arithmetic",
            "interpretation of the staged caller",
            "nested interpretation",
            "partial evaluation with a known input",
            "partial evaluation with an unknown input",
            "caller with a bounded dynamic argument",
            "eager JVP",
            "eager VJP",
            "caller with a scan",
            "caller with a condition",
            "caller with a while loop",
        ];

        let mut outcomes = Vec::new();
        for row in &rows {
            let callee = &row.callee;
            let executed = callee.execute(&session, row.input.clone()).unwrap();
            let expected = observe(&executed, false);

            // The results of an eager call, like those of an execution, live on the devices of a compiled program,
            // which every process knows, rather than on devices that this process chose on its own.
            assert_eq!(
                callee.call(row.input.clone()).map(|output| output.is_locally_assigned()),
                Ok(executed.is_locally_assigned()),
            );
            // The staged caller takes the captures of the callee as leading inputs, which its call receives first.
            let staged: StagedFunction<ArrayType, ArrayType> =
                stage(|x| callee.call(x).unwrap(), row.input_type.clone(), Options::new(row.caller_mesh.clone()))
                    .unwrap();
            let program = staged.source_program().to_program_with_lifted_captures().unwrap();
            let mut inputs = staged.source_program().captures().to_vec();
            inputs.push(Value::Array(row.input.clone()));
            let known = |input_known: bool| {
                let mut known = vec![true; inputs.len()];
                known[inputs.len() - 1] = input_known;
                known
            };
            let call = program
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
                .unwrap();
            let Operation::JitCall(operation) = call.operation() else { unreachable!("selected a `jit_call`") };
            let call_callee = Arc::new(program.entry_region_ref().with_id(call.regions()[0]).unwrap().to_program());
            let call_regions = AttachedRegions::Callees(std::slice::from_ref(&call_callee));
            let policy = ResidualPolicyReference::new(NothingSavable);
            let caller_upload = |r#type: &ArrayType, values: &[f32]| match row.caller_mesh.devices() {
                [_] => upload(r#type, None, None, values),
                _ => upload(r#type, Some(&row.caller_mesh), None, values),
            };
            let dynamic_input = caller_upload(&short_vector_type, &values[..3]);
            let predicate = match row.caller_mesh.devices() {
                [_] => Array::from_host_buffer(&session, predicate_type.clone(), None, None, [1u8]).unwrap(),
                _ => Array::from_host_buffer(&session, predicate_type.clone(), row.caller_mesh.clone(), None, [1u8])
                    .unwrap(),
            };
            let observations: [Result<Observation, String>; 13] = [
                callee
                    .call(row.input.clone())
                    .map(|output| observe(&output, false))
                    .map_err(|error| error.to_string()),
                compile(|x| callee.call(x).unwrap(), row.input_type.clone(), &session, row.caller_mesh.clone())
                    .and_then(|caller: CompiledFunction<ArrayType, ArrayType>| {
                        caller.executable().execute(&session, row.input.clone())
                    })
                    .map(|output| observe(&output, false))
                    .map_err(|error| error.to_string()),
                stage_with_flat_captures(
                    |_, _, x: CompileTracer| Ok(Neg::neg(&callee.call(x)?)?),
                    Vec::new(),
                    row.input_type.clone(),
                    Options::new(row.caller_mesh.clone()),
                )
                .and_then(|caller: StagedFunction<ArrayType, ArrayType>| session.compile_staged_function(caller))
                .and_then(|caller| caller.executable().execute(&session, row.input.clone()))
                .map(|output| observe(&output, true))
                .map_err(|error| error.to_string()),
                program
                    .interpret_in_context(&session, inputs.clone())
                    .map_err(|error| error.to_string())
                    .and_then(|mut outputs| observe_value(outputs.remove(0))),
                operation
                    .interpret(&session, &EagerInterpretationDriver::new(&call_regions), &inputs)
                    .map_err(|error| error.to_string())
                    .and_then(|mut outputs| observe_value(outputs.remove(0))),
                program
                    .partition_with_residual_policy(&known(true), &policy)
                    .and_then(|partition| evaluate_partition(partition, &inputs))
                    .map_err(|error| error.to_string())
                    .and_then(observe_value),
                program
                    .partition_with_residual_policy(&known(false), &policy)
                    .and_then(|partition| evaluate_partition(partition, &inputs))
                    .map_err(|error| error.to_string())
                    .and_then(observe_value),
                compile(
                    |(x, length): (CompileTracer, CompileTracer)| (callee.call(x).unwrap(), length),
                    (row.input_type.clone(), dynamic_type.clone()),
                    &session,
                    row.caller_mesh.clone(),
                )
                .and_then(|caller: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)>| {
                    caller.executable().execute(&session, (row.input.clone(), dynamic_input))
                })
                .map(|(output, _)| observe(&output, false))
                .map_err(|error| error.to_string()),
                session
                    .differentiate_at(Value::Array(row.input.clone()))
                    .jvp(Value::Array(row.input.clone()), |x| {
                        Ok(callee.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
                    })
                    .map_err(|error| error.to_string())
                    .and_then(|(output, _)| observe_value(output)),
                session
                    .differentiate_at(Value::Array(row.input.clone()))
                    .vjp(|x| Ok(callee.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()))
                    .and_then(|(output, pullback)| Ok((pullback.apply_materialized(output.clone())?, output)))
                    .map_err(|error| error.to_string())
                    .and_then(|(_, output)| observe_value(output)),
                stage_with_flat_captures(
                    |_, _, x: CompileTracer| {
                        // The loop carries the result of a call, whose type the calls in its body return.
                        let carry = callee.call(x)?.into_value();
                        let (carry, ()) = scan(
                            |_, carry, ()| {
                                Ok((
                                    callee.call(ValueProjection::<ArrayType>::into_projected(carry)?)?.into_value(),
                                    (),
                                ))
                            },
                            carry,
                            (),
                            ScanOptions::new().with_iteration_count(2usize),
                        )?;
                        Ok(ValueProjection::<ArrayType>::into_projected(carry)?)
                    },
                    Vec::new(),
                    row.input_type.clone(),
                    Options::new(row.caller_mesh.clone()),
                )
                .and_then(|caller: StagedFunction<ArrayType, ArrayType>| session.compile_staged_function(caller))
                .and_then(|caller| caller.executable().execute(&session, row.input.clone()))
                .map(|output| observe(&output, false))
                .map_err(|error| error.to_string()),
                stage_with_flat_captures(
                    |_, _, (x, predicate): (CompileTracer, CompileTracer)| {
                        let output = predicate.into_value().condition(
                            |x| Ok(callee.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()),
                            |x| Ok(callee.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()),
                            x.into_value(),
                        )?;
                        Ok(ValueProjection::<ArrayType>::into_projected(output)?)
                    },
                    Vec::new(),
                    (row.input_type.clone(), predicate_type.clone()),
                    Options::new(row.caller_mesh.clone()),
                )
                .and_then(|caller: StagedFunction<(ArrayType, ArrayType), ArrayType>| {
                    session.compile_staged_function(caller)
                })
                .and_then(|caller| caller.executable().execute(&session, (row.input.clone(), predicate.clone())))
                .map(|output| observe(&output, false))
                .map_err(|error| error.to_string()),
                stage_with_flat_captures(
                    |_, _, (x, predicate): (CompileTracer, CompileTracer)| {
                        // The loop carries the result of a call, whose type the calls in its body return.
                        let state = (predicate.into_value(), callee.call(x)?.into_value());
                        let (_, output) = r#while(
                            |(predicate, _)| Ok(predicate.clone()),
                            |(predicate, x)| {
                                let output = callee.call(ValueProjection::<ArrayType>::into_projected(x)?)?;
                                Ok((predicate.not_equal(&predicate)?, output.into_value()))
                            },
                            state,
                            WhileOptions::new(),
                        )?;
                        Ok(ValueProjection::<ArrayType>::into_projected(output)?)
                    },
                    Vec::new(),
                    (row.input_type.clone(), predicate_type.clone()),
                    Options::new(row.caller_mesh.clone()),
                )
                .and_then(|caller: StagedFunction<(ArrayType, ArrayType), ArrayType>| {
                    session.compile_staged_function(caller)
                })
                .and_then(|caller| caller.executable().execute(&session, (row.input.clone(), predicate)))
                .map(|output| observe(&output, false))
                .map_err(|error| error.to_string()),
            ];
            for (column, observation) in columns.iter().zip(observations) {
                let outcome = match observation {
                    Ok(observation) if observation == expected => "agrees".to_string(),
                    Ok(observation)
                        if (&observation.devices, observation.committed) == (&expected.devices, expected.committed) =>
                    {
                        format!("is placed as `{}` instead of `{}`", observation.sharding, expected.sharding)
                    }
                    Ok(observation) => format!("disagrees with {expected:?} as {observation:?}"),
                    Err(error) => format!("fails with `{error}`"),
                };
                outcomes.push(format!("{}, {column}: {outcome}", row.name));
            }
        }

        // Every entry point agrees with the execution of the callee, except where a caller compiled for other devices
        // calls it, where a caller that runs as replicas because of a bounded dynamic argument calls a callee whose
        // boundary splits a value, which that caller cannot enforce, and where a caller negates a bufferless value. A
        // bufferless result of a loop or a condition is placed as its type records, because the compiler reports no
        // placement for it and the placement of a call inside a nested region does not reach the result of the
        // operation that owns that region. A caller compiled ahead of time places its argument as its type records,
        // which replicates the split input of the dispatcher before the call. Eager reverse mode differentiation cannot
        // return the cotangent of an input that a call moves from the canonical mesh of one device onto a callee mesh
        // of another name, because the type of that cotangent is over the callee mesh while the input that it belongs
        // to is over the canonical mesh.
        let other_devices = "fails with `\
            `jit_call` cannot call a function compiled for the devices [1] of mesh `['x'=1:explicit]` inside a program \
            compiled for the devices [0] of mesh `['x'=1:explicit]``";
        let split_replicas = "fails with `cannot lower `jit_call`: bounded-dynamic multi-device programs currently \
                              require fully replicated sharding overrides of called functions because Shardy rejects \
                              dynamic tensors`";
        let replicated = "is placed as `{mesh<['x'=2:auto]>, [{}]}` instead of `{mesh<['x'=2:auto]>, [{'x'}]}`";
        let moved_cotangent = "fails with `cotangent contribution has type f32[4] but its accumulator expects \
                               f32[4][sharding={mesh<['device'=1:auto]>, [{}]}]`";
        let moved_seed = "fails with `mesh `['device'=1:auto]` does not have the topology of the module mesh \
                          `['x'=1:explicit]``";
        let exceptions = [
            ("split output override", "caller with a bounded dynamic argument", split_replicas),
            ("other single device", "nested call", other_devices),
            ("other single device", "nested call consumed by arithmetic", other_devices),
            ("other single device", "caller with a bounded dynamic argument", other_devices),
            ("other single device", "eager VJP", moved_cotangent),
            ("other single device", "caller with a scan", other_devices),
            ("other single device", "caller with a condition", other_devices),
            ("other single device", "caller with a while loop", other_devices),
            ("single device replicated output override", "eager VJP", moved_cotangent),
            (
                "bufferless result",
                "nested call consumed by arithmetic",
                "fails with ``neg` does not support input data type `zero``",
            ),
            ("bufferless result", "caller with a bounded dynamic argument", split_replicas),
            ("bufferless result", "caller with a scan", replicated),
            ("bufferless result", "caller with a condition", replicated),
            ("bufferless result", "caller with a while loop", replicated),
            ("placement in the body", "eager VJP", moved_cotangent),
            ("device in the body", "eager VJP", moved_cotangent),
            ("committed capture", "eager VJP", moved_seed),
            ("three levels of nesting", "eager VJP", moved_cotangent),
            ("dispatcher over auto axes", "nested call", replicated),
            ("dispatcher over auto axes", "nested call consumed by arithmetic", replicated),
            ("dispatcher over auto axes", "caller with a bounded dynamic argument", replicated),
            ("dispatcher over auto axes", "caller with a scan", replicated),
            ("dispatcher over auto axes", "caller with a condition", replicated),
            ("dispatcher over auto axes", "caller with a while loop", replicated),
        ];
        let expected_outcomes = rows
            .iter()
            .flat_map(|row| columns.iter().map(move |column| (row.name, *column)))
            .map(|(row, column)| {
                let outcome = exceptions
                    .iter()
                    .find(|exception| exception.0 == row && exception.1 == column)
                    .map_or("agrees", |exception| exception.2);
                format!("{row}, {column}: {outcome}")
            })
            .collect::<Vec<_>>();
        assert_eq!(outcomes.join("\n"), expected_outcomes.join("\n"));

        // A `jit` caller of the dispatcher is specialized for the split input itself, so the call inside it keeps that
        // placement, as a JAX `jit` function nested in another one does.
        let Row { callee: Callee::Dispatcher(dispatcher), input, .. } = rows.last().unwrap() else {
            unreachable!("the last row calls a dispatcher");
        };
        let caller: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> =
            jit(|(), x: CompileTracer| dispatcher.call((), x), &session, mesh.clone());
        assert_eq!(
            observe(&caller.execute((), input.clone()).unwrap(), false),
            observe(&dispatcher.execute((), input.clone()).unwrap(), false),
        );
    }

    #[test]
    fn test_compiled_function_call_rejects_inputs_that_its_input_types_forbid() {
        // A callee input type without a sharding describes a value that is invariant along manual mesh axes and fully
        // reduced, so a call rejects an input that varies along a manual axis or that is unreduced or reduced, at every
        // entry point, as JAX rejects a `shard_map` whose replicated output it cannot infer.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            mesh.devices()[..1].to_vec(),
        )
        .unwrap();
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let manual = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual_split = Sharding::new(manual.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let manual_replicated = Sharding::replicated(manual.clone(), 1);
        let local_type = ArrayType::new_static(DataType::F32, [2]);
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let split_type = vector_type.clone().with_sharding(split.clone()).unwrap();
        let inner: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.clone() + x, local_type, &session, single_device_mesh).unwrap();
        let input =
            Array::from_host_buffer(&session, split_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let mapped = |output_sharding: &Sharding| {
            stage_with_flat_captures(
                |_, _, x: CompileTracer| {
                    Ok(shard_map(
                        |local| { eprintln!("local {}", local.r#type()); let o = inner.call(local); if let Ok(o) = &o { eprintln!("out {}", o.r#type()); } o },
                        x,
                        manual.clone(),
                        manual_split.clone(),
                        output_sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .map_err(ProgramError::from)?)
                },
                Vec::new(),
                split_type.clone(),
                Options::new(mesh.clone()),
            )
            .and_then(|caller: StagedFunction<ArrayType, ArrayType>| session.compile_staged_function(caller))
            .and_then(|caller| caller.executable().execute(&session, input.clone()))
            .map(|output| (output.r#type().to_string(), read_sharded_f32_array(&output)))
            .map_err(|error| error.to_string())
        };
        eprintln!("replicated: {:?}", mapped(&manual_replicated));
        eprintln!("split: {:?}", mapped(&manual_split));

        // An unreduced or a reduced input of a callee over a plain input type is rejected eagerly and inside a compiled
        // caller alike.
        let plain: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.clone() + x, vector_type.clone(), &session, mesh.clone()).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        for sharding in [replicated.clone().with_unreduced_axes(["x"]).unwrap(), replicated.with_reduced_axes(["x"]).unwrap()] {
            let input_type = vector_type.clone().with_sharding(sharding).unwrap();
            let input = Array::from_host_buffer(
                &session,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            eprintln!("eager: {:?}", plain.call(input.clone()).map(|output| output.r#type().to_string()).map_err(|error| error.to_string()));
            let nested = stage_with_flat_captures(
                |_, _, x: CompileTracer| Ok(plain.call(x)?),
                Vec::new(),
                input_type,
                Options::new(mesh.clone()),
            )
            .and_then(|caller: StagedFunction<ArrayType, ArrayType>| session.compile_staged_function(caller))
            .and_then(|caller| caller.executable().execute(&session, input.clone()))
            .map(|output| output.r#type().to_string())
            .map_err(|error| error.to_string());
            eprintln!("nested: {nested:?}");
        }
    }

    #[test]
    fn test_compiled_function_call_accepts_inputs_that_refine_its_input_types() {
        // A call accepts every input that an execution of its callee accepts, which is one whose type refines the
        // callee input type up to its sharding, in every context. A static input of a callee over a bounded dynamic
        // extent refines the outputs that name that extent, so a caller over the static extent returns static outputs.
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]));
        let static_type = ArrayType::new_static(DataType::F32, [3]);
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.clone() + x, dynamic_type, &session, mesh.clone()).unwrap();
        let input =
            Array::from_host_buffer(&session, static_type.clone(), None, None, values_to_bytes(&[1f32, 2., 3.]))
                .unwrap();
        let read = |array: &Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();
        assert_eq!(read(&function.call(input.clone()).unwrap()), vec![2., 4., 6.]);
        let caller: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| function.call(x).unwrap(), static_type.clone(), &session, mesh).unwrap();
        assert_eq!(caller.output_types(), &[static_type]);
        assert_eq!(read(&caller.executable().execute(&session, input.clone()).unwrap()), vec![2., 4., 6.]);
        let (primal, tangent) = session
            .differentiate_at(Value::Array(input.clone()))
            .jvp(Value::Array(input.clone()), |x| {
                Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
            })
            .unwrap();
        assert_eq!(read(&ValueProjection::<ArrayType>::into_projected(primal).unwrap()), vec![2., 4., 6.]);
        assert_eq!(read(&ValueProjection::<ArrayType>::into_projected(tangent).unwrap()), vec![2., 4., 6.]);
        let (output, pullback) = session
            .differentiate_at(Value::Array(input.clone()))
            .vjp(|x| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()))
            .unwrap();
        assert_eq!(read(&ValueProjection::<ArrayType>::into_projected(output).unwrap()), vec![2., 4., 6.]);
        let cotangent = pullback.apply_materialized(Value::Array(input)).unwrap();
        assert_eq!(read(&ValueProjection::<ArrayType>::into_projected(cotangent).unwrap()), vec![2., 4., 6.]);
    }

    #[test]
    fn test_compiled_function_call_without_outputs() {
        // A callee without outputs is called eagerly and inside another compiled function like any other callee, and
        // the call stages nothing that its caller would return.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let input =
            Array::from_host_buffer(&session, vector_type.clone(), mesh.clone(), None, values_to_bytes(&[1.0f32; 4]))
                .unwrap();
        let function: CompiledFunction<ArrayType, ()> =
            compile(|_: CompileTracer| (), vector_type.clone(), &session, mesh.clone()).unwrap();
        assert_eq!(function.call(input.clone()), Ok(()));
        let caller: CompiledFunction<ArrayType, ArrayType> = compile(
            |x: CompileTracer| {
                function.call(x.clone()).unwrap();
                x
            },
            vector_type,
            &session,
            mesh,
        )
        .unwrap();
        let output = caller.executable().execute(&session, input).unwrap();
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn test_compiled_function_call_with_donated_input() {
        // Donation belongs to the executable of a function. An execution may reuse the buffer of a donated argument,
        // while a call, eager or inside another compiled function, leaves its input readable and computes the same
        // values.
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x.clone() + x,
            vector_type.clone(),
            &session,
            Options::new(mesh.clone()).with_donate(true),
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();
        let read = |array: &Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();
        let called = function.call(input.clone()).unwrap();
        assert_eq!(read(&called), vec![2., 4., 6.]);
        assert_eq!(read(&input), vec![1., 2., 3.]);
        let caller: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| function.call(x).unwrap(), vector_type, &session, mesh).unwrap();
        assert_eq!(read(&caller.executable().execute(&session, input.clone()).unwrap()), vec![2., 4., 6.]);
        assert_eq!(read(&input), vec![1., 2., 3.]);
        assert_eq!(read(&function.executable().execute(&session, input).unwrap()), vec![2., 4., 6.]);
    }

    #[test]
    fn test_compiled_function_call_keeps_replicated_output_override_inside_compile() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values =
            |array: &Array| array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();

        // The callee splits its value along `x` but requests a replicated output, which stages nothing in its body.
        // Executed directly or called inside another compiled function, it returns two replicas.
        let vector_type = ArrayType::new_static(DataType::I32, [4]);
        let vector = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let placing: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |x| x.place(&target, None).unwrap(),
            vector_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let wrapped: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| placing.call(x).unwrap(), vector_type.clone(), &engine, mesh.clone()).unwrap();
        for output in [
            placing.executable().execute(&engine, vector.clone()).unwrap(),
            wrapped.executable().execute(&engine, vector.clone()).unwrap(),
        ] {
            output.block_until_ready().unwrap();
            assert_eq!(output.sharding(), &replicated);
            assert_eq!(shapes(&output), vec!["[4]", "[4]"]);
            assert_eq!(values(&output), vec![values_to_bytes(&[1i32, 2, 3, 4]), values_to_bytes(&[1i32, 2, 3, 4])]);
        }
    }

    #[test]
    fn test_compiled_function_call_keeps_unconstrained_output_override_inside_compile() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values =
            |array: &Array| array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();

        // An identity with an output override that leaves a dimension unconstrained returns that dimension whole,
        // whose runtime placement records it as unconstrained, directly and inside another compiled function.
        let unconstrained = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Unconstrained],
        )
        .unwrap();
        let matrix_type = ArrayType::new_static(DataType::I32, [4, 4]);
        let matrix_values = (0..16).collect::<Vec<i32>>();
        let matrix =
            Array::from_host_buffer(&engine, matrix_type.clone(), mesh.clone(), None, values_to_bytes(&matrix_values))
                .unwrap();
        let identity: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x,
            matrix_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_output_shardings(vec![unconstrained.clone()]),
        )
        .unwrap();
        let wrapped: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| identity.call(x).unwrap(), matrix_type, &engine, mesh.clone()).unwrap();
        for output in [
            identity.executable().execute(&engine, matrix.clone()).unwrap(),
            wrapped.executable().execute(&engine, matrix.clone()).unwrap(),
        ] {
            output.block_until_ready().unwrap();
            assert_eq!(output.sharding(), &unconstrained);
            assert_eq!(shapes(&output), vec!["[2, 4]", "[2, 4]"]);
            assert_eq!(
                values(&output),
                vec![values_to_bytes(&matrix_values[..8]), values_to_bytes(&matrix_values[8..])],
            );
        }
    }

    #[test]
    fn test_compiled_function_call_keeps_split_output_override_of_zero_arrays_inside_compile() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();

        // An identity over a bufferless array with a split output override keeps the split, which no buffer could
        // report, directly and inside another compiled function.
        let zero_type = ArrayType::new_static(DataType::Zero, [4]);
        let zero = Array::from_host_buffer(&engine, zero_type.clone(), mesh.clone(), None, []).unwrap();
        let identity: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x,
            zero_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]),
        )
        .unwrap();
        let wrapped: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| identity.call(x).unwrap(), zero_type, &engine, mesh.clone()).unwrap();
        for output in [
            identity.executable().execute(&engine, zero.clone()).unwrap(),
            wrapped.executable().execute(&engine, zero.clone()).unwrap(),
        ] {
            output.block_until_ready().unwrap();
            assert_eq!(output.sharding(), &sharded);
            assert_eq!(shapes(&output), vec!["[2]", "[2]"]);
            assert!(output.shards().iter().all(|shard| shard.buffer().is_none()));
        }
    }

    #[test]
    fn test_compiled_function_call_keeps_replicated_output_override_of_consumed_results_inside_compile() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values =
            |array: &Array| array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();

        // An identity with a replicated output override returns two replicas of a split input, and so does every call
        // of it inside another compiled function whose results that function consumes rather than returns. The call
        // constrains its result to the override, which the body of the identity cannot show.
        let vector_type = ArrayType::new_static(DataType::I32, [4]);
        let vector = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        let identity: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x,
            vector_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let doubled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| identity.call(x.clone()).unwrap() + identity.call(x).unwrap(),
            vector_type,
            &engine,
            Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]),
        )
        .unwrap();
        assert_eq!(
            doubled.function.lowered().lowered_program().stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func private @jit_call_0(%arg0: tensor<4xi32>) -> tensor<4xi32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{?}]> : tensor<4xi32>
                    return %0 : tensor<4xi32>
                  }
                  func.func @main(%arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{?}]>}) {
                    %0 = call @jit_call_0(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<4xi32>
                    %2 = call @jit_call_0(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}]> : tensor<4xi32>
                    %4 = stablehlo.add %1, %3 : tensor<4xi32>
                    return %4 : tensor<4xi32>
                  }
                }
            "#},
        );
        let identity_output = identity.executable().execute(&engine, vector.clone()).unwrap();
        let doubled_output = doubled.executable().execute(&engine, vector).unwrap();
        identity_output.block_until_ready().unwrap();
        doubled_output.block_until_ready().unwrap();
        assert_eq!(identity_output.sharding(), &replicated);
        assert!(identity_output.is_committed());
        assert_eq!(shapes(&identity_output), vec!["[4]", "[4]"]);
        assert_eq!(values(&identity_output), vec![values_to_bytes(&[1i32, 2, 3, 4]); 2]);
        assert_eq!(doubled_output.sharding(), &replicated);
        assert!(doubled_output.is_committed());
        assert_eq!(shapes(&doubled_output), vec!["[4]", "[4]"]);
        assert_eq!(values(&doubled_output), vec![values_to_bytes(&[2i32, 4, 6, 8]); 2]);
    }

    #[test]
    fn test_compiled_function_call_keeps_replicated_input_override_of_consumed_results_inside_compile() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values =
            |array: &Array| array.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();

        // An identity with a replicated input override gathers a split input and returns it, and so does every call of
        // it inside another compiled function, which constrains the operand of the call to the override.
        let vector_type = ArrayType::new_static(DataType::I32, [4]);
        let vector = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        let identity: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x,
            vector_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let doubled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| identity.call(x.clone()).unwrap() + identity.call(x).unwrap(),
            vector_type,
            &engine,
            Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]),
        )
        .unwrap();
        assert_eq!(
            doubled.function.lowered().lowered_program().stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func private @jit_call_0(%arg0: tensor<4xi32>) -> tensor<4xi32> {
                    return %arg0 : tensor<4xi32>
                  }
                  func.func @main(%arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{?}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xi32>
                    %1 = call @jit_call_0(%0) : (tensor<4xi32>) -> tensor<4xi32>
                    %2 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xi32>
                    %3 = call @jit_call_0(%2) : (tensor<4xi32>) -> tensor<4xi32>
                    %4 = stablehlo.add %1, %3 : tensor<4xi32>
                    return %4 : tensor<4xi32>
                  }
                }
            "#},
        );
        let identity_output = identity.executable().execute(&engine, vector.clone()).unwrap();
        let doubled_output = doubled.executable().execute(&engine, vector).unwrap();
        identity_output.block_until_ready().unwrap();
        doubled_output.block_until_ready().unwrap();
        assert_eq!(identity_output.sharding(), &replicated);
        assert!(identity_output.is_committed());
        assert_eq!(shapes(&identity_output), vec!["[4]", "[4]"]);
        assert_eq!(values(&identity_output), vec![values_to_bytes(&[1i32, 2, 3, 4]); 2]);
        assert_eq!(doubled_output.sharding(), &replicated);
        assert!(doubled_output.is_committed());
        assert_eq!(shapes(&doubled_output), vec!["[4]", "[4]"]);
        assert_eq!(values(&doubled_output), vec![values_to_bytes(&[2i32, 4, 6, 8]); 2]);
    }

    #[test]
    fn test_compiled_function_call_under_eager_parents() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let read = |value: Value| {
            read_first_shard::<f32>(client, &ValueProjection::<ArrayType>::into_projected(value).unwrap())
        };

        // `f(x) = x * c` captures the runtime array `c`, so every call below also imports a capture into its parent.
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&vector_type, &[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // An eager parent executes the call immediately on concrete arrays.
        let output = function.call(array(&vector_type, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &output), vec![2.0, -2.0, 1.5]);

        // Eager forward mode differentiation treats the capture as a constant, so the tangent of `x * c` is `t * c`.
        let (primal, tangent) = engine
            .differentiate_at(Value::Array(array(&vector_type, &[1.0, 2.0, 3.0])))
            .jvp(Value::Array(array(&vector_type, &[1.0, 1.0, 2.0])), |x| {
                Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
            })
            .unwrap();
        assert_eq!(read(primal), vec![2.0, -2.0, 1.5]);
        assert_eq!(read(tangent), vec![2.0, -1.0, 1.0]);

        // Eager batching treats the capture as replicated and batches the call itself.
        let batched = batch(
            |x| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()),
            Value::Array(array(&matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(read(batched), vec![2.0, -2.0, 1.5, 8.0, -5.0, 3.0]);
    }

    #[test]
    fn test_compiled_function_call_inside_rematerialize_and_composite() {
        type Body = Tracer<TracingContext<Constant, Operation>>;
        // Composite rules must be `'static`, and so the compiled function that they call must outlive every borrow.
        let client = Box::leak(Box::new(execution_client()));
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };

        // `f(x) = x * c` captures the runtime array `c`. Rematerialized and composite bodies are traced over constants
        // alone, so calling `f` inside them imports `c` as a session constant.
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let rematerialized =
            rematerialize(|x: Body| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()));
        let composed = composite({
            let function = function.clone();
            move |x: Body| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
        });

        // The pullbacks of both bodies scale the cotangent by `c`, which reaches the outer executable as a hidden
        // argument rather than as a literal.
        let outer: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType, ArrayType)> = compile(
            |(x, cotangent)| {
                let x = x.into_value();
                let cotangent = cotangent.into_value();
                let context = x.context().clone();
                let (output, pullback) = context.differentiate_at(x.clone()).vjp(|x| rematerialized.call(x)).unwrap();
                let rematerialized_gradient = pullback.apply_materialized(cotangent.clone()).unwrap();
                let (_, pullback) = context.differentiate_at(x).vjp(|x| composed.call(x)).unwrap();
                let composed_gradient = pullback.apply_materialized(cotangent).unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(rematerialized_gradient).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(composed_gradient).unwrap(),
                )
            },
            (vector_type.clone(), vector_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let (output, rematerialized_gradient, composed_gradient) =
            outer.executable().execute(&engine, (array(&[1.0, 2.0, 3.0]), array(&[1.0, 1.0, 2.0]))).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &output), vec![2.0, -2.0, 1.5]);
        assert_eq!(read_first_shard::<f32>(client, &rematerialized_gradient), vec![2.0, -1.0, 1.0]);
        assert_eq!(read_first_shard::<f32>(client, &composed_gradient), vec![2.0, -1.0, 1.0]);

        // Eager parents resolve the session constant when lifting it, and eager dispatch passes its array to the
        // executables of the rematerialized and composite applications.
        let read = |value: Value| {
            read_first_shard::<f32>(client, &ValueProjection::<ArrayType>::into_projected(value).unwrap())
        };
        assert_eq!(read(rematerialized.call(Value::Array(array(&[1.0, 2.0, 3.0]))).unwrap()), vec![2.0, -2.0, 1.5]);
        assert_eq!(read(composed.call(Value::Array(array(&[1.0, 2.0, 3.0]))).unwrap()), vec![2.0, -2.0, 1.5]);
    }

    #[test]
    fn test_compiled_function_call_inside_nested_body_regions_and_custom_rules() {
        type Body = Tracer<TracingContext<Constant, Operation>>;
        let client = Box::leak(Box::new(execution_client()));
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // A `scan` nested in a rematerialized body imports the capture of `f` through its own trace, so its session
        // constant lives in a region attached to the rematerialized body, which computes `x ↦ x * c³`.
        let rematerialized = rematerialize(|x: Body| {
            let (carry, ()) = scan(
                |_, carry, _| {
                    Ok((function.call(ValueProjection::<ArrayType>::into_projected(carry)?)?.into_value(), ()))
                },
                x.clone(),
                x,
                ScanOptions::new(),
            )?;
            Ok(carry)
        });

        // A custom forward mode rule that calls `f` on the tangent is traced lazily, after the primal body.
        let composed = composite({
            let function = function.clone();
            move |x: Body| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value())
        })
        .with_jvp({
            let function = function.clone();
            move |x: Body, tangent: Body| {
                let call = |value: Body| -> Result<Body, ProgramError> {
                    Ok(function.call(ValueProjection::<ArrayType>::into_projected(value)?)?.into_value())
                };
                Ok((call(x)?, call(tangent)?))
            }
        });

        let outer: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(x, tangent)| {
                let x = x.into_value();
                let tangent = tangent.into_value();
                let context = x.context().clone();
                let (_, pullback) = context.differentiate_at(x.clone()).vjp(|x| rematerialized.call(x)).unwrap();
                let gradient = pullback.apply_materialized(tangent.clone()).unwrap();
                let (_, tangent) = context.differentiate_at(x).jvp(tangent, |x| composed.call(x)).unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(tangent).unwrap(),
                )
            },
            (vector_type.clone(), vector_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let (gradient, tangent) =
            outer.executable().execute(&engine, (array(&[1.0, 2.0, 3.0]), array(&[1.0, 1.0, 2.0]))).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &gradient), vec![8.0, -1.0, 0.25]);
        assert_eq!(read_first_shard::<f32>(client, &tangent), vec![2.0, -1.0, 1.0]);
    }

    #[test]
    fn test_compiled_function_call_inside_batch_in_parent() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&vector_type, &[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // Batching array leaves in their array IR parent lets the batched closure call compiled functions, both on
        // eager arrays and on the array leaves of an enclosing compilation.
        let batched = batch_in_parent(
            |x| function.call(x),
            array(&matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &batched), vec![2.0, -2.0, 1.5, 8.0, -5.0, 3.0]);
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            |xs| {
                batch_in_parent(
                    |x| function.call(x),
                    xs,
                    BatchAxis::new(0),
                    BatchAxis::new(1),
                    BatchAxisSpecification::default(),
                )
                .unwrap()
            },
            matrix_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, array(&matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &output), vec![2.0, 8.0, -2.0, -5.0, 1.5, 3.0]);
    }

    #[test]
    fn test_compiled_function_call_inside_differentiate_in_parent_at() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let read = |array: &Array| read_first_shard::<f32>(client, array);
        let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // Differentiating array leaves in their array IR parent lets the differentiated closures call compiled
        // functions, and every derivative result keeps arrays at its boundary. `f(x) = x * c` has the Jacobian
        // `diag(c)`, and `x ↦ f(x) · x` has the gradient `2 * c * x`.
        let (output, tangent) = differentiate_in_parent_at(array(&[1.0, 2.0, 3.0]))
            .jvp(array(&[1.0, 1.0, 2.0]), |x| function.call(x))
            .unwrap();
        assert_eq!((read(&output), read(&tangent)), (vec![2.0, -2.0, 1.5], vec![2.0, -1.0, 1.0]));
        let (_, pushforward) =
            differentiate_in_parent_at(array(&[1.0, 2.0, 3.0])).linearize(|x| function.call(x)).unwrap();
        assert_eq!(read(&pushforward.apply(array(&[1.0, 1.0, 2.0])).unwrap()), vec![2.0, -1.0, 1.0]);
        let (_, pullback) = differentiate_in_parent_at(array(&[1.0, 2.0, 3.0])).vjp(|x| function.call(x)).unwrap();
        assert_eq!(read(&pullback.apply_materialized(array(&[1.0, 1.0, 2.0])).unwrap()), vec![2.0, -1.0, 1.0]);
        let MaybeZero::Value(cotangent) = pullback.apply(array(&[1.0, 1.0, 2.0])).unwrap() else {
            panic!("the cotangent of a connected input should not be a structural zero");
        };
        assert_eq!(read(&cotangent), vec![2.0, -1.0, 1.0]);

        // Derivative results apply in descendant contexts too, such as a batching level over their own context.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let matrix = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                matrix_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let tangents = batch_in_parent(
            |tangent| pushforward.apply_in_context(&tangent.domain().parent().clone(), tangent),
            matrix(&[1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(read(&tangents), vec![2.0, -1.0, 1.0, 4.0, -2.0, 1.0]);
        let cotangents = batch_in_parent(
            |cotangent| pullback.apply_materialized_in_context(&cotangent.domain().parent().clone(), cotangent),
            matrix(&[1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(read(&cotangents), vec![2.0, -1.0, 1.0, 4.0, -2.0, 1.0]);
        let gradient = differentiate_in_parent_at(array(&[1.0, 2.0, 3.0]))
            .gradient(|x| function.call(x.clone())?.dot(&x, &dimensions))
            .unwrap();
        assert_eq!(read(&gradient), vec![4.0, -4.0, 3.0]);
        let jacobian =
            differentiate_in_parent_at(array(&[1.0, 2.0, 3.0])).jacobian_forward(|x| function.call(x)).unwrap();
        assert_eq!(read(&jacobian.values()[0]), vec![2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5]);

        // The same entry point serves the array leaves of an enclosing compilation.
        let outer: CompiledFunction<ArrayType, (ArrayType, ArrayType)> = compile(
            |x| {
                let jacobian = differentiate_in_parent_at(x.clone()).jacobian_reverse(|x| function.call(x)).unwrap();
                let hessian =
                    differentiate_in_parent_at(x).hessian(|x| function.call(x.clone())?.dot(&x, &dimensions)).unwrap();
                (jacobian.into_values().remove(0), hessian.into_values().remove(0))
            },
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let (jacobian, hessian) = outer.executable().execute(&engine, array(&[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(read(&jacobian), vec![2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5]);
        assert_eq!(read(&hessian), vec![4.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_compiled_function_derived_functions_keep_lowering_options() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let strategy = RaggedDotLoweringStrategy::Instruction;
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| x.clone() * x,
            scalar_type,
            &engine,
            Options::new(mesh).with_ragged_dot_lowering_strategy(strategy),
        )
        .unwrap();

        // Derived functions lower like their source, so they keep the options that select how programs lower.
        let gradient = function.gradient(&engine).unwrap();
        let jvp = function.jvp(&engine).unwrap();
        let batched = function
            .batch(&engine, BatchAxis::new(0), BatchAxis::new(0), BatchAxisSpecification::default().with_extent(2))
            .unwrap();
        assert_eq!(gradient.function.staged().options().ragged_dot_lowering_strategy(), strategy);
        assert_eq!(jvp.function.staged().options().ragged_dot_lowering_strategy(), strategy);
        assert_eq!(batched.function.staged().options().ragged_dot_lowering_strategy(), strategy);
    }

    #[test]
    fn test_compiled_function_gradient_materializes_disconnected_symbolic_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
            compile(|(_, live)| live, (dynamic_type, scalar_type.clone()), &engine, mesh.clone()).unwrap();
        let gradient: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compiled.gradient(&engine).unwrap();
        let concrete_type = ArrayType::new(DataType::F32, Shape::new(vec![3.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let disconnected = Array::from_host_buffer(
            &engine,
            concrete_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let live = Array::from_host_buffer(&engine, scalar_type, mesh, None, values_to_bytes::<f32>(&[7.0]).as_slice())
            .unwrap();
        let (zero, one) = gradient.executable().execute(&engine, (disconnected, live)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &zero), vec![0.0; 3]);
        assert_eq!(read_first_shard::<f32>(client, &one), vec![1.0]);
    }

    #[test]
    fn test_compiled_function_gradient_materializes_disconnected_condition_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
            |(dead, live)| {
                let predicate = live.compare(&(-live.clone()), ComparisonDirection::GreaterThan).unwrap();
                predicate
                    .condition(|(_, live)| Ok(live.clone() * live), |(_, live)| Ok(live), (dead, live))
                    .unwrap()
            },
            (dynamic_type, scalar_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compiled.gradient(&engine).unwrap();
        assert_eq!(
            gradient.source_program().program().to_string(),
            indoc! {r#"
                lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:dimension<extent ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}] = zero [type=f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}]] %2
                    %4:f32[][sharding={mesh<['x'=1:auto]>, []}] = one [type=f32[][sharding={mesh<['x'=1:auto]>, []}]]
                    %5:f32[][sharding={mesh<['x'=1:auto]>, []}], %6:bool[][sharding={mesh<['x'=1:auto]>, []}], %7:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 %1 [
                        callee={
                            lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                            let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = neg %1
                                %3:bool[][sharding={mesh<['x'=1:auto]>, []}] = compare [direction=GreaterThan] %1 %2
                                %4:f32[][sharding={mesh<['x'=1:auto]>, []}], %5:f32[][sharding={mesh<['x'=1:auto]>, []}] = condition %3 %0 %1 [
                                    true={
                                        lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                        let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %1 %1
                                        in (%2, %1)
                                    },
                                    false={
                                        lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                        let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = zero [type=f32[][sharding={mesh<['x'=1:auto]>, []}]]
                                        in (%1, %2)
                                    },
                                ]
                            in (%4, %3, %5)
                        },
                    ]
                    %8:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %4 %6 %7 [
                        callee={
                            lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:bool[][sharding={mesh<['x'=1:auto]>, []}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                            let %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = condition %1 %0 %2 [
                                true={
                                    lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                    let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %1 %0
                                        %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %1 %0
                                        %4:f32[][sharding={mesh<['x'=1:auto]>, []}] = add %2 %3
                                    in (%4)
                                },
                                false={
                                    lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                    in (%0)
                                },
                            ]
                            in (%3)
                        },
                    ]
                in (%3, %8)"#},
        );
        let concrete_type = replicated_vector_type(&mesh, 3);
        let dead = Array::from_host_buffer(
            &engine,
            concrete_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        for (value, expected_derivative) in [(7.0, 14.0), (-7.0, 1.0)] {
            let live = Array::from_host_buffer(
                &engine,
                scalar_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[value]).as_slice(),
            )
            .unwrap();
            let (zero, derivative) = gradient.executable().execute(&engine, (dead.clone(), live)).unwrap();
            assert_eq!(read_first_shard::<f32>(client, &zero), vec![0.0; 3]);
            assert_eq!(read_first_shard::<f32>(client, &derivative), vec![expected_derivative]);
        }
    }

    #[test]
    fn test_compiled_function_gradient_materializes_disconnected_scanned_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let slice_type = scalar_type.clone();
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.clone().into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let mut body = ProgramBuilder::<Constant, Operation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar_type.clone().into());
        body.add_input(slice_type.into());
        let doubled = body
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![carry, carry], None)
            .unwrap()[0];
        let body = Arc::new(
            body.build::<Vec<Constant>, Vec<Constant>>(vec![doubled], vec![Placeholder; 3], vec![Placeholder])
                .unwrap(),
        );
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
            |(elements, initial)| {
                let size = DimensionSizeOperation::new(elements.r#type().as_ref(), 0).unwrap();
                let elements = elements.into_value();
                let context = elements.context().clone();
                let length = context
                    .bind(Operation::DimensionSize(size), Vec::new(), std::slice::from_ref(&elements))
                    .unwrap()
                    .remove(0);
                let inputs = vec![initial.into_value(), elements, length];
                let outputs = context
                    .bind(
                        Operation::Scan(ScanOperation::new(1, Dimension::Dynamic(extent.clone()))),
                        AttachedRegions::Callees(std::slice::from_ref(&body)),
                        &inputs,
                    )
                    .unwrap();
                ValueProjection::<ArrayType>::into_projected(outputs.into_iter().next().unwrap()).unwrap()
            },
            (stacked_type, scalar_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compiled.gradient(&engine).unwrap();
        assert_eq!(
            gradient.source_program().program().to_string(),
            indoc! {r#"
                lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:dimension<extent ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}] = zero [type=f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}]] %2
                    %4:f32[][sharding={mesh<['x'=1:auto]>, []}] = one [type=f32[][sharding={mesh<['x'=1:auto]>, []}]]
                    %5:f32[][sharding={mesh<['x'=1:auto]>, []}], %6:dimension<extent ∈ [1, 5)> = jit_call %0 %1 [
                        callee={
                            lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                            let %2:dimension<extent ∈ [1, 5)> = dimension_size [axis=0] %0
                                %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = scan [carry_count=1, iteration_count=extent, reverse=false] %1 %0 %2 [
                                    body={
                                        lambda %0:i64[], %1:f32[][sharding={mesh<['x'=1:auto]>, []}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                        let %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = add %1 %1
                                        in (%3)
                                    },
                                ]
                            in (%3, %2)
                        },
                    ]
                    %7:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %4 %6 [
                        callee={
                            lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:dimension<extent ∈ [1, 5)> .
                            let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = scan [carry_count=1, iteration_count=extent, reverse=true] %0 %1 [
                                body={
                                    lambda %0:i64[], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                    let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = add %1 %1
                                    in (%2)
                                },
                            ]
                            in (%2)
                        },
                    ]
                in (%3, %7)"#},
        );
        let concrete_type = replicated_vector_type(&mesh, 3);
        let elements = Array::from_host_buffer(
            &engine,
            concrete_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let initial =
            Array::from_host_buffer(&engine, scalar_type, mesh, None, values_to_bytes::<f32>(&[7.0]).as_slice())
                .unwrap();
        let (zero, derivative) = gradient.executable().execute(&engine, (elements, initial)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &zero), vec![0.0; 3]);
        assert_eq!(read_first_shard::<f32>(client, &derivative), vec![8.0]);
    }

    #[test]
    fn test_compiled_function_gradient_omits_dead_seeds_of_staged_compiled_functions() {
        // `inner(x, s) = (x · x, s · s)` has a dynamically shaped output that `outer(x, s) = inner(x, s).1` discards.
        // The pullback of the staged `inner` call receives a structural zero seed for `x · x`, so the transposed call
        // has no input for it and the pullback reads no `extent` for it. The only `f32[extent]` zero is the gradient
        // for `x`, which the outer boundary builds. Forward mode over the gradient and the gradient of `∂outer/∂s` both
        // differentiate through that transposed call again.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let input_types = (dynamic_type, scalar_type.clone());
        let inner: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compile(|(x, s)| (x.clone() * x, s.clone() * s), input_types.clone(), &engine, mesh.clone()).unwrap();
        let outer: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
            compile(move |inputs| inner.call(inputs).unwrap().1, input_types.clone(), &engine, mesh.clone()).unwrap();
        let gradient: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            outer.gradient(&engine).unwrap();
        assert_eq!(
            gradient.source_program().program().to_string(),
            indoc! {r#"
                lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:dimension<extent ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}] = zero [type=f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}]] %2
                    %4:f32[][sharding={mesh<['x'=1:auto]>, []}] = one [type=f32[][sharding={mesh<['x'=1:auto]>, []}]]
                    %5:f32[][sharding={mesh<['x'=1:auto]>, []}], %6:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %7:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 %1 [
                        callee={
                            lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                            let %2:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %3:f32[][sharding={mesh<['x'=1:auto]>, []}], %4:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %5:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 %1 [
                                callee={
                                    lambda %0:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                    let %2:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}] = mul %0 %0
                                        %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %1 %1
                                    in (%2, %3, %0, %1)
                                },
                            ]
                            in (%3, %4, %5)
                        },
                    ]
                    %8:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %4 %6 %7 [
                        callee={
                            lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                            let %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 %1 %2 [
                                callee={
                                    lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[extent][sharding={mesh<['x'=1:auto]>, [{}]}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                                    let %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %2 %0
                                        %4:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %2 %0
                                        %5:f32[][sharding={mesh<['x'=1:auto]>, []}] = add %3 %4
                                    in (%5)
                                },
                            ]
                            in (%3)
                        },
                    ]
                in (%3, %8)"#},
        );

        let concrete_type = replicated_vector_type(&mesh, 3);
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let x = array(&concrete_type, &[2.0, 3.0, 4.0]);
        let s = array(&scalar_type, &[3.0]);
        let (x_gradient, s_gradient) = gradient.executable().execute(&engine, (x.clone(), s.clone())).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &x_gradient), vec![0.0; 3]);
        assert_eq!(read_first_shard::<f32>(client, &s_gradient), vec![6.0]);

        // Under forward over reverse mode, the tangent of the gradient `(0, 2s)` along `(ẋ, ṡ)` is `(0, 2ṡ)`.
        let forward_over_reverse = gradient.jvp(&engine).unwrap();
        let ((x_gradient, s_gradient), (x_gradient_tangent, s_gradient_tangent)) = forward_over_reverse
            .executable()
            .execute(
                &engine,
                ((x.clone(), s.clone()), (array(&concrete_type, &[1.0, 1.0, 1.0]), array(&scalar_type, &[1.0]))),
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &x_gradient), vec![0.0; 3]);
        assert_eq!(read_first_shard::<f32>(client, &s_gradient), vec![6.0]);
        assert_eq!(read_first_shard::<f32>(client, &x_gradient_tangent), vec![0.0; 3]);
        assert_eq!(read_first_shard::<f32>(client, &s_gradient_tangent), vec![2.0]);

        // Under reverse over reverse mode, the gradient of `∂outer/∂s = 2s` is `(0, 2)`.
        let s_derivative: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
            compile(move |inputs| gradient.call(inputs).unwrap().1, input_types, &engine, mesh.clone()).unwrap();
        let reverse_over_reverse = s_derivative.gradient(&engine).unwrap();
        let (x_second, s_second) = reverse_over_reverse.executable().execute(&engine, (x, s)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &x_second), vec![0.0; 3]);
        assert_eq!(read_first_shard::<f32>(client, &s_second), vec![2.0]);
    }

    #[test]
    fn test_compiled_function_gradient_preserves_captures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<ArrayType, ArrayType> = inner.gradient(&engine).unwrap();

        assert_eq!(gradient.source_program().captures().len(), 1);

        let input = Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes::<f32>(&[3.0]).as_slice())
            .unwrap();
        let output = gradient.executable().execute(&engine, input).unwrap();

        assert_eq!(read_first_shard::<f32>(client, &output), vec![2.0]);
    }

    #[test]
    fn test_compiled_function_gradient_retains_derived_function_per_session() {
        // For `f = |x| x * x`, repeated `f.gradient(&session)` calls with one session must reuse the retained derived
        // function without deriving it again or querying the compilation cache, while different runtime inputs to
        // the shared derived function must still produce their own correct gradients, which depend on those inputs.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.clone() * x, input_type.clone(), &engine, mesh.clone()).unwrap();

        let first: CompiledFunction<ArrayType, ArrayType> = compiled.gradient(&engine).unwrap();
        let statistics = engine.compilation_cache().statistics();
        let second = compiled.gradient(&engine).unwrap();

        // The warm call reuses the retained executable and never reaches the shared compilation cache.
        assert!(std::ptr::eq(first.function.compiled_program(), second.function.compiled_program()));
        assert_eq!(engine.compilation_cache().statistics(), statistics);

        // The retained derived function keeps residuals as runtime values. Different primals through the shared
        // executable produce different, correct gradients, which depend on those primals (i.e., `d(x*x)/dx = 2x`).
        for point in [3.0f32, -5.0f32] {
            let input = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = second.executable().execute(&engine, input).unwrap();
            assert_eq!(read_first_shard::<f32>(client, &output), vec![2.0 * point]);
        }

        // A different session identity misses, produces a fresh derivative, and replaces the retained slot.
        let other_engine = Session::builder().client(client).build().unwrap();
        let third = compiled.gradient(&other_engine).unwrap();
        assert!(!std::ptr::eq(first.function.compiled_program(), third.function.compiled_program()));
        let other_statistics = other_engine.compilation_cache().statistics();
        let fourth = compiled.gradient(&other_engine).unwrap();
        assert!(std::ptr::eq(third.function.compiled_program(), fourth.function.compiled_program()));
        assert_eq!(other_engine.compilation_cache().statistics(), other_statistics);
    }

    #[test]
    fn test_compiled_function_gradient_rejects_non_scalar_output() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 2);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type, &session, mesh).unwrap();
        let expected = DifferentiationError::NonScalarGradientOutput {
            output_type: "f32[2][sharding={mesh<['x'=1:auto]>, [{}]}]".to_string(),
        };
        assert!(matches!(
            compiled.gradient(&session),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(error) if error.downcast_custom::<DifferentiationError>() == Some(&expected),
            ),
        ));
    }

    #[test]
    fn test_compiled_function_gradient_keeps_sharding_overrides() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let input = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();

        // The function requests an input split along `x` and a replicated output, neither of which its body shows, as
        // types record no placement over auto mesh axes. The call in the gradient trace carries those overrides, and
        // transposing it places the gradient as the input override, as JAX places the gradient of a `jit` function
        // with `in_shardings`, whether the gradient is executed directly or called inside another compiled function.
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x| (x.clone() * x).reduce(&[0], ReductionKind::Sum).unwrap(),
            vector_type.clone(),
            &engine,
            Options::new(mesh.clone())
                .with_input_shardings(vec![sharded.clone()])
                .with_output_shardings(vec![Sharding::replicated(mesh.logical_mesh().clone(), 0)]),
        )
        .unwrap();
        let gradient = function.gradient(&engine).unwrap();
        let wrapped: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| gradient.call(x).unwrap(), vector_type, &engine, mesh.clone()).unwrap();
        for output in [
            gradient.executable().execute(&engine, input.clone()).unwrap(),
            wrapped.executable().execute(&engine, input).unwrap(),
        ] {
            output.block_until_ready().unwrap();
            assert_eq!(output.sharding(), &sharded);
            assert!(output.is_committed());
            assert_eq!(
                output.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
                ["[2]", "[2]"]
            );
            assert_eq!(
                output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
                [values_to_bytes(&[2f32, 4.]), values_to_bytes(&[6f32, 8.])],
            );
        }
    }

    #[test]
    fn test_compiled_function_jvp() {
        // For `f = |x| x.sin()`, `f.jvp()` produces a compiled function that stages one fused `jit_call`, and executing
        // it at `(primal, tangent)` returns `(sin(primal), cos(primal) * tangent)` within `1e-5`.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp_compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();
        assert_eq!(
            jvp_compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:f32[][sharding={mesh<['x'=1:auto]>, []}], %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 %1 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = sin %0
                            %3:f32[][sharding={mesh<['x'=1:auto]>, []}] = cos %0
                            %4:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%2, %3)"},
        );

        for &(primal, tangent) in &[(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let primal_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([tangent].as_slice()).as_slice(),
            )
            .unwrap();
            let (primal_out, tangent_out) =
                jvp_compiled.executable().execute(&engine, (primal_array, tangent_array)).unwrap();
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
    fn test_compiled_function_jvp_preserves_captures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), None, values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = jvp_compiled.executable().execute(&engine, (primal, tangent)).unwrap();

        assert_eq!(read_first_shard::<f32>(client, &primal_output), vec![5.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_compiled_function_jvp_and_gradient_of_deep_callee_chain_on_default_stack() {
        // Transforming and compiling a function whose `jit_call` callee contains a long elementwise chain used to
        // overflow the default libtest thread stack in debug builds (around 150 chained operations aborted), because
        // linearization and transposition rebuilt programs by recursing along use-def chains. The rebuilds are driven
        // by worklists now, so deriving `jvp` and `gradient` functions across a callee chain well past that threshold
        // must succeed on a test thread with the default stack size.
        const CALLEE_OPERATIONS: usize = 600;
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let callee: CompiledFunction<ArrayType, ArrayType> = compile(
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
        let outer: CompiledFunction<ArrayType, ArrayType> =
            compile(move |x| callee.call(x).unwrap().sin().unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();

        let jvp: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = outer.jvp(&engine).unwrap();
        assert_eq!(jvp.output_types(), &[input_type.clone(), input_type.clone()]);
        let gradient: CompiledFunction<ArrayType, ArrayType> = outer.gradient(&engine).unwrap();
        assert_eq!(gradient.output_types(), std::slice::from_ref(&input_type));

        // A reference chain evaluated in `f64` pins the gradient's value. Each step contributes its local derivative
        // at the value before the step, and the appended outer `sin` contributes the final `cos` factor.
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
        let input = Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes::<f32>(&[0.5]).as_slice())
            .unwrap();
        let output = gradient.executable().execute(&engine, input).unwrap();
        assert!((read_first_shard::<f32>(client, &output)[0] as f64 - expected_gradient).abs() < 1e-3);
    }

    #[test]
    fn test_compiled_function_jvp_retains_derived_function_per_session() {
        // Mirrors `test_compiled_function_gradient_retains_derived_function_per_domain` for the forward mode
        // `CompiledFunction::jvp` path.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.clone() * x, input_type.clone(), &engine, mesh.clone()).unwrap();

        let first: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compiled.jvp(&engine).unwrap();
        let statistics = engine.compilation_cache().statistics();
        let second = compiled.jvp(&engine).unwrap();

        assert!(std::ptr::eq(first.function.compiled_program(), second.function.compiled_program()));
        assert_eq!(engine.compilation_cache().statistics(), statistics);

        // The retained JVP computes `(x * x, 2 * x * t)` for runtime `(x, t)` supplied per call.
        let primal = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = second.executable().execute(&engine, (primal, tangent)).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &primal_output), vec![9.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent_output), vec![24.0]);
    }

    #[test]
    fn test_compiled_function_jvp_executes_zero_space_boundaries() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let primal_type = ArrayType::new(DataType::Boolean, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let tangent_type = primal_type.tangent().unwrap();
        let primal: CompiledFunction<ArrayType, ArrayType> =
            compile(|value| value, primal_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = primal.jvp(&engine).unwrap();
        let zero_identity: CompiledFunction<ArrayType, ArrayType> =
            compile(|value| value, tangent_type.clone(), &engine, mesh.clone()).unwrap();

        let primal_input = Array::from_host_buffer(&engine, primal_type.clone(), mesh.clone(), None, [1u8]).unwrap();
        let tangent_input = Array::from_host_buffer(&engine, tangent_type.clone(), mesh, None, []).unwrap();
        let (primal_output, tangent_output) = jvp.executable().execute(&engine, (primal_input, tangent_input)).unwrap();

        assert_eq!(primal_output.r#type().as_ref(), &primal_type);
        assert_eq!(tangent_output.r#type().as_ref(), &tangent_type);
        assert!(tangent_output.addressable_shards().next().unwrap().buffer().is_none());

        let chained_tangent = zero_identity.executable().execute(&engine, tangent_output).unwrap();
        assert_eq!(chained_tangent.r#type().as_ref(), &tangent_type);
        assert!(chained_tangent.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_compiled_function_jvp_keeps_replicated_output_override() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let primals = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let tangents = Array::from_host_buffer(
            &engine,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[5f32, 6., 7., 8.]),
        )
        .unwrap();

        // The function splits its value along `x` but requests a replicated output, which stages nothing in its body.
        // The call in the JVP trace carries that override, so the primal and tangent outputs of the JVP are replicated,
        // whether it is executed directly or called inside another compiled function.
        let target = Placement::new(mesh.clone(), sharded).unwrap();
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |x| x.place(&target, None).unwrap(),
            vector_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_output_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let jvp: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = function.jvp(&engine).unwrap();
        let wrapped: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compile(|inputs| jvp.call(inputs).unwrap(), (vector_type.clone(), vector_type), &engine, mesh.clone())
                .unwrap();
        for (primal_output, tangent_output) in [
            jvp.executable().execute(&engine, (primals.clone(), tangents.clone())).unwrap(),
            wrapped.executable().execute(&engine, (primals, tangents)).unwrap(),
        ] {
            for (output, values) in [(primal_output, [1f32, 2., 3., 4.]), (tangent_output, [5f32, 6., 7., 8.])] {
                output.block_until_ready().unwrap();
                assert_eq!(output.sharding(), &replicated);
                assert!(output.is_committed());
                assert_eq!(
                    output.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
                    ["[4]", "[4]"]
                );
                assert_eq!(
                    output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
                    [values_to_bytes(&values), values_to_bytes(&values)],
                );
            }
        }
    }

    #[test]
    fn test_compiled_function_batch() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);
        let vector_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(replicated(1)).unwrap();
        let array = |r#type: ArrayType, values: &[f32]| {
            Array::from_host_buffer(&engine, r#type, mesh.clone(), None, values_to_bytes::<f32>(values).as_slice())
                .unwrap()
        };

        // `f(x, y) = x * y + c` with a captured `c`, so the batched function also inherits a capture.
        let function: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile_with_captures(
            |captures, (x, y)| x * y + captures[0].clone(),
            vec![array(vector_type.clone(), &[10.0, 20.0])],
            (vector_type.clone(), vector_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // `x` is mapped along its leading axis and `y` is replicated, so the batched function takes a `[3, 2]` array
        // and a `[2]` array, and its output gains the batch axis at the requested trailing position.
        let batched = function
            .batch(
                &engine,
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(-1),
                BatchAxisSpecification::default().with_extent(3),
            )
            .unwrap();
        assert_eq!(batched.source_program().captures().len(), 1);
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 2]).with_sharding(replicated(2)).unwrap();
        let output = batched
            .executable()
            .execute(
                &engine,
                (array(matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), array(vector_type.clone(), &[1.0, -1.0])),
            )
            .unwrap();
        assert_eq!(output.shape().dimensions(), &[2, 3]);
        assert_eq!(read_first_shard::<f32>(client, &output), vec![11.0, 13.0, 15.0, 18.0, 16.0, 14.0]);

        // The batched function is ordinary staged source, so it can be differentiated in turn.
        let jvp = batched.jvp(&engine).unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 2]).with_sharding(replicated(2)).unwrap();
        let (primal, tangent) = jvp
            .executable()
            .execute(
                &engine,
                (
                    (
                        array(matrix_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                        array(vector_type.clone(), &[1.0, -1.0]),
                    ),
                    (array(matrix_type, &[1.0; 6]), array(vector_type, &[0.0, 0.0])),
                ),
            )
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &primal), vec![11.0, 13.0, 15.0, 18.0, 16.0, 14.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent), vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_compiled_function_batch_places_batch_dimension() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let matrix_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(matrix_sharding.clone()).unwrap();
        let matrix = Array::from_host_buffer(
            &engine,
            matrix_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]).as_slice(),
        )
        .unwrap();
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), vector_type, &engine, mesh.clone()).unwrap();

        // The requested placement splits the batch items across `x`, exactly like uncompiled batching of an input
        // whose batch dimension is sharded along `x`.
        let batched = function
            .batch(
                &engine,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::default().with_extent(2).with_sharding(ShardingDimension::sharded(["x"])),
            )
            .unwrap();
        assert_eq!(batched.output_types(), std::slice::from_ref(&matrix_type));
        let uncompiled = batch(
            |x| x.sin(),
            Value::Array(matrix.clone()),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        let uncompiled = ValueProjection::<ArrayType>::into_projected(uncompiled).unwrap();
        let executed = batched.executable().execute(&engine, matrix).unwrap();
        assert_eq!(executed.r#type(), uncompiled.r#type());
        assert_eq!(read_sharded_f32_array(&executed), read_sharded_f32_array(&uncompiled));
    }

    #[test]
    fn test_compiled_function_batch_keeps_replicated_output_override() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let matrix = Array::from_host_buffer(
            &engine,
            ArrayType::new_static(DataType::I32, [2, 2]),
            mesh.clone(),
            None,
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();

        // The function splits its value along `x` but requests a replicated output, which stages nothing in its body.
        // The batched call carries that override, so the batched function returns two replicas as well.
        let target = Placement::new(mesh.clone(), sharded).unwrap();
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |x| x.place(&target, None).unwrap(),
            ArrayType::new_static(DataType::I32, [2]),
            &engine,
            Options::new(mesh.clone())
                .with_output_shardings(vec![Sharding::replicated(mesh.logical_mesh().clone(), 1)]),
        )
        .unwrap();
        let batched = function
            .batch(&engine, BatchAxis::new(0), BatchAxis::new(0), BatchAxisSpecification::default().with_extent(2))
            .unwrap();
        let output = batched.executable().execute(&engine, matrix).unwrap();
        output.block_until_ready().unwrap();
        assert_eq!(output.sharding(), &replicated);
        assert!(output.is_committed());
        assert_eq!(
            output.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>(),
            ["[2, 2]", "[2, 2]"]
        );
        assert_eq!(
            output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            [values_to_bytes(&[1i32, 2, 3, 4]), values_to_bytes(&[1i32, 2, 3, 4])],
        );
    }

    #[test]
    fn test_compiled_function_batch_requires_static_extent() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let function: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), scalar_type, &engine, mesh).unwrap();
        assert!(matches!(
            function.batch(&engine, BatchAxis::new(0), BatchAxis::new(0), BatchAxisSpecification::named("i")),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidArgument { message })
                    if *message == "compiled function batching requires an explicit static batch extent",
            ),
        ));
    }

    #[test]
    fn test_compile() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

        let observed = read_first_shard::<f32>(client, &output);
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_broadcast_capture() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let value =
            Array::from_host_buffer(&engine, scalar_type, mesh.clone(), None, values_to_bytes(&[4f32]).as_slice())
                .unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, exemplar| captures[0].broadcast(exemplar.r#type().into_owned(), &[]).unwrap(),
            vec![value],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(compiled.source_program().captures().len(), 1);
        assert_eq!(
            compiled.source_program().to_program_with_lifted_captures().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[2, 3][sharding={mesh<['x'=1:auto]>, [{}, {}]}] .
                let %2:f32[2, 3][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = broadcast [output_type=f32[2, 3][sharding={mesh<['x'=1:auto]>, [{}, {}]}], output_axes=[]] %0
                in (%2)"},
        );
        let input =
            Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes(&[9f32; 6]).as_slice()).unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &output), vec![4f32; 6]);
    }

    #[test]
    fn test_compile_discharges_local_reference_state_without_changing_the_array_abi() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_scalar_type(&mesh, DataType::F32);

        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input| {
                let input = input.into_value();
                let reference = input.reference_new().unwrap();
                reference.add_update(&input).unwrap();
                ValueProjection::<ArrayType>::into_projected(reference.freeze().unwrap()).unwrap()
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let executable = compiled.executable();
        assert_eq!(executable.function.input_types(), &[Type::Array(input_type.clone())]);
        assert_eq!(executable.output_types(), std::slice::from_ref(&input_type));

        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes::<f32>(&[3.0]).as_slice())
                .unwrap();
        let output = executable.execute(&session, input).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &output), vec![6.0]);
    }

    #[test]
    fn test_compile_local_reference_value_and_gradient() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_scalar_type(&mesh, DataType::F32);

        // The inner gradient differentiates reference contents at entry, while the outer compiled function returns
        // the scalar objective, the mutated primal state, and an ordinary gradient. Both local references discharge
        // before lowering, so the public executable retains its ordinary array boundary and placement metadata.
        let compiled: CompiledFunction<ArrayType, (ArrayType, ArrayType, ArrayType)> = compile(
            |input| {
                let reference = input.into_value().reference_new().unwrap();
                let (value, gradient) = differentiate_at(reference.clone())
                    .value_and_gradient(|reference| {
                        let current = reference.read()?;
                        reference.add_update(&current)?;
                        let current = ValueProjection::<ArrayType>::into_projected(reference.read()?)?;
                        current.mul(&current).map(|value| value.into_value())
                    })
                    .unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(reference.freeze().unwrap()).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                )
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let executable = compiled.executable();
        assert_eq!(executable.function.input_types(), &[Type::Array(input_type.clone())]);
        assert_eq!(executable.output_types(), &[input_type.clone(), input_type.clone(), input_type.clone()]);

        // Reusing one executable must start each invocation's cotangent state at zero. In particular, returning to
        // the first input after another call must recover the same derivative instead of accumulating prior results.
        for (initial, expected_value, expected_state, expected_gradient) in
            [(3.0f32, 36.0, 6.0, 24.0), (5.0, 100.0, 10.0, 40.0), (3.0, 36.0, 6.0, 24.0)]
        {
            let input = Array::from_host_buffer(
                &session,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[initial]).as_slice(),
            )
            .unwrap();
            let (value, state, gradient) = executable.execute(&session, input).unwrap();
            assert_eq!(read_first_shard::<f32>(client, &value), vec![expected_value]);
            assert_eq!(read_first_shard::<f32>(client, &state), vec![expected_state]);
            assert_eq!(read_first_shard::<f32>(client, &gradient), vec![expected_gradient]);
        }
    }

    #[test]
    fn test_compile_scan_indexes_its_captured_reference_root_in_nested_call() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        for (length, nested) in [(3, false), (3, true), (256, false), (256, true)] {
            let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]));
            let reference_type = Type::Reference(ReferenceType::new(input_type.clone()));

            // The callee captures the complete root and receives the scan index as an ordinary input. It forms its
            // scalar view locally, so no view crosses the call boundary. Capture 0 must still resolve to that same root
            // when reference discharge runs before compilation lifts the captured constant into an ordinary operand.
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(reference_type.clone());
            let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
            let root = builder.add_constant(Constant::Captured(CaptureReference::new(0, reference_type.clone())));
            let current = builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Dynamic,
                    }]),
                    Vec::new(),
                    vec![root, index],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Dynamic,
                    }]),
                    Vec::new(),
                    vec![root, current, index],
                    None,
                )
                .unwrap();
            let updated = builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Dynamic,
                    }]),
                    Vec::new(),
                    vec![root, index],
                    None,
                )
                .unwrap()[0];
            let callee = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![updated], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
            let root = builder.add_input(ReferenceType::new(input_type.clone()).into());
            // Compare selection inside the call with the direct discharge path over slices.
            let outputs = if nested {
                let callee = builder.import_region(callee.entry_region_ref());
                builder
                    .add_instruction(
                        Operation::JitCall(JitCallOperation::new(1)),
                        vec![callee],
                        vec![root, index],
                        None,
                    )
                    .unwrap()
                    .to_vec()
            } else {
                let current = builder
                    .add_instruction(
                        ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                            axis: 0,
                            index: ArrayReferenceTransformIndex::Dynamic,
                        }]),
                        Vec::new(),
                        vec![root, index],
                        None,
                    )
                    .unwrap()[0];
                builder
                    .add_instruction(
                        ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                            axis: 0,
                            index: ArrayReferenceTransformIndex::Dynamic,
                        }]),
                        Vec::new(),
                        vec![root, current, index],
                        None,
                    )
                    .unwrap();
                builder
                    .add_instruction(
                        ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                            axis: 0,
                            index: ArrayReferenceTransformIndex::Dynamic,
                        }]),
                        Vec::new(),
                        vec![root, index],
                        None,
                    )
                    .unwrap()
                    .to_vec()
            };
            let body = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder])
                .unwrap();

            // Discharge the source directly before compilation can lift captured constants into ordinary operands.
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.clone().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            let body_region = builder.import_region(body.entry_region_ref());
            let outputs = builder
                .add_instruction(
                    Operation::Scan(ScanOperation::new(0, length)),
                    vec![body_region],
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            let frozen =
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let source = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![outputs, frozen], vec![Placeholder], vec![Placeholder; 2])
                .unwrap();
            let program = source.discharge_references(0).unwrap().into_program_without_external_references().unwrap();

            let compiled: CompiledFunction<ArrayType, (ArrayType, ArrayType)> = compile(
                |input| {
                    let input = input.into_value();
                    let outputs = program.interpret_in_context(input.context(), vec![input.clone()]).unwrap();
                    let mut outputs = outputs.into_iter();
                    (
                        ValueProjection::<ArrayType>::into_projected(outputs.next().unwrap()).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(outputs.next().unwrap()).unwrap(),
                    )
                },
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            // A larger root exercises repeated updates through the same compiled loop and nested call.
            let values = (1..=length).map(|value| value as f32).collect::<Vec<_>>();
            let expected = values.iter().map(|value| value * 2.0).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &session,
                input_type,
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&values).as_slice(),
            )
            .unwrap();
            let (outputs, frozen) = compiled.executable().execute(&session, input).unwrap();
            assert_eq!(read_first_shard::<f32>(client, &outputs), expected);
            assert_eq!(read_first_shard::<f32>(client, &frozen), expected);
        }
    }

    #[test]
    fn test_compile_rematerialized_function_rounds_residuals_saved_through_custom_storage() {
        /// [`ResidualStorage`] that stores a residual of shape `[8]` with shape `[8, 1]` and restores it by reshaping
        /// it back, which reproduces the stored value exactly.
        #[derive(Clone, Debug)]
        struct ReshapeStorage;

        impl ResidualStorage<Type> for ReshapeStorage {
            fn name(&self) -> String {
                "reshape".to_owned()
            }

            fn store_payloads(&self, residual_type: &Type) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                let _ = residual_type;
                Ok(vec![ErasedOperation::new(ReshapeOperation::new(vec![8], vec![8, 1]))])
            }

            fn restore_payloads(
                &self,
                stored_type: &Type,
                residual_type: &Type,
            ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                let _ = (stored_type, residual_type);
                Ok(vec![ErasedOperation::new(ReshapeOperation::new(vec![8, 1], vec![8]))])
            }
        }

        /// [`ResidualPolicy`] that saves the value tagged `y` through a [`ReshapeStorage`] and recomputes everything
        /// else.
        #[derive(Clone)]
        struct SaveTaggedThroughReshape;

        impl ResidualPolicy<Type> for SaveTaggedThroughReshape {
            type Storage = ReshapeStorage;

            fn name(&self) -> &str {
                "save_tagged_through_reshape"
            }

            fn classify(
                &self,
                candidate: &ResidualCandidate<'_, Type>,
            ) -> Result<ResidualDecision<ReshapeStorage>, ResidualRejection> {
                let saved = candidate
                    .producers()
                    .iter()
                    .any(|producer| producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "y"));
                Ok(if saved { ResidualDecision::SaveWith(ReshapeStorage) } else { ResidualDecision::Recompute })
            }
        }

        // `(x, s) ↦ f32(y) · s` with `y = tag(bf16(x) · bf16(x), "y")` and `x` excluded from differentiation saves `y`
        // through a `ReshapeStorage`, while the forward computation also converts `y` to `f32`. XLA may compute `y` in
        // `f32`, so `y` must be rounded to `bf16` before it is stored. With `s` and the seed set to one, the forward
        // output is the value of `y` that the forward computation uses, and the gradient with respect to `s` is the
        // value of `y` that the backward computation restores, so the two must agree bit for bit. Three of the squares
        // below are not representable in `bf16`.
        type Body = Tracer<TracingContext<Constant, Operation>>;
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 8);
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let function = rematerialize(|(x, s): (Body, Body)| {
            let x = ValueProjection::<ArrayType>::into_projected(x)?
                .stop_gradient()?
                .convert_element_type(DataType::BF16)?;
            let y = x.mul(&x)?.tag("y")?.convert_element_type(DataType::F32)?;
            Ok(y.mul(&ValueProjection::<ArrayType>::into_projected(s)?)?.into_value())
        })
        .with_policy(SaveTaggedThroughReshape);
        let outer: CompiledFunction<(ArrayType, ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(x, s, seed)| {
                let x = x.into_value();
                let context = x.context().clone();
                let (output, pullback) =
                    context.differentiate_at((x, s.into_value())).vjp(|(x, s)| function.call((x, s))).unwrap();
                let (_, s_gradient) = pullback.apply_materialized(seed.into_value()).unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(s_gradient).unwrap(),
                )
            },
            (vector_type.clone(), vector_type.clone(), vector_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let ones = [1f32; 8];
        let (output, s_gradient) = outer
            .executable()
            .execute(
                &engine,
                (array(&[1.0078125, 1.125, 1.375, 1.1015625, 3.25, 2.75, 1.5, 1.0625]), array(&ones), array(&ones)),
            )
            .unwrap();
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let expected = bits(vec![1.015625, 1.265625, 1.890625, 1.2109375, 10.5625, 7.5625, 2.25, 1.125]);
        assert_eq!(bits(read_first_shard::<f32>(client, &output)), expected);
        assert_eq!(bits(read_first_shard::<f32>(client, &s_gradient)), expected);
    }

    #[test]
    fn test_compile_dense_differentiation_of_array_ir_values() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 2);

        // The dense builders run directly on array IR values, rather than on their array projections, whose packed
        // replay batches the derivative program in the array IR universe.
        let function: CompiledFunction<ArrayType, (ArrayType, ArrayType, ArrayType)> = compile(
            |x| {
                let x = x.into_value();
                let context = x.context().clone();
                // Both cube closures differentiate `x ↦ x³`, and the Hessian differentiates `x ↦ x²`, elementwise.
                let forward = context
                    .differentiate_at(x.clone())
                    .jacobian_forward(|x| {
                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                        Ok(x.mul(&x)?.mul(&x)?.into_value())
                    })
                    .unwrap();
                let reverse = context
                    .differentiate_at(x.clone())
                    .jacobian_reverse(|x| {
                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                        Ok(x.mul(&x)?.mul(&x)?.into_value())
                    })
                    .unwrap();
                let hessian = context
                    .differentiate_at(x)
                    .hessian(|x| {
                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                        Ok(x.mul(&x)?.into_value())
                    })
                    .unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(forward.values()[0].clone()).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(reverse.values()[0].clone()).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(hessian.values()[0].clone()).unwrap(),
                )
            },
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let x =
            Array::from_host_buffer(&engine, vector_type, mesh, None, values_to_bytes::<f32>(&[1.0, -2.0]).as_slice())
                .unwrap();
        let (forward, reverse, hessian) = function.executable().execute(&engine, x).unwrap();
        assert_eq!(read_first_shard::<f32>(client, &forward), vec![3.0, 0.0, 0.0, 12.0]);
        assert_eq!(read_first_shard::<f32>(client, &reverse), vec![3.0, 0.0, 0.0, 12.0]);
        assert_eq!(read_first_shard::<f32>(client, &hessian), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_compile_dense_differentiation_as_part_of_the_enclosing_program() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input_type = replicated_scalar_type(&mesh, DataType::F32);

        let forward: CompiledFunction<ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| input.domain().differentiate_at(input).jacobian_forward(|value| Mul::mul(&value, &value)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let reverse: CompiledFunction<ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| input.domain().differentiate_at(input).jacobian_reverse(|value| Mul::mul(&value, &value)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let second: CompiledFunction<ArrayType, Hessian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| input.domain().differentiate_at(input).hessian(|value| Mul::mul(&value, &value)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(session.cached_program_count(), 2);
        let _: CompiledFunction<ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| input.domain().differentiate_at(input).jacobian_forward(|value| Mul::mul(&value, &value)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            session.cached_program_count(),
            2,
            "an equivalent dense transform should reuse its enclosing compilation"
        );

        assert_eq!(
            forward.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_type=f32[1][sharding={mesh<['x'=1:auto]>, [{}]}], output_axes=[]] %0
                    %2:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=0]
                    %3:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=1]
                    %4:bool[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = compare [direction=Equal] %2 %3
                    %5:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = one [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %6:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = zero [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %7:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = select %4 %5 %6
                    %8:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = reshape [shape=[1], output_sharding={mesh<['x'=1:auto]>, [{}]}] %7
                    %9:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = mul %1 %8
                    %10:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_type=f32[1][sharding={mesh<['x'=1:auto]>, [{}]}], output_axes=[]] %0
                    %11:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = mul %10 %8
                    %12:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = add %9 %11
                    %13:f32[][sharding={mesh<['x'=1:auto]>, []}] = reshape [shape=[]] %12
                in (%13)"},
        );
        assert_eq!(
            reverse.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_type=f32[1][sharding={mesh<['x'=1:auto]>, [{}]}], output_axes=[]] %0
                    %2:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=0]
                    %3:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=1]
                    %4:bool[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = compare [direction=Equal] %2 %3
                    %5:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = one [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %6:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = zero [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %7:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = select %4 %5 %6
                    %8:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = reshape [shape=[1], output_sharding={mesh<['x'=1:auto]>, [{}]}] %7
                    %9:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = mul %1 %8
                    %10:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_type=f32[1][sharding={mesh<['x'=1:auto]>, [{}]}], output_axes=[]] %0
                    %11:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = mul %10 %8
                    %12:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = add %9 %11
                    %13:f32[][sharding={mesh<['x'=1:auto]>, []}] = reshape [shape=[]] %12
                in (%13)"},
        );
        assert_eq!(
            second.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=0]
                    %2:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=1]
                    %3:bool[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = compare [direction=Equal] %1 %2
                    %4:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = one [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %5:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = zero [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %6:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = select %3 %4 %5
                    %7:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = reshape [shape=[1], output_sharding={mesh<['x'=1:auto]>, [{}]}] %6
                    %8:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = broadcast [output_type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], output_axes=[1]] %7
                    %9:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=0]
                    %10:u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = iota [type=u64[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], dimension=1]
                    %11:bool[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = compare [direction=Equal] %9 %10
                    %12:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = one [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %13:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = zero [type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}]]
                    %14:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = select %11 %12 %13
                    %15:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = reshape [shape=[1], output_sharding={mesh<['x'=1:auto]>, [{}]}] %14
                    %16:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = broadcast [output_type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], output_axes=[0]] %15
                    %17:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = mul %8 %16
                    %18:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = broadcast [output_type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], output_axes=[1]] %7
                    %19:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = broadcast [output_type=f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}], output_axes=[0]] %15
                    %20:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = mul %18 %19
                    %21:f32[1, 1][sharding={mesh<['x'=1:auto]>, [{}, {}]}] = add %17 %20
                    %22:f32[1][sharding={mesh<['x'=1:auto]>, [{}]}] = reshape [shape=[1]] %21
                    %23:f32[][sharding={mesh<['x'=1:auto]>, []}] = reshape [shape=[]] %22
                in (%23)"},
        );

        for (compiled, expected) in [(forward, 6.0), (reverse, 6.0)] {
            let input = Array::from_host_buffer(
                &session,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(&[3.0]).as_slice(),
            )
            .unwrap();
            let jacobian: Jacobian<ArrayType, Array, ArrayType, ArrayType> =
                compiled.executable().execute(&session, input).unwrap();
            assert_eq!(read_first_shard::<f32>(client, jacobian.iter_blocks().next().unwrap().value()), vec![expected]);
        }

        let input =
            Array::from_host_buffer(&session, input_type, mesh, None, values_to_bytes::<f32>(&[3.0]).as_slice())
                .unwrap();
        let hessian: Hessian<ArrayType, Array, ArrayType, ArrayType> =
            second.executable().execute(&session, input).unwrap();
        assert_eq!(read_first_shard::<f32>(client, hessian.iter_blocks().next().unwrap().value()), vec![2.0]);
    }

    #[test]
    fn test_compile_dense_differentiation_of_promoted_broadcast() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        let forward: CompiledFunction<
            (ArrayType, ArrayType),
            Jacobian<ArrayType, ArrayType, (ArrayType, ArrayType), ArrayType>,
        > = compile(
            |inputs| {
                inputs
                    .0
                    .domain()
                    .differentiate_at(inputs)
                    .jacobian_forward(|(scalar, vector)| scalar.atan2(&vector))
                    .unwrap()
            },
            (scalar_type.clone(), vector_type.clone()),
            &session,
            mesh.clone(),
        )
        .unwrap();

        let scalar = Array::from_host_buffer(
            &session,
            scalar_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let vector =
            Array::from_host_buffer(&session, vector_type, mesh, None, values_to_bytes::<f64>(&[1.0, 4.0]).as_slice())
                .unwrap();
        let jacobian: Jacobian<ArrayType, Array, (ArrayType, ArrayType), ArrayType> =
            forward.executable().execute(&session, (scalar, vector)).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        for (actual, expected) in read_first_shard::<f64>(client, blocks[0].value()).into_iter().zip([0.2, 0.2]) {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }
        for (actual, expected) in
            read_first_shard::<f64>(client, blocks[1].value()).into_iter().zip([-0.4, 0.0, 0.0, -0.1])
        {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }
    }

    #[test]
    fn test_compile_stable_cross_entropy_gradients() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let label_type = replicated_scalar_type(&mesh, DataType::I32);
        let soft_target = softmax_cross_entropy_with_target_logits::<ProjectedContext<Session, ArrayType>>(-1);
        // Compile each complete stabilized derivative so optimizer rewrites cannot hide behind eager operation
        // boundaries. Integer labels and teacher logits parameterize their respective losses without differentiation.
        let compiled: CompiledFunction<
            (ArrayType, ArrayType, ArrayType, ArrayType),
            (ArrayType, ArrayType, ArrayType),
        > = compile(
            move |(logits, labels, student, teacher)| {
                let (loss, hard_gradient) = differentiate_at(logits)
                    .with_captures(labels)
                    .value_and_gradient(|logits, labels| logits.softmax_cross_entropy_with_integer_labels(&labels, 0))
                    .unwrap();
                let soft_gradient = differentiate_at(student)
                    .with_captures(teacher)
                    .gradient(|student, teacher| soft_target.call(student, teacher))
                    .unwrap();
                (loss, hard_gradient, soft_gradient)
            },
            (vector_type.clone(), label_type.clone(), vector_type.clone(), vector_type.clone()),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let student_values = [2.0f32 - 1e-7, 1.0f32 + 1e-7, 0.0];
        let logits = Array::from_host_buffer(
            &session,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[9.0f32, -9.0, -9.0]),
        )
        .unwrap();
        let labels =
            Array::from_host_buffer(&session, label_type, mesh.clone(), None, values_to_bytes(&[0i32])).unwrap();
        let student = Array::from_host_buffer(
            &session,
            vector_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&student_values),
        )
        .unwrap();
        let teacher =
            Array::from_host_buffer(&session, vector_type, mesh, None, values_to_bytes(&[2.0f32, 1.0, 0.0])).unwrap();
        let (loss, hard_gradient, soft_gradient) =
            compiled.executable().execute(&session, (logits, labels, student, teacher)).unwrap();

        let tail = (-18.0f64).exp();
        let loss = read_first_shard::<f32>(client, &loss)[0] as f64;
        assert!((loss - (2.0 * tail).ln_1p()).abs() < 1e-13);
        let hard_gradient = read_first_shard::<f32>(client, &hard_gradient);
        let expected = tail / (1.0 + 2.0 * tail);
        assert!((f64::from(hard_gradient[0]) + 2.0 * expected).abs() < 1e-13);
        assert!((f64::from(hard_gradient[1]) - expected).abs() < 1e-13);
        assert!((f64::from(hard_gradient[2]) - expected).abs() < 1e-13);
        assert!((hard_gradient.iter().map(|&value| f64::from(value)).sum::<f64>()).abs() < 1e-13);

        // Use the represented f32 logits in the f64 oracle so the comparison measures probability cancellation,
        // rather than a discrepancy between intended and representable student perturbations.
        let student_denominator = student_values.iter().map(|&value| (f64::from(value) - 2.0).exp()).sum::<f64>();
        let teacher_denominator = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        for (class, actual) in read_first_shard::<f32>(client, &soft_gradient).into_iter().enumerate() {
            let expected = (f64::from(student_values[class]) - 2.0).exp() / student_denominator
                - (-(class as f64)).exp() / teacher_denominator;
            assert!((f64::from(actual) - expected).abs() < 1e-13, "class {class}: {actual} differs from {expected}");
        }
    }

    #[test]
    fn test_compile_softmax_sharded_class_axis() {
        // A class axis sharded over two devices normalizes across both shards. The partitioner combines the argmax
        // and the sums across the devices, the probabilities keep the input's sharding, and the loss and gradient of
        // a replicated integer label match the unsharded arithmetic.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let engine = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let logits_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let label_type = replicated_scalar_type(&mesh, DataType::I32);
        let compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType, ArrayType)> = compile(
            |(logits, labels)| {
                let probabilities = logits.softmax(0).unwrap();
                let (loss, gradient) = differentiate_at(logits)
                    .with_captures(labels)
                    .value_and_gradient(|logits, labels| logits.softmax_cross_entropy_with_integer_labels(&labels, 0))
                    .unwrap();
                (probabilities, loss, gradient)
            },
            (logits_type.clone(), label_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let values = [9.0f32, -9.0, 1.0, -9.0];
        let logits =
            Array::from_host_buffer(&engine, logits_type, mesh.clone(), None, values_to_bytes(&values)).unwrap();
        let labels = Array::from_host_buffer(&engine, label_type, mesh, None, values_to_bytes(&[2i32])).unwrap();
        let (probabilities, loss, gradient) = compiled.executable().execute(&engine, (logits, labels)).unwrap();
        assert_eq!(probabilities.sharding(), &sharding);
        assert_eq!(gradient.sharding(), &sharding);
        let read_shards = |array: &Array| {
            let mut observed = Vec::new();
            for device in client.addressable_devices().unwrap().iter().take(2) {
                let bytes = array
                    .device_shard(device.id().unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                observed.extend(values_from_bytes::<f32>(&bytes));
            }
            observed
        };
        let denominator = values.iter().map(|&value| (f64::from(value) - 9.0).exp()).sum::<f64>();
        let expected_probabilities =
            values.iter().map(|&value| (f64::from(value) - 9.0).exp() / denominator).collect::<Vec<_>>();
        for (actual, expected) in read_shards(&probabilities).into_iter().zip(&expected_probabilities) {
            assert!((f64::from(actual) - expected).abs() < 1e-7, "{actual} differs from {expected}");
        }
        let expected_loss = 8.0 + denominator.ln();
        assert!((f64::from(read_first_shard::<f32>(client, &loss)[0]) - expected_loss).abs() < 1e-5);
        for (class, actual) in read_shards(&gradient).into_iter().enumerate() {
            let expected = expected_probabilities[class] - if class == 2 { 1.0 } else { 0.0 };
            assert!((f64::from(actual) - expected).abs() < 1e-7, "class {class}: {actual} differs from {expected}");
        }
    }

    #[test]
    fn test_compile_softmax_dynamic_shapes_matches_reference_backend() {
        // One compiled program normalizes dynamic batch and class extents at their runtime sizes. For dynamic shapes
        // the normalization stages its class coordinates, fills, and broadcasts relative to the logits, whose
        // runtime extents the lowered program reads.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let classes = DimensionVariable::new("classes", DimensionBounds::new(1, Some(5)).unwrap());
        let replicated = |r#type: ArrayType| {
            let rank = r#type.rank();
            r#type.with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), rank)).unwrap()
        };
        let logits_type =
            replicated(ArrayType::new(DataType::F32, Shape::new(vec![batch.clone().into(), classes.into()])));
        let labels_type = replicated(ArrayType::new(DataType::I32, Shape::new(vec![batch.into()])));
        let compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(logits, labels)| {
                let log_probabilities = logits.log_softmax(1).unwrap();
                let loss = logits.softmax_cross_entropy_with_integer_labels(&labels, 1).unwrap();
                (log_probabilities, loss)
            },
            (logits_type, labels_type),
            &session,
            mesh.clone(),
        )
        .unwrap();
        for (rows, columns) in [(2, 3), (1, 4), (4, 1)] {
            let values = (0..rows * columns).map(|index| 3.0 * (0.75 * index as f32).sin()).collect::<Vec<_>>();
            let label_values = (0..rows).map(|row| (row % columns) as i32).collect::<Vec<_>>();
            let logits = Array::from_host_buffer(
                &session,
                replicated(ArrayType::new_static(DataType::F32, [rows, columns])),
                mesh.clone(),
                None,
                values_to_bytes(&values),
            )
            .unwrap();
            let labels = Array::from_host_buffer(
                &session,
                replicated(ArrayType::new_static(DataType::I32, [rows])),
                mesh.clone(),
                None,
                values_to_bytes(&label_values),
            )
            .unwrap();
            let (log_probabilities, loss) = compiled.executable().execute(&session, (logits, labels)).unwrap();

            // The reference arrays compute the same functions at the static runtime shape.
            let expected_logits = CoreArray::matrix(rows, columns, values).unwrap();
            let expected_labels = CoreArray::vector(label_values).unwrap();
            let expected_log_probabilities = expected_logits.log_softmax(1).unwrap().elements::<f32>().unwrap();
            let expected_loss = expected_logits
                .softmax_cross_entropy_with_integer_labels(&expected_labels, 1)
                .unwrap()
                .elements::<f32>()
                .unwrap();
            let log_probabilities = read_first_shard::<f32>(client, &log_probabilities);
            let loss = read_first_shard::<f32>(client, &loss);
            assert_eq!(log_probabilities.len(), expected_log_probabilities.len());
            assert_eq!(loss.len(), expected_loss.len());
            for (actual, expected) in log_probabilities.iter().zip(&expected_log_probabilities) {
                assert!((actual - expected).abs() < 1e-5, "{actual} differs from {expected}");
            }
            for (actual, expected) in loss.iter().zip(&expected_loss) {
                assert!((actual - expected).abs() < 1e-5, "{actual} differs from {expected}");
            }
        }
    }

    #[test]
    fn test_compile_stages_jvp_of_compiled_function_call_with_captures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal, tangent)| {
                let primal = primal.into_value();
                let tangent = tangent.into_value();
                let context = primal.context().clone();
                let (primal_output, tangent_output) = context
                    .jvp(
                        move |x, ()| {
                            let x = ValueProjection::<ArrayType>::into_projected(x).map_err(ProgramError::from)?;
                            Ok(inner.call(x)?.into_value())
                        },
                        primal,
                        tangent,
                        (),
                    )
                    .unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(primal_output).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(tangent_output).unwrap(),
                )
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), None, values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = jvp_compiled.executable().execute(&engine, (primal, tangent)).unwrap();

        assert_eq!(read_first_shard::<f32>(client, &primal_output), vec![5.0]);
        assert_eq!(read_first_shard::<f32>(client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_compile_binary_function_with_tuple_input() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let shape = Shape::new(vec![Dimension::Static(3)]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharding.clone()).unwrap();
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
            compile(|(a, b)| a + b, (input_type.clone(), input_type.clone()), &engine, mesh.clone()).unwrap();

        let a_values = [10.0f32, 20.0, 30.0];
        let b_values = [1.0f32, 2.0, 3.0];
        let a = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&a_values).as_slice(),
        )
        .unwrap();
        let b = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&b_values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, (a, b)).unwrap();

        let observed = read_first_shard::<f32>(client, &output);
        assert_eq!(observed, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_compile_reuses_cached_lowering_on_repeated_call_site() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 4);

        assert_eq!(engine.cached_program_count(), 0);
        // Equivalent lowerings share one executable, so the second invocation is a cache hit.
        for _ in 0..2 {
            let _: CompiledFunction<ArrayType, ArrayType> =
                compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cached_program_count(), 1, "an equivalent repeated lowering should hit the cache");
    }

    #[test]
    fn test_compile_shares_cache_across_call_sites_for_equivalent_lowerings() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 4);

        assert_eq!(engine.cached_program_count(), 0);
        // Cache identity is the complete lowering plus options and target, not the Rust source location.
        let _: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let _: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type, &engine, mesh).unwrap();
        assert_eq!(engine.cached_program_count(), 1);
    }

    #[test]
    fn test_compile_reuses_lowering_when_unused_input_metadata_differs() {
        // Unused input metadata that is not a parameter does not partition compilation when the complete lowering is
        // identical.

        #[derive(Parameterized, Debug, Clone, PartialEq, Eq, Hash)]
        #[ryft(crate = "ryft_core")]
        struct HyperparamInput<P: Parameter> {
            array: P,
            batch_size: usize,
        }

        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let array_type = replicated_vector_type(&mesh, 4);

        for batch_size in [32usize, 64usize] {
            let input = HyperparamInput { array: array_type.clone(), batch_size };
            let _: CompiledFunction<HyperparamInput<ArrayType>, ArrayType> =
                compile(|input| input.array.sin().unwrap(), input, &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cached_program_count(), 1);
    }

    #[test]
    fn test_compile_rematerialized_function_with_imported_session_constants() {
        type Body = Tracer<TracingContext<Constant, Operation>>;
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 3);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                r#type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let read = |value: Value| {
            read_first_shard::<f32>(client, &ValueProjection::<ArrayType>::into_projected(value).unwrap())
        };

        // XLA's elementary functions may differ from the host ones in the last bit.
        let assert_close = |actual: Vec<f32>, expected: Vec<f32>| {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual.iter().zip(&expected).all(|(actual, expected)| (actual - expected).abs() < 1e-6),
                "expected {expected:?} but got {actual:?}",
            );
        };

        // `x ↦ sin(x · w)` closes over the runtime array `w`, which every trace of its body imports as a session
        // constant through `ThroughSession`. The constant is not differentiated, so the gradient is `cos(x · w) w`.
        let x = [0.1f32, 0.2, 0.3];
        let w = [2.0f32, -1.0, 0.5];
        let dot = x.iter().zip(&w).map(|(x, w)| x * w).sum::<f32>();
        let gradient = w.iter().map(|w| dot.cos() * w).collect::<Vec<_>>();
        let weights = array(&vector_type, &w);
        let function = rematerialize(|x: Body| {
            let w = ImportingContext::<Session, ThroughSession>::import(x.context(), Value::Array(weights.clone()))?;
            let x = ValueProjection::<ArrayType>::into_projected(x)?;
            let w = ValueProjection::<ArrayType>::into_projected(w)?;
            Ok(x.dot(&w, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.sin()?.into_value())
        });

        // Eager calls and eager differentiation resolve the session constant on the session.
        assert_close(read(function.call(Value::Array(array(&vector_type, &x))).unwrap()), vec![dot.sin()]);
        let (value, eager_gradient) = engine
            .differentiate_at(Value::Array(array(&vector_type, &x)))
            .value_and_gradient(|x| function.call(x))
            .unwrap();
        assert_close(read(value), vec![dot.sin()]);
        assert_close(read(eager_gradient), gradient.clone());

        // A compiled pullback receives the session constant as a hidden argument of its executable.
        let compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(x, cotangent)| {
                let x = x.into_value();
                let context = x.context().clone();
                let (output, pullback) = context.differentiate_at(x).vjp(|x| function.call(x)).unwrap();
                let gradient = pullback.apply_materialized(cotangent.into_value()).unwrap();
                (
                    ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                )
            },
            (vector_type.clone(), scalar_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let (output, compiled_gradient) = compiled
            .executable()
            .execute(&engine, (array(&vector_type, &x), array(&scalar_type, &[2.0])))
            .unwrap();
        assert_close(read_first_shard::<f32>(client, &output), vec![dot.sin()]);
        assert_close(
            read_first_shard::<f32>(client, &compiled_gradient),
            gradient.iter().map(|gradient| 2.0 * gradient).collect(),
        );
    }

    #[test]
    fn test_compile_applies_trace_local_pushforward_under_batch() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let vector_type = replicated_vector_type(&mesh, 2);
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();

        // `x ↦ sin(x) * x` is linearized inside the compilation, so its residuals are tracers of that trace. Applying
        // the pushforward under a batching level of the same trace imports them as replicated values of that level.
        let function: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
            |(x, tangents)| {
                let x = x.into_value();
                let context = x.context().clone();
                let (_, pushforward) = context
                    .differentiate_at(x)
                    .linearize(|x| {
                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                        Ok(x.sin()?.mul(&x)?.into_value())
                    })
                    .unwrap();
                let output = batch(
                    |tangent| {
                        let context = tangent.context().clone();
                        pushforward.apply_in_context(&context, tangent)
                    },
                    tangents.into_value(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::default(),
                )
                .unwrap();
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            (vector_type.clone(), matrix_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let x = [0.5f32, -1.0];
        let tangents = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0];
        let output = function
            .executable()
            .execute(
                &engine,
                (
                    Array::from_host_buffer(
                        &engine,
                        vector_type,
                        mesh.clone(),
                        None,
                        values_to_bytes::<f32>(&x).as_slice(),
                    )
                    .unwrap(),
                    Array::from_host_buffer(
                        &engine,
                        matrix_type,
                        mesh,
                        None,
                        values_to_bytes::<f32>(&tangents).as_slice(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        let derivative = x.map(|x| x.cos() * x + x.sin());
        let expected = tangents.iter().enumerate().map(|(index, tangent)| tangent * derivative[index % 2]);
        assert!(
            read_first_shard::<f32>(client, &output)
                .iter()
                .zip(expected)
                .all(|(observed, expected)| (observed - expected).abs() < 1e-5)
        );
    }

    #[test]
    fn test_compile_with_captures() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let input_type = replicated_vector_type(&mesh, 4);
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();

        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(compiled.source_program().captures().len(), 1);
        assert_eq!(
            compiled.source_program().to_program_with_lifted_captures().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}], %1:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] .
                let %2:f32[4][sharding={mesh<['x'=1:auto]>, [{}]}] = add %1 %0
                in (%2)"},
        );

        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();

        assert_eq!(read_first_shard::<f32>(client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_compile_with_captures_erases_zero_space_capture_argument() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let capture = Array::from_host_buffer(&engine, zero_type.clone(), mesh.clone(), None, []).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, _| captures[0].clone(),
            vec![capture],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&engine, input_type, mesh, None, values_to_bytes::<f32>(&[1.0])).unwrap();

        let output = compiled.executable().execute(&engine, input).unwrap();

        assert_eq!(output.r#type().as_ref(), &zero_type);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
        output.block_until_ready().unwrap();
    }

    #[test]
    fn test_compile_with_captures_keeps_captured_array_placement_over_auto_axes() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let bytes = values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5]);
        let captured = Array::from_host_buffer(
            &engine,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharded.clone()),
            bytes.as_slice(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), None, bytes.as_slice()).unwrap();

        // The captured array enters the executable at its runtime placement over the auto mesh axis rather than at
        // the replicated placement of its type, so it needs no gathering and the compiler shards the sum like it.
        let function: CompiledFunction<ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x.add(&captures[0]).unwrap(),
            vec![captured],
            input_type,
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let output = function.executable().execute(&engine, input).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&replicated));
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0],);
    }

    #[test]
    fn test_compile_with_options_donates_argument() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 3);
        let options = Options::new(mesh.clone()).with_donate(true);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0];
        let source = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

        // The output remains independently readable after the donating call returns. Donation is opaque from the host
        // side, because PJRT may reuse the input's device buffer for the output, but the public API only observes the
        // resulting `Array`.
        let observed = read_first_shard::<f32>(client, &output);
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_donation_arity_mismatch() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 2);
        // A tree of three donation flags does not match the one flat input of the function.
        let options = Options::new(mesh.clone()).with_donate((true, false, false));
        let result: Result<CompiledFunction<ArrayType, ArrayType>, Error> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(
            result,
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 1, actual_count: 3 })),
        ));
    }

    #[test]
    fn test_compile_with_options_input_shardings_override_replaces_input_sharding() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();

        // `input_type` carries the abstract shape and data type but a replicated sharding, which the `input_shardings`
        // override replaces with a sharding along `x` before tracing, so the compiled program shards the input across
        // both devices of the mesh.
        let shape = Shape::new(vec![Dimension::Static(4)]);
        let abstract_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let abstract_input_type =
            ArrayType::new(DataType::F32, shape.clone()).with_sharding(abstract_sharding).unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), abstract_input_type, &engine, options).unwrap();

        // Build the input array under the overridden sharding so it matches the executable's expected layout.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            ArrayType::new(DataType::F32, shape),
            mesh.clone(),
            Some(sharded),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

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
    fn test_compile_with_options_rejects_input_shardings_arity_mismatch() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 2);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // Two shardings for one flat input must fail.
        let options = Options::new(mesh).with_input_shardings(vec![sharding.clone(), sharding]);
        let result: Result<CompiledFunction<ArrayType, ArrayType>, Error> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(
            result,
            Err(Error::Compilation(CompilationError::ShardingCountMismatch { kind: "input", expected: 1, actual: 2 })),
        ));
    }

    #[test]
    fn test_compile_with_options_output_shardings_override_propagates_to_output_array() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let engine = Session::builder().client(client).build().unwrap();

        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape).with_sharding(replicated.clone()).unwrap();
        // Override the output sharding to shard along `x` across both devices, so the partitioner emits a fully
        // sharded output and the sharding metadata of the `Array` matches.
        let options = Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]);
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        // Types record no placement over auto mesh axes, so the staged output type stays replicated while the returned
        // Array carries the overridden sharding.
        assert_eq!(compiled.output_types()[0].sharding(), Some(&replicated));

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(output.r#type().sharding(), Some(&replicated));

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
    fn test_compile_with_options_rejects_output_shardings_arity_mismatch() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 2);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let options = Options::new(mesh).with_output_shardings(vec![sharding.clone(), sharding]);
        let result: Result<CompiledFunction<ArrayType, ArrayType>, Error> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(
            result,
            Err(Error::Compilation(CompilationError::ShardingCountMismatch { kind: "output", expected: 1, actual: 2 })),
        ));
    }

    #[test]
    fn test_stage_applies_input_shardings() {
        // Input sharding overrides change the input types before tracing begins.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        // The input type declares no sharding, so the sharding of the staged input can only come from the override.
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()));

        let options = Options::new(mesh).with_input_shardings(vec![sharding.clone()]);
        let staged: StagedFunction<ArrayType, ArrayType> = stage(|x| x.sin().unwrap(), input_type, options).unwrap();
        let Type::Array(staged_input_type) = &staged.source_program().program().input_types()[0] else {
            panic!("public staged input should remain an array");
        };
        assert_eq!(staged_input_type.sharding(), Some(&sharding));
        assert_eq!(engine.cached_program_count(), 0, "staging must not build an executable");
    }

    #[test]
    fn test_infer_output_types() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();
        let input_type = replicated_vector_type(&mesh, 7);

        let output_type: ArrayType = infer_output_types(|x| x.sin().unwrap(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `infer_output_types` must not have populated the compile cache.
        assert_eq!(engine.cached_program_count(), 0);
    }
}
