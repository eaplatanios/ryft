//! XLA function staging, retained JIT dispatch, compilation, and execution.
//!
//! This module wraps the lifecycle from [`ryft_core::compilation`] with XLA array projections and options. [`jit`]
//! creates a [`CompiledFunctionDispatcher`] that selects specializations from runtime inputs. Its first execution of a
//! specialization stages, lowers, and requests compilation, while warm executions reuse the retained executable.
//! [`compile`] stages and obtains a [`CompiledFunction`] for one explicit abstract signature immediately. Both paths
//! share the domain's [`CompilationContext`](ryft_core::CompilationContext) cache.
//!
//! [`stage`] returns a [`StagedFunction`] for functions that are composed into larger programs before compilation.
//! Staged functions register runtime captures in their retained capture table instead of embedding runtime arrays in
//! the IR. Runtime invocation uses [`CompiledFunctionDispatcher::execute`] or an [`ExecutableFunction`] obtained
//! through [`CompiledFunction::executable`]. [`ExecutableFunction::execute`] returns structured array outputs, and
//! [`ExecutableFunction::execute_async`] also retains whole-invocation completion. Staged functions, compiled
//! functions, and dispatchers can also be called from any context, which is how they compose with transforms and
//! with other compilations.
//!
//! [`jit_statefully`] and [`compile_statefully`] expose the heterogeneous boundary used by external array
//! references. Asynchronous stateful calls atomically publish read leases and pending mutation generations immediately
//! after submission, commit hidden final states without holding host mutexes through device execution, and return a
//! [`StatefulExecution`] that reports completion failures. [`StatefulCompiledFunctionDispatcher::execute_statefully`]
//! returns that completion, and [`StatefulCompiledFunctionDispatcher::execute_statefully_and_wait`] waits for it before
//! returning public outputs. [`StatefulCompiledFunction`] supplies the corresponding fixed-signature entry points.
//!
//! # Calling Compiled Functions
//!
//! [`StagedFunction::call`], [`CompiledFunction::call`], and [`CompiledFunctionDispatcher::call`] bind a `jit_call`
//! operation, which carries the retained source program of the called function, in the context that their inputs
//! belong to. These inputs and outputs are the `ArrayType` leaves of a [`ProjectedContext`] over an array IR context,
//! which the call recovers from the first input. They are [`Array`]s under the eager [`Domain`] and [`ProjectedValue`]s
//! of tracers under a trace or a transform (e.g., the [`CompileTracer`]s of an enclosing compilation). The
//! `call_in_context` variants take the array IR context explicitly, which also serves nullary functions. Every parent
//! context is supported:
//!
//!   - An eager parent executes the call immediately through eager dispatch, which lowers it on its first use rather
//!     than reusing the retained executable of a [`CompiledFunction`]. [`ExecutableFunction::execute`] and
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
//! import captures through [`ThroughSession`](crate::ThroughSession) instead. The [`Session`](crate::Session) of each
//! such array retains it for as long as a handle to it exists, and the body stages a lifetime-free
//! [`SessionConstant`](crate::experimental::ops::SessionConstant) handle to it, which keeps the callee closed. An
//! eager parent resolves the handle when it lifts it, and lowering passes every distinct handle of a program as a
//! hidden argument of the entry function after its captures, so the executable depends only on the types of these
//! arrays. Composite bodies and rules must be `'static`, so they can only call compiled functions of a `'static`
//! client (i.e., `CompiledFunction<'static, ..>`).
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
//! automatic differentiation and preserved-kernel semantics remain under development. Ordinary array-only staging,
//! compilation, and execution are unaffected.
//!
//! Reference lowering opens each external reference as an ordinary array input. A read-only reference has no hidden
//! result. A mutated reference gains one hidden final-state output after the public output prefix, and the runtime
//! commits that value atomically into the same reference after execution. The entry argument also receives a
//! `tf.aliasing_output` relation to the hidden result. That relation is a non-semantic may-alias hint: it permits, but
//! never requires, backend buffer reuse, and it never promises physical in-place mutation. Reference-state inputs are
//! themselves never donated, so each reference retains the snapshot it passed in, and correctness always comes from
//! committing the returned final value. Read snapshots and retained initializers consequently remain immutable even
//! when a backend chooses to reuse storage.
//!
//! Calls reject unsupported state before launch. The admitted boundary is device-memory state on a mesh in which the
//! calling process participates (i.e., every device is visible to its client under the process that owns it, and at
//! least one is addressable by this process): static references may be replicated or sharded with identical
//! input/final-state sharding; finite bounded-dynamic references are read-only and replicated, and only on a fully
//! addressable, single-process mesh. Over a mesh that spans processes (multi-host execution), every process calls with
//! its own reference over the same mesh, binds and commits only its addressable shards, and so each process commits its
//! own part of the final state; as for any multi-controller output, a failure on one process does not roll back the
//! commits of the others. Zero-space, host-memory, unbounded-dynamic, dynamic nonreplicated, input-bucketed,
//! foreign/unknown-device or non-participating-mesh, and bounded-dynamic mutation cases are rejected. Reference
//! arguments must be distinct root handles. Internal accesses carry transform paths and dynamic indices, which
//! discharge rewrites as array operations before compilation. Views are not separate program values and cannot cross
//! this boundary.
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
//! that use the same mutable reference dependency-chain their generations; read-only calls publish leases and may
//! overlap. Dropping [`StatefulExecution`] does not cancel work. Awaiting it reports whole-invocation failure. A
//! failure between submission handoff and hidden-state replacement commit poisons the complete affected mutation
//! group, so later reference access cannot silently observe stale state. Poisoning is terminal there because no buffer
//! can be proven current after that boundary, and recovery requires constructing a new reference from independently
//! trusted state. Once the hidden final-state replacements are committed, the pending generations are disarmed, so a
//! later failure—such as public-output reconstruction—surfaces as an ordinary error while the committed reference
//! state stays valid.
//!
//! Local references—including local views and references inside supported conditions, bounded loops, scans, and
//! nested calls—use the ordinary [`jit`] or [`compile`] surface: they are discharged to array SSA before StableHLO
//! lowering and do not change the public ABI. Local-reference batching and automatic differentiation likewise
//! discharge first. Rematerialization recomputes local reference lifecycles inside its body, saves reads of external
//! references, and never recomputes their mutations. Automatic differentiation of external references (e.g.,
//! `value_and_gradient`, `linearize`, or `jvp` with reference tangents) and batching over mapped references, or over
//! shared references that are only read, behave inside [`compile_statefully`] and [`jit_statefully`] exactly as they
//! do without a compilation boundary, including around nested calls of functions that take references. Batching that
//! stores batched values into a shared reference is rejected on both paths. Custom-derivative rule references and
//! preserved references outside the experimental kernel boundary are rejected explicitly.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use ryft_core::{
    ArrayIrBatchingPolicy, ArrayIrType, ArrayType, Batch, BatchAxis, BatchAxisSpecification, BatchingError,
    BatchingTracer, ClosedProgram, Compile, CompiledFunction as CoreCompiledFunction,
    CompiledFunctionDispatcher as CoreCompiledFunctionDispatcher, ConstrainSharding, Context, ContextProjection,
    DeviceMesh, DifferentiableType, Dimension as CoreDimension, DimensionValue, DispatcherStatistics,
    DomainTracingContext, ExecutableFunction as CoreExecutableFunction, ForwardModeDifferentiate, ImportingContext,
    MeshAxisType, MeshContext, NamedAxes, Parameter, Parameterized, ParameterizedFamily, ProgramError,
    ProjectedContext, ProjectedValue, Reshard, ReverseModeDifferentiate, Sharding, ShardingDimension,
    StagedFunction as CoreStagedFunction, StatefulExecution, Tracer as CoreTracer, Typed, Value as CoreValue,
    ValueProjection,
};
use ryft_pjrt::Execution;

use crate::domains::execution::{validate_input_type, validate_runtime_outputs};
use crate::domains::{CompiledProgram, with_input_shardings};
use crate::experimental::ops::{Constant, Operation};
use crate::{
    AdaptiveProfileGuidedFunction, AdaptiveProfileGuidedOptions, Array, CompilationError, Domain, Error,
    ExecutionError, Options, Value,
};

/// Array IR tracer retained by the experimental stateful XLA program.
///
/// # Stability
///
/// This alias is experimental while the external-reference ABI is evolving.
pub type StatefulCompileTracer<'c> = CoreTracer<DomainTracingContext<Domain<'c>, Value<'c>>>;

/// Tracer leaf exposed by the public array-only XLA compilation facade.
pub type CompileTracer<'c> = ProjectedValue<ArrayType, StatefulCompileTracer<'c>>;

/// Internal reparameterization of one public array parameter tree into the production array IR type family.
type ProgramParameters<P> = <P as Parameterized<ArrayType>>::To<ArrayIrType>;

/// Internal value tree corresponding to one public array parameter tree.
type ProgramParameterValues<P, V> = <ProgramParameters<P> as Parameterized<ArrayIrType>>::To<V>;

/// Captured-constant tree corresponding to one public array parameter tree.
type ProgramConstants<P> = ProgramParameterValues<P, Constant>;

/// Heterogeneous XLA runtime value used by the stateful compilation facade.
///
/// # Stability
///
/// This alias is experimental while the external-reference ABI is evolving.
pub type StatefulValue<'c> = Value<'c>;

/// Projects one internal array IR tracer tree to the public array-only tracer tree.
fn project_tracers<'c, P>(
    values: ProgramParameterValues<P, StatefulCompileTracer<'c>>,
) -> Result<P::To<CompileTracer<'c>>, Error>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<ArrayIrType>,
{
    let structure = values.parameter_structure();
    let parameters = values
        .into_parameters()
        .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    P::To::<CompileTracer<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Lifts one public array-only tracer tree back into the internal array IR tracer tree.
fn lift_tracers<'c, P>(
    values: P::To<CompileTracer<'c>>,
) -> Result<ProgramParameterValues<P, StatefulCompileTracer<'c>>, Error>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
{
    let structure = values.parameter_structure();
    let parameters = values.into_parameters().map(ProjectedValue::into_value);
    ProgramParameterValues::<P, StatefulCompileTracer<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Lifts one public runtime array tree into the production array IR value family.
fn lift_arrays<'c, P>(values: P::To<Array<'c>>) -> Result<ProgramParameterValues<P, StatefulValue<'c>>, Error>
where
    P: Parameterized<ArrayType>,
    P::Family:
        ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Array<'c>> + ParameterizedFamily<StatefulValue<'c>>,
{
    let structure = values.parameter_structure();
    let parameters = values.into_parameters().map(Value::Array);
    ProgramParameterValues::<P, StatefulValue<'c>>::from_parameters(structure, parameters)
        .map_err(ProgramError::from)
        .map_err(Into::into)
}

/// Projects one production array IR output tree back to public runtime arrays.
fn project_arrays<'c, P>(values: ProgramParameterValues<P, StatefulValue<'c>>) -> Result<P::To<Array<'c>>, Error>
where
    P: Parameterized<ArrayType>,
    P::Family:
        ParameterizedFamily<ArrayIrType> + ParameterizedFamily<StatefulValue<'c>> + ParameterizedFamily<Array<'c>>,
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

/// Projects one array IR output tree of any context to the `ArrayType` projection of its values.
fn project_values<P, V>(
    values: ProgramParameterValues<P, V>,
) -> Result<P::To<<V as ValueProjection<ArrayType>>::Projected>, ProgramError>
where
    P: Parameterized<ArrayType>,
    P::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<V>
        + ParameterizedFamily<<V as ValueProjection<ArrayType>>::Projected>,
    V: CoreValue<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Parameter>,
{
    let structure = values.parameter_structure();
    let parameters = values
        .into_parameters()
        .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(P::To::<<V as ValueProjection<ArrayType>>::Projected>::from_parameters(structure, parameters)?)
}

/// Stages the transition of `value` to the placement `sharding` that a sharding override of a compiled function
/// boundary requests. A `reshard` brings the placement that types record (i.e., the placement over explicit mesh axes)
/// to that of `sharding`, and a `constrain_sharding` then constrains the placement over the auto mesh axes that
/// `sharding` names, which types never record. Afterwards, the type of `value` carries `sharding` without its auto
/// mesh axes, which is also the type that the corresponding input or output of the compiled function has.
fn place<V>(value: V, sharding: &Sharding) -> Result<V, ProgramError>
where
    V: CoreValue<Type = ArrayIrType> + Reshard + ConstrainSharding,
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
    V: CoreValue<Type = ArrayIrType> + Reshard + ConstrainSharding,
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
            ArrayIrType::Array(_) => place(output, sharding),
            ArrayIrType::Dimension(_) | ArrayIrType::Reference(_) => Ok(output),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(P::from_parameters(structure, outputs).map_err(ProgramError::from)?)
}

/// Brings each of the flat array `inputs` of a call to the placement of the corresponding input of the called function
/// (refer to `place`): to the sharding override in `overrides`, if any, which may also place it over auto mesh axes,
/// and otherwise to the callee input type if the input type differs from it only in its sharding. The call thereby
/// constrains its inputs to the placement of the called function in the context of the caller, which composes with
/// transforms, much like an execution reshards concrete inputs. Other inputs are returned unchanged, so that binding
/// the call reports any other mismatch.
fn place_inputs<V>(
    inputs: Vec<V>,
    input_types: &[ArrayIrType],
    overrides: Option<&[Sharding]>,
) -> Result<Vec<V>, ProgramError>
where
    V: CoreValue<Type = ArrayIrType> + Reshard + ConstrainSharding,
{
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            if let Some(sharding) = overrides.and_then(|overrides| overrides.get(index))
                && matches!(input.r#type().as_ref(), ArrayIrType::Array(_))
            {
                return place(input, sharding);
            }
            let target = match (input.r#type().as_ref(), input_types.get(index)) {
                (ArrayIrType::Array(actual), Some(ArrayIrType::Array(expected))) if actual != expected => {
                    expected.sharding().filter(|&sharding| {
                        actual.clone().with_sharding(sharding.clone()).is_ok_and(|actual| &actual == expected)
                    })
                }
                _ => None,
            }
            .cloned();
            match target {
                Some(sharding) => place(input, &sharding),
                None => Ok(input),
            }
        })
        .collect()
}

/// Retained array-only XLA JIT dispatcher with explicit host-side static parameters.
///
/// [`Self::execute`] runs the dispatcher on concrete arrays. Its first execution for each `(static parameters, dynamic
/// parameter structure, dynamic abstract types)` specialization stages, lowers, and requests compilation, and warm
/// executions dispatch directly to the retained executable. Static values should be low-cardinality configuration such
/// as axes, shapes, or Boolean branch choices, while arrays remain dynamic.
///
/// [`Self::call`] instead calls the closure as a `jit_call` operation in any context, including eager, staging, and
/// transform contexts. Executables retain no source program, so a call traces the closure again on demand into a source
/// specialization, which is additionally keyed by the named axes and default mesh of the calling context and retained
/// in a cache separate from the executables. Transformed calls therefore never evict executables, and warm executions
/// never trace. Host side effects of the closure run once per executable specialization and once per source
/// specialization. Both kinds of specialization stage the output sharding overrides of the dispatcher's [`Options`],
/// and calls bring their inputs to the input sharding overrides first, so that calls and executions agree on the
/// placement of values at the boundary (refer to the [module documentation](crate::compilation)).
pub struct CompiledFunctionDispatcher<
    'c,
    F,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
> where
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Array IR domain dispatcher hidden behind the public array projection boundary.
    function: CoreCompiledFunctionDispatcher<Domain<'c>, Static, ProgramParameters<In>, ProgramParameters<Out>, F>,
}

impl<'c, F, Static, In, Out> Clone for CompiledFunctionDispatcher<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    CoreCompiledFunctionDispatcher<Domain<'c>, Static, ProgramParameters<In>, ProgramParameters<Out>, F>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, F, Static, In, Out> CompiledFunctionDispatcher<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<Array<'c>>
        + ParameterizedFamily<StatefulValue<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<Array<'c>>
        + ParameterizedFamily<StatefulValue<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
{
    /// Executes a specialization with public array inputs, compiling it on a miss and reusing it on warm invocations.
    /// The array IR runtime outputs are projected back to the input/output container family's array representation.
    pub fn execute(&self, static_parameters: Static, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, Error>
    where
        F: Fn(
            Static,
            ProgramParameterValues<In, StatefulCompileTracer<'c>>,
        ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer<'c>>, Error>,
        ProgramParameters<In>: Parameterized<ArrayIrType, To<ArrayIrType> = ProgramParameters<In>>,
        ProgramParameters<Out>: Parameterized<ArrayIrType, To<ArrayIrType> = ProgramParameters<Out>>,
        ProgramParameterValues<In, StatefulValue<'c>>:
            Parameterized<StatefulValue<'c>, To<ArrayIrType> = ProgramParameters<In>>,
        ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
                StatefulCompileTracer<'c>,
                To<ArrayIrType> = ProgramParameters<Out>,
                To<Constant> = ProgramConstants<Out>,
            >,
    {
        let inputs = lift_arrays::<In>(inputs)?;
        project_arrays::<Out>(self.function.execute(static_parameters, inputs)?)
    }

    /// Calls the staged specialization of this dispatcher for `static_parameters` and the abstract types of `inputs` as
    /// a `jit_call` operation in the array IR context recovered from `inputs`. Unlike [`Self::execute`], it works under
    /// any context, including eager, staging, and transform contexts, so this dispatcher's closure composes with
    /// enclosing transforms and compilations. Inputs and outputs are `ArrayType` values of a [`ProjectedContext`] over
    /// that context (e.g., [`Array`]s under the eager [`Domain`] and [`ProjectedValue`]s under traces and transforms),
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
            ProgramParameterValues<In, StatefulCompileTracer<'c>>,
        ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer<'c>>, Error>,
        L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
        C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>
            + ContextProjection<ArrayType>
            + ImportingContext<Domain<'c>, Path>
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
    /// Returns an error when the domain rejects the staged input types, when tracing the closure fails, when the same
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
            ProgramParameterValues<In, StatefulCompileTracer<'c>>,
        ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer<'c>>, Error>,
        C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>
            + ImportingContext<Domain<'c>, Path>
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
        let inputs = match self.function.options().input_shardings.as_deref() {
            None => inputs,
            Some(overrides) => {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                place_inputs(inputs, &with_input_shardings(input_types, self.function.options())?, Some(overrides))?
            }
        };
        let inputs = ProgramParameterValues::<In, C::Value>::from_parameters(input_structure, inputs)
            .map_err(ProgramError::from)?;
        let outputs = self.function.call_in_context(context, static_parameters, inputs)?;
        Ok(project_values::<Out, C::Value>(outputs)?)
    }
}

impl<'c, F, Static, In, Out> CompiledFunctionDispatcher<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
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
pub fn jit_with_options<'c, F, Static, In, Out>(
    function: F,
    domain: &Domain<'c>,
    options: Options,
) -> CompiledFunctionDispatcher<
    'c,
    impl Fn(
        Static,
        ProgramParameterValues<In, StatefulCompileTracer<'c>>,
    ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer<'c>>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<CompileTracer<'c>>) -> Result<Out::To<CompileTracer<'c>>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileTracer<'c>>,
{
    // Executable and staged specializations both trace this closure, so both stage the output sharding overrides.
    let output_shardings = options.output_shardings.clone();
    let function = move |static_parameters, inputs| {
        let outputs = function(static_parameters, project_tracers::<In>(inputs)?)?;
        place_outputs(lift_tracers::<Out>(outputs)?, output_shardings.as_deref())
    };
    CompiledFunctionDispatcher { function: domain.jit_with_options(function, options) }
}

/// Constructs a retained dispatcher for `function` on `mesh` with otherwise default options. Refer to
/// [`jit_with_options`] for more information.
///
/// `function` is fallible so that it can reject its inputs while it is traced, and an infallible function returns its
/// outputs wrapped in [`Ok`].
#[inline]
pub fn jit<'c, F, Static, In, Out>(
    function: F,
    domain: &Domain<'c>,
    mesh: DeviceMesh,
) -> CompiledFunctionDispatcher<
    'c,
    impl Fn(
        Static,
        ProgramParameterValues<In, StatefulCompileTracer<'c>>,
    ) -> Result<ProgramParameterValues<Out, StatefulCompileTracer<'c>>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<CompileTracer<'c>>) -> Result<Out::To<CompileTracer<'c>>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileTracer<'c>>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileTracer<'c>>,
{
    jit_with_options(function, domain, Options::new(mesh))
}

/// Retained heterogeneous XLA JIT dispatcher whose calls may bind external references.
///
/// The dispatcher retains one compiled specialization per static-parameter and input-type combination, exactly like
/// the array-only JIT surface, but its boundary is the heterogeneous [`StatefulValue`] family: reference-typed
/// leaves bind live [`ryft_core::ArrayReference`] values at call time instead of immutable arrays. Every reference
/// leaf must be an unrenamed root handle whose type refines the effective compiled boundary, including its declared
/// sharding and memory. A read-only bounded-dynamic reference may refine its finite replicated declaration. The runtime
/// snapshots every reference, releases its state locks at the backend submission handoff, and publishes
/// completion-bearing read leases or hidden final states before public outputs are exposed.
///
/// # Stability
///
/// This external-state surface is experimental. Ordinary [`CompiledFunctionDispatcher`] calls remain the stable
/// array-only boundary.
pub struct StatefulCompiledFunctionDispatcher<
    'c,
    F,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayIrType>,
    Out: Parameterized<ArrayIrType>,
> where
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Constant>,
{
    /// Backend-neutral dispatcher retaining heterogeneous runtime values and compiled specializations.
    function: CoreCompiledFunctionDispatcher<Domain<'c>, Static, In, Out, F>,
}

impl<'c, F, Static, In, Out> Clone for StatefulCompiledFunctionDispatcher<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayIrType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>,
    Out: Parameterized<ArrayIrType>,
    Out::Family: ParameterizedFamily<Constant>,
    CoreCompiledFunctionDispatcher<Domain<'c>, Static, In, Out, F>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, F, Static, In, Out> StatefulCompiledFunctionDispatcher<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayIrType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulValue<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::To<StatefulValue<'c>>: Parameterized<
            StatefulValue<'c>,
            Family = In::Family,
            ParameterStructure = In::ParameterStructure,
            To<ArrayIrType> = In,
        >,
    Out: Parameterized<ArrayIrType>,
    Out::Family: ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulValue<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<StatefulValue<'c>>:
        Parameterized<StatefulValue<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    Out::To<StatefulCompileTracer<'c>>:
        Parameterized<StatefulCompileTracer<'c>, To<ArrayIrType> = Out, To<Constant> = Out::To<Constant>>,
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
        inputs: In::To<StatefulValue<'c>>,
    ) -> Result<Out::To<StatefulValue<'c>>, Error>
    where
        F: Fn(Static, In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    {
        self.function.execute_statefully_and_wait(static_parameters, inputs)
    }

    /// Returns the completion-bearing stateful call shape for one retained specialization.
    ///
    /// The returned [`StatefulExecution`] may already carry reconstructed public outputs backed by pending device
    /// values; awaiting it observes whole-invocation completion, including asynchronous execution errors that would
    /// otherwise surface only when the affected reference is next accessed. Dropping the result never cancels
    /// submitted work.
    pub fn execute_statefully(
        &self,
        static_parameters: Static,
        inputs: In::To<StatefulValue<'c>>,
    ) -> StatefulExecution<Out::To<StatefulValue<'c>>, Error>
    where
        F: Fn(Static, In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    {
        self.function.execute_statefully(static_parameters, inputs)
    }
}

impl<'c, F, Static, In, Out> StatefulCompiledFunctionDispatcher<'c, F, Static, In, Out>
where
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<ArrayIrType>,
    In::ParameterStructure: Eq + Hash,
    In::Family: ParameterizedFamily<Constant>,
    Out: Parameterized<ArrayIrType>,
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
/// This external-state entry point is experimental.
pub fn jit_statefully_with_options<'c, F, Static, In, Out>(
    function: F,
    domain: &Domain<'c>,
    options: Options,
) -> StatefulCompiledFunctionDispatcher<
    'c,
    impl Fn(Static, In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<
            ArrayIrType,
            Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
        >,
    In::ParameterStructure: Eq + Hash,
    Out: Parameterized<
            ArrayIrType,
            Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
        >,
{
    let output_shardings = options.output_shardings.clone();
    let function = move |static_parameters, inputs| {
        place_outputs(function(static_parameters, inputs)?, output_shardings.as_deref())
    };
    StatefulCompiledFunctionDispatcher { function: domain.jit_with_options(function, options) }
}

/// Constructs a retained stateful dispatcher for a heterogeneous XLA `function` on `mesh` with otherwise default
/// options.
///
/// `function` is fallible so that it can reject its inputs while it is traced, and an infallible function returns its
/// outputs wrapped in [`Ok`].
///
/// # Stability
///
/// This external-state entry point is experimental.
#[inline]
pub fn jit_statefully<'c, F, Static, In, Out>(
    function: F,
    domain: &Domain<'c>,
    mesh: DeviceMesh,
) -> StatefulCompiledFunctionDispatcher<
    'c,
    impl Fn(Static, In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    Static,
    In,
    Out,
>
where
    F: Fn(Static, In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    Static: Clone + Debug + Eq + Hash,
    In: Parameterized<
            ArrayIrType,
            Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
        >,
    In::ParameterStructure: Eq + Hash,
    Out: Parameterized<
            ArrayIrType,
            Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
        >,
{
    jit_statefully_with_options(function, domain, Options::new(mesh))
}

/// Compiled heterogeneous XLA function whose invocation may bind external references.
///
/// This is the fixed-signature counterpart of [`StatefulCompiledFunctionDispatcher`]. Its reference-typed input leaves
/// bind unrenamed root [`ryft_core::ArrayReference`] handles whose types refine the effective compiled boundary,
/// including its declared sharding and memory. Read-only bounded-dynamic references may refine finite replicated
/// declarations. Each invocation uses the same snapshot, generation-revalidation, asynchronous publication, and
/// completion protocol as the retained dispatcher.
///
/// # Stability
///
/// This external-state surface is experimental. Ordinary [`CompiledFunction`] calls remain the stable array-only
/// boundary.
pub struct StatefulCompiledFunction<'c, In: Parameterized<ArrayIrType>, Out: Parameterized<ArrayIrType>>
where
    In::Family: ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<Constant>,
{
    /// Backend-neutral compiled function retaining the array IR runtime boundary.
    function: CoreCompiledFunction<Domain<'c>, In, Out>,
}

impl<'c, In, Out> StatefulCompiledFunction<'c, In, Out>
where
    In: Parameterized<ArrayIrType>,
    Out: Parameterized<ArrayIrType>,
    In::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulValue<'c>>,
    Out::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulValue<'c>>,
    In::To<StatefulValue<'c>>: Parameterized<StatefulValue<'c>>,
    Out::To<StatefulValue<'c>>:
        Parameterized<StatefulValue<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
{
    /// Returns the runtime-only executable handle.
    #[inline]
    pub fn executable(&self) -> &CoreExecutableFunction<Domain<'c>, In, Out> {
        self.function.executable()
    }

    /// Executes synchronously and returns public outputs after the reference transaction completes.
    ///
    /// The call waits for the completion-bearing protocol after pending reference state and read leases are published.
    /// A failed completion therefore cannot leave silently stale reference state behind. Refer to the documentation of
    /// [`StatefulCompiledFunctionDispatcher::execute_statefully_and_wait`] for the complete transaction contract.
    pub fn execute_statefully_and_wait(
        &self,
        domain: &Domain<'c>,
        inputs: In::To<StatefulValue<'c>>,
    ) -> Result<Out::To<StatefulValue<'c>>, Error> {
        self.function.executable().execute_statefully_and_wait(domain, inputs)
    }

    /// Submits a stateful call and returns its non-cancelling completion-bearing result.
    ///
    /// Refer to the documentation of [`StatefulCompiledFunctionDispatcher::execute_statefully`] for the completion and
    /// error-observation contract.
    pub fn execute_statefully(
        &self,
        domain: &Domain<'c>,
        inputs: In::To<StatefulValue<'c>>,
    ) -> StatefulExecution<Out::To<StatefulValue<'c>>, Error> {
        self.function.executable().execute_statefully(domain, inputs)
    }

    /// Returns the public flat output types, excluding the hidden final-state output suffix that reference
    /// discharge appends for mutated external references.
    #[inline]
    pub fn output_types(&self) -> Cow<'_, [ArrayIrType]> {
        self.function.executable().output_types()
    }
}

/// Compiles one heterogeneous signature for stateful invocation.
///
/// # Stability
///
/// This external-state entry point is experimental.
pub fn compile_statefully<'domain, 'c: 'domain, F, In, Out>(
    function: F,
    input_types: In,
    domain: &'domain Domain<'c>,
    options: Options,
) -> Result<StatefulCompiledFunction<'c, In, Out>, Error>
where
    F: FnOnce(In::To<StatefulCompileTracer<'c>>) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    In: Parameterized<ArrayIrType>,
    In::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out: Parameterized<ArrayIrType>,
    Out::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<StatefulCompileTracer<'c>>:
        Parameterized<StatefulCompileTracer<'c>, To<ArrayIrType> = Out, To<Constant> = Out::To<Constant>>,
{
    compile_statefully_with_captures(move |_, inputs| function(inputs), Vec::new(), input_types, domain, options)
}

/// Compiles one heterogeneous signature with explicit array or reference captures for stateful invocation.
///
/// # Stability
///
/// This external-state entry point is experimental. A captured reference participates in the same reference
/// transaction as a public reference input and cannot alias any other capture or public argument in the invocation.
pub fn compile_statefully_with_captures<'domain, 'c: 'domain, F, In, Out>(
    function: F,
    captures: Vec<StatefulValue<'c>>,
    input_types: In,
    domain: &'domain Domain<'c>,
    options: Options,
) -> Result<StatefulCompiledFunction<'c, In, Out>, Error>
where
    F: FnOnce(
        Vec<StatefulCompileTracer<'c>>,
        In::To<StatefulCompileTracer<'c>>,
    ) -> Result<Out::To<StatefulCompileTracer<'c>>, Error>,
    In: Parameterized<ArrayIrType>,
    In::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out: Parameterized<ArrayIrType>,
    Out::Family: ParameterizedFamily<Constant> + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<StatefulCompileTracer<'c>>:
        Parameterized<StatefulCompileTracer<'c>, To<ArrayIrType> = Out, To<Constant> = Out::To<Constant>>,
{
    let input_types = with_input_shardings(input_types, &options)?;
    let output_shardings = options.output_shardings.clone();
    let staged = CoreStagedFunction::<Domain<'_>, In, Out>::stage(
        |_, capture_tracers, inputs| place_outputs(function(capture_tracers, inputs)?, output_shardings.as_deref()),
        captures,
        input_types,
        options,
    )?;
    let lowered = staged.lower(&domain)?;
    Ok(StatefulCompiledFunction { function: lowered.compile(&domain)? })
}

/// Staged XLA function that has not been compiled. Returned by [`stage`] and [`stage_with_captures`].
///
/// Holds the traced source [`Program`](ryft_core::programs::Program) of one closure together with its captured
/// runtime [`Array`]s and input and output type metadata, without compiling a PJRT executable. This is the right
/// handle for functions that are only composed into larger programs. [`Self::call`] calls it in any context as a
/// `jit_call` boundary that carries its source program (refer to the [module documentation](crate::compilation)),
/// and [`Domain::compile_staged_function`] produces a [`CompiledFunction`] when an executable is needed. Executable
/// cache identity is derived from the complete lowered computation and compile-relevant backend state, so equivalent
/// lowerings can share an executable even when they were produced at different Rust call sites.
pub struct StagedFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Backend-neutral staged function carrying the source program, captures, structures, and retained options.
    function: CoreStagedFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for StagedFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> StagedFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Returns the flat staged source [`Program`](ryft_core::Program) together with its captured runtime values.
    /// Use it for program-level transforms (e.g., transposition or partial evaluation of the program itself) and for
    /// diagnostics such as printed IR, instruction counts, and graph rendering. Value-level transforms instead call
    /// this function through [`Self::call`].
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<StatefulValue<'c>, Constant, Operation, Vec<Constant>, Vec<Constant>> {
        self.function.source_program()
    }

    /// Calls this function as a `jit_call` operation in the array IR context `C` recovered from `inputs`.
    ///
    /// The call binds a boundary that carries this function's retained source program, so enclosing transforms
    /// rewrite it through the ordinary rules of the `jit_call` operation. Inputs and outputs are `ArrayType` values of
    /// a [`ProjectedContext`] over `C`, which the call recovers from the first input. Under the eager [`Domain`], they
    /// are [`Array`]s and the call executes immediately through eager dispatch. Under a trace or a transform (e.g.,
    /// forward mode differentiation), they are [`ProjectedValue`]s of its tracers and the call is recorded or
    /// rewritten there. The captured runtime arrays of this function enter `C` through [`ImportingContext`], which
    /// treats them as values that do not vary across a batch, have structurally zero tangents, and are known to partial
    /// evaluation. Traces over constants alone (e.g., the bodies of [`rematerialize`](ryft_core::rematerialize) and
    /// [`composite`](ryft_core::composite)) import them as session constants instead (refer to
    /// [`ThroughSession`](crate::ThroughSession)). An input whose type differs from the corresponding input type of
    /// this function only in its sharding is first brought to that type in `C` (refer to the
    /// [module documentation](crate::compilation)). Use [`Self::call_in_context`] for nullary functions, because no
    /// input exists from which to recover the context.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `inputs` has no leaves, and an error when importing a capture,
    /// binding the call, or reassembling the structured outputs fails.
    #[inline]
    pub fn call<L, C, Path>(&self, inputs: In::To<L>) -> Result<Out::To<L>, ProgramError>
    where
        L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
        C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>
            + ContextProjection<ArrayType>
            + ImportingContext<Domain<'c>, Path>,
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
        C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation> + ImportingContext<Domain<'c>, Path>,
        C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
        In::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
        Out::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    {
        call_staged_in_context::<In, Out, C, Path>(&self.function, context, inputs)
    }

    /// Consumes this facade and returns its array IR staged artifact for crate-internal verification.
    #[cfg(test)]
    pub(crate) fn into_inner(self) -> CoreStagedFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>> {
        self.function
    }
}

/// Calls `function` as a `jit_call` operation in `context` on behalf of [`StagedFunction::call_in_context`] and
/// [`CompiledFunction::call_in_context`]. Inputs whose type differs from the input type of `function` only in its
/// sharding are first brought to that type in `context` (refer to `place_inputs`).
fn call_staged_in_context<'c, In, Out, C, Path>(
    function: &CoreStagedFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>>,
    context: &C,
    inputs: In::To<<C::Value as ValueProjection<ArrayType>>::Projected>,
) -> Result<Out::To<<C::Value as ValueProjection<ArrayType>>::Projected>, ProgramError>
where
    In: Parameterized<ArrayType>,
    In::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<C::Value>
        + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    Out: Parameterized<ArrayType>,
    Out::Family: ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<C::Value>
        + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation> + ImportingContext<Domain<'c>, Path>,
    C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
{
    let input_structure = inputs.parameter_structure();
    let inputs = inputs.into_parameters().map(<C::Value as ValueProjection<ArrayType>>::from_projected).collect();
    let inputs = ProgramParameterValues::<In, C::Value>::from_parameters(
        input_structure,
        place_inputs(inputs, function.input_types(), function.options().input_shardings.as_deref())?,
    )?;
    project_values::<Out, C::Value>(function.call_in_context(context, inputs)?)
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
    From<CoreStagedFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>>> for StagedFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    #[inline]
    fn from(function: CoreStagedFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>>) -> Self {
        Self { function }
    }
}

/// Runtime-only XLA executable handle.
///
/// Unlike [`CompiledFunction`], this type does not retain the `Arc`-backed staged program or lowering metadata used
/// by transforms. It can only execute and inspect its runtime signature, so it cannot be called inside traces or
/// transforms, which require [`CompiledFunction::call`] and the source program that it carries. Its thread safety is
/// derived from its backend state, and no unsafe blanket implementation is used.
pub struct ExecutableFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    function: CoreExecutableFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for ExecutableFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { function: self.function.clone() }
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> ExecutableFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
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

    /// Executes this function through `domain` on concrete [`Array`] inputs and reconstructs its structured outputs.
    /// Returned arrays can still have pending device work. Use [`Self::execute_async`] when a whole-invocation
    /// completion handle is needed, including for a function with no array outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the executable belongs to another client, input validation or output reconstruction fails,
    /// or execution reports a failure. Deferred device failures can also surface when an output is awaited.
    #[inline]
    pub fn execute(&self, domain: &Domain<'c>, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, Error>
    where
        In: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<StatefulValue<'c>>,
            >,
        Out: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<StatefulValue<'c>>,
            >,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs = lift_arrays::<In>(inputs)?;
        project_arrays::<Out>(self.function.execute(domain, inputs)?)
    }

    /// Submits this function through `domain` and returns its structured outputs together with whole-invocation
    /// completion, including for a function with no array outputs. Await the returned [`Execution`] to observe
    /// asynchronous execution failures.
    ///
    /// Erased zero-space inputs await producer completion before submission because no physical input buffer can
    /// carry their dependency. Input preparation and output reconstruction for bounded dynamic shapes can also wait
    /// for transfers.
    ///
    /// # Errors
    ///
    /// Returns an error if the executable requires stateful invocation, belongs to another client, or fails input
    /// validation, submission, or output reconstruction.
    pub fn execute_async(
        &self,
        domain: &Domain<'c>,
        inputs: In::To<Array<'c>>,
    ) -> Result<Execution<Out::To<Array<'c>>>, Error>
    where
        In: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<StatefulValue<'c>>,
            >,
        Out: Parameterized<
                ArrayType,
                Family: ParameterizedFamily<ArrayIrType>
                            + ParameterizedFamily<Constant>
                            + ParameterizedFamily<Array<'c>>
                            + ParameterizedFamily<StatefulValue<'c>>,
            >,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        if self.function.compiled_program().requires_stateful_call() {
            return Err(ExecutionError::StatefulInvocationRequired.into());
        }
        let flat_inputs = lift_arrays::<In>(inputs)?.into_parameters().collect::<Vec<_>>();
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
        let execution = domain.execute_compiled_async(self.function.compiled_program(), inputs)?;
        let (flat_outputs, fence) = execution.into_parts();
        validate_runtime_outputs(self.function.compiled_program().output_types(), flat_outputs.as_slice())?;
        let outputs = ProgramParameterValues::<Out, Value<'c>>::from_parameters(
            self.function.output_structure().clone(),
            flat_outputs.into_iter().map(Value::Array),
        )
        .map_err(ProgramError::from)?;
        Ok(Execution::new(project_arrays::<Out>(outputs)?, fence))
    }
}

impl<'c> Domain<'c> {
    /// Lowers and compiles a staged XLA function using the options that it retained when it was staged.
    pub fn compile_staged_function<In, Out>(
        &self,
        staged: StagedFunction<'c, In, Out>,
    ) -> Result<CompiledFunction<'c, In, Out>, Error>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    {
        let function = staged.function.lower(self)?.compile(self)?;
        Ok(CompiledFunction { function, derived: Arc::new(DerivedFunctionSlots::new()) })
    }

    /// Replaces an executable program while preserving and validating its runtime signature.
    pub(crate) fn replace_executable_program<In, Out>(
        &self,
        executable: &ExecutableFunction<'c, In, Out>,
        program: std::sync::Arc<CompiledProgram<'c>>,
    ) -> Result<ExecutableFunction<'c, In, Out>, Error>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    {
        self.validate_replacement(executable.function.compiled_program(), &program)?;
        Ok(ExecutableFunction { function: executable.function.with_compiled_program(program.clone()) })
    }

    /// Wraps a compiled function in an adaptive profile-guided dispatcher.
    ///
    /// The returned runtime-only handle samples a bounded number of baseline executions, recompiles the already
    /// lowered StableHLO through this domain's ordinary compilation cache with the aggregated XLA profile, and then
    /// atomically directs subsequent calls to the compatible optimized executable.
    pub fn adaptive_profile_guided_recompilation<In, Out>(
        &self,
        function: &CompiledFunction<'c, In, Out>,
        options: AdaptiveProfileGuidedOptions,
    ) -> Result<AdaptiveProfileGuidedFunction<'c, In, Out>, Error>
    where
        In: Parameterized<ArrayType>,
        In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
        Out: Parameterized<ArrayType>,
        Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
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
/// [`compile_with_options`], and [`Domain::compile_staged_function`].
///
/// Holds the cached PJRT-backed [`CompiledProgram`] together with the [`StagedFunction`] it was compiled from, whose
/// input and output type metadata marshals a [`Parameterized`] tree of [`Array`]s into the executable and reassembles
/// the outputs into the expected output structure. [`Self::executable`] extracts the runtime-only
/// [`ExecutableFunction`] that executes this signature directly.
///
/// The retained staged function also keeps the source [`Program`](ryft_core::Program) that was compiled, which makes
/// this handle transformable. [`Self::call`] calls the function in any context as a `jit_call` boundary that carries
/// this source (refer to the [module documentation](crate::compilation)), so it composes with enclosing transforms and
/// compilations. [`Self::batch`], [`Self::gradient`], and [`Self::jvp`] compile transformed functions around such a
/// call, and [`Self::source_program`] exposes the source for program-level transforms and diagnostics (e.g., printed
/// IR, instruction counts, and graph rendering).
pub struct CompiledFunction<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Backend-neutral compiled function backed by an XLA executable. Retains the staged source function,
    /// captured runtime buffers, output structure, and compilation options through its lowered metadata.
    function: CoreCompiledFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<Out>>,

    /// Derived transformed functions retained by [`Self::gradient`] and [`Self::jvp`]. Shared across clones so a
    /// wrapper's clones reuse one retained derivative per transform.
    derived: Arc<DerivedFunctions<'c, In, Out>>,
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> Clone for CompiledFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    fn clone(&self) -> Self {
        Self { function: self.function.clone(), derived: Arc::clone(&self.derived) }
    }
}

/// Retention slots for the derived functions of one [`CompiledFunction`] (i.e., the reverse mode gradient and the
/// forward mode JVP compiled around a call of its retained source program).
type DerivedFunctions<'c, In, Out> = DerivedFunctionSlots<
    'c,
    CoreCompiledFunction<Domain<'c>, ProgramParameters<In>, ProgramParameters<In>>,
    CoreCompiledFunction<Domain<'c>, ProgramParameters<(In, In)>, ProgramParameters<(Out, Out)>>,
>;

/// One-entry retention slots for functions derived from a [`CompiledFunction`].
///
/// [`CompiledFunction::gradient`] and [`CompiledFunction::jvp`] take no per-call transform configuration, so each
/// derived function has exactly one valid form per domain identity. The slot retains the most recently produced
/// artifact together with a witness of the domain that produced it. A later call with a matching domain reuses the
/// retained artifact and skips differentiation, staging, and lowering entirely, while a call with a different domain
/// identity produces a fresh artifact and replaces the slot. Failed productions are never stored, so they retry
/// naturally.
///
/// A slot is a retention slot and not a single-flight barrier: its lock is released before the artifact is derived, so
/// concurrent cold calls may each derive their own copy and the last insert wins. That is sound because the copies are
/// interchangeable derivations of the same source program, and it is deliberate because waiting single-flight belongs
/// exclusively to the shared [`CompilationContext`](ryft_core::compilation::CompilationContext), which still
/// deduplicates the backend compilation those duplicate derivations request.
struct DerivedFunctionSlots<'c, Gradient, Jvp> {
    /// Retained reverse mode gradient function, if any.
    gradient: Mutex<Option<(DomainWitness<'c>, Gradient)>>,

    /// Retained forward mode JVP function, if any.
    jvp: Mutex<Option<(DomainWitness<'c>, Jvp)>>,
}

impl<'c, Gradient, Jvp> DerivedFunctionSlots<'c, Gradient, Jvp> {
    /// Creates empty retention slots.
    fn new() -> Self {
        Self { gradient: Mutex::new(None), jvp: Mutex::new(None) }
    }
}

/// Identity of the [`Domain`] that produced a retained derived function.
///
/// [`Domain`] handles are cheap clones of shared state, so two handles denote interchangeable compilation
/// pipelines exactly when they share one compilation context, one client, equal compilation-option templates, and an
/// equal mesh. The witness stores a full domain clone, which both provides those comparisons and keeps the shared
/// compilation context alive so its address can never be reused by a later allocation.
struct DomainWitness<'c> {
    /// Domain handle captured when the retained artifact was produced.
    domain: Domain<'c>,
}

impl<'c> DomainWitness<'c> {
    /// Captures the identity of `domain`.
    fn new(domain: &Domain<'c>) -> Self {
        Self { domain: domain.clone() }
    }

    /// Returns whether `domain` is interchangeable with the domain that produced the retained artifact.
    fn matches(&self, domain: &Domain<'c>) -> bool {
        std::ptr::eq(self.domain.compilation_context(), domain.compilation_context())
            && std::ptr::eq(self.domain.client(), domain.client())
            && self.domain.compilation_options() == domain.compilation_options()
            && self.domain.mesh().ok() == domain.mesh().ok()
    }
}

impl<'c, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>> CompiledFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
    Out::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<Constant>,
{
    /// Returns the flat output [`ArrayType`]s in the order the executor produces them.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.function.compiled_program().output_types()
    }

    /// Returns a runtime-only handle that omits staged and lowered transform metadata.
    #[inline]
    pub fn executable(&self) -> ExecutableFunction<'c, In, Out> {
        ExecutableFunction { function: self.function.executable().clone() }
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable(self) -> ExecutableFunction<'c, In, Out> {
        ExecutableFunction { function: self.function.into_executable() }
    }

    /// Returns the staged function this executable was compiled from.
    #[inline]
    pub fn staged(&self) -> StagedFunction<'c, In, Out> {
        self.function.staged().clone().into()
    }

    /// Returns the flat source [`Program`](ryft_core::Program) that produced the compiled artifact. Use it for
    /// program-level transforms (e.g., transposition or partial evaluation of the program itself) and for diagnostics
    /// such as printed IR, instruction counts, and graph rendering. Value-level transforms instead call this function
    /// through [`Self::call`].
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<StatefulValue<'c>, Constant, Operation, Vec<Constant>, Vec<Constant>> {
        self.function.source_program()
    }

    /// Returns the device mesh the compiled program runs against. Delegates to the cached
    /// [`CompiledProgram::mesh`](crate::domains::CompiledProgram::mesh).
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.function.compiled_program().mesh()
    }

    /// Returns the [`Options`] with which functions derived from this one (e.g., its gradient) are compiled. They keep
    /// the options that select how programs lower (i.e., the mesh, the kernel compiler, and the ragged dot lowering
    /// strategy) and drop the ones that describe this function's own signature or executable (i.e., its input and
    /// output shardings, donation flags, input bound bucketing, and feedback-directed profile), which do not apply to a
    /// derived function.
    fn derived_options(&self) -> Options {
        let options = self.function.staged().options();
        Options {
            kernel_compiler: options.kernel_compiler.clone(),
            ragged_dot_lowering_strategy: options.ragged_dot_lowering_strategy,
            ..Options::new(self.mesh().clone())
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

    /// Calls this compiled function as a `jit_call` operation in the array IR context recovered from `inputs`, exactly
    /// as [`StagedFunction::call`] calls its retained staged function. Under the eager [`Domain`], the call executes
    /// through eager dispatch rather than through the retained executable, so use [`ExecutableFunction::execute`] to
    /// run that executable directly on concrete arrays.
    ///
    /// Refer to the documentation of [`StagedFunction::call`] for more information.
    #[inline]
    pub fn call<L, C, Path>(&self, inputs: In::To<L>) -> Result<Out::To<L>, ProgramError>
    where
        L: CoreValue<Type = ArrayType, Domain = ProjectedContext<C, ArrayType>>,
        C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>
            + ContextProjection<ArrayType>
            + ImportingContext<Domain<'c>, Path>,
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
        C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation> + ImportingContext<Domain<'c>, Path>,
        C::Value: ValueProjection<ArrayType, Projected: Parameter> + Reshard + ConstrainSharding,
        In::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
        Out::Family:
            ParameterizedFamily<C::Value> + ParameterizedFamily<<C::Value as ValueProjection<ArrayType>>::Projected>,
    {
        call_staged_in_context::<In, Out, C, Path>(self.function.staged(), context, inputs)
    }
}

// Reverse mode differentiation packaged as a function. Mirrors `jax.grad(jax.jit(f))`. The original closure is never
// traced again. The derived trace calls the retained staged function through `call_flat_in_context`, which binds a
// `jit_call` boundary that reverse mode differentiation rewrites through its ordinary JVP and transpose rules.
impl<'c, In: Parameterized<ArrayType, To<ArrayType> = In>> CompiledFunction<'c, In, ArrayType>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: std::fmt::Debug + Hash + PartialEq,
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
    /// interchangeable `domain` return the retained compiled function without repeating differentiation, staging, or
    /// lowering, while a call with a different domain identity produces a fresh derivative and replaces the retained
    /// one. The retained artifact is a compiled program whose residuals are runtime values, so reuse across calls with
    /// different runtime inputs is sound by construction. Retention is not single-flight. Concurrent cold calls may
    /// each derive a gradient, and because those derivations are interchangeable, the inserts are idempotent and the
    /// last one wins, while the shared [`CompilationContext`](ryft_core::compilation::CompilationContext) still
    /// deduplicates their backend compilation.
    #[track_caller]
    pub fn gradient(&self, domain: &Domain<'c>) -> Result<CompiledFunction<'c, In, In>, Error>
    where
        In::Family: ParameterizedFamily<CompileTracer<'c>, To = In::To<CompileTracer<'c>>>,
        In::To<CompileTracer<'c>>: Parameterized<
                CompileTracer<'c>,
                Family = In::Family,
                ParameterStructure = In::ParameterStructure,
                To<ArrayType> = In,
                To<Constant> = In::To<Constant>,
            >,
        ProgramParameterValues<In, StatefulCompileTracer<'c>>: Parameterized<
                StatefulCompileTracer<'c>,
                To<ArrayIrType> = ProgramParameters<In>,
                To<Constant> = ProgramConstants<In>,
            >,
    {
        {
            let slot = self.derived.gradient.lock().expect("derived-gradient retention slot mutex poisoned");
            if let Some((witness, artifact)) = slot.as_ref() {
                if witness.matches(domain) {
                    return Ok(CompiledFunction {
                        function: artifact.clone(),
                        derived: Arc::new(DerivedFunctionSlots::new()),
                    });
                }
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
                In::To::<CompileTracer<'c>>::from_parameters(input_structure, gradients)
                    .map_err(ProgramError::from)
                    .map_err(Into::into)
            },
            Vec::new(),
            input_signature,
            domain,
            self.derived_options(),
        )?;
        *self.derived.gradient.lock().expect("derived-gradient retention slot mutex poisoned") =
            Some((DomainWitness::new(domain), compiled.function.clone()));
        Ok(compiled)
    }
}

// Forward mode differentiation packaged as a function. Mirrors `jax.jvp(jax.jit(f))`. The derived trace calls the
// retained staged function through `call_flat_in_context`, which binds a `jit_call` boundary that forward mode
// differentiation rewrites through its ordinary JVP rule.
impl<'c, In: Clone + Parameterized<ArrayType, To<ArrayType> = In>, Out: Parameterized<ArrayType>>
    CompiledFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: std::fmt::Debug + Hash + PartialEq,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
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
    /// interchangeable `domain` return the retained compiled function without repeating differentiation, staging, or
    /// lowering, while a call with a different domain identity produces a fresh derivative and replaces the retained
    /// one. Retention is not single-flight. Concurrent cold calls may each derive a JVP, and because those derivations
    /// are interchangeable, the inserts are idempotent and the last one wins, while the shared
    /// [`CompilationContext`](ryft_core::compilation::CompilationContext) still deduplicates their backend compilation.
    #[track_caller]
    pub fn jvp(&self, domain: &Domain<'c>) -> Result<CompiledFunction<'c, (In, In), (Out, Out)>, Error>
    where
        In::Family: ParameterizedFamily<CompileTracer<'c>, To = In::To<CompileTracer<'c>>>,
        In::To<CompileTracer<'c>>:
            Parameterized<CompileTracer<'c>, Family = In::Family, ParameterStructure = In::ParameterStructure>,
        Out::Family: ParameterizedFamily<CompileTracer<'c>, To = Out::To<CompileTracer<'c>>>,
        Out::To<CompileTracer<'c>>: Parameterized<
                CompileTracer<'c>,
                Family = Out::Family,
                ParameterStructure = Out::ParameterStructure,
                To<ArrayType> = Out,
                To<Constant> = Out::To<Constant>,
            >,
        ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
                StatefulCompileTracer<'c>,
                To<ArrayIrType> = ProgramParameters<Out>,
                To<Constant> = ProgramConstants<Out>,
            >,
    {
        {
            let slot = self.derived.jvp.lock().expect("derived-jvp retention slot mutex poisoned");
            if let Some((witness, artifact)) = slot.as_ref() {
                if witness.matches(domain) {
                    return Ok(CompiledFunction {
                        function: artifact.clone(),
                        derived: Arc::new(DerivedFunctionSlots::new()),
                    });
                }
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
            domain,
            self.derived_options(),
        )?;
        *self.derived.jvp.lock().expect("derived-jvp retention slot mutex poisoned") =
            Some((DomainWitness::new(domain), compiled.function.clone()));
        Ok(compiled)
    }
}

/// Batched tracer of one array IR batching level directly inside an XLA compilation trace.
type CompileBatchTracer<'c> = BatchingTracer<DomainTracingContext<Domain<'c>, Value<'c>>, ArrayIrBatchingPolicy>;

// Batching packaged as a function. Mirrors `jax.jit(jax.vmap(f))` for a static batch extent. The derived trace calls
// the retained staged function through `call_in_context` under `batch`, which rewrites the resulting `jit_call`
// boundary through its ordinary batching rule.
impl<'c, In: Parameterized<ArrayType, To<ArrayType> = In>, Out: Parameterized<ArrayType>> CompiledFunction<'c, In, Out>
where
    In::Family: ParameterizedFamily<ArrayType, To = In>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<BatchAxis>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileBatchTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<BatchAxis>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>
        + ParameterizedFamily<CompileBatchTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
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
        domain: &Domain<'c>,
        input_batch_axes: InputBatchAxes,
        output_batch_axes: OutputBatchAxes,
        batch_axis: BatchAxisSpecification<usize>,
    ) -> Result<CompiledFunction<'c, In, Out>, Error> {
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
                    Ok(r#type.batched(position, CoreDimension::Static(extent), batch_sharding.clone())?)
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
                        |inputs: Vec<CompileBatchTracer<'c>>| {
                            let context = inputs
                                .first()
                                .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                                .context()
                                .clone();
                            let inputs = ProgramParameterValues::<In, CompileBatchTracer<'c>>::from_parameters(
                                input_structure,
                                inputs,
                            )?;
                            let outputs = staged.call_in_context::<CompileBatchTracer<'c>, _, _>(&context, inputs)?;
                            Ok(outputs.into_parameters().collect::<Vec<CompileBatchTracer<'c>>>())
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
                Ok(Out::To::<CompileTracer<'c>>::from_parameters(output_structure, outputs)
                    .map_err(ProgramError::from)?)
            },
            Vec::new(),
            batched_input_types,
            domain,
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
/// compiled and executed against `domain`. Its [`CompilationContext`](ryft_core::compilation::CompilationContext)
/// structurally shares equivalent lowerings even when they originate at different Rust call sites.
#[track_caller]
pub fn compile<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    domain: &'domain Domain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledFunction<'c, In, Out>, Error>
where
    F: FnOnce(In::To<CompileTracer<'c>>) -> Out::To<CompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    compile_with_options(function, input_types, domain, Options::new(mesh))
}

/// Same as [`compile`] but makes runtime arrays explicit captures of the compiled program.
///
/// The closure receives capture tracers first and ordinary input tracers second. Captures are compiled as hidden
/// executable arguments and are supplied from the returned [`CompiledFunction`] at execution time, so callers of the
/// compiled function still pass only `In` inputs.
#[track_caller]
pub fn compile_with_captures<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain Domain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledFunction<'c, In, Out>, Error>
where
    F: FnOnce(Vec<CompileTracer<'c>>, In::To<CompileTracer<'c>>) -> Out::To<CompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    compile_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        domain,
        Options::new(mesh),
    )
}

/// Same as [`compile`] but accepts an explicit [`Options`] payload.
#[track_caller]
pub fn compile_with_options<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    domain: &'domain Domain<'c>,
    options: Options,
) -> Result<CompiledFunction<'c, In, Out>, Error>
where
    F: FnOnce(In::To<CompileTracer<'c>>) -> Out::To<CompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    compile_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, domain, options)
}

#[track_caller]
fn compile_with_flat_captures<'domain, 'c: 'domain, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain Domain<'c>,
    options: Options,
) -> Result<CompiledFunction<'c, In, Out>, Error>
where
    F: FnOnce(
        Vec<Constant>,
        Vec<CompileTracer<'c>>,
        In::To<CompileTracer<'c>>,
    ) -> Result<Out::To<CompileTracer<'c>>, Error>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    let staged = stage_with_flat_captures::<F, In, Out>(function, captures, input_types, options)?;
    domain.compile_staged_function(staged)
}

/// Traces `function` into a staged program and returns a [`StagedFunction`] handle, without compiling a PJRT
/// executable.
///
/// This is the staging-only counterpart of [`compile`]. Use it for functions that are composed into larger programs
/// or transforms through [`StagedFunction::call`] rather than executed directly, so that no executable is built for
/// them. Compile the staged handle later with [`Domain::compile_staged_function`] when direct execution is needed.
/// `options` are applied to the abstract input signature before tracing and retained for subsequent lowering and
/// compilation.
#[track_caller]
pub fn stage<'c, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
    options: Options,
) -> Result<StagedFunction<'c, In, Out>, Error>
where
    F: FnOnce(In::To<CompileTracer<'c>>) -> Out::To<CompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    stage_with_flat_captures(|_, _, inputs| Ok(function(inputs)), Vec::new(), input_types, options)
}

/// Same as [`stage`] but makes runtime arrays explicit captures of the staged program.
///
/// The closure receives capture tracers first and ordinary input tracers second, mirroring
/// [`compile_with_captures`]. Captures are retained on the staged handle, and every [`StagedFunction::call`] imports
/// them into the calling context as leading inputs of its `jit_call` boundary.
#[track_caller]
pub fn stage_with_captures<'c, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    options: Options,
) -> Result<StagedFunction<'c, In, Out>, Error>
where
    F: FnOnce(Vec<CompileTracer<'c>>, In::To<CompileTracer<'c>>) -> Out::To<CompileTracer<'c>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    stage_with_flat_captures(
        |_, capture_tracers, inputs| Ok(function(capture_tracers, inputs)),
        captures,
        input_types,
        options,
    )
}

#[track_caller]
fn stage_with_flat_captures<'c, F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    options: Options,
) -> Result<StagedFunction<'c, In, Out>, Error>
where
    F: FnOnce(
        Vec<Constant>,
        Vec<CompileTracer<'c>>,
        In::To<CompileTracer<'c>>,
    ) -> Result<Out::To<CompileTracer<'c>>, Error>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    In::ParameterStructure: Hash,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'c>>
        + ParameterizedFamily<StatefulCompileTracer<'c>>,
    Out::To<CompileTracer<'c>>: Parameterized<CompileTracer<'c>, To<ArrayType> = Out, To<Constant> = Out::To<Constant>>,
    ProgramParameterValues<Out, StatefulCompileTracer<'c>>: Parameterized<
            StatefulCompileTracer<'c>,
            To<ArrayIrType> = ProgramParameters<Out>,
            To<Constant> = ProgramConstants<Out>,
        >,
{
    let captures = captures.into_iter().map(Value::Array).collect();
    let input_structure = input_types.parameter_structure();
    let input_types =
        ProgramParameters::<In>::from_parameters(input_structure, input_types.into_parameters().map(ArrayIrType::from))
            .map_err(ProgramError::from)?;
    let output_shardings = options.output_shardings.clone();
    let function = move |capture_references,
                         capture_tracers: Vec<StatefulCompileTracer<'c>>,
                         inputs: ProgramParameterValues<In, StatefulCompileTracer<'c>>| {
        let capture_tracers = capture_tracers
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let inputs = project_tracers::<In>(inputs)?;
        let outputs = lift_tracers::<Out>(function(capture_references, capture_tracers, inputs)?)?;
        place_outputs(outputs, output_shardings.as_deref())
    };
    let input_types = with_input_shardings(input_types, &options)?;
    let function = CoreStagedFunction::<Domain<'_>, ProgramParameters<In>, ProgramParameters<Out>>::stage(
        function,
        captures,
        input_types,
        options,
    )?;
    Ok(StagedFunction { function })
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
#[track_caller]
pub fn infer_output_types<F, In: Parameterized<ArrayType>, Out: Parameterized<ArrayType>>(
    function: F,
    input_types: In,
) -> Result<Out, ProgramError>
where
    F: FnOnce(In::To<CompileTracer<'static>>) -> Out::To<CompileTracer<'static>>,
    In::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'static>>
        + ParameterizedFamily<StatefulCompileTracer<'static>>,
    Out::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Constant>
        + ParameterizedFamily<CompileTracer<'static>>
        + ParameterizedFamily<StatefulCompileTracer<'static>>,
    Out::To<CompileTracer<'static>>: Parameterized<CompileTracer<'static>, To<ArrayType> = Out>,
{
    let input_structure = input_types.parameter_structure();
    let input_types = ProgramParameters::<In>::from_parameters(
        input_structure,
        input_types.into_parameters().map(ArrayIrType::from),
    )?;
    let output_types = DomainTracingContext::<Domain<'static>, StatefulValue<'static>>::infer_output_type(
        |tracers| {
            let input_structure = tracers.parameter_structure();
            let input_parameters = tracers
                .into_parameters()
                .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let tracers = In::To::<CompileTracer<'static>>::from_parameters(input_structure, input_parameters)?;
            let outputs = function(tracers);
            let output_structure = outputs.parameter_structure();
            ProgramParameterValues::<Out, StatefulCompileTracer<'static>>::from_parameters(
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
pub(crate) mod tests {
    use std::sync::{Arc, Barrier, Condvar, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        Add, AddOperation, ArgMax, Array as CoreArray, ArrayIrType, ArrayOperation, ArrayReference,
        ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, Atan2, AttachedRegions, AxisIndex, BatchAxis,
        BatchAxisSpecification, Broadcast, CaptureReference, Compare, ComparisonDirection, Condition,
        ConstrainSharding, Context, ConvertElementType, Cos, CotangentDestinationKind, Cumulative, CustomCall,
        CustomCallOperation, DataType, Device, DeviceMesh, DifferentiableType, Differentiate,
        Dimension as CoreDimension, DimensionBounds, DimensionSize, DimensionSizeOperation, DimensionVariable, Div,
        DomainTracer, DomainTracingContext, Dot, DotDimensionNumbers, DynamicSlice, DynamicUpdateSlice, EagerContext,
        Error as CoreError, Exp, Fill, ForwardModeDifferentiate, Hessian, Iota, Jacobian, LogicalMesh, Logistic,
        MaybeZero, Memory, MeshAxis, MeshAxisType, Mul, MulOperation, OneLike, ParallelVaryOperation, Placeholder,
        ProgramBuilder as CoreProgramBuilder, ProgramError, ProjectedValue, Random, Reduce, ReductionKind,
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceCompletion as CoreReferenceCompletion,
        ReferenceCompletionBackend, ReferenceError, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew,
        ReferenceNewOperation, ReferenceRead, ReferenceReadOperation, ReferenceType, Reshape, Reshard, ScanOperation,
        ScanOptions, Select, Shape, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, Sin, Softmax,
        StagedFunction as CoreStagedFunction, StopGradient, StopGradientOperation, Sub, Tanh, TopK, Trace, Tracer,
        TracingContext, TransferToMemory, Typed, TypedProgram, Value as CoreValue, ValueProjection, WhileOperation,
        ZeroLike, batch, batch_in_parent, composite, differentiate_at, differentiate_in_parent_at, rematerialize, scan,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::compilation::{
        CompileTracer, CompiledFunction, CompiledFunctionDispatcher, ExecutableFunction, StagedFunction,
        StatefulCompileTracer, compile, compile_statefully, compile_statefully_with_captures, compile_with_captures,
        compile_with_options, infer_output_types, jit, jit_statefully, jit_with_options, stage, stage_with_captures,
    };
    use crate::domains::tests::{StatefulFailureInjection, inject_stateful_completion, inject_stateful_failure};
    use crate::experimental::ops::{Constant, Operation};
    use crate::operations::jit_call::JitCallOperation;
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{
        AdaptiveProfileGuidedOptions, Array, ArrayError, CompilationError, Domain, Error, ExecutionError, FromPjrt,
        Options, RaggedDotLoweringStrategy, Reference, Session, Value,
    };

    /// Deterministic completion gate used by stateful asynchronous integration tests.
    #[derive(Clone)]
    struct ControlledReferenceCompletion {
        /// Shared terminal result and wait notification.
        state: Arc<(Mutex<ControlledReferenceCompletionState>, Condvar)>,
    }

    /// Mutable state of one [`ControlledReferenceCompletion`].
    struct ControlledReferenceCompletionState {
        /// Terminal result, absent while the completion is pending.
        result: Option<Result<(), Arc<str>>>,

        /// Optional one-shot notification emitted when a caller starts waiting, carrying this gate's label so that
        /// tests sharing one channel across several gates can pin which gate was awaited first.
        await_started: Option<(&'static str, mpsc::Sender<&'static str>)>,
    }

    impl ControlledReferenceCompletion {
        /// Creates a pending completion gate.
        fn new() -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledReferenceCompletionState { result: None, await_started: None }),
                    Condvar::new(),
                )),
            }
        }

        /// Creates a pending completion that reports `label` when its blocking wait begins.
        fn with_await_notification(label: &'static str, await_started: mpsc::Sender<&'static str>) -> Self {
            Self {
                state: Arc::new((
                    Mutex::new(ControlledReferenceCompletionState {
                        result: None,
                        await_started: Some((label, await_started)),
                    }),
                    Condvar::new(),
                )),
            }
        }

        /// Completes this gate exactly once and wakes blocking waiters.
        fn complete(&self, result: Result<(), Arc<str>>) {
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            assert!(state.result.is_none());
            state.result = Some(result);
            ready.notify_all();
        }
    }

    impl ReferenceCompletionBackend for ControlledReferenceCompletion {
        fn r#await(&self) -> Result<(), Arc<str>> {
            let (state, ready) = &*self.state;
            let mut state = state.lock().unwrap();
            if let Some((label, await_started)) = state.await_started.take() {
                await_started.send(label).unwrap();
            }
            while state.result.is_none() {
                state = ready.wait(state).unwrap();
            }
            state.result.clone().unwrap()
        }

        fn is_ready(&self) -> Result<bool, Arc<str>> {
            let state = self.state.0.lock().unwrap();
            state.result.clone().map_or(Ok(false), |result| result.map(|_| true))
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    fn single_device_mesh(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap()
    }

    /// Creates a two-device mesh with the axis interpretation required by the test.
    fn two_device_mesh(client: &ryft_pjrt::Client<'_>, axis_type: MeshAxisType) -> DeviceMesh {
        let devices: Vec<Device> = client
            .addressable_devices()
            .unwrap()
            .iter()
            .take(2)
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap(), devices).unwrap()
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

    fn read_sharded_f32_array(array: &Array<'_>) -> Vec<f32> {
        let mut values = Vec::new();
        for device in array.mesh().devices() {
            let shard_bytes = array
                .device_shard(device.id())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            values.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        values
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

    /// Builds a nonlinear reference body with per-device reads and products. Its derivatives require saved values
    /// from both operands, so different shard values must survive the primal-to-tangent boundary.
    fn nonlinear_reference_shard_map_program(
        mesh: &LogicalMesh,
        replicated_reference: bool,
    ) -> ryft_core::Program<Constant, Operation, Vec<Constant>, Vec<Constant>> {
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let reference_sharding =
            if replicated_reference { Sharding::replicated(mesh.clone(), 1) } else { sharded.clone() };
        let reference_type = ArrayType::new_static(DataType::F32, [if replicated_reference { 2 } else { 4 }])
            .with_sharding(reference_sharding.clone())
            .unwrap();
        let value_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![reference_sharding.clone(), sharded.clone()],
            vec![sharded.clone()],
            vec!["x".to_string()],
        )
        .unwrap();
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
        let reference =
            body.add_input(ReferenceType::new(boundary.local_input_type(0, &reference_type).unwrap()).into());
        let value = body.add_input(boundary.local_input_type(1, &value_type).unwrap().into());
        let state = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let state = if replicated_reference {
            body.add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![state], None)
                .unwrap()[0]
        } else {
            state
        };
        let output = body.add_instruction(MulOperation::new(), Vec::new(), vec![state, value], None).unwrap()[0];
        let body = body
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let inputs = vec![ArrayIrType::Reference(ReferenceType::new(reference_type)), ArrayIrType::Array(value_type)];
        let operation = ShardMapOperation::from_program(&body, inputs.clone(), boundary).unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let inputs = inputs.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let body = builder.import_program(body);
        let outputs =
            builder.add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], inputs, None).unwrap();
        let outputs = outputs.to_vec();
        builder.build(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_jit_unary_function_runs_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

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
    fn test_compile_converted_broadcast() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let output_type = ArrayType::new_static(DataType::C64, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            |value| value.convert_element_type(DataType::C64).unwrap().broadcast(output_type.clone(), &[1]).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(compiled.source_program().program().input_ids().len(), 1);
        assert_eq!(compiled.source_program().captures().len(), 0);
        let input =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes(&[1f32, 2., 3.]).as_slice())
                .unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        // Complex storage interleaves real and imaginary components after scalar element conversion.
        assert_eq!(values_from_bytes::<f32>(&bytes), vec![1., 0., 2., 0., 3., 0., 1., 0., 2., 0., 3., 0.]);
    }

    #[test]
    fn test_compile_broadcast_capture() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let value =
            Array::from_host_buffer(&engine, scalar_type, mesh.clone(), values_to_bytes(&[4f32]).as_slice()).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, exemplar| captures[0].broadcast(exemplar.r#type().into_owned(), &[]).unwrap(),
            vec![value],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(compiled.source_program().captures().len(), 1);
        assert_eq!(compiled.source_program().to_program_with_lifted_captures().unwrap().input_ids().len(), 2);
        let input = Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes(&[9f32; 6]).as_slice()).unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();
        assert_eq!(read_f32_array(&client, &output), vec![4f32; 6]);
    }

    #[test]
    fn test_compile_reduce_product() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|value| value.reduce_product(&[0]).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let gradient = compiled.gradient(&engine).unwrap();

        // A zero factor retains the product of the other factors in its derivative; two zeros erase every term.
        for (values, expected, expected_gradient) in [
            ([2f32, 3., 4.], 24., vec![12., 8., 6.]),
            ([0f32, 3., 4.], 0., vec![12., 0., 0.]),
            ([0f32, 0., 4.], 0., vec![0., 0., 0.]),
        ] {
            let input =
                Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), &values_to_bytes(&values)).unwrap();
            let output = compiled.executable().execute(&engine, input.clone()).unwrap();
            assert_eq!(read_f32_array(&client, &output), vec![expected]);
            let output = gradient.executable().execute(&engine, input).unwrap();
            assert_eq!(read_f32_array(&client, &output), expected_gradient);
        }
    }

    #[test]
    fn test_compile_reduce_product_sharded_gradient() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Explicit);
        let engine = Session::new(&client).domain();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|value| value.reduce_product(&[0]).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let gradient = compiled.gradient(&engine).unwrap();

        // Each input cotangent depends on factors held by the other device, including the sole zero's cotangent.
        for (values, expected) in
            [([2f32, 3., 4., 5.], vec![60f32, 40., 30., 24.]), ([0f32, 3., 4., 5.], vec![60f32, 0., 0., 0.])]
        {
            let input =
                Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), &values_to_bytes(&values)).unwrap();
            let output = gradient.executable().execute(&engine, input).unwrap();
            assert_eq!(output.sharding(), input_type.sharding().unwrap());
            let mut observed = Vec::new();
            for device in client.addressable_devices().unwrap().iter().take(2) {
                let bytes = output
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
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn test_compile_reduce_narrow_floating_point() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        // Product retains the small growth of each factor; the larger mean also needs a widened divisor.
        for (kind, count, input, expected) in [
            (ReductionKind::Sum, 4096, 1f32, 4096f32),
            (ReductionKind::Product, 4096, 1.0009765625f32, 54.5f32),
            (ReductionKind::Mean, 4096, 1f32, 1f32),
            (ReductionKind::Mean, 70000, 1f32, 1f32),
            (ReductionKind::LogSumExp, 4096, 0f32, 8.3203125f32),
        ] {
            let input_type = ArrayType::new_static(DataType::F32, [count])
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap();
            let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
                |value| {
                    value
                        .convert_element_type(DataType::F16)
                        .unwrap()
                        .reduce(&[0], kind)
                        .unwrap()
                        .convert_element_type(DataType::F32)
                        .unwrap()
                },
                input_type.clone(),
                &engine,
                mesh.clone(),
            )
            .unwrap();
            let input =
                Array::from_host_buffer(&engine, input_type, mesh.clone(), &values_to_bytes(&vec![input; count]))
                    .unwrap();
            let output = compiled.executable().execute(&engine, input).unwrap();
            assert_eq!(read_f32_array(&client, &output), vec![expected]);
        }
    }

    #[test]
    fn test_compile_reduce_mean_empty() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F32, [0])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            |value| value.reduce(&[0], ReductionKind::Mean).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&engine, input_type, mesh, &[]).unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();
        assert!(read_f32_array(&client, &output)[0].is_nan());
    }

    #[test]
    fn test_compile_reduce_complex() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        // Complex storage interleaves the real and imaginary components. Infinite real components force extrema
        // to compare the imaginary component against the identity rather than deciding on the real component.
        for (kind, components, expected) in [
            (ReductionKind::Product, vec![2f32, 4., 6., 8.], vec![-20f32, 40.]),
            (ReductionKind::Product, vec![], vec![1f32, 0.]),
            (ReductionKind::Mean, vec![2f32, 4., 6., 8.], vec![4f32, 6.]),
            (ReductionKind::Max, vec![f32::NEG_INFINITY, -1., f32::NEG_INFINITY, -1.], vec![f32::NEG_INFINITY, -1.]),
            (ReductionKind::Min, vec![f32::INFINITY, 1., f32::INFINITY, 1.], vec![f32::INFINITY, 1.]),
        ] {
            let input_type = ArrayType::new_static(DataType::C64, [components.len() / 2])
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap();
            let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
                compile(|value| value.reduce(&[0], kind).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
            let input =
                Array::from_host_buffer(&engine, input_type, mesh.clone(), &values_to_bytes(&components)).unwrap();
            let output = compiled.executable().execute(&engine, input).unwrap();
            assert_eq!(read_f32_array(&client, &output), expected);
        }
    }

    #[test]
    fn test_jit_cumulative_sum_runs_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let source = || {
            Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&values).as_slice(),
            )
            .unwrap()
        };

        // The compiled `reduce_window` scan reproduces the exact prefix sums of the operand, and its reverse
        // direction the exact suffix sums.
        let forward: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.cumulative_sum(0).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let output = forward.executable().execute(&engine, source()).unwrap();
        assert_eq!(read_f32_array(&client, &output), vec![1.0, 3.0, 6.0, 10.0]);

        let reverse: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.reverse_cumulative_sum(0).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let output = reverse.executable().execute(&engine, source()).unwrap();
        assert_eq!(read_f32_array(&client, &output), vec![10.0, 9.0, 7.0, 4.0]);

        // Reverse-mode differentiation through the compiled scan exercises the transposed direction end to end:
        // summing every prefix weights element `i` by the number of prefixes that contain it, so the gradient is
        // the reverse scan of ones.
        let total: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            |x| x.cumulative_sum(0).unwrap().reduce(&[0], ReductionKind::Sum).unwrap(),
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<'_, ArrayType, ArrayType> = total.gradient(&engine).unwrap();
        let output = gradient.executable().execute(&engine, source()).unwrap();
        assert_eq!(read_f32_array(&client, &output), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_compile_reduce_log_sum_exp_gradient() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F64, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|value| value.reduce_log_sum_exp(&[0]).unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();
        let gradient = compiled.gradient(&engine).unwrap();

        // Rounding the primal output at a large offset must not erase the derivative's normalization.
        let input =
            Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), &values_to_bytes(&[1e20f64, 1e20]))
                .unwrap();
        let output = gradient.executable().execute(&engine, input).unwrap();
        assert_eq!(read_f32_array(&client, &output.convert_element_type(DataType::F32).unwrap()), vec![0.5, 0.5]);

        // The compiled finite-shift guard must retain the undefined weight at an infinite input.
        let input = Array::from_host_buffer(&engine, input_type, mesh, &values_to_bytes(&[f64::INFINITY, 0.])).unwrap();
        let output = gradient.executable().execute(&engine, input).unwrap();
        let values = read_f32_array(&client, &output.convert_element_type(DataType::F32).unwrap());
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.);
    }

    #[test]
    fn test_compile_reduce_log_sum_exp_narrow_jvp() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F16, [65536])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|value| value.reduce_log_sum_exp(&[0]).unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();
        let differentiated = compiled.jvp(&engine).unwrap();
        // The normalization count exceeds the largest finite `f16`, so derivative intermediates must stay widened.
        let input =
            Array::from_host_buffer(&engine, input_type, mesh, &values_to_bytes(&vec![0x3c00u16; 65536])).unwrap();
        let (_, tangent) = differentiated.executable().execute(&engine, (input.clone(), input)).unwrap();
        assert_eq!(read_f32_array(&client, &tangent.convert_element_type(DataType::F32).unwrap()), vec![1.]);
    }

    #[test]
    fn test_jit_log_sum_exp_and_cumulative_log_sum_exp_run_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source = |values: [f32; 4]| {
            Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&values).as_slice(),
            )
            .unwrap()
        };
        // Both lowerings are expansions over transcendental StableHLO operations, so the compiled results are
        // compared within a tolerance scaled to the magnitude of the expected `f32` value.
        let assert_close = |observed: &[f32], expected: &[f32]| {
            assert_eq!(observed.len(), expected.len());
            for (index, (observed, expected)) in observed.iter().zip(expected).enumerate() {
                let tolerance = 1e-6 * expected.abs().max(1.0);
                assert!(
                    (observed - expected).abs() <= tolerance,
                    "element {index}: compiled {observed} but expected {expected}",
                );
            }
        };

        // The compiled guarded expansion returns `log(4)` for four equal zeros, and the same value shifted by the
        // maximum for four equal thousands, where a shift-free `sum(exp(x))` would have overflowed `f32` outright.
        let reduced: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.reduce_log_sum_exp(&[0]).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let output = reduced.executable().execute(&engine, source([0.0; 4])).unwrap();
        assert_close(&read_f32_array(&client, &output), &[4.0f32.ln()]);
        let output = reduced.executable().execute(&engine, source([1000.0; 4])).unwrap();
        assert_close(&read_f32_array(&client, &output), &[1000.0 + 4.0f32.ln()]);

        // The reverse prefix scan folds the pairwise stable `log_add_exp` over each suffix, so position `i` of an
        // all-zero operand holds `log(4 - i)` and the final position holds `log(1) = 0`.
        let scanned: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.reverse_cumulative_log_sum_exp(0).unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();
        let output = scanned.executable().execute(&engine, source([0.0; 4])).unwrap();
        assert_close(&read_f32_array(&client, &output), &[4.0f32.ln(), 3.0f32.ln(), 2.0f32.ln(), 0.0]);
    }

    #[test]
    fn test_public_jit_discharges_local_reference_state_without_changing_the_array_abi() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            |input| {
                let input = input.into_value();
                let reference = input.reference_new().unwrap();
                reference.add_update(&input).unwrap();
                ValueProjection::<ArrayType>::into_projected(reference.freeze().unwrap()).unwrap()
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let executable = compiled.executable();
        assert_eq!(executable.function.input_types(), &[ArrayIrType::Array(input_type.clone())]);
        assert_eq!(executable.output_types(), &[input_type.clone()]);

        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&[3.0]).as_slice()).unwrap();
        let output = executable.execute(&domain, input).unwrap();
        assert_eq!(read_f32_array(&client, &output), vec![6.0]);
    }

    #[test]
    fn test_public_jit_reference_input_value_and_gradient() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        // The inner gradient differentiates reference contents at entry, while the outer compiled function returns
        // the scalar objective, the mutated primal state, and an ordinary gradient. Both local references discharge
        // before lowering, so the public executable retains its ordinary array boundary and placement metadata.
        let compiled: CompiledFunction<'_, ArrayType, (ArrayType, ArrayType, ArrayType)> = compile(
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
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let executable = compiled.executable();
        assert_eq!(executable.function.input_types(), &[ArrayIrType::Array(input_type.clone())]);
        assert_eq!(executable.output_types(), &[input_type.clone(), input_type.clone(), input_type.clone()]);

        // Reusing one executable must start each invocation's cotangent state at zero. In particular, returning to
        // the first input after another call must recover the same derivative instead of accumulating prior results.
        for (initial, expected_value, expected_state, expected_gradient) in
            [(3.0_f32, 36.0, 6.0, 24.0), (5.0, 100.0, 10.0, 40.0), (3.0, 36.0, 6.0, 24.0)]
        {
            let input = Array::from_host_buffer(
                &domain,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[initial]).as_slice(),
            )
            .unwrap();
            let (value, state, gradient) = executable.execute(&domain, input).unwrap();
            assert_eq!(read_f32_array(&client, &value), vec![expected_value]);
            assert_eq!(read_f32_array(&client, &state), vec![expected_state]);
            assert_eq!(read_f32_array(&client, &gradient), vec![expected_gradient]);
        }
    }

    #[test]
    fn test_public_jit_scan_indexes_its_captured_reference_root_in_nested_call() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        for (length, nested) in [(3, false), (3, true), (256, false), (256, true)] {
            let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(length)]));
            let reference_type = ArrayIrType::Reference(ReferenceType::new(input_type.clone()));

            // The callee captures the complete root and receives the scan index as an ordinary input. It forms its
            // scalar view locally, so no view crosses the call boundary. Capture 0 must still resolve to that same root
            // when reference discharge runs before compilation lifts the captured constant into an ordinary operand.
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let index = builder.add_input(ArrayType::scalar(DataType::I64).into());
            let root = builder.add_input(ReferenceType::new(input_type.clone()).into());
            // Compare selection inside the call with the direct slice-sized discharge path.
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
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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

            let compiled: CompiledFunction<'_, ArrayType, (ArrayType, ArrayType)> = compile(
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
                &domain,
                mesh.clone(),
            )
            .unwrap();
            // A larger root exercises repeated updates through the same compiled loop and nested call.
            let values = (1..=length).map(|value| value as f32).collect::<Vec<_>>();
            let expected = values.iter().map(|value| value * 2.0).collect::<Vec<_>>();
            let input =
                Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                    .unwrap();
            let (outputs, frozen) = compiled.executable().execute(&domain, input).unwrap();
            assert_eq!(read_f32_array(&client, &outputs), expected);
            assert_eq!(read_f32_array(&client, &frozen), expected);
        }
    }

    #[test]
    fn test_stateful_compiled_function_updates_public_holder_across_calls() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let initial =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap();
        assert!(initial.r#type().sharding().is_some());
        let reference = ArrayReference::new(initial);
        let retained_snapshot = reference.read().unwrap();
        let rejected_update =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        assert!(matches!(
            compiled.executable().execute(
                &domain,
                (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(rejected_update),)
            ),
            Err(Error::Execution(ExecutionError::StatefulInvocationRequired)),
        ));
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![1.0]);

        for expected in [3.0f32, 5.0] {
            let update =
                Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                    .unwrap();
            let output = compiled
                .execute_statefully_and_wait(
                    &domain,
                    (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
                )
                .unwrap();
            let Value::Array(output) = output else { panic!("stateful public output must be an array") };
            assert_eq!(read_f32_array(&client, &output), vec![expected]);
            assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![expected]);
        }
        assert_eq!(read_f32_array(&client, &retained_snapshot), vec![1.0]);
    }

    #[test]
    fn test_stateful_reference_transforms_match_uncompiled_transforms() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(scalar_type.clone()));
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), values_to_bytes::<f32>(values).as_slice())
                .unwrap()
        };
        let bind = |reference: &ArrayReference<_>| Value::Reference(Reference::new(reference.clone(), domain.clone()));
        let read = |value: Value<'_>| {
            let Value::Array(array) = value else { panic!("expected an array output") };
            read_f32_array(&client, &array)
        };

        // `r ↦ { r += read(r); read(r)² }` mutates its reference, and its gradient is taken with respect to the contents
        // of the reference at entry. Differentiating it inside a stateful compilation matches doing so eagerly.
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
        let compiled = compile_statefully::<_, ArrayIrType, (ArrayIrType, ArrayIrType)>(
            |reference| {
                Ok(differentiate_at(reference)
                    .value_and_gradient(|reference| square_after_doubling!(reference))
                    .map_err(ProgramError::from)?)
            },
            reference_type.clone(),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let compiled_reference = ArrayReference::new(array(&scalar_type, &[3.0]));
        let (compiled_value, compiled_gradient) =
            compiled.execute_statefully_and_wait(&domain, bind(&compiled_reference)).unwrap();
        assert_eq!((read(value), read(gradient)), (vec![36.0], vec![24.0]));
        assert_eq!((read(compiled_value), read(compiled_gradient)), (vec![36.0], vec![24.0]));
        assert_eq!(read_f32_array(&client, &eager_reference.read().unwrap()), vec![6.0]);
        assert_eq!(read_f32_array(&client, &compiled_reference.read().unwrap()), vec![6.0]);

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
        let compiled_error = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
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
            (reference_type, ArrayIrType::Array(vector_type)),
            &domain,
            Options::new(mesh),
        )
        .err()
        .unwrap();
        let message = "cannot store a batched value into an unbatched reference";
        assert!(eager_error.to_string().contains(message), "{eager_error}");
        assert!(compiled_error.to_string().contains(message), "{compiled_error}");
        assert_eq!(read_f32_array(&client, &shared_reference.read().unwrap()), vec![1.0]);
    }

    /// Traces `f(reference, update) = { reference += update; read(reference) }` over a `ref<f32[]>` and an `f32[]` into
    /// the XLA program universe, so that program-level transforms of a reference-bearing program can be replayed into a
    /// stateful compilation.
    fn accumulate_and_read_program(
        reference_type: &ArrayIrType,
        array_type: &ArrayType,
    ) -> ryft_core::Program<Constant, Operation, Vec<Constant>, Vec<Constant>> {
        let (_, program) = <Domain<'_>>::trace(
            |(reference, update): (DomainTracer<Domain<'_>>, DomainTracer<Domain<'_>>)| {
                reference.add_update(&update)?;
                reference.read()
            },
            (reference_type.clone(), ArrayIrType::Array(array_type.clone())),
        )
        .unwrap();
        program.into_flat_program()
    }

    #[test]
    fn test_stateful_compiled_function_commits_forward_mode_tangent_references() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));

        // The program-level transform derives `[r, x, ṫr, ẋ] ↦ [y, ẏ]`, and replaying it into the stateful trace makes
        // both the reference and its tangent reference public holders of the compiled function.
        let jvp = accumulate_and_read_program(&reference_type, &array_type).jvp().unwrap();
        assert_eq!(
            jvp.input_types(),
            vec![
                reference_type.clone(),
                ArrayIrType::Array(array_type.clone()),
                reference_type.clone(),
                ArrayIrType::Array(array_type.clone()),
            ],
        );
        let compiled =
            compile_statefully::<_, (ArrayIrType, ArrayIrType, ArrayIrType, ArrayIrType), (ArrayIrType, ArrayIrType)>(
                |(reference, update, tangent_reference, tangent_update)| {
                    let context = reference.context().clone();
                    let mut outputs =
                        jvp.interpret_in_context(&context, vec![reference, update, tangent_reference, tangent_update])?;
                    let tangent = outputs.remove(1);
                    Ok((outputs.remove(0), tangent))
                },
                (
                    reference_type.clone(),
                    ArrayIrType::Array(array_type.clone()),
                    reference_type,
                    ArrayIrType::Array(array_type.clone()),
                ),
                &domain,
                Options::new(mesh.clone()),
            )
            .unwrap();

        // Both references are committed after the call: the primal holder accumulates the update and the tangent
        // holder accumulates the tangent update, and the outputs read the committed states.
        let scalar = |value: f32| {
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };
        let reference = ArrayReference::new(scalar(1.0));
        let tangent_reference = ArrayReference::new(scalar(10.0));
        let (primal, tangent) = compiled
            .execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(reference.clone(), domain.clone())),
                    Value::Array(scalar(3.0)),
                    Value::Reference(Reference::new(tangent_reference.clone(), domain.clone())),
                    Value::Array(scalar(100.0)),
                ),
            )
            .unwrap();
        let Value::Array(primal) = primal else { panic!("stateful primal output must be an array") };
        let Value::Array(tangent) = tangent else { panic!("stateful tangent output must be an array") };
        assert_eq!(read_f32_array(&client, &primal), vec![4.0]);
        assert_eq!(read_f32_array(&client, &tangent), vec![110.0]);
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![4.0]);
        assert_eq!(read_f32_array(&client, &tangent_reference.read().unwrap()), vec![110.0]);
    }

    #[test]
    fn test_stateful_compiled_function_commits_reverse_mode_cotangent_references() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));

        // The program-level pullback of `y = { r += x; read(r) }` under a `Reference` destination for `r` consumes
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
        assert_eq!(pullback.input_types(), vec![ArrayIrType::Array(array_type.clone()), reference_type.clone()]);
        assert_eq!(pullback.output_types(), vec![reference_type.clone(), ArrayIrType::Array(array_type.clone())]);
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(cotangent_reference, output_cotangent)| {
                let context = cotangent_reference.context().clone();
                let mut outputs =
                    pullback.interpret_in_context(&context, vec![output_cotangent, cotangent_reference])?;
                Ok(outputs.remove(1))
            },
            (reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();

        // The read accumulates `ȳ` into the destination and the accumulation hands the accumulated cotangent to `x̄`
        // while leaving the destination unchanged, so both end at `5 + 2`.
        let scalar = |value: f32| {
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };
        let cotangent_reference = ArrayReference::new(scalar(5.0));
        let input_cotangent = compiled
            .execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(cotangent_reference.clone(), domain.clone())),
                    Value::Array(scalar(2.0)),
                ),
            )
            .unwrap();
        let Value::Array(input_cotangent) = input_cotangent else {
            panic!("stateful cotangent output must be an array")
        };
        assert_eq!(read_f32_array(&client, &input_cotangent), vec![7.0]);
        assert_eq!(read_f32_array(&client, &cotangent_reference.read().unwrap()), vec![7.0]);
    }

    #[test]
    fn test_array_ir_value_and_gradient_with_captures_on_the_domain() {
        use ryft_core::Add;

        // Array IR tracers dispatch `add` and `dot` through the array-IR capability universe without projecting to
        // arrays first, and the captured gradient of `f(x; c) = (x + c) · c` is `c`.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let vector = |values: &[f32]| {
            Value::Array(
                Array::from_host_buffer(
                    &domain,
                    array_type.clone(),
                    mesh.clone(),
                    values_to_bytes::<f32>(values).as_slice(),
                )
                .unwrap(),
            )
        };
        let read =
            |value: Value<'_>| read_f32_array(&client, &ValueProjection::<ArrayType>::into_projected(value).unwrap());
        let (value, gradient) = domain
            .differentiate_at(vector(&[1.0, 2.0]))
            .with_captures(vector(&[3.0, 4.0]))
            .value_and_gradient(|x, c| x.add(&c)?.dot(&c, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])))
            .unwrap();
        assert_eq!(read(value), vec![36.0]);
        assert_eq!(read(gradient), vec![3.0, 4.0]);
    }

    #[test]
    fn test_rematerialized_function_differentiates_on_the_domain() {
        use ryft_core::{DotsSavable, ResidualPolicy, SaveAndOffloadOnlyTheseNames, Tag, rematerialize};

        // `x ↦ sin(tag(x · x, "dot"))` executes its value and gradient on the XLA domain under a policy that saves the
        // dot product and under one that offloads it to pinned host memory, and both match `sin(x · x)` and
        // `2 cos(x · x) x`, which the residual placement must not change.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = [0.1f32, 0.2, 0.3];
        let dot = values.iter().map(|value| value * value).sum::<f32>();
        type ProgramTracer<'c> = DomainTracer<Domain<'c>>;
        fn sine_of_dot(x: ProgramTracer<'_>) -> Result<ProgramTracer<'_>, ProgramError> {
            let x = ValueProjection::<ArrayType>::into_projected(x)?;
            let dot = x.dot(&x, &DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]))?.tag("dot")?;
            Ok(dot.sin()?.into_value())
        }
        fn value_and_gradient<'c, P: Clone + ResidualPolicy<ArrayIrType>>(
            domain: &Domain<'c>,
            input: Array<'c>,
            policy: P,
        ) -> (Value<'c>, Value<'c>) {
            let function = rematerialize(sine_of_dot).with_policy(policy);
            domain.differentiate_at(Value::Array(input)).value_and_gradient(|x| function.call(x)).unwrap()
        }
        let input = || {
            Array::from_host_buffer(
                &domain,
                array_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&values).as_slice(),
            )
            .unwrap()
        };
        let offload = SaveAndOffloadOnlyTheseNames::new(Vec::<String>::new(), ["dot"], Memory::Host { pinned: true });
        for (value, gradient) in
            [value_and_gradient(&domain, input(), DotsSavable), value_and_gradient(&domain, input(), offload.unwrap())]
        {
            // XLA's elementary functions may differ from the host ones in the last bit.
            let assert_close = |actual: Value<'_>, expected: Vec<f32>| {
                let actual = read_f32_array(&client, &ValueProjection::<ArrayType>::into_projected(actual).unwrap());
                assert!(actual.iter().zip(&expected).all(|(actual, expected)| (actual - expected).abs() < 1e-6));
                assert_eq!(actual.len(), expected.len());
            };
            assert_close(value, vec![dot.sin()]);
            assert_close(gradient, values.iter().map(|value| 2.0 * dot.cos() * value).collect());
        }
    }

    #[test]
    fn test_stateful_compiled_function_commits_accumulating_custom_backward_buffers() {
        use ryft_core::{ArraySliceAxis, CotangentAccumulator, MaybeZero, Slice, TranspositionContext, composite};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::new_static(DataType::F32, [4]);

        // `x ↦ x[1]`, whose accumulating backward rule adds its seed to only the affected entry of the caller buffer.
        // The compiled pullback therefore updates the buffer in place, without a full-sized cotangent temporary.
        type ProgramTracer<'c> = DomainTracer<Domain<'c>>;
        let element = |x: &ProgramTracer<'_>| -> Result<ProgramTracer<'_>, ProgramError> {
            Ok(ValueProjection::<ArrayType>::into_projected(x.clone())?.slice(&[1], &[2usize], &[1])?.into_value())
        };
        let function = composite(move |x: ProgramTracer<'_>| element(&x)).with_accumulating_vjp(
            move |x: ProgramTracer<'_>| Ok((element(&x)?, ())),
            |context: &mut TranspositionContext<Constant, Operation>,
             (): (),
             seed: MaybeZero<ProgramTracer<'_>>,
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
            Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), values_to_bytes::<f32>(values)).unwrap()
        };
        let input = Value::Array(array(&array_type, &[1.0, 2.0, 3.0, 4.0]));
        let (_, pullback) = domain.differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert!(pullback.residuals().is_empty());
        let pullback = pullback.transposed_program(&[CotangentDestinationKind::Reference]).unwrap().into_program();
        let [seed_type, buffer_type] = <[ArrayIrType; 2]>::try_from(pullback.input_types().into_owned()).unwrap();
        assert!(matches!(&buffer_type, ArrayIrType::Reference(_)));
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
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ()>(
            |(buffer, seed)| {
                let context = buffer.context().clone();
                pullback.interpret_in_context(&context, vec![seed, buffer])?;
                Ok(())
            },
            (buffer_type, seed_type),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let buffer = ArrayReference::new(array(&array_type, &[1.0, 1.0, 1.0, 1.0]));
        compiled
            .execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(buffer.clone(), domain.clone())),
                    Value::Array(array(&ArrayType::new_static(DataType::F32, [1]), &[5.0])),
                ),
            )
            .unwrap();
        assert_eq!(read_f32_array(&client, &buffer.read().unwrap()), vec![1.0, 6.0, 1.0, 1.0]);
    }

    #[test]
    fn test_stateful_compilation_rejects_escaping_reference_outputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32);

        // A reference allocated inside the program has no host identity the ABI could hand back.
        let error = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |input| input.reference_new().map_err(Into::into),
            ArrayIrType::Array(array_type.clone()),
            &domain,
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
        let error = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            Ok,
            ArrayIrType::Reference(ReferenceType::new(array_type)),
            &domain,
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
    fn test_stateful_compilation_stages_reference_outputs_and_rejects_them_when_lowered() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        fn allocate_reference<'c>(
            _: Vec<Constant>,
            _: Vec<StatefulCompileTracer<'c>>,
            input: StatefulCompileTracer<'c>,
        ) -> Result<StatefulCompileTracer<'c>, Error> {
            input.reference_new().map_err(Into::into)
        }

        // The stateful XLA ABI only restricts the outputs of a lowered program, so a function that returns a reference
        // stages successfully and is rejected when it is lowered on its own.
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            allocate_reference,
            Vec::new(),
            ArrayIrType::Array(array_type),
            Options::new(mesh),
        )
        .unwrap();
        let error = staged.lower(&domain).err().unwrap();
        assert!(matches!(
            error,
            Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason })
                if reason == "output 0 is a reference allocated inside the program and cannot be returned through the \
                              stateful XLA ABI, which has no protocol to turn a backend array into a host reference",
        ));
    }

    #[test]
    fn test_staged_function_explicit_context_calls_without_input_leaves() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32);
        let capture = Value::Array(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 7f32.to_ne_bytes().as_slice()).unwrap(),
        );
        let inner = CoreStagedFunction::<Domain<'_>, (), ArrayIrType>::stage(
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

        let outer = CoreStagedFunction::<Domain<'_>, ArrayIrType, (ArrayIrType, ArrayIrType)>::stage(
            |_, _, input| {
                // No input leaf can recover the enclosing trace when the callee has no public inputs.
                assert!(matches!(
                    inner.call::<StatefulCompileTracer<'_>, _>(()),
                    Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
                ));
                let context = input.context();
                let typed = inner.call_in_context(context, ())?;
                let mut flat = inner.call_flat_in_context(context, Vec::new())?;
                Ok((typed, flat.remove(0)))
            },
            Vec::new(),
            ArrayIrType::Array(array_type.clone()),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let compiled = outer.lower(&domain).unwrap().compile(&domain).unwrap();
        let input =
            Value::Array(Array::from_host_buffer(&domain, array_type, mesh, 1f32.to_ne_bytes().as_slice()).unwrap());
        let (Value::Array(typed), Value::Array(flat)) = compiled.executable().execute(&domain, input).unwrap() else {
            panic!("capture-only calls must return arrays");
        };
        assert_eq!(read_f32_array(&client, &typed), vec![7.0]);
        assert_eq!(read_f32_array(&client, &flat), vec![7.0]);
    }

    #[test]
    fn test_stateful_compilation_forwards_references_through_nested_calls() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));

        fn forward<'c>(
            _: Vec<Constant>,
            _: Vec<StatefulCompileTracer<'c>>,
            input: StatefulCompileTracer<'c>,
        ) -> Result<StatefulCompileTracer<'c>, Error> {
            Ok(input)
        }

        // A nested call may return the external reference that it receives, because that reference never crosses the
        // ABI as an output of the enclosing program.
        let inner = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            forward,
            Vec::new(),
            reference_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let outer = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            move |reference| inner.call(reference)?.read().map_err(Into::into),
            reference_type,
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type, mesh, 7.0f32.to_ne_bytes().as_slice()).unwrap(),
        );
        let output = outer
            .execute_statefully_and_wait(&domain, Value::Reference(Reference::new(reference.clone(), domain.clone())))
            .unwrap();
        let Value::Array(output) = output else { panic!("stateful public output must be an array") };
        assert_eq!(read_f32_array(&client, &output), vec![7.0]);
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![7.0]);
    }

    #[test]
    fn test_stateful_compilation_rejects_nested_calls_that_return_allocated_references() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        fn allocate_reference<'c>(
            _: Vec<Constant>,
            _: Vec<StatefulCompileTracer<'c>>,
            input: StatefulCompileTracer<'c>,
        ) -> Result<StatefulCompileTracer<'c>, Error> {
            input.reference_new().map_err(Into::into)
        }

        // A reference that a nested call allocates has no caller allocation that could receive it, so embedding the
        // call is rejected with a diagnostic that names the offending output.
        let inner = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            allocate_reference,
            Vec::new(),
            ArrayIrType::Array(array_type.clone()),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let error = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            move |input| inner.call(input)?.freeze().map_err(Into::into),
            ArrayIrType::Array(array_type),
            &domain,
            Options::new(mesh),
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            Error::Core(ryft_core::Error::Program(ProgramError::UnsupportedOperation { message }))
                if message == "output 0 of the called function is a reference that the function allocates itself, \
                               which a nested call cannot return; allocate the reference in the caller and pass it to \
                               the function instead",
        ));
    }

    #[test]
    fn test_stateful_compiled_function_delegates_an_all_array_signature() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |input| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                Ok((input.clone() + input).into_value())
            },
            ArrayIrType::Array(array_type.clone()),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let input = Array::from_host_buffer(&domain, array_type, mesh, 3.0f32.to_ne_bytes().as_slice()).unwrap();

        let output = compiled.execute_statefully_and_wait(&domain, Value::Array(input)).unwrap();
        let Value::Array(output) = output else { panic!("stateful all-array output must be an array") };
        assert_eq!(read_f32_array(&client, &output), vec![6.0]);
    }

    #[test]
    fn test_stateful_async_reference_free_signature_reports_completion_through_its_await() {
        // A reference-free invocation binds no holders, but the asynchronous surface still promises that awaiting the
        // returned execution observes whole-invocation completion, so it returns a pending execution carrying the
        // execution fence rather than an already-completed result.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |input| {
                let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
                Ok((input.clone() + input).into_value())
            },
            ArrayIrType::Array(array_type.clone()),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let input = Array::from_host_buffer(&domain, array_type, mesh, 3.0f32.to_ne_bytes().as_slice()).unwrap();

        let output = compiled.execute_statefully(&domain, Value::Array(input)).r#await().unwrap();
        let Value::Array(output) = output else { panic!("stateful all-array output must be an array") };
        assert_eq!(read_f32_array(&client, &output), vec![6.0]);
    }

    #[test]
    fn test_stateful_read_only_holder_remains_ready_across_calls() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |reference| reference.read().map_err(Into::into),
            ArrayIrType::Reference(ReferenceType::new(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type, mesh, 7.0f32.to_ne_bytes().as_slice()).unwrap(),
        );
        for _ in 0..2 {
            let output = compiled
                .execute_statefully_and_wait(
                    &domain,
                    Value::Reference(Reference::new(reference.clone(), domain.clone())),
                )
                .unwrap();
            let Value::Array(output) = output else { panic!("stateful public output must be an array") };
            assert_eq!(read_f32_array(&client, &output), vec![7.0]);
            assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![7.0]);
        }
    }

    #[test]
    fn test_executable_function_execute_async_rejects_a_reference_bearing_executable() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |reference| reference.read().map_err(Into::into),
            ArrayIrType::Reference(ReferenceType::new(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();

        // A stateful compilation produces an ordinary runtime executable over the same flat signature, so the
        // array-only asynchronous entry point is reachable and must refuse the holder transaction up front rather
        // than enqueueing an execution whose hidden final-state outputs nothing would publish.
        let executable: ExecutableFunction<'_, ArrayType, ArrayType> =
            ExecutableFunction { function: compiled.executable().clone() };
        let input = Array::from_host_buffer(&domain, array_type, mesh, 7.0f32.to_ne_bytes().as_slice()).unwrap();
        assert!(matches!(
            executable.execute_async(&domain, input),
            Err(Error::Execution(ExecutionError::StatefulInvocationRequired)),
        ));
    }

    #[test]
    fn test_stateful_async_read_leases_overlap_and_block_mutation_until_both_complete() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let read = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |reference| reference.read().map_err(Into::into),
            reference_type.clone(),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let mutate = compile_statefully::<_, (ArrayIrType, ArrayIrType), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        // Both gates report into one labelled channel, so the assertions below pin which read lease the mutation
        // blocks on first instead of silently depending on the order the transaction awaits its leases.
        let (await_started, await_observed) = mpsc::channel();
        let first_gate = ControlledReferenceCompletion::with_await_notification("first read", await_started.clone());
        inject_stateful_completion(CoreReferenceCompletion::new(first_gate.clone()));
        let first_read =
            read.execute_statefully(&domain, Value::Reference(Reference::new(reference.clone(), domain.clone())));
        let second_gate = ControlledReferenceCompletion::with_await_notification("second read", await_started);
        inject_stateful_completion(CoreReferenceCompletion::new(second_gate.clone()));
        let second_read =
            read.execute_statefully(&domain, Value::Reference(Reference::new(reference.clone(), domain.clone())));

        std::thread::scope(|scope| {
            let (mutation_finished, mutation_result) = mpsc::channel();
            let mutation_reference = reference.clone();
            let domain = &domain;
            let mutate = &mutate;
            scope.spawn(move || {
                let update =
                    Array::from_host_buffer(domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();
                mutation_finished
                    .send(mutate.execute_statefully_and_wait(
                        domain,
                        (Value::Reference(Reference::new(mutation_reference, domain.clone())), Value::Array(update)),
                    ))
                    .unwrap();
            });

            assert_eq!(await_observed.recv_timeout(Duration::from_secs(5)).unwrap(), "first read");
            assert!(matches!(mutation_result.try_recv(), Err(mpsc::TryRecvError::Empty)));
            first_gate.complete(Ok(()));
            assert_eq!(await_observed.recv_timeout(Duration::from_secs(5)).unwrap(), "second read");
            assert!(matches!(mutation_result.try_recv(), Err(mpsc::TryRecvError::Empty)));
            second_gate.complete(Ok(()));
            mutation_result.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        });
        first_read.r#await().unwrap();
        second_read.r#await().unwrap();
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_stateful_zero_public_output_waits_for_reference_replacement_commit() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (ArrayIrType::Reference(ReferenceType::new(array_type.clone())), ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let update = Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        compiled
            .execute_statefully(
                &domain,
                (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
            )
            .r#await()
            .unwrap();
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_stateful_async_chains_unawaited_mutations_and_preserves_prior_failure() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (ArrayIrType::Reference(ReferenceType::new(array_type.clone())), ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let predecessor = ControlledReferenceCompletion::new();
        inject_stateful_completion(CoreReferenceCompletion::new(predecessor.clone()));
        let first_update =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        let first = compiled.execute_statefully(
            &domain,
            (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(first_update)),
        );
        let second_update =
            Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        let second = compiled.execute_statefully(
            &domain,
            (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(second_update)),
        );

        predecessor.complete(Err(Arc::from("injected predecessor failure")));
        for execution in [first, second] {
            assert!(matches!(
                execution.r#await(),
                Err(Error::Execution(ExecutionError::AsynchronousFailure { reason }))
                    if reason.as_ref() == "injected predecessor failure",
            ));
        }
        let error = reference.read().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ExecutionPoisoned { reason: "injected predecessor failure".to_string() }),
        );
    }

    #[test]
    fn test_stateful_async_drop_still_reconciles_pending_holder_on_access() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (ArrayIrType::Reference(ReferenceType::new(array_type.clone())), ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let completion = ControlledReferenceCompletion::new();
        inject_stateful_completion(CoreReferenceCompletion::new(completion.clone()));
        let update = Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        drop(compiled.execute_statefully(
            &domain,
            (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
        ));

        completion.complete(Ok(()));
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_stateful_call_rejects_non_root_external_view_without_consuming_root() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |reference| reference.read().map_err(Into::into),
            ArrayIrType::Reference(ReferenceType::new(scalar_type)),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let root = ArrayReference::new(
            Array::from_host_buffer(&domain, vector_type, mesh, values_to_bytes::<f32>(&[4.0, 9.0]).as_slice())
                .unwrap(),
        );
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(&domain, Value::Reference(Reference::new(view, domain.clone()))),
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message: reason })))
                if reason
                    == "external state input 0 must be a root reference handle that uses its allocation's stored \
                        type identities",
        ));
        assert_eq!(read_f32_array(&client, &root.read().unwrap()), vec![4.0, 9.0]);
    }

    #[test]
    fn test_stateful_external_holder_accepts_static_sharding_and_rejects_mismatched_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        // Types record placement over explicit mesh axes only, so the alternate static sharding uses an explicit axis.
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let domain = Session::new(&client).domain();
        let shape = Shape::new(vec![CoreDimension::Static(2)]);
        let replicated_type = ArrayType::new(DataType::F32, shape.clone())
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let alternate_type = ArrayType::new(DataType::F32, shape)
            .with_sharding(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let sharded = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |reference| reference.read().map_err(Into::into),
            ArrayIrType::Reference(ReferenceType::new(alternate_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &domain,
                alternate_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[1.0, 2.0]).as_slice(),
            )
            .unwrap(),
        );
        let Value::Array(output) = sharded
            .execute_statefully_and_wait(&domain, Value::Reference(Reference::new(reference.clone(), domain.clone())))
            .unwrap()
        else {
            panic!("stateful read output must be an array")
        };
        assert_eq!(read_f32_array(&client, &output), vec![1.0, 2.0]);
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (
                ArrayIrType::Reference(ReferenceType::new(replicated_type.clone())),
                ArrayIrType::Array(replicated_type.clone()),
            ),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let update = Array::from_host_buffer(
            &domain,
            replicated_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[3.0, 4.0]).as_slice(),
        )
        .unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
            ),
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message: reason })))
                if reason == format!(
                    "external state input 0 reference type `{alternate_type}` does not refine effective compiled state \
                     type `{replicated_type}`",
                ),
        ));
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![1.0, 2.0]);
    }

    #[test]
    fn test_stateful_external_holder_updates_two_device_sharded_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        // Types record placement over explicit mesh axes only, so the statically sharded state uses an explicit axis.
        let mesh = two_device_mesh(&client, MeshAxisType::Explicit);
        let domain = Session::new(&client).domain();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(reference, update)| {
                reference.add_update(&update)?;
                reference.read().map_err(Into::into)
            },
            (ArrayIrType::Reference(ReferenceType::new(array_type.clone())), ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &domain,
                array_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &domain,
            array_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();

        let Value::Array(output) = compiled
            .execute_statefully_and_wait(
                &domain,
                (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
            )
            .unwrap()
        else {
            panic!("stateful sharded output must be an array")
        };
        let expected = vec![11.0, 22.0, 33.0, 44.0];
        assert_eq!(read_sharded_f32_array(&output), expected);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), expected);

        let reversed_mesh =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().cloned().collect()).unwrap();
        let reversed_reference = ArrayReference::new(
            Array::from_host_buffer(
                &domain,
                array_type.clone(),
                reversed_mesh,
                values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &domain,
            array_type,
            mesh,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(reversed_reference.clone(), domain.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message: reason })))
                if reason == "external state input 0 reference mesh does not match the compiled device mesh",
        ));
        assert_eq!(read_sharded_f32_array(&reversed_reference.read().unwrap()), vec![1.0, 2.0, 3.0, 4.0],);
    }

    #[test]
    fn test_stateful_shard_map_jvp_preserves_nonlinear_residuals_on_each_device() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let domain = Session::new(&client).domain();
        let program = nonlinear_reference_shard_map_program(&logical_mesh, false).jvp().unwrap();
        let inputs = program.input_types().into_owned();
        let array_type = <&ReferenceType<ArrayType>>::try_from(&inputs[0]).unwrap().referent().clone();
        let compiled = compile_statefully::<_, Vec<ArrayIrType>, Vec<ArrayIrType>>(
            |inputs| program.interpret_in_context(inputs[0].context(), inputs.clone()).map_err(Into::into),
            inputs,
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), values_to_bytes(values).as_slice())
                .unwrap()
        };
        let reference = ArrayReference::new(array(&[2.0, 3.0, 5.0, 7.0]));
        let tangent_reference = ArrayReference::new(array(&[23.0, 29.0, 31.0, 37.0]));
        let outputs = compiled
            .execute_statefully_and_wait(
                &domain,
                vec![
                    Value::Reference(Reference::new(reference.clone(), domain.clone())),
                    Value::Array(array(&[11.0, 13.0, 17.0, 19.0])),
                    Value::Reference(Reference::new(tangent_reference.clone(), domain.clone())),
                    Value::Array(array(&[41.0, 43.0, 47.0, 53.0])),
                ],
            )
            .unwrap();
        let outputs = outputs
            .iter()
            .map(|output| match output {
                Value::Array(array) => read_sharded_f32_array(array),
                _ => panic!("nonlinear outputs are arrays"),
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![vec![22.0, 39.0, 85.0, 133.0], vec![335.0, 506.0, 762.0, 1074.0]]);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![2.0, 3.0, 5.0, 7.0]);
        assert_eq!(read_sharded_f32_array(&tangent_reference.read().unwrap()), vec![23.0, 29.0, 31.0, 37.0]);
    }

    #[test]
    fn test_stateful_shard_map_reverse_sums_replicated_reference_contributions() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let domain = Session::new(&client).domain();
        let program = nonlinear_reference_shard_map_program(&logical_mesh, true);
        let mut input_types = program.input_types().into_owned();
        let reference_type = <&ReferenceType<ArrayType>>::try_from(&input_types[0]).unwrap().referent().clone();
        let array_type = <&ArrayType>::try_from(&input_types[1]).unwrap().clone();
        input_types.extend(input_types.clone());
        let (primal, tangent, ..) = program.linearize().unwrap().into_parts();
        let pullback = tangent
            .transpose_with_respect_to(
                &[0, 1],
                &[CotangentDestinationKind::Reference, CotangentDestinationKind::Return],
            )
            .unwrap();
        let compiled = compile_statefully::<_, Vec<ArrayIrType>, Vec<ArrayIrType>>(
            |inputs| {
                let context = inputs[0].context().clone();
                let mut primals = primal.interpret_in_context(&context, inputs[..2].to_vec())?;
                let output = primals.remove(0);
                let mut operands = vec![inputs[3].clone(), inputs[2].clone()];
                operands.extend(primals);
                let mut cotangents = pullback.interpret_in_context(&context, operands)?;
                Ok(vec![output, cotangents.remove(1)])
            },
            input_types,
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&domain, r#type.clone(), mesh.clone(), values_to_bytes(values).as_slice()).unwrap()
        };
        let reference = ArrayReference::new(array(&reference_type, &[2.0, 3.0]));
        let destination = ArrayReference::new(array(&reference_type, &[5.0, 7.0]));
        let outputs = compiled
            .execute_statefully_and_wait(
                &domain,
                vec![
                    Value::Reference(Reference::new(reference.clone(), domain.clone())),
                    Value::Array(array(&array_type, &[11.0, 13.0, 17.0, 19.0])),
                    Value::Reference(Reference::new(destination.clone(), domain.clone())),
                    Value::Array(array(&array_type, &[1.0, 2.0, 3.0, 4.0])),
                ],
            )
            .unwrap();
        let outputs = outputs
            .iter()
            .map(|output| match output {
                Value::Array(array) => read_sharded_f32_array(array),
                _ => panic!("nonlinear outputs are arrays"),
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![vec![22.0, 39.0, 34.0, 57.0], vec![2.0, 6.0, 6.0, 12.0]]);
        assert_eq!(read_sharded_f32_array(&destination.read().unwrap()), vec![67.0, 109.0, 67.0, 109.0]);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn test_stateful_shard_map_reverse_adds_replicated_reference_cotangents_once() {
        // The output seed `[1, 2]` reaches the replicated reference's cotangent destination `[5, 7]` exactly once,
        // without any normalization by the size of the manual axis or per-device duplication of the destination.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let domain = Session::new(&client).domain();
        let sharding = Sharding::replicated(logical_mesh.clone(), 1);
        let array_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = body.add_input(reference_type.clone());
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let body = body
            .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = ShardMapOperation::from_program(
            &body,
            vec![reference_type.clone()],
            ShardMapBoundary::new(
                logical_mesh.clone(),
                vec![sharding.clone()],
                vec![sharding.clone()],
                vec!["x".to_string()],
            )
            .unwrap(),
        )
        .unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = builder.add_input(reference_type.clone());
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![reference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let compiled = compile_statefully::<_, Vec<ArrayIrType>, Vec<ArrayIrType>>(
            |inputs| {
                let context = inputs[0].context().clone();
                let outputs = pullback.interpret_in_context(&context, inputs)?;
                Ok(vec![context.bind(ReferenceReadOperation::new(), Vec::new(), &outputs)?.remove(0)])
            },
            vec![ArrayIrType::Array(array_type.clone()), reference_type.clone()],
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), values_to_bytes(values).as_slice())
                .unwrap()
        };
        let destination = ArrayReference::new(array(&[5.0, 7.0]));
        compiled
            .execute_statefully_and_wait(
                &domain,
                vec![
                    Value::Array(array(&[1.0, 2.0])),
                    Value::Reference(Reference::new(destination.clone(), domain.clone())),
                ],
            )
            .unwrap();
        assert_eq!(read_sharded_f32_array(&destination.read().unwrap()), vec![6.0, 9.0, 6.0, 9.0]);
    }

    /// Builds the program `r -> shard_map(|r| r += read(r))(r)` for an unchecked map over a two-device mesh whose
    /// reference input is reduced along the manual axis `x`, together with the mesh, the domain, and the referent type.
    pub(crate) fn doubling_reduced_reference_program(
        logical_mesh: &LogicalMesh,
    ) -> (ryft_core::Program<Constant, Operation, Vec<Constant>, Vec<Constant>>, ArrayType) {
        let reduced = Sharding::replicated(logical_mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let boundary = ShardMapBoundary::new(logical_mesh.clone(), vec![reduced.clone()], Vec::new(), Vec::new())
            .unwrap()
            .with_check_manual_variation(false);
        let referent_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(reduced).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
        let local_type = boundary.local_input_type(0, &referent_type).unwrap();
        let reference = body.add_input(ArrayIrType::Reference(ReferenceType::new(local_type)));
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let body = body.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_program(&body, vec![reference_type.clone()], boundary).unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = builder.add_input(reference_type);
        let body = builder.import_program(body);
        builder
            .add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![reference], None)
            .unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        (program, referent_type)
    }

    #[test]
    fn test_stateful_unchecked_shard_map_writes_back_reduced_references() {
        // Every device receives the whole reduced referent `[1, 2]`, doubles it, and writes back its copy, so both
        // devices hold `[2, 4]` afterwards.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let domain = Session::new(&client).domain();
        let (program, referent_type) = doubling_reduced_reference_program(mesh.logical_mesh());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
        let compiled = compile_statefully::<_, Vec<ArrayIrType>, Vec<ArrayIrType>>(
            |inputs| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)?;
                Ok(Vec::new())
            },
            vec![reference_type],
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, referent_type, mesh, values_to_bytes(&[1f32, 2.0]).as_slice()).unwrap(),
        );
        compiled
            .execute_statefully_and_wait(
                &domain,
                vec![Value::Reference(Reference::new(reference.clone(), domain.clone()))],
            )
            .unwrap();
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![2.0, 4.0, 2.0, 4.0]);
    }

    #[test]
    fn test_stateful_unchecked_shard_map_reverse_accumulates_mutated_reduced_gradient_references() {
        // The map doubles a reduced reference in place, so the cotangent of its initial state is twice that of its
        // final state. The gradient reference is unreduced, so each device doubles its own summand in place: the
        // logical destination `[5, 7]`, uploaded as the summands `[5, 7]` and `[0, 0]`, becomes `[10, 14]`.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let domain = Session::new(&client).domain();
        let (program, referent_type) = doubling_reduced_reference_program(mesh.logical_mesh());
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let cotangent_type =
            referent_type.clone().with_sharding(referent_type.sharding().unwrap().cotangent().unwrap()).unwrap();
        let cotangent_reference_type = ArrayIrType::Reference(ReferenceType::new(cotangent_type.clone()));
        let compiled = compile_statefully::<_, Vec<ArrayIrType>, Vec<ArrayIrType>>(
            |inputs| {
                let context = inputs[0].context().clone();
                pullback.interpret_in_context(&context, inputs)?;
                Ok(Vec::new())
            },
            vec![cotangent_reference_type],
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let destination = ArrayReference::new(
            Array::from_host_buffer(&domain, cotangent_type, mesh, values_to_bytes(&[5f32, 7.0]).as_slice()).unwrap(),
        );
        compiled
            .execute_statefully_and_wait(
                &domain,
                vec![Value::Reference(Reference::new(destination.clone(), domain.clone()))],
            )
            .unwrap();
        let destination = destination.read().unwrap();
        assert_eq!(read_sharded_f32_array(&destination), vec![10.0, 14.0, 0.0, 0.0]);
        assert_eq!(
            values_from_bytes::<f32>(&crate::arrays_v0::host::materialize_dense_array_bytes(&destination).unwrap()),
            vec![10.0, 14.0],
        );
    }

    /// A manual `shard_map` over a reference sharded along its manual axis, run through the stateful ABI: each device
    /// adds its shard of the update into the shard of the referent it owns and reads that shard back, so the output
    /// and the committed global referent both carry every device's write in mesh order.
    #[test]
    fn test_stateful_shard_map_updates_two_device_sharded_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let domain = Session::new(&client).domain();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let sharded_type = global_type.clone().with_sharding(sharded.clone()).unwrap();

        // The boundary is `[r: ref<f32[4]>, x: f32[4]] -> f32[4]`, everything sharded along `x`, and the local body
        // performs `add_update(r_local, x_local); read(r_local)` over the `f32[2]` shards the device owns.
        let boundary = ShardMapBoundary::new(
            logical_mesh,
            vec![sharded.clone(), sharded.clone()],
            vec![sharded.clone()],
            vec!["x".to_string()],
        )
        .unwrap();
        let body = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let local_referent = boundary.local_input_type(0, &global_type).unwrap();
            let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(local_referent)));
            let update = builder.add_input(ArrayIrType::Array(boundary.local_input_type(1, &global_type).unwrap()));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![
                ArrayIrType::Reference(ReferenceType::new(sharded_type.clone())),
                ArrayIrType::Array(sharded_type.clone()),
            ],
            boundary,
        )
        .unwrap();
        assert_eq!(operation.global_output_types(), &[ArrayIrType::Array(sharded_type.clone())]);
        let operation = Operation::ShardMap(Box::new(operation));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(reference, update)| {
                let context = reference.domain();
                let mut outputs = context.bind(operation, vec![body], &[reference, update])?;
                Ok(outputs.pop().unwrap())
            },
            (
                ArrayIrType::Reference(ReferenceType::new(sharded_type.clone())),
                ArrayIrType::Array(sharded_type.clone()),
            ),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();

        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &domain,
                sharded_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &domain,
            sharded_type,
            mesh,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        let Value::Array(output) = compiled
            .execute_statefully_and_wait(
                &domain,
                (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
            )
            .unwrap()
        else {
            panic!("stateful shard-map output must be an array")
        };

        // Device 0 owns `[1, 2]` and device 1 owns `[3, 4]`; the concatenation of the per-device shards in mesh order
        // shows that each device mutated exactly its own shard, both in the read output and in the committed referent.
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(read_sharded_f32_array(&output), vec![11.0, 22.0, 33.0, 44.0]);
        let committed = reference.read().unwrap();
        assert_eq!(committed.sharding(), &sharded);
        assert_eq!(read_sharded_f32_array(&committed), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// A manual `shard_map` reading a reference replicated along its manual axis, run through the stateful ABI: every
    /// device sees the whole referent and adds it to the shard of the value input it owns, while the referent itself is
    /// left untouched on every device.
    #[test]
    fn test_stateful_shard_map_reads_replicated_reference_on_every_shard() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Manual);
        let logical_mesh = mesh.logical_mesh().clone();
        let domain = Session::new(&client).domain();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let referent_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]));
        let replicated_type = referent_type.clone().with_sharding(replicated.clone()).unwrap();
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let sharded_type = global_type.clone().with_sharding(sharded.clone()).unwrap();

        // The boundary is `[r: ref<f32[2]>, x: f32[4]] -> f32[4]` with `r` replicated and `x` and the output sharded
        // along `x`, and the local body performs `read(r_local) + x_local` over the whole `f32[2]` referent and the
        // `f32[2]` shard of `x` the device owns.
        let boundary = ShardMapBoundary::new(
            logical_mesh,
            vec![replicated, sharded.clone()],
            vec![sharded.clone()],
            vec!["x".to_string()],
        )
        .unwrap();
        let body = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let local_referent = boundary.local_input_type(0, &referent_type).unwrap();
            let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(local_referent)));
            let update = builder.add_input(ArrayIrType::Array(boundary.local_input_type(1, &global_type).unwrap()));
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let state = builder
                .add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![state], None)
                .unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, update], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![
                ArrayIrType::Reference(ReferenceType::new(replicated_type.clone())),
                ArrayIrType::Array(sharded_type.clone()),
            ],
            boundary,
        )
        .unwrap();
        assert_eq!(operation.global_output_types(), &[ArrayIrType::Array(sharded_type.clone())]);
        let operation = Operation::ShardMap(Box::new(operation));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(reference, update)| {
                let context = reference.domain();
                let mut outputs = context.bind(operation, vec![body], &[reference, update])?;
                Ok(outputs.pop().unwrap())
            },
            (
                ArrayIrType::Reference(ReferenceType::new(replicated_type.clone())),
                ArrayIrType::Array(sharded_type.clone()),
            ),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();

        let reference = ArrayReference::new(
            Array::from_host_buffer(
                &domain,
                replicated_type,
                mesh.clone(),
                values_to_bytes::<f32>(&[100.0, 200.0]).as_slice(),
            )
            .unwrap(),
        );
        let update = Array::from_host_buffer(
            &domain,
            sharded_type,
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let Value::Array(output) = compiled
            .execute_statefully_and_wait(
                &domain,
                (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
            )
            .unwrap()
        else {
            panic!("stateful shard-map output must be an array")
        };

        // Both devices add the whole `[100, 200]` referent to their own shard of `x`, and the replicated referent
        // (read here as its per-device copies in mesh order) is unchanged on both devices.
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(read_sharded_f32_array(&output), vec![101.0, 202.0, 103.0, 204.0]);
        assert_eq!(read_sharded_f32_array(&reference.read().unwrap()), vec![100.0, 200.0, 100.0, 200.0]);
    }

    #[test]
    fn test_stateful_external_holder_reads_bounded_dynamic_state_and_rejects_mutation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let extent = DimensionVariable::new("state_extent", DimensionBounds::new(1, Some(5)).unwrap());
        let declared_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let read = compile_statefully::<_, ArrayIrType, ArrayIrType>(
            |reference| reference.read().map_err(Into::into),
            ArrayIrType::Reference(ReferenceType::new(declared_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let mutation = compile_statefully::<_, (ArrayIrType, ArrayIrType), ()>(
            |(reference, update)| reference.add_update(&update).map_err(Into::into),
            (ArrayIrType::Reference(ReferenceType::new(declared_type.clone())), ArrayIrType::Array(declared_type)),
            &domain,
            Options::new(mesh.clone()),
        );
        assert!(matches!(
            mutation,
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "mutated bounded-dynamic external state input 0 is unsupported because backend alias \
                              compatibility has not been verified",
        ));

        for expected in [vec![1.0_f32, 2.0], vec![1.0_f32, 2.0, 3.0, 4.0]] {
            let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(expected.len())]))
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap();
            let reference = ArrayReference::new(
                Array::from_host_buffer(
                    &domain,
                    actual_type,
                    mesh.clone(),
                    values_to_bytes(expected.as_slice()).as_slice(),
                )
                .unwrap(),
            );
            let Value::Array(output) = read
                .execute_statefully_and_wait(
                    &domain,
                    Value::Reference(Reference::new(reference.clone(), domain.clone())),
                )
                .unwrap()
            else {
                panic!("stateful bounded-dynamic output must be an array")
            };
            assert_eq!(read_f32_array(&client, &output), expected);
            assert_eq!(read_f32_array(&client, &reference.read().unwrap()), expected);
        }
    }

    #[test]
    fn test_retained_stateful_jit_reuses_public_holder_specialization() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let function = jit_statefully::<_, (), (ArrayIrType, ArrayIrType), ArrayIrType>(
            |(), (reference, update): (StatefulCompileTracer<'_>, StatefulCompileTracer<'_>)| {
                reference.add_update(&update)?;
                Ok(reference.read()?)
            },
            &domain,
            mesh.clone(),
        );
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );

        for expected in [3.0f32, 5.0] {
            let update =
                Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                    .unwrap();
            let output = function
                .execute_statefully_and_wait(
                    (),
                    (Value::Reference(Reference::new(reference.clone(), domain.clone())), Value::Array(update)),
                )
                .unwrap();
            let Value::Array(output) = output else { panic!("stateful public output must be an array") };
            assert_eq!(read_f32_array(&client, &output), vec![expected]);
        }
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
    }

    #[test]
    fn test_stateful_compiled_function_updates_captured_holder() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let compiled = compile_statefully_with_captures::<_, ArrayIrType, ArrayIrType>(
            |captures, update| {
                captures[0].add_update(&update)?;
                captures[0].read().map_err(Into::into)
            },
            vec![Value::Reference(Reference::new(reference.clone(), domain.clone()))],
            ArrayIrType::Array(array_type.clone()),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let update = Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        let output = compiled.execute_statefully_and_wait(&domain, Value::Array(update)).unwrap();
        let Value::Array(output) = output else { panic!("stateful public output must be an array") };
        assert_eq!(read_f32_array(&client, &output), vec![3.0]);
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_stateful_call_rejects_duplicate_capture_and_public_holder_before_mutation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type, mesh.clone(), 1.0f32.to_ne_bytes().as_slice()).unwrap(),
        );
        let compiled = compile_statefully_with_captures::<_, ArrayIrType, ArrayIrType>(
            |captures, public_reference| {
                let update = public_reference.read()?;
                captures[0].add_update(&update)?;
                captures[0].read().map_err(Into::into)
            },
            vec![Value::Reference(Reference::new(reference.clone(), domain.clone()))],
            reference_type.clone(),
            &domain,
            Options::new(mesh),
        )
        .unwrap();
        let error = compiled
            .execute_statefully_and_wait(&domain, Value::Reference(Reference::new(reference.clone(), domain.clone())))
            .err();
        assert!(
            matches!(
                &error,
                Some(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                    if message == "capture 0 and input 0 bind the same reference allocation",
            ),
            "{error:?}",
        );
        assert_eq!(read_f32_array(&client, &reference.read().unwrap()), vec![1.0]);
    }

    #[test]
    fn test_stateful_compilation_rejects_duplicate_reference_captures_before_staging() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let result = compile_statefully_with_captures::<_, ArrayIrType, ArrayIrType>(
            |captures, _| captures[0].read().map_err(Into::into),
            vec![
                Value::Reference(Reference::new(reference.clone(), domain.clone())),
                Value::Reference(Reference::new(reference, domain.clone())),
            ],
            ArrayIrType::Array(array_type),
            &domain,
            Options::new(mesh),
        );
        assert!(matches!(
            result,
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                if message == "capture 1 and capture 0 bind the same reference allocation",
        ));
    }

    #[test]
    fn test_stateful_call_commits_two_mutated_references_bound_in_reverse_identity_order() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType, ArrayIrType), (ArrayIrType, ArrayIrType)>(
            |(first, second, update)| {
                first.add_update(&update)?;
                second.add_update(&update)?;
                Ok((first.read()?, second.read()?))
            },
            (reference_type.clone(), reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let first = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let second = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let (lower_reference, higher_reference) =
            if first.id() < second.id() { (first, second) } else { (second, first) };
        let update = Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();

        let (higher_output, lower_output) = compiled
            .execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(higher_reference.clone(), domain.clone())),
                    Value::Reference(Reference::new(lower_reference.clone(), domain.clone())),
                    Value::Array(update),
                ),
            )
            .unwrap();
        let Value::Array(higher_output) = higher_output else {
            panic!("first stateful public output must be an array")
        };
        let Value::Array(lower_output) = lower_output else { panic!("second stateful public output must be an array") };
        assert_eq!(read_f32_array(&client, &higher_output), vec![3.0]);
        assert_eq!(read_f32_array(&client, &lower_output), vec![3.0]);
        assert_eq!(read_f32_array(&client, &higher_reference.read().unwrap()), vec![3.0]);
        assert_eq!(read_f32_array(&client, &lower_reference.read().unwrap()), vec![3.0]);
    }

    #[test]
    fn test_overlapping_stateful_calls_lock_shared_holders_in_identity_order() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType, ArrayIrType), ()>(
            |(first, second, update)| {
                first.add_update(&update)?;
                second.add_update(&update).map_err(Into::into)
            },
            (reference_type.clone(), reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let first = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let second = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let (lower_reference, higher_reference) =
            if first.id() < second.id() { (first, second) } else { (second, first) };
        let higher_first_update =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        let lower_first_update =
            Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();

        // The two calls present the same holders in opposite argument order and start together. Both must finish:
        // internal identity ordering serializes their retained-guard windows without creating a lock cycle.
        std::thread::scope(|scope| {
            let start = Arc::new(Barrier::new(3));
            let (finished, results) = mpsc::channel();
            for (first, second, update) in [
                (higher_reference.clone(), lower_reference.clone(), higher_first_update),
                (lower_reference.clone(), higher_reference.clone(), lower_first_update),
            ] {
                let start = Arc::clone(&start);
                let finished = finished.clone();
                let compiled = &compiled;
                let domain = &domain;
                scope.spawn(move || {
                    start.wait();
                    finished
                        .send(compiled.execute_statefully_and_wait(
                            domain,
                            (
                                Value::Reference(Reference::new(first, domain.clone())),
                                Value::Reference(Reference::new(second, domain.clone())),
                                Value::Array(update),
                            ),
                        ))
                        .unwrap();
                });
            }
            drop(finished);
            start.wait();
            for _ in 0..2 {
                results.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
            }
        });
        assert_eq!(read_f32_array(&client, &lower_reference.read().unwrap()), vec![5.0]);
        assert_eq!(read_f32_array(&client, &higher_reference.read().unwrap()), vec![5.0]);
    }

    #[test]
    fn test_stateful_failure_boundaries_restore_poison_or_commit_reference_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType, ArrayIrType, ArrayIrType), ArrayIrType>(
            |(first, second, read_only, update)| {
                let output = read_only.read()?;
                first.add_update(&update)?;
                second.add_update(&update)?;
                Ok(output)
            },
            (reference_type.clone(), reference_type.clone(), reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference_new = |value: f32| {
            ArrayReference::new(
                Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice())
                    .unwrap(),
            )
        };

        // Failures during pre-execution holder validation leave every holder Ready and unchanged.
        let pre_handoff_first = reference_new(1.0);
        let pre_handoff_second = reference_new(10.0);
        let pre_handoff_read_only = reference_new(20.0);
        let wrong_update_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let wrong_update =
            Array::from_host_buffer(&domain, wrong_update_type, mesh.clone(), 2.0f64.to_ne_bytes().as_slice()).unwrap();
        let declared_update_type = <&ArrayType>::try_from(&compiled.executable().input_types()[3]).unwrap();
        let expected_error = format!(
            "runtime input type {} does not refine declared type {declared_update_type}",
            wrong_update.r#type(),
        );
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(pre_handoff_first.clone(), domain.clone())),
                    Value::Reference(Reference::new(pre_handoff_second.clone(), domain.clone())),
                    Value::Reference(Reference::new(pre_handoff_read_only.clone(), domain.clone())),
                    Value::Array(wrong_update),
                ),
            ),
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                if message == expected_error,
        ));
        assert_eq!(read_f32_array(&client, &pre_handoff_first.read().unwrap()), vec![1.0]);
        assert_eq!(read_f32_array(&client, &pre_handoff_second.read().unwrap()), vec![10.0]);
        assert_eq!(read_f32_array(&client, &pre_handoff_read_only.read().unwrap()), vec![20.0]);

        // A fully prepared call that fails before PJRT submission publishes no lease or pending mutation.
        let pre_submission_first = reference_new(1.0);
        let pre_submission_second = reference_new(10.0);
        let pre_submission_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        inject_stateful_failure(StatefulFailureInjection::BeforeSubmission);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(pre_submission_first.clone(), domain.clone())),
                    Value::Reference(Reference::new(pre_submission_second.clone(), domain.clone())),
                    Value::Reference(Reference::new(pre_submission_read_only.clone(), domain.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure before execution submission",
        ));
        assert_eq!(read_f32_array(&client, &pre_submission_first.read().unwrap()), vec![1.0]);
        assert_eq!(read_f32_array(&client, &pre_submission_second.read().unwrap()), vec![10.0]);
        assert_eq!(read_f32_array(&client, &pre_submission_read_only.read().unwrap()), vec![20.0]);

        // Once mutable buffers cross the execution handoff, every mutated holder is poisoned together while a
        // read-only peer remains Ready.
        let post_handoff_first = reference_new(1.0);
        let post_handoff_second = reference_new(10.0);
        let post_handoff_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        inject_stateful_failure(StatefulFailureInjection::AfterHandoff);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(post_handoff_first.clone(), domain.clone())),
                    Value::Reference(Reference::new(post_handoff_second.clone(), domain.clone())),
                    Value::Reference(Reference::new(post_handoff_read_only.clone(), domain.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure after execution handoff",
        ));
        for reference in [&post_handoff_first, &post_handoff_second] {
            let Err(error) = reference.read() else { panic!("mutated holder must be poisoned after handoff") };
            assert_eq!(
                error.downcast_custom::<ReferenceError>(),
                Some(&ReferenceError::ExecutionPoisoned {
                    reason: "injected failure after execution handoff".to_string(),
                }),
            );
        }
        assert_eq!(read_f32_array(&client, &post_handoff_read_only.read().unwrap()), vec![20.0]);

        // Every ordinary error after publication but before the validate-all/commit-all boundary poisons the complete
        // mutated batch with that exact error. The published read lease on a read-only peer remains valid and releases
        // normally when the submitted execution completes.
        let pre_commit_first = reference_new(1.0);
        let pre_commit_second = reference_new(10.0);
        let pre_commit_read_only = reference_new(20.0);
        let update =
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap();
        inject_stateful_failure(StatefulFailureInjection::BeforeHiddenReplacementCommit);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(pre_commit_first.clone(), domain.clone())),
                    Value::Reference(Reference::new(pre_commit_second.clone(), domain.clone())),
                    Value::Reference(Reference::new(pre_commit_read_only.clone(), domain.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure before hidden state replacement commit",
        ));
        for reference in [&pre_commit_first, &pre_commit_second] {
            let Err(error) = reference.read() else { panic!("mutated reference must be poisoned before commit") };
            assert_eq!(
                error.downcast_custom::<ReferenceError>(),
                Some(&ReferenceError::ExecutionPoisoned {
                    reason: "injected failure before hidden state replacement commit".to_string(),
                }),
            );
        }
        assert_eq!(read_f32_array(&client, &pre_commit_read_only.read().unwrap()), vec![20.0]);

        // A failure confined to public reconstruction occurs only after every hidden final-state replacement commits.
        let committed_first = reference_new(1.0);
        let committed_second = reference_new(10.0);
        let committed_read_only = reference_new(20.0);
        let update = Array::from_host_buffer(&domain, array_type, mesh, 2.0f32.to_ne_bytes().as_slice()).unwrap();
        inject_stateful_failure(StatefulFailureInjection::BeforePublicReconstruction);
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(committed_first.clone(), domain.clone())),
                    Value::Reference(Reference::new(committed_second.clone(), domain.clone())),
                    Value::Reference(Reference::new(committed_read_only.clone(), domain.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Pjrt(ryft_pjrt::Error::Internal { message, .. }))
                if message == "injected failure before public output reconstruction",
        ));
        assert_eq!(read_f32_array(&client, &committed_first.read().unwrap()), vec![3.0]);
        assert_eq!(read_f32_array(&client, &committed_second.read().unwrap()), vec![12.0]);
        assert_eq!(read_f32_array(&client, &committed_read_only.read().unwrap()), vec![20.0]);
    }

    #[test]
    fn test_stateful_multi_holder_validation_failure_leaves_ready_peer_unchanged() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType, ArrayIrType), ArrayIrType>(
            |(first, second, update)| {
                first.add_update(&update)?;
                second.add_update(&update)?;
                first.read().map_err(Into::into)
            },
            (reference_type.clone(), reference_type, ArrayIrType::Array(array_type.clone())),
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let first = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 1.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        let frozen = ArrayReference::new(
            Array::from_host_buffer(&domain, array_type.clone(), mesh.clone(), 2.0f32.to_ne_bytes().as_slice())
                .unwrap(),
        );
        frozen.freeze().unwrap();
        let update = Array::from_host_buffer(&domain, array_type, mesh, 3.0f32.to_ne_bytes().as_slice()).unwrap();
        assert!(matches!(
            compiled.execute_statefully_and_wait(
                &domain,
                (
                    Value::Reference(Reference::new(first.clone(), domain.clone())),
                    Value::Reference(Reference::new(frozen, domain.clone())),
                    Value::Array(update),
                ),
            ),
            Err(Error::Core(ryft_core::Error::Reference(ReferenceError::Frozen))),
        ));
        assert_eq!(read_f32_array(&client, &first.read().unwrap()), vec![1.0]);
    }

    #[test]
    fn test_dense_differentiation_of_array_ir_values() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        // The dense builders run directly on array IR values (rather than on their array projections), whose packed
        // replay batches the derivative program in the array IR universe.
        let function: CompiledFunction<'_, ArrayType, (ArrayType, ArrayType, ArrayType)> = compile(
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
        let x = Array::from_host_buffer(&engine, vector_type, mesh, values_to_bytes::<f32>(&[1.0, -2.0]).as_slice())
            .unwrap();
        let (forward, reverse, hessian) = function.executable().execute(&engine, x).unwrap();
        assert_eq!(read_f32_array(&client, &forward), vec![3.0, 0.0, 0.0, 12.0]);
        assert_eq!(read_f32_array(&client, &reverse), vec![3.0, 0.0, 0.0, 12.0]);
        assert_eq!(read_f32_array(&client, &hessian), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_dense_differentiation_compiles_as_part_of_the_enclosing_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();

        let forward: CompiledFunction<'_, ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| {
                input
                    .domain()
                    .differentiate_at(input)
                    .jacobian_forward(|value| Mul::mul(&value, &value))
                    .expect("forward Jacobian should stage")
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let reverse: CompiledFunction<'_, ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| {
                input
                    .domain()
                    .differentiate_at(input)
                    .jacobian_reverse(|value| Mul::mul(&value, &value))
                    .expect("reverse Jacobian should stage")
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let second: CompiledFunction<'_, ArrayType, Hessian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| {
                input
                    .domain()
                    .differentiate_at(input)
                    .hessian(|value| Mul::mul(&value, &value))
                    .expect("Hessian should stage")
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(domain.program_count(), 2);
        let _: CompiledFunction<'_, ArrayType, Jacobian<ArrayType, ArrayType, ArrayType, ArrayType>> = compile(
            |input| {
                input
                    .domain()
                    .differentiate_at(input)
                    .jacobian_forward(|value| Mul::mul(&value, &value))
                    .expect("forward Jacobian should stage")
            },
            input_type.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(domain.program_count(), 2, "an equivalent dense transform should reuse its enclosing compilation");

        assert!(forward.source_program().program().regions().iter().any(|region| {
            region
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Iota(_))))
        }));
        assert!(reverse.source_program().program().regions().iter().any(|region| {
            region
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Iota(_))))
        }));
        assert!(second.source_program().program().regions().iter().any(|region| {
            region
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Iota(_))))
        }));

        for (compiled, expected) in [(forward, 6.0), (reverse, 6.0)] {
            let input = Array::from_host_buffer(
                &domain,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[3.0]).as_slice(),
            )
            .unwrap();
            let jacobian: Jacobian<ArrayType, Array<'_>, ArrayType, ArrayType> =
                compiled.executable().execute(&domain, input).unwrap();
            assert_eq!(read_f32_array(&client, jacobian.iter_blocks().next().unwrap().value()), vec![expected]);
        }

        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&[3.0]).as_slice()).unwrap();
        let hessian: Hessian<ArrayType, Array<'_>, ArrayType, ArrayType> =
            second.executable().execute(&domain, input).unwrap();
        assert_eq!(read_f32_array(&client, hessian.iter_blocks().next().unwrap().value()), vec![2.0]);
    }

    #[test]
    fn test_promoted_broadcast_dense_differentiation_compiles_as_part_of_the_enclosing_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        let forward: CompiledFunction<
            '_,
            (ArrayType, ArrayType),
            Jacobian<ArrayType, ArrayType, (ArrayType, ArrayType), ArrayType>,
        > = compile(
            |inputs| {
                inputs
                    .0
                    .domain()
                    .differentiate_at(inputs)
                    .jacobian_forward(|(scalar, vector)| scalar.atan2(&vector))
                    .expect("forward Jacobian should stage")
            },
            (scalar_type.clone(), vector_type.clone()),
            &domain,
            mesh.clone(),
        )
        .unwrap();

        let scalar =
            Array::from_host_buffer(&domain, scalar_type, mesh.clone(), values_to_bytes::<f32>(&[2.0]).as_slice())
                .unwrap();
        let vector =
            Array::from_host_buffer(&domain, vector_type, mesh, values_to_bytes::<f64>(&[1.0, 4.0]).as_slice())
                .unwrap();
        let jacobian: Jacobian<ArrayType, Array<'_>, (ArrayType, ArrayType), ArrayType> =
            forward.executable().execute(&domain, (scalar, vector)).unwrap();
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
    fn test_executable_program_runs_without_transform_metadata() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable();
        let input =
            Array::from_host_buffer(&domain, input_type.clone(), mesh, values_to_bytes::<f32>(&[0.5]).as_slice())
                .unwrap();

        let output = executable.execute(&domain, input).unwrap();

        assert_eq!(executable.output_types(), &[input_type]);
        assert!((read_f32_array(&client, &output)[0] - 0.5_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_executable_program_rejects_foreign_client_before_enqueue() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let other_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let other_mesh = single_device_mesh(&other_client);
        let domain = Session::new(&client).domain();
        let other_domain = Session::new(&other_client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable();
        let other_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(other_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let other_executable: ExecutableFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), other_input_type, &other_domain, other_mesh)
                .unwrap()
                .into_executable();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();

        assert!(matches!(
            executable.execute(&other_domain, input.clone()),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
        assert!(matches!(
            executable.execute_async(&other_domain, input),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
        assert!(matches!(
            domain.validate_replacement(
                executable.function.compiled_program(),
                other_executable.function.compiled_program(),
            ),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
        assert!(matches!(
            other_domain.validate_replacement(
                executable.function.compiled_program(),
                other_executable.function.compiled_program(),
            ),
            Err(Error::Execution(ExecutionError::ClientMismatch)),
        ));
    }

    #[test]
    fn test_compiled_and_staged_functions_are_send_and_sync() {
        assert_send_sync::<CompiledFunction<'static, ArrayType, ArrayType>>();
        assert_send_sync::<StagedFunction<'static, ArrayType, ArrayType>>();
    }

    #[test]
    fn test_executable_program_executes_across_threads() {
        assert_send_sync::<ExecutableFunction<'static, ArrayType, ArrayType>>();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let executable: ExecutableFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type.clone(), &domain, mesh.clone())
                .unwrap()
                .into_executable();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();

        let output =
            std::thread::scope(|scope| scope.spawn(move || executable.execute(&domain, input)).join().unwrap())
                .unwrap();
        assert!((read_f32_array(&client, &output)[0] - 0.5_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_profile_guided_recompilation_rejects_cpu_at_construction() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|input| input.sin().unwrap(), input_type, &domain, mesh).unwrap();

        assert!(matches!(
            domain.adaptive_profile_guided_recompilation(&compiled, AdaptiveProfileGuidedOptions::default()),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason })) if reason.contains("CUDA or ROCm"),
        ));
    }

    #[test]
    fn test_retained_jit_reuses_static_specializations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let function: CompiledFunctionDispatcher<'_, _, bool, ArrayType, ArrayType> = jit(
            |apply_sine, input: CompileTracer<'_>| Ok(if apply_sine { input.sin()? } else { input }),
            &domain,
            mesh.clone(),
        );

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source =
            Array::from_host_buffer(&domain, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();

        let sine = function.execute(true, source.clone()).unwrap();
        let warm_sine = function.execute(true, source.clone()).unwrap();
        let identity = function.execute(false, source).unwrap();
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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x.stop_gradient().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

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

        // Lowering preserves every result of one variadic barrier without materializing an MLIR operation.
        let variadic: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(first, second)| {
                let context = first.domain();
                let mut outputs = context.bind(StopGradientOperation::new(), Vec::new(), &[first, second]).unwrap();
                let second = outputs.pop().unwrap();
                let first = outputs.pop().unwrap();
                (first, second)
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let other_values = [4.0f32, 3.0, 2.0, 1.0];
        let first = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let second =
            Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes::<f32>(&other_values).as_slice())
                .unwrap();
        let (first, second) = variadic.executable().execute(&engine, (first, second)).unwrap();
        assert_eq!(read_f32_array(&client, &first), values);
        assert_eq!(read_f32_array(&client, &second), other_values);
    }

    #[test]
    fn test_jit_transfer_to_memory_round_trip_runs_end_to_end() {
        let client = execution_client();
        let devices = client.addressable_devices().unwrap();
        let device = &devices[0];
        let device_kind = device.default_memory().unwrap().kind().unwrap().into_owned();
        let device_id = device.id().unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::I64, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        for (memory, memory_kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let available =
                device.addressable_memories().unwrap().iter().any(|memory| memory.kind().unwrap() == memory_kind);
            let host_type = input_type.clone().with_memory(memory);
            let values = [i64::MIN, i64::MAX, 9_007_199_254_740_993, -9_007_199_254_740_993];
            // CUDA exposes pinned host buffers, but not an unpinned host allocation tier. Reject an unavailable
            // placement explicitly rather than silently allocating the buffer in device memory.
            if client.platform_name().unwrap().eq_ignore_ascii_case("cuda")
                && memory == (Memory::Host { pinned: false })
            {
                assert!(!available);
                assert!(matches!(
                    Array::from_host_buffer(&engine, host_type, mesh.clone(), values_to_bytes(&values).as_slice()),
                    Err(Error::Array(error)) if matches!(error.as_ref(),
                        ArrayError::UnsupportedMemory { device_id: actual_device, memory: actual_memory }
                            if *actual_device == device_id && *actual_memory == memory,
                    ),
                ));
                continue;
            }
            assert!(available, "execution device does not expose required memory kind `{memory_kind}`");
            let source =
                Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), values_to_bytes(&values).as_slice())
                    .unwrap();

            // A compiled host result must occupy the requested host memory, not just carry host metadata on a device buffer.
            let to_host: CompiledFunction<'_, ArrayType, ArrayType> =
                compile(|input| input.transfer_to_memory(memory).unwrap(), input_type.clone(), &engine, mesh.clone())
                    .unwrap();
            let host = to_host.executable().execute(&engine, source.clone()).unwrap();
            assert_eq!(host.r#type().as_ref(), &host_type);
            let buffer = host.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), memory_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );

            // Host placement survives an identity executable boundary, even without a transfer instruction.
            let identity: CompiledFunction<'_, ArrayType, ArrayType> =
                compile(|input| input, host_type.clone(), &engine, mesh.clone()).unwrap();
            let host = identity.executable().execute(&engine, host).unwrap();
            assert_eq!(host.r#type().as_ref(), &host_type);
            let buffer = host.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), memory_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );

            // The host input signature must permit a real host input to return to device memory.
            let to_device: CompiledFunction<'_, ArrayType, ArrayType> =
                compile(|input| input.transfer_to_memory(Memory::Device).unwrap(), host_type, &engine, mesh.clone())
                    .unwrap();
            let output = to_device.executable().execute(&engine, host).unwrap();
            assert_eq!(output.r#type().as_ref(), &input_type);
            let buffer = output.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), device_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );

            let round_trip: CompiledFunction<'_, ArrayType, ArrayType> = compile(
                |input| input.transfer_to_memory(memory).unwrap().transfer_to_memory(Memory::Device).unwrap(),
                input_type.clone(),
                &engine,
                mesh.clone(),
            )
            .unwrap();
            let output = round_trip.executable().execute(&engine, source).unwrap();
            assert_eq!(output.r#type().as_ref(), &input_type);
            let buffer = output.device_shard(device_id).unwrap().buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), device_kind);
            assert_eq!(
                values_from_bytes::<i64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice()),
                values,
            );
        }
    }

    /// Smoke test: after `compile` runs, the returned handle exposes the source
    /// [`Program`] that was traced. This is the foundation for diagnostics (printing the IR)
    /// and for transform composition / inner staging via
    /// [`CompiledFunction::call`].
    #[test]
    fn test_compiled_function_retains_source_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
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
    fn test_compile_stable_cross_entropy_gradients() {
        use ryft_core::{ProjectedContext, Softmax, softmax_cross_entropy_with_target_logits};

        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let label_type = ArrayType::scalar(DataType::I32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let soft_target = softmax_cross_entropy_with_target_logits::<ProjectedContext<Domain<'_>, ArrayType>>(-1);
        // Compile each complete stabilized derivative so optimizer rewrites cannot hide behind eager operation
        // boundaries. Integer labels and teacher logits parameterize their respective losses without differentiation.
        let compiled: CompiledFunction<
            '_,
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
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let student_values = [2.0f32 - 1e-7, 1.0f32 + 1e-7, 0.0];
        let logits = Array::from_host_buffer(
            &domain,
            vector_type.clone(),
            mesh.clone(),
            &values_to_bytes(&[9.0f32, -9.0, -9.0]),
        )
        .unwrap();
        let labels = Array::from_host_buffer(&domain, label_type, mesh.clone(), &values_to_bytes(&[0i32])).unwrap();
        let student =
            Array::from_host_buffer(&domain, vector_type.clone(), mesh.clone(), &values_to_bytes(&student_values))
                .unwrap();
        let teacher =
            Array::from_host_buffer(&domain, vector_type, mesh, &values_to_bytes(&[2.0f32, 1.0, 0.0])).unwrap();
        let (loss, hard_gradient, soft_gradient) =
            compiled.executable().execute(&domain, (logits, labels, student, teacher)).unwrap();

        let tail = (-18.0f64).exp();
        let loss = read_f32_array(&client, &loss)[0] as f64;
        assert!((loss - (2.0 * tail).ln_1p()).abs() < 1e-13);
        let hard_gradient = read_f32_array(&client, &hard_gradient);
        let expected = tail / (1.0 + 2.0 * tail);
        assert!((f64::from(hard_gradient[0]) + 2.0 * expected).abs() < 1e-13);
        assert!((f64::from(hard_gradient[1]) - expected).abs() < 1e-13);
        assert!((f64::from(hard_gradient[2]) - expected).abs() < 1e-13);
        assert!((hard_gradient.iter().map(|&value| f64::from(value)).sum::<f64>()).abs() < 1e-13);

        // Use the represented f32 logits in the f64 oracle so the comparison measures probability cancellation,
        // rather than a discrepancy between intended and representable student perturbations.
        let student_denominator = student_values.iter().map(|&value| (f64::from(value) - 2.0).exp()).sum::<f64>();
        let teacher_denominator = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        for (class, actual) in read_f32_array(&client, &soft_gradient).into_iter().enumerate() {
            let expected = (f64::from(student_values[class]) - 2.0).exp() / student_denominator
                - (-(class as f64)).exp() / teacher_denominator;
            assert!((f64::from(actual) - expected).abs() < 1e-13, "class {class}: {actual} differs from {expected}");
        }
    }

    #[test]
    fn test_compile_softmax_sharded_class_axis() {
        // A class axis sharded over two devices normalizes across both shards: the partitioner combines the argmax
        // and the sums across the devices, the probabilities keep the input's sharding, and the loss and gradient of
        // a replicated integer label match the unsharded arithmetic.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Explicit);
        let engine = Session::new(&client).domain();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let logits_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let label_type = ArrayType::scalar(DataType::I32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType, ArrayType)> = compile(
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
        let logits = Array::from_host_buffer(&engine, logits_type, mesh.clone(), &values_to_bytes(&values)).unwrap();
        let labels = Array::from_host_buffer(&engine, label_type, mesh, &values_to_bytes(&[2i32])).unwrap();
        let (probabilities, loss, gradient) = compiled.executable().execute(&engine, (logits, labels)).unwrap();
        assert_eq!(probabilities.sharding(), &sharding);
        assert_eq!(gradient.sharding(), &sharding);
        let read_shards = |array: &Array<'_>| {
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
        assert!((f64::from(read_f32_array(&client, &loss)[0]) - expected_loss).abs() < 1e-5);
        for (class, actual) in read_shards(&gradient).into_iter().enumerate() {
            let expected = expected_probabilities[class] - if class == 2 { 1.0 } else { 0.0 };
            assert!((f64::from(actual) - expected).abs() < 1e-7, "class {class}: {actual} differs from {expected}");
        }
    }

    #[test]
    fn test_executable_function_execute_async_retains_zero_output_completion() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ()> =
            compile(|_| (), input_type.clone(), &engine, mesh.clone()).unwrap();
        let input =
            Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes::<f32>(&[1.0]).as_slice()).unwrap();

        let execution = compiled.executable().execute_async(&engine, input).unwrap();
        assert_eq!(execution.output(), &());
        execution.fence().block_until_ready().unwrap();
        execution.block_until_ready().unwrap();
    }

    /// Inner-composition smoke test: a compiled function can be staged into another
    /// `compile` closure as a sub-routine, producing the same result as if the
    /// whole computation were a single closure. Mirrors JAX's
    /// `jit(lambda x: jit(f)(x).cos())` pattern.
    ///
    /// Exercises [`CompiledFunction::call`], which stages the retained source program behind a
    /// `jit_call` boundary in the active outer trace.
    #[test]
    fn test_compiled_function_staged_inside_compile() {
        use ryft_core::Cos;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        // Inner: compile `f = |x| x.sin()`.
        let inner: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // Outer: compile `g = |x| cos(inner(x))` by staging `inner` as one `jit_call` and applying `cos` to its
        // output.
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jit_call_count = outer
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
            .count();
        let inlined_sin_count = outer
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Sin(_))))
            .count();
        assert_eq!(jit_call_count, 1, "inner compiled function should stage as one jit_call");
        assert_eq!(inlined_sin_count, 0, "inner function body should not be inlined into the outer trace");

        // Execute and compare against the mathematical reference.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, source).unwrap();

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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();

        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
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
            &engine,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_compile_with_captures_erases_zero_space_capture_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let capture = Array::from_host_buffer(&engine, zero_type.clone(), mesh.clone(), []).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, _| captures[0].clone(),
            vec![capture],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes::<f32>(&[1.0])).unwrap();

        let output = compiled.executable().execute(&engine, input).unwrap();

        assert_eq!(output.r#type().as_ref(), &zero_type);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
        output.block_until_ready().unwrap();
    }

    #[test]
    fn test_captured_compiled_function_stages_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        assert_eq!(outer.source_program().captures().len(), 1);

        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, input).unwrap();
        let observed = read_f32_array(&client, &output);
        for (got, expected) in observed.iter().zip([3.0f32.sin(), 4.0f32.sin(), 5.0f32.sin(), 6.0f32.sin()]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_multiple_captured_compiled_functions_stage_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let left_bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let right_bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 10.0, 10.0, 10.0]).as_slice(),
        )
        .unwrap();
        let left: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![left_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let right: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![right_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = outer.executable().execute(&engine, input).unwrap();
        let observed = read_f32_array(&client, &output);
        for (got, expected) in observed.iter().zip([14.0, 16.0, 18.0, 20.0]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_jvp_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = jvp_compiled.executable().execute(&engine, (primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_gradient_method_materializes_disconnected_symbolic_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile(|(_, live)| live, (dynamic_type, scalar_type.clone()), &engine, mesh.clone()).unwrap();
        let gradient: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compiled.gradient(&engine).unwrap();
        let concrete_type = ArrayType::new(DataType::F32, Shape::new(vec![3.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let disconnected = Array::from_host_buffer(
            &engine,
            concrete_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let live =
            Array::from_host_buffer(&engine, scalar_type, mesh, values_to_bytes::<f32>(&[7.0]).as_slice()).unwrap();
        let (zero, one) = gradient.executable().execute(&engine, (disconnected, live)).unwrap();
        assert_eq!(read_f32_array(&client, &zero), vec![0.0; 3]);
        assert_eq!(read_f32_array(&client, &one), vec![1.0]);
    }

    #[test]
    fn test_gradient_method_materializes_disconnected_condition_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> = compile(
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
        let gradient: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
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
        let concrete_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let dead = Array::from_host_buffer(
            &engine,
            concrete_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        for (value, expected_derivative) in [(7.0, 14.0), (-7.0, 1.0)] {
            let live = Array::from_host_buffer(
                &engine,
                scalar_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[value]).as_slice(),
            )
            .unwrap();
            let (zero, derivative) = gradient.executable().execute(&engine, (dead.clone(), live)).unwrap();
            assert_eq!(read_f32_array(&client, &zero), vec![0.0; 3]);
            assert_eq!(read_f32_array(&client, &derivative), vec![expected_derivative]);
        }
    }

    #[test]
    fn test_gradient_method_materializes_disconnected_scanned_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let slice_type = scalar_type.clone();
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.clone().into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let mut body = CoreProgramBuilder::<Constant, Operation>::new();
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
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> = compile(
            |(elements, initial)| {
                let size = DimensionSizeOperation::new(elements.r#type().as_ref(), 0).unwrap();
                let elements = elements.into_value();
                let context = elements.context().clone();
                let length =
                    context.bind(Operation::DimensionSize(size), Vec::new(), &[elements.clone()]).unwrap().remove(0);
                let inputs = vec![initial.into_value(), elements, length];
                let outputs = context
                    .bind(
                        Operation::Scan(ScanOperation::new(1, CoreDimension::Dynamic(extent.clone()))),
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
        let gradient: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
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
        let concrete_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let elements = Array::from_host_buffer(
            &engine,
            concrete_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let initial =
            Array::from_host_buffer(&engine, scalar_type, mesh, values_to_bytes::<f32>(&[7.0]).as_slice()).unwrap();
        let (zero, derivative) = gradient.executable().execute(&engine, (elements, initial)).unwrap();
        assert_eq!(read_f32_array(&client, &zero), vec![0.0; 3]);
        assert_eq!(read_f32_array(&client, &derivative), vec![8.0]);
    }

    #[test]
    fn test_gradient_method_omits_dead_seeds_of_staged_compiled_functions() {
        // `inner(x, s) = (x · x, s · s)` has a dynamically shaped output that `outer(x, s) = inner(x, s).1` discards. The
        // pullback of the staged `inner` call receives a structural-zero seed for `x · x`, so the transposed call has no
        // input for it and the pullback reads no `extent` for it; the only `f32[extent]` zero is the gradient for `x`,
        // which the outer boundary builds. Forward mode over the gradient and the gradient of `∂outer/∂s` both
        // differentiate through that transposed call again.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let input_types = (dynamic_type, scalar_type.clone());
        let inner: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compile(|(x, s)| (x.clone() * x, s.clone() * s), input_types.clone(), &engine, mesh.clone()).unwrap();
        let outer: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile(move |inputs| inner.call(inputs).unwrap().1, input_types.clone(), &engine, mesh.clone()).unwrap();
        let gradient: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
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

        let concrete_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&engine, r#type.clone(), mesh.clone(), values_to_bytes::<f32>(values).as_slice())
                .unwrap()
        };
        let x = array(&concrete_type, &[2.0, 3.0, 4.0]);
        let s = array(&scalar_type, &[3.0]);
        let (x_gradient, s_gradient) = gradient.executable().execute(&engine, (x.clone(), s.clone())).unwrap();
        assert_eq!(read_f32_array(&client, &x_gradient), vec![0.0; 3]);
        assert_eq!(read_f32_array(&client, &s_gradient), vec![6.0]);

        // Forward over reverse: the tangent of the gradient `(0, 2s)` along `(ẋ, ṡ)` is `(0, 2ṡ)`.
        let forward_over_reverse = gradient.jvp(&engine).unwrap();
        let ((x_gradient, s_gradient), (x_gradient_tangent, s_gradient_tangent)) = forward_over_reverse
            .executable()
            .execute(
                &engine,
                ((x.clone(), s.clone()), (array(&concrete_type, &[1.0, 1.0, 1.0]), array(&scalar_type, &[1.0]))),
            )
            .unwrap();
        assert_eq!(read_f32_array(&client, &x_gradient), vec![0.0; 3]);
        assert_eq!(read_f32_array(&client, &s_gradient), vec![6.0]);
        assert_eq!(read_f32_array(&client, &x_gradient_tangent), vec![0.0; 3]);
        assert_eq!(read_f32_array(&client, &s_gradient_tangent), vec![2.0]);

        // Reverse over reverse: the gradient of `∂outer/∂s = 2s` is `(0, 2)`.
        let s_derivative: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile(move |inputs| gradient.call(inputs).unwrap().1, input_types, &engine, mesh.clone()).unwrap();
        let reverse_over_reverse = s_derivative.gradient(&engine).unwrap();
        let (x_second, s_second) = reverse_over_reverse.executable().execute(&engine, (x, s)).unwrap();
        assert_eq!(read_f32_array(&client, &x_second), vec![0.0; 3]);
        assert_eq!(read_f32_array(&client, &s_second), vec![2.0]);
    }

    #[test]
    fn test_gradient_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<'_, ArrayType, ArrayType> = inner.gradient(&engine).unwrap();

        assert_eq!(gradient.source_program().captures().len(), 1);

        let input =
            Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes::<f32>(&[3.0]).as_slice()).unwrap();
        let output = gradient.executable().execute(&engine, input).unwrap();

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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let callee: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(move |x| callee.call(x).unwrap().sin().unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();

        let jvp: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = outer.jvp(&engine).unwrap();
        assert_eq!(jvp.output_types().len(), 2);
        let gradient: CompiledFunction<'_, ArrayType, ArrayType> = outer.gradient(&engine).unwrap();
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
            Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes::<f32>(&[0.5]).as_slice()).unwrap();
        let output = gradient.executable().execute(&engine, input).unwrap();
        assert!((read_f32_array(&client, &output)[0] as f64 - expected_gradient).abs() < 1e-3);
    }

    /// For `f = |x| x * x`, repeated `f.gradient(&domain)` calls with one domain must reuse the retained derived
    /// function (no frontend re-derivation and no compilation-context traffic), while different runtime inputs to the
    /// shared derived function must still produce their own correct value-dependent gradients.
    #[test]
    fn test_gradient_method_retains_derived_function_per_domain() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x, input_type.clone(), &engine, mesh.clone()).unwrap();

        let first: CompiledFunction<'_, ArrayType, ArrayType> = compiled.gradient(&engine).unwrap();
        let statistics = engine.compilation_context().statistics();
        let second = compiled.gradient(&engine).unwrap();

        // The warm call reuses the retained executable and never reaches the shared compilation context.
        assert!(std::ptr::eq(first.function.compiled_program(), second.function.compiled_program()));
        assert_eq!(engine.compilation_context().statistics(), statistics);

        // The retained derived function keeps residuals as runtime values: different primals through the shared
        // executable produce different, correct value-dependent gradients (`d(x*x)/dx = 2x`).
        for point in [3.0f32, -5.0f32] {
            let input = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = second.executable().execute(&engine, input).unwrap();
            assert_eq!(read_f32_array(&client, &output), vec![2.0 * point]);
        }

        // A different domain identity misses, produces a fresh derivative, and replaces the retained slot.
        let other_engine = Session::new(&client).domain();
        let third = compiled.gradient(&other_engine).unwrap();
        assert!(!std::ptr::eq(first.function.compiled_program(), third.function.compiled_program()));
        let other_statistics = other_engine.compilation_context().statistics();
        let fourth = compiled.gradient(&other_engine).unwrap();
        assert!(std::ptr::eq(third.function.compiled_program(), fourth.function.compiled_program()));
        assert_eq!(other_engine.compilation_context().statistics(), other_statistics);
    }

    /// Mirrors [`test_gradient_method_retains_derived_function_per_domain`] for the forward-mode
    /// [`CompiledFunction::jvp`] path.
    #[test]
    fn test_jvp_method_retains_derived_function_per_domain() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.clone() * x, input_type.clone(), &engine, mesh.clone()).unwrap();

        let first: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compiled.jvp(&engine).unwrap();
        let statistics = engine.compilation_context().statistics();
        let second = compiled.jvp(&engine).unwrap();

        assert!(std::ptr::eq(first.function.compiled_program(), second.function.compiled_program()));
        assert_eq!(engine.compilation_context().statistics(), statistics);

        // The retained JVP computes `(x * x, 2 * x * t)` for runtime `(x, t)` supplied per call.
        let primal = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&engine, input_type, mesh, values_to_bytes::<f32>(&[4.0]).as_slice()).unwrap();
        let (primal_output, tangent_output) = second.executable().execute(&engine, (primal, tangent)).unwrap();
        assert_eq!(read_f32_array(&client, &primal_output), vec![9.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![24.0]);
    }

    #[test]
    fn test_compile_can_stage_jvp_of_captured_compiled_function() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
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
                    .expect("the JVP of a nested compiled call with captures should stage");
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
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = jvp_compiled.executable().execute(&engine, (primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    /// Outer-transform smoke test: applying `grad` to a `CompiledFunction` produces a new compiled function that
    /// For `f = |x| x.sin()`, `f.jvp()` produces a compiled function for which interpreting its executable function at
    /// `(primal, tangent)` returns `(sin(primal), cos(primal) * tangent)` within `1e-5`.
    #[test]
    fn test_jvp_method_returns_primal_and_tangent() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp_compiled: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp(&engine).unwrap();
        let jit_call_count = jvp_compiled
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
            .count();
        let inlined_sin_count = jvp_compiled
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Sin(_))))
            .count();
        assert_eq!(
            jit_call_count, 2,
            "the JVP of a compiled call should stage separate primal and pushforward `jit_call` boundaries",
        );
        assert_eq!(inlined_sin_count, 0, "the JVP of a compiled call should not inline the callee body");

        for &(primal, tangent) in &[(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let primal_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
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
    fn test_jvp_method_executes_zero_space_boundaries() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let primal_type = ArrayType::new(DataType::Boolean, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let tangent_type = primal_type.tangent().unwrap();
        let primal: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|value| value, primal_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = primal.jvp(&engine).unwrap();
        let zero_identity: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|value| value, tangent_type.clone(), &engine, mesh.clone()).unwrap();

        let primal_input = Array::from_host_buffer(&engine, primal_type.clone(), mesh.clone(), [1u8]).unwrap();
        let tangent_input = Array::from_host_buffer(&engine, tangent_type.clone(), mesh, []).unwrap();
        let (primal_output, tangent_output) = jvp.executable().execute(&engine, (primal_input, tangent_input)).unwrap();

        assert_eq!(primal_output.r#type().as_ref(), &primal_type);
        assert_eq!(tangent_output.r#type().as_ref(), &tangent_type);
        assert!(tangent_output.addressable_shards().next().unwrap().buffer().is_none());

        let chained_tangent = zero_identity.executable().execute(&engine, tangent_output).unwrap();
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
    /// `(x, dx) -> (primal, tangent)` function reproduces the legacy [`jvp`](CompiledFunction::jvp) result —
    /// `(sin(x), cos(x) * dx)` — within `1e-5`.
    #[test]
    fn test_jvp_of_jit_call_preserves_boundary_and_matches_legacy_jvp() {
        use ryft_core::StagingContext;

        use crate::experimental::ops::FlatProgram;
        use crate::operations::jit_call::JitCallOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        // Inner `jit(sin)` staged as one `jit_call` inside an outer program, so the outer source program holds exactly
        // one `jit_call` instruction for the replay to differentiate.
        let inner: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let inner_for_outer = inner.clone();
        let outer: StagedFunction<'_, ArrayType, ArrayType> =
            stage(move |x| inner_for_outer.call(x).unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        let outer_program = outer.source_program().program();
        assert_eq!(
            outer_program
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
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
        let count_operations = |program: &FlatProgram| {
            let jit_calls = program
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
                .count();
            let inlined = program
                .instructions()
                .iter()
                .filter(|instruction| {
                    matches!(instruction.operation(), Operation::Array(ArrayOperation::Sin(_) | ArrayOperation::Cos(_)))
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
        let (primal_half, tangent_half, residual_count, _) = linearization.into_parts();
        let jvp_compiled: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal_input, tangent_input)| {
                let context = primal_input.value().context().clone();
                let mut primal_outputs = context
                    .stage_operation(
                        Operation::JitCall(JitCallOperation::new(0)),
                        AttachedRegions::Callees(&[primal_half.clone()]),
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
                        Operation::JitCall(JitCallOperation::new(0)),
                        AttachedRegions::Callees(&[tangent_half.clone()]),
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
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([tangent].as_slice()).as_slice(),
            )
            .unwrap();
            let (jvp_primal, jvp_tangent) =
                jvp_compiled.executable().execute(&engine, (primal_array.clone(), tangent_array.clone())).unwrap();
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

    #[test]
    fn test_compiled_function_derived_functions_keep_lowering_options() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let strategy = RaggedDotLoweringStrategy::Instruction;
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_options(
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
        assert_eq!(gradient.function.staged().options().ragged_dot_lowering_strategy, strategy);
        assert_eq!(jvp.function.staged().options().ragged_dot_lowering_strategy, strategy);
        assert_eq!(batched.function.staged().options().ragged_dot_lowering_strategy, strategy);
    }

    #[test]
    fn test_compiled_function_batch() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);
        let vector_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(replicated(1)).unwrap();
        let array = |r#type: ArrayType, values: &[f32]| {
            Array::from_host_buffer(&engine, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
        };

        // `f(x, y) = x * y + c` with a captured `c`, so the batched function also inherits a capture.
        let function: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> = compile_with_captures(
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
        assert_eq!(read_f32_array(&client, &output), vec![11.0, 13.0, 15.0, 18.0, 16.0, 14.0]);

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
        assert_eq!(read_f32_array(&client, &primal), vec![11.0, 13.0, 15.0, 18.0, 16.0, 14.0]);
        assert_eq!(read_f32_array(&client, &tangent), vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_compiled_function_batch_places_batch_dimension() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Explicit);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
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
            values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]).as_slice(),
        )
        .unwrap();
        let function: CompiledFunction<'_, ArrayType, ArrayType> =
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
            |x| Ok(x.sin()?),
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
    fn test_compiled_function_batch_requires_static_extent() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let function: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), scalar_type, &engine, mesh).unwrap();
        assert!(matches!(
            function.batch(&engine, BatchAxis::new(0), BatchAxis::new(0), BatchAxisSpecification::named("i")),
            Err(Error::Core(CoreError::Program(ProgramError::InvalidArgument { message })))
                if message == "compiled function batching requires an explicit static batch extent",
        ));
    }

    #[test]
    fn test_jit_binary_function_with_tuple_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let shape = Shape::new(vec![CoreDimension::Static(3)]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharding.clone()).unwrap();
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile(|(a, b)| a + b, (input_type.clone(), input_type.clone()), &engine, mesh.clone()).unwrap();

        let a_values = [10.0f32, 20.0, 30.0];
        let b_values = [1.0f32, 2.0, 3.0];
        let a = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&a_values).as_slice(),
        )
        .unwrap();
        let b =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&b_values).as_slice())
                .unwrap();
        let output = compiled.executable().execute(&engine, (a, b)).unwrap();

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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        assert_eq!(engine.program_count(), 0);
        // Equivalent lowerings share one executable, so the second invocation is a cache hit.
        for _ in 0..2 {
            let _: CompiledFunction<'_, ArrayType, ArrayType> =
                compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.program_count(), 1, "an equivalent repeated lowering should hit the cache");
    }

    #[test]
    fn test_jit_equivalent_lowerings_share_cache_across_call_sites() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        assert_eq!(engine.program_count(), 0);
        // Cache identity is the complete lowering plus options and target, not the Rust source location.
        let _: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let _: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type, &engine, mesh).unwrap();
        assert_eq!(engine.program_count(), 1);
    }

    #[test]
    fn test_compile_with_options_donates_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = Options::new(mesh.clone()).with_donate(true);
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0];
        let source =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        // Build a `donation_flags` vec whose length does not match the function's flat input
        // arity. `with_donate` enforces matching shape via the `Parameterized<bool>` bound on
        // its argument, so producing an arity mismatch requires setting `donation_flags`
        // directly on `Options`.
        let mut options = Options::new(mesh.clone());
        options.donation_flags = Some(vec![true, false, false]); // 3 entries, 1 input
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, Error> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(
            result,
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 1, actual_count: 3 })),
        ));
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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        for batch_size in [32usize, 64usize] {
            let input = HyperparamInput { array: array_type.clone(), batch_size };
            let _: CompiledFunction<'_, HyperparamInput<ArrayType>, ArrayType> =
                compile(|input| input.array.sin().unwrap(), input, &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.program_count(), 1);
    }

    #[test]
    fn test_compile_with_options_input_shardings_override_replaces_input_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();

        // input_type carries the abstract shape & dtype but a "wrong" sharding (replicated). The
        // `input_shardings` override replaces it with a 2-way shard along "x" before tracing, so
        // the compiled program shards the input across the 2-device mesh.
        let shape = Shape::new(vec![CoreDimension::Static(4)]);
        let abstract_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let abstract_input_type =
            ArrayType::new(DataType::F32, shape.clone()).with_sharding(abstract_sharding).unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), abstract_input_type, &engine, options).unwrap();

        // Build the input array under the overridden sharding so it matches the executable's
        // expected layout.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer_with_sharding(
            &engine,
            ArrayType::new(DataType::F32, shape),
            sharded,
            mesh.clone(),
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
    fn test_compiled_function_call_constrains_inputs_to_input_sharding_overrides() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Explicit);
        let engine = Session::new(&client).domain();
        let replicated = |rank| Sharding::replicated(mesh.logical_mesh().clone(), rank);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated(1)).unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(replicated(2)).unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&engine, r#type.clone(), mesh.clone(), values_to_bytes::<f32>(values).as_slice())
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
        let function: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), vector_type.clone(), &engine, options.clone()).unwrap();
        let executed = function.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(executed.sharding(), &sharded);
        let called = function.call(input.clone()).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
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
        let dispatcher: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> =
            jit_with_options(|(), x: CompileTracer<'_>| Ok::<_, Error>(x.sin()?), &engine, options);
        let executed = dispatcher.execute((), input.clone()).unwrap();
        let called = dispatcher.call((), input).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
    }

    #[test]
    fn test_compile_with_options_rejects_input_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // Two shardings for one flat input — should fail.
        let options = Options::new(mesh).with_input_shardings(vec![sharding.clone(), sharding]);
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, Error> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(
            result,
            Err(Error::Compilation(CompilationError::ShardingCountMismatch { kind: "input", expected: 1, actual: 2 })),
        ));
    }

    #[test]
    fn test_compile_with_options_output_shardings_override_propagates_to_output_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();

        let shape = Shape::new(vec![CoreDimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape).with_sharding(replicated.clone()).unwrap();
        // Override the output sharding to a 2-way shard along "x" so the partitioner emits a
        // fully-sharded output and `Array`'s sharding metadata matches.
        let options = Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]);
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options).unwrap();

        // Types record no placement over auto mesh axes, so the staged output type stays replicated while the returned
        // Array carries the overridden sharding.
        assert_eq!(compiled.output_types()[0].sharding(), Some(&replicated));

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
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
    fn test_compiled_function_call_returns_output_sharding_overrides() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Explicit);
        let engine = Session::new(&client).domain();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5]).as_slice(),
        )
        .unwrap();
        let options = Options::new(mesh.clone()).with_output_shardings(vec![sharded.clone()]);

        // The override is staged on the output of the traced function, so calls return the output types that
        // executions return, both eagerly and inside another compilation.
        let function: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), input_type.clone(), &engine, options.clone()).unwrap();
        let executed = function.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(executed.sharding(), &sharded);
        let called = function.call(input.clone()).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| function.call(x).unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(outer.output_types(), function.output_types());

        // Executable and staged specializations of a dispatcher stage the override alike.
        let dispatcher: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> =
            jit_with_options(|(), x: CompileTracer<'_>| Ok::<_, Error>(x.sin()?), &engine, options);
        let executed = dispatcher.execute((), input.clone()).unwrap();
        assert_eq!(executed.sharding(), &sharded);
        let called = dispatcher.call((), input).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(read_sharded_f32_array(&called), read_sharded_f32_array(&executed));
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| dispatcher.call((), x).unwrap(), input_type, &engine, mesh).unwrap();
        assert_eq!(outer.output_types(), std::slice::from_ref(executed.r#type().as_ref()));
    }

    #[test]
    fn test_jit_implicitly_reshards_mismatched_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();

        // The executable expects a 2-way shard along "x", but the caller will pass a fully
        // replicated array. `CompiledFunction::interpret` should silently reshard before executing.
        let shape = Shape::new(vec![CoreDimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let replicated_input_type = ArrayType::new(DataType::F32, shape).with_sharding(replicated).unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]);
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), replicated_input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &engine,
            replicated_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        // Calling with a replicated source against a sharded-expecting executable would error
        // without implicit reshard. With reshard it should succeed and produce correct output.
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
    fn test_compiled_function_calls_agree_with_executions_for_auto_axis_overrides() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let read = |array: &Array<'_>| {
            values_from_bytes::<f32>(&crate::arrays_v0::host::materialize_dense_array_bytes(array).unwrap())
        };
        let assert_close = |actual: Vec<f32>, expected: Vec<f32>| {
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().zip(&expected).all(|(actual, expected)| (actual - expected).abs() < 1e-5));
        };
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let input =
            Array::from_host_buffer(&engine, vector_type.clone(), mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let options = Options::new(mesh.clone())
            .with_input_shardings(vec![sharded.clone()])
            .with_output_shardings(vec![sharded.clone()]);
        let function: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin().unwrap(), vector_type.clone(), &engine, options).unwrap();

        // Types record no placement over auto mesh axes, so calls and executions return the same (replicated) output
        // type, and the call no longer rejects the replicated input. The runtime placement of the executed and eagerly
        // called outputs is the override.
        let executed = function.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(executed.r#type().sharding(), Some(&replicated));
        assert_eq!(executed.sharding(), &sharded);
        let called = function.call(input.clone()).unwrap();
        assert_eq!(called.r#type(), executed.r#type());
        assert_eq!(called.sharding(), &sharded);
        assert_close(read(&called), values.iter().copied().map(f32::sin).collect());
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
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
        let total: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            |x| function.call(x).unwrap().reduce(&[0], ReductionKind::Sum).unwrap(),
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledFunction<'_, ArrayType, ArrayType> = total.gradient(&engine).unwrap();
        let gradient = gradient.executable().execute(&engine, input.clone()).unwrap();
        assert_eq!(gradient.r#type().as_ref(), &vector_type);
        assert_close(read(&gradient), values.iter().copied().map(f32::cos).collect());

        // Batching places the batched input over the override, with a replicated batch dimension.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let matrix_values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let matrix =
            Array::from_host_buffer(&engine, matrix_type.clone(), mesh.clone(), values_to_bytes::<f32>(&matrix_values))
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
    fn test_jit_dispatch_specializes_on_input_placement_over_auto_axes() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let bytes = values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5]);
        let sharded_input = Array::from_host_buffer_with_sharding(
            &engine,
            input_type.clone(),
            sharded.clone(),
            mesh.clone(),
            bytes.as_slice(),
        )
        .unwrap();
        let replicated_input =
            Array::from_host_buffer_with_sharding(&engine, input_type, replicated.clone(), mesh.clone(), bytes.as_slice())
                .unwrap();
        assert_eq!(sharded_input.r#type(), replicated_input.r#type());

        // Inputs whose types agree but whose placements over auto mesh axes differ get their own executables, which keep
        // each placement instead of resharding to the other, and the compiler chooses the output placements.
        let dispatcher: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> =
            jit_with_options(|(), x: CompileTracer<'_>| Ok::<_, Error>(x.sin()?), &engine, Options::new(mesh.clone()));
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
    fn test_compiled_function_places_captured_arrays_over_auto_axes_as_they_are() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let bytes = values_to_bytes::<f32>(&[0.0, 0.5, 1.0, 1.5]);
        let captured = Array::from_host_buffer_with_sharding(
            &engine,
            ArrayType::new_static(DataType::F32, [4]),
            sharded.clone(),
            mesh.clone(),
            bytes.as_slice(),
        )
        .unwrap();
        let input = Array::from_host_buffer(&engine, input_type.clone(), mesh.clone(), bytes.as_slice()).unwrap();

        // The captured array enters the executable at its runtime placement over the auto mesh axis rather than at
        // the replicated placement of its type, so it needs no gathering and the compiler shards the sum like it.
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
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
        assert_eq!(
            values_from_bytes::<f32>(&crate::arrays_v0::host::materialize_dense_array_bytes(&output).unwrap()),
            vec![0.0, 1.0, 2.0, 3.0],
        );
    }

    #[test]
    fn test_compile_with_options_rejects_output_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let options = Options::new(mesh).with_output_shardings(vec![sharding.clone(), sharding]);
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, Error> =
            compile_with_options(|x| x.sin().unwrap(), input_type, &engine, options);
        assert!(matches!(
            result,
            Err(Error::Compilation(CompilationError::ShardingCountMismatch { kind: "output", expected: 1, actual: 2 })),
        ));
    }

    #[test]
    fn test_infer_output_types_returns_output_types_without_compiling() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(7)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        let output_type: ArrayType = infer_output_types(|x| x.sin().unwrap(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `infer_output_types` must not have populated the compile cache.
        assert_eq!(engine.program_count(), 0);
    }

    /// Verifies that [`ConstrainSharding::constrain_sharding`] works inside a `compile`-compiled function over an
    /// auto mesh axis: the propagation hint is staged into the trace and lowers to `sdy.sharding_constraint`, and the
    /// output array carries the (input-derived) sharding on each device.
    #[test]
    fn test_jit_constrain_sharding_constrains_output_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = two_device_mesh(&client, MeshAxisType::Auto);
        let engine = Session::new(&client).domain();

        let shape = Shape::new(vec![CoreDimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, shape);
        let target_sharding = sharded.clone();

        // The user invokes `constrain_sharding` directly inside the staged closure — it's compiled into the
        // same MLIR program as the rest of the function body. Types record no placement over auto mesh axes,
        // so the input placement is an `input_shardings` override.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_options(
            move |x| {
                let constrained =
                    x.constrain_sharding(&target_sharding).expect("staged sharding constraint should succeed");
                constrained.sin().unwrap()
            },
            input_type.clone(),
            &engine,
            Options::new(mesh.clone()).with_input_shardings(vec![sharded.clone()]),
        )
        .unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer_with_sharding(
            &engine,
            input_type,
            sharded.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
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

    /// Multiple staged reshards inside one `compile` body compile into a single MLIR program with chained
    /// `sdy.sharding_constraint` ops — exactly one cache entry, exactly one PJRT execute per call. This is the
    /// async-pipelined regime: PJRT runs the whole compiled program in one shot without per-reshard host sync. The
    /// mesh axis is explicit so each reshard is a tracked transition and the final (replicated) sharding governs the
    /// output buffer.
    #[test]
    fn test_jit_with_chained_reshards_compiles_to_one_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
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
        let engine = Session::new(&client).domain();

        let shape = Shape::new(vec![CoreDimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharded.clone()).unwrap();
        let constraint_a = replicated.clone();
        let constraint_b = sharded.clone();
        let constraint_c = replicated;

        // Three staged reshards compose inside one closure. Each emits a `sdy.sharding_constraint` op into
        // the same MLIR program. After trace+compile, the executable runs all three in one PJRT dispatch.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let a = x.reshard(&constraint_a).unwrap();
                let b = a.sin().unwrap().reshard(&constraint_b).unwrap();
                b.sin().unwrap().reshard(&constraint_c).unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // One compile means one cache entry for the whole pipeline.
        assert_eq!(engine.program_count(), 1, "three staged reshards should compile into one program");

        let values = [0.1f32, 0.2, 0.3, 0.4];
        let source =
            Array::from_host_buffer(&engine, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let staged: StagedFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        assert_eq!(engine.program_count(), 0, "staging must not compile a PJRT executable");

        let staged_compiled = engine.compile_staged_function(staged).unwrap();
        assert_eq!(engine.program_count(), 1, "compiling the staged handle should build exactly one executable");

        let direct: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let input_value = 0.5f32;
        let make_input = || {
            Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
            )
            .unwrap()
        };
        let staged_output =
            read_f32_array(&client, &staged_compiled.executable().execute(&engine, make_input()).unwrap());
        let direct_output = read_f32_array(&client, &direct.executable().execute(&engine, make_input()).unwrap());
        assert_eq!(staged_output, direct_output);
        assert!((staged_output[0] - input_value.sin()).abs() < 1e-6);
    }

    /// A staged-but-never-compiled function with captures can be called inside an outer [`compile`], threading its
    /// captures into the outer compiled function. Only the outer program is compiled.
    #[test]
    fn test_staged_function_call_stages_into_outer_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let bias = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.25]).as_slice(),
        )
        .unwrap();
        let inner: StagedFunction<'_, ArrayType, ArrayType> = stage_with_captures(
            |captures, x| (x + captures[0].clone()).sin().unwrap(),
            vec![bias],
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        assert_eq!(engine.program_count(), 0, "staging the inner function must not compile it");

        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.call(x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        assert_eq!(engine.program_count(), 1, "only the outer program should be compiled");
        assert_eq!(outer.source_program().captures().len(), 1);

        let input_value = 0.5f32;
        let input = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let observed = read_f32_array(&client, &outer.executable().execute(&engine, input).unwrap());
        assert_eq!(observed.len(), 1);
        let expected = (input_value + 0.25).sin().cos();
        assert!((observed[0] - expected).abs() < 1e-5, "expected cos(sin(x + bias)) = {expected}, got {}", observed[0]);
    }

    #[test]
    fn test_compiled_function_call_under_eager_parents() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&engine, r#type.clone(), mesh.clone(), values_to_bytes::<f32>(values).as_slice())
                .unwrap()
        };
        let read =
            |value: Value<'_>| read_f32_array(&client, &ValueProjection::<ArrayType>::into_projected(value).unwrap());

        // `f(x) = x * c` captures the runtime array `c`, so every call below also imports a capture into its parent.
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&vector_type, &[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // An eager parent executes the call immediately on concrete arrays.
        let output = function.call(array(&vector_type, &[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(read_f32_array(&client, &output), vec![2.0, -2.0, 1.5]);

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
        let mesh = single_device_mesh(client);
        let engine = Session::new(client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };

        // `f(x) = x * c` captures the runtime array `c`. Rematerialized and composite bodies are traced over constants
        // alone, so calling `f` inside them imports `c` as a session constant.
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
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
        let outer: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType, ArrayType)> = compile(
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
        assert_eq!(read_f32_array(client, &output), vec![2.0, -2.0, 1.5]);
        assert_eq!(read_f32_array(client, &rematerialized_gradient), vec![2.0, -1.0, 1.0]);
        assert_eq!(read_f32_array(client, &composed_gradient), vec![2.0, -1.0, 1.0]);

        // Eager parents resolve the session constant when lifting it, and eager dispatch passes its array to the
        // executables of the rematerialized and composite applications.
        let read =
            |value: Value<'_>| read_f32_array(client, &ValueProjection::<ArrayType>::into_projected(value).unwrap());
        assert_eq!(read(rematerialized.call(Value::Array(array(&[1.0, 2.0, 3.0]))).unwrap()), vec![2.0, -2.0, 1.5]);
        assert_eq!(read(composed.call(Value::Array(array(&[1.0, 2.0, 3.0]))).unwrap()), vec![2.0, -2.0, 1.5]);
    }

    #[test]
    fn test_compiled_function_call_inside_nested_body_regions_and_custom_rules() {
        type Body = Tracer<TracingContext<Constant, Operation>>;
        let client = Box::leak(Box::new(execution_client()));
        let mesh = single_device_mesh(client);
        let engine = Session::new(client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x * captures[0].clone(),
            vec![array(&[2.0, -1.0, 0.5])],
            vector_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // A `scan` nested in a rematerialized body imports the capture of `f` through its own trace, so its session
        // constant lives in a region attached to the rematerialized body: `x ↦ x * c³`.
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

        // A custom forward-mode rule that calls `f` on the tangent is traced lazily, after the primal body.
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

        let outer: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
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
        assert_eq!(read_f32_array(client, &gradient), vec![8.0, -1.0, 0.25]);
        assert_eq!(read_f32_array(client, &tangent), vec![2.0, -1.0, 1.0]);
    }

    #[test]
    fn test_compiled_function_call_inside_batch_in_parent() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let array = |r#type: &ArrayType, values: &[f32]| {
            Array::from_host_buffer(&engine, r#type.clone(), mesh.clone(), values_to_bytes::<f32>(values).as_slice())
                .unwrap()
        };
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
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
            |x| Ok(function.call(x)?),
            array(&matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::default(),
        )
        .unwrap();
        assert_eq!(read_f32_array(&client, &batched), vec![2.0, -2.0, 1.5, 8.0, -5.0, 3.0]);
        let outer: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            |xs| {
                batch_in_parent(
                    |x| Ok(function.call(x)?),
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
        assert_eq!(read_f32_array(&client, &output), vec![2.0, 8.0, -2.0, -5.0, 1.5, 3.0]);
    }

    #[test]
    fn test_compiled_function_call_inside_differentiate_in_parent_at() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(
                &engine,
                vector_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };
        let read = |array: &Array<'_>| read_f32_array(&client, array);
        let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
        let function: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_captures(
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
            .jvp(array(&[1.0, 1.0, 2.0]), |x| Ok(function.call(x)?))
            .unwrap();
        assert_eq!((read(&output), read(&tangent)), (vec![2.0, -2.0, 1.5], vec![2.0, -1.0, 1.0]));
        let (_, pushforward) =
            differentiate_in_parent_at(array(&[1.0, 2.0, 3.0])).linearize(|x| Ok(function.call(x)?)).unwrap();
        assert_eq!(read(&pushforward.apply(array(&[1.0, 1.0, 2.0])).unwrap()), vec![2.0, -1.0, 1.0]);
        let (_, pullback) = differentiate_in_parent_at(array(&[1.0, 2.0, 3.0])).vjp(|x| Ok(function.call(x)?)).unwrap();
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
            .gradient(|x| Ok(function.call(x.clone())?.dot(&x, &dimensions)?))
            .unwrap();
        assert_eq!(read(&gradient), vec![4.0, -4.0, 3.0]);
        let jacobian = differentiate_in_parent_at(array(&[1.0, 2.0, 3.0]))
            .jacobian_forward(|x| Ok(function.call(x)?))
            .unwrap();
        assert_eq!(read(&jacobian.values()[0]), vec![2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5]);

        // The same entry point serves the array leaves of an enclosing compilation.
        let outer: CompiledFunction<'_, ArrayType, (ArrayType, ArrayType)> = compile(
            |x| {
                let jacobian =
                    differentiate_in_parent_at(x.clone()).jacobian_reverse(|x| Ok(function.call(x)?)).unwrap();
                let hessian = differentiate_in_parent_at(x)
                    .hessian(|x| Ok(function.call(x.clone())?.dot(&x, &dimensions)?))
                    .unwrap();
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
    fn test_axis_index_inside_jvp_matches_across_the_call_boundary() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let xs = Array::from_host_buffer(
            &engine,
            vector_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0]).as_slice(),
        )
        .unwrap();

        // `x ↦ x + i` reads the index `i` of the enclosing batch axis from inside a `jvp` trace. Calling it as a compiled
        // function (whose staged specialization is traced in the caller's named axes) matches calling it inline.
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
        let dispatcher: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> =
            jit(|(), x: CompileTracer<'_>| Ok::<_, Error>(plus_axis_index!(x)), &engine, mesh.clone());
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
            assert_eq!(read_f32_array(&client, &output), vec![10.0, 21.0, 32.0]);
            assert_eq!(read_f32_array(&client, &tangent), vec![10.0, 20.0, 30.0]);
        }
    }

    #[test]
    fn test_compile_applies_trace_local_pushforward_under_batch() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let vector_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();

        // `x ↦ sin(x) * x` is linearized inside the compilation, so its residuals are tracers of that trace. Applying
        // the pushforward under a batching level of the same trace imports them as replicated values of that level.
        let function: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> = compile(
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
                    Array::from_host_buffer(&engine, vector_type, mesh.clone(), values_to_bytes::<f32>(&x).as_slice())
                        .unwrap(),
                    Array::from_host_buffer(&engine, matrix_type, mesh, values_to_bytes::<f32>(&tangents).as_slice())
                        .unwrap(),
                ),
            )
            .unwrap();
        let derivative = x.map(|x| x.cos() * x + x.sin());
        let expected = tangents.iter().enumerate().map(|(index, tangent)| tangent * derivative[index % 2]);
        assert!(
            read_f32_array(&client, &output)
                .iter()
                .zip(expected)
                .all(|(observed, expected)| (observed - expected).abs() < 1e-5)
        );
    }

    #[test]
    fn test_compiled_function_dispatcher_call() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let function: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> =
            jit(|(), x: CompileTracer<'_>| Ok::<_, Error>(x.sin()?), &engine, mesh.clone());

        // A call traces the closure into the calling trace, and neither lowers, compiles, nor executes it.
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| function.call((), x).unwrap().cos().unwrap(), input_type.clone(), &engine, mesh.clone())
                .unwrap();
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (0, 1, 1));
        assert_eq!((statistics.dispatch_misses, statistics.lowerings, statistics.compilation_requests), (0, 0, 0));
        let input = Array::from_host_buffer(
            &engine,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.5, -1.0]).as_slice(),
        )
        .unwrap();
        let observed = read_f32_array(&client, &outer.executable().execute(&engine, input).unwrap());
        assert!(observed.iter().zip([0.5f32, -1.0]).all(|(observed, x)| (observed - x.sin().cos()).abs() < 1e-6));

        // Another call with the same types and scope reuses the staged specialization.
        compile::<_, ArrayType, ArrayType>(|x| function.call((), x).unwrap(), input_type, &engine, mesh).unwrap();
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (1, 1, 1));
    }

    #[test]
    fn test_compiled_function_dispatcher_call_with_named_axes() {
        let client = execution_client();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();

        // The closure reads the index along the named batch axis `i` of the level that calls it. The call traces
        // it in that level's scope, and batching the resulting call resolves the axis once per batch item.
        let function: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> = jit(
            |(), x: CompileTracer<'_>| {
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
        let outer: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
            values_to_bytes::<f32>(&[0.0; 6]).as_slice(),
        )
        .unwrap();
        assert_eq!(read_f32_array(&client, &outer.executable().execute(&engine, input).unwrap()), vec![0.0, 1.0, 2.0]);

        // With only replicated inputs and an explicit extent, the call must still be rebatched at the named level,
        // because its callee reads that level by name. Binding the callee unchanged would let `axis_index` escape it.
        let outer: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
            Array::from_host_buffer(&engine, matrix_type, mesh, values_to_bytes::<f32>(&[0.0; 6]).as_slice()).unwrap();
        assert_eq!(read_f32_array(&client, &outer.executable().execute(&engine, input).unwrap()), vec![0.0, 1.0, 2.0]);
    }

    /// Input sharding overrides change the input types before tracing begins.
    #[test]
    fn test_staging_applies_input_shardings() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding.clone()).unwrap();

        let options = Options::new(mesh).with_input_shardings(vec![sharding]);
        let staged: StagedFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type, options).unwrap();
        let ArrayIrType::Array(staged_input_type) = &staged.source_program().program().input_types()[0] else {
            panic!("public staged input should remain an array");
        };
        assert!(staged_input_type.sharding().is_some());
        assert_eq!(engine.program_count(), 0, "staging must not build an executable");
    }

    /// End-to-end check that a function calling the same staged block twice — whose two `jit_call`s deduplicate into
    /// one shared `func.func` at lowering — still compiles and computes the correct result through PJRT. Here
    /// `g(x) = sin(x) + sin(x) = 2 sin(x)`.
    #[test]
    fn test_repeated_staged_call_round_trips() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        let inner: StagedFunction<'_, ArrayType, ArrayType> =
            stage(|x| x.sin().unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        let outer: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let observed = read_f32_array(&client, &outer.executable().execute(&engine, input).unwrap());
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
    fn print_baseline_compilation_statistics(domain: &Domain<'_>) {
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
            domain.program_count(),
            statistics.memory_lookup_duration.as_secs_f64() * 1e3,
            statistics.compilation_duration.as_secs_f64() * 1e3,
        );
    }

    /// Caching-plan measurement for repeated [`CompiledFunction::jvp`] construction.
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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();

        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]))
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
                .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap().cos().unwrap(), input_type, &domain, mesh).unwrap();

        domain.compilation_context().clear_statistics();
        let mut durations = Vec::with_capacity(BASELINE_REPETITIONS);
        for _ in 0..BASELINE_REPETITIONS {
            let start = Instant::now();
            let derived: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
                compiled.jvp(&domain).unwrap();
            durations.push(start.elapsed());
            assert_eq!(derived.output_types().len(), 2);
        }

        print_baseline_durations("repeated CompiledFunction::jvp", &durations);
        print_baseline_compilation_statistics(&domain);
    }

    /// Caching-plan measurement for repeated [`CompiledFunction::gradient`] construction. Mirrors
    /// [`test_baseline_repeated_compiled_function_jvp`] for the reverse-mode path.
    #[test]
    #[ignore = "phase 0 baseline measurement"]
    fn test_baseline_repeated_compiled_function_gradient() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();

        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4), CoreDimension::Static(4)]))
                .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
                .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap().reduce(&[0, 1], ReductionKind::Sum).unwrap(), input_type, &domain, mesh)
                .unwrap();

        domain.compilation_context().clear_statistics();
        let mut durations = Vec::with_capacity(BASELINE_REPETITIONS);
        for _ in 0..BASELINE_REPETITIONS {
            let start = Instant::now();
            let derived: CompiledFunction<'_, ArrayType, ArrayType> = compiled.gradient(&domain).unwrap();
            durations.push(start.elapsed());
            assert_eq!(derived.output_types().len(), 1);
        }

        print_baseline_durations("repeated CompiledFunction::gradient", &durations);
        print_baseline_compilation_statistics(&domain);
    }

    /// Phase 0 caching-plan composition baseline: a retained JIT dispatcher closure that internally runs an eager reverse-mode
    /// gradient. The JIT boundary stages the eager transform exactly once per specialization, so repeated calls with
    /// same-shaped inputs are served entirely by retained dispatch. The recorded counters are the reference point the
    /// caching redesign must preserve, and they also show whether the current trace and lowering fallback tiers ever
    /// hit under this workload.
    #[test]
    #[ignore = "phase 0 baseline measurement"]
    fn test_baseline_jit_composition_caches_eager_transforms() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();

        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let function: CompiledFunctionDispatcher<'_, _, (), ArrayType, ArrayType> = jit(
            |_, input: CompileTracer<'_>| {
                Ok(input.domain().differentiate_at(input).gradient(|value| Mul::mul(&value, &value))?)
            },
            &domain,
            mesh.clone(),
        );

        domain.compilation_context().clear_statistics();
        let mut durations = Vec::with_capacity(BASELINE_REPETITIONS);
        for index in 0..BASELINE_REPETITIONS {
            let input = Array::from_host_buffer(
                &domain,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[index as f32]).as_slice(),
            )
            .unwrap();
            let start = Instant::now();
            let output = function.execute((), input).unwrap();
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
            "  dispatcher: dispatch_hits={} dispatch_misses={} traces={} lowerings={} compilation_requests={}",
            statistics.dispatch_hits,
            statistics.dispatch_misses,
            statistics.traces,
            statistics.lowerings,
            statistics.compilation_requests,
        );
        println!(
            "  dispatcher: dispatch_preparation_ms={:.3} dispatch_lookup_ms={:.3} tracing_ms={:.3} \
             lowering_ms={:.3}",
            statistics.dispatch_preparation_duration.as_secs_f64() * 1e3,
            statistics.dispatch_lookup_duration.as_secs_f64() * 1e3,
            statistics.tracing_duration.as_secs_f64() * 1e3,
            statistics.lowering_duration.as_secs_f64() * 1e3,
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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let domain = Session::new(&client).domain();

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
            let callee: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
                let outer: CompiledFunction<'_, ArrayType, ArrayType> = compile(
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
                    .filter(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
                    .count();
                let inlined_cosine_count = instructions
                    .iter()
                    .filter(|instruction| matches!(instruction.operation(), Operation::Array(ArrayOperation::Cos(_))))
                    .count();
                assert_eq!(jit_call_count, 1, "each outer specialization should stage one jit_call boundary");
                assert_eq!(inlined_cosine_count, 0, "the outer program should not inline the callee body");
                outers.push(outer);
            }

            let mut jvp_totals = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            let mut jvp_backends = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            for outer in &outers {
                let before = domain.compilation_context().statistics().compilation_duration;
                let start = Instant::now();
                let derived: CompiledFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
                    outer.jvp(&domain).unwrap();
                jvp_totals.push(start.elapsed());
                let after = domain.compilation_context().statistics().compilation_duration;
                jvp_backends.push(after - before);
                assert_eq!(derived.output_types().len(), 2);
            }

            let mut gradient_totals = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            let mut gradient_backends = Vec::with_capacity(CALLEE_OUTER_SPECIALIZATIONS);
            for outer in &outers {
                let before = domain.compilation_context().statistics().compilation_duration;
                let start = Instant::now();
                let derived: CompiledFunction<'_, ArrayType, ArrayType> = outer.gradient(&domain).unwrap();
                gradient_totals.push(start.elapsed());
                let after = domain.compilation_context().statistics().compilation_duration;
                gradient_backends.push(after - before);
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
    /// domain universe: its tracers are exactly [`CompileTracer`]s, so the shared [`decode_step`] model runs
    /// unchanged inside the staged regions and in the eager reference loop.
    type DecodeTraceContext<'c> = DomainTracingContext<Domain<'c>, Value<'c>>;

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
        Shape::new(dimensions.iter().map(|&dimension| CoreDimension::Static(dimension)).collect())
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
    /// function executes eagerly over reference [`CoreArray`](ryft_core::Array)s and stages symbolically over [`CompileTracer`]s
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
            + CoreValue<Type = ArrayType>
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
        let zero_index = position.zero_like()?;
        let embedding = embeddings
            .dynamic_slice(&[token, zero_index.clone()], &[1, dimension])?
            .reshape(static_shape(&[dimension]))?;
        let vector_times_matrix = DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new());
        let query = embedding.dot(query_weights, &vector_times_matrix).unwrap();
        let key = embedding.dot(key_weights, &vector_times_matrix).unwrap();
        let value = embedding.dot(value_weights, &vector_times_matrix).unwrap();
        let cache_keys = state[2].dynamic_update_slice(
            &key.reshape(static_shape(&[1, dimension]))?,
            &[position.clone(), zero_index.clone()],
        )?;
        let cache_values = state[3]
            .dynamic_update_slice(&value.reshape(static_shape(&[1, dimension]))?, &[position.clone(), zero_index])?;

        // Masked scaled dot-product attention over the visible cache prefix `[0, position]`.
        let attended = match attention {
            DecodeAttention::Composed => {
                let scores = cache_keys
                    .dot(&query, &DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()))
                    .unwrap();
                let scores_type = scores.r#type().into_owned();
                let scale = context.fill(&scores_type, 1.0 / (dimension as f32).sqrt())?;
                let scores = scores.mul(&scale)?;
                let positions_type = tokens.r#type().into_owned();
                let positions = context.iota(&positions_type, 0)?;
                let visible = positions
                    .compare(&position.broadcast(positions_type, &[])?, ComparisonDirection::LessThanOrEqual)?;
                let masked = visible.select(&scores, &context.fill(&scores_type, -1.0e30f32)?)?;
                let stabilized = masked
                    .sub(&masked.reduce(&[0], ReductionKind::Max).unwrap().broadcast(scores_type.clone(), &[])?)?;
                let exponentials = stabilized.exp()?;
                let weights = exponentials
                    .div(&exponentials.reduce(&[0], ReductionKind::Sum).unwrap().broadcast(scores_type, &[])?)?;
                weights
                    .dot(&cache_values, &DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new()))
                    .unwrap()
            }
            DecodeAttention::CustomCall => {
                let operation =
                    CustomCallOperation::new(DECODE_ATTENTION_CUSTOM_CALL_TARGET, vec![query.r#type().into_owned()]);
                let inputs = [cache_keys.clone(), cache_values.clone(), query, position.clone()];
                V::custom_call(&operation, &inputs)?.remove(0)
            }
        };

        // Gated multilayer perceptron, output projection, and next-token selection.
        let hidden = attended.dot(hidden_weights, &vector_times_matrix).unwrap().tanh()?;
        let gate = attended.dot(gate_weights, &vector_times_matrix).unwrap().logistic()?;
        let logits = hidden.mul(&gate)?.dot(output_weights, &vector_times_matrix).unwrap();
        let (generator, next_token) = match sampling {
            DecodeSampling::Greedy => (generator, logits.argmax(0)?),
            DecodeSampling::TopK => {
                let (top_logits, top_indices) = logits.top_k(configuration.top_k, 0)?;
                let (generator, choice) = generator.random_categorical(&top_logits, 0)?;
                let next_token = top_indices.dynamic_slice(&[choice], &[1])?.reshape(static_shape(&[]))?;
                (generator, next_token)
            }
        };
        let tokens = tokens.dynamic_update_slice(&next_token.reshape(static_shape(&[1]))?, &[position.clone()])?;
        let position = position.add(&position.one_like()?)?;
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
        inputs: Vec<CompileTracer<'c>>,
        configuration: &DecodeConfiguration,
        sampling: DecodeSampling,
        attention: DecodeAttention,
    ) -> Vec<CompileTracer<'c>> {
        let context = inputs[0].value().context().clone();
        let carry_types = inputs.iter().map(|input| input.value().r#type().into_owned()).collect::<Vec<ArrayIrType>>();
        let steps = configuration.steps;
        let (_, condition) = <DecodeTraceContext<'c>>::trace(
            |state: Vec<StatefulCompileTracer<'c>>| {
                let state = state
                    .into_iter()
                    .map(|value| {
                        ValueProjection::<ArrayType>::into_projected(value)
                            .expect("decode condition state should remain array-valued")
                    })
                    .collect::<Vec<_>>();
                let position = &state[0];
                let limit = position.domain().fill(&position.r#type().into_owned(), steps as i32)?;
                Ok(vec![position.compare(&limit, ComparisonDirection::LessThan)?.into_value()])
            },
            carry_types.clone(),
        )
        .unwrap();
        let (_, body) = <DecodeTraceContext<'c>>::trace(
            |state: Vec<StatefulCompileTracer<'c>>| {
                let state = state
                    .into_iter()
                    .map(|value| {
                        ValueProjection::<ArrayType>::into_projected(value)
                            .expect("decode body state should remain array-valued")
                    })
                    .collect::<Vec<_>>();
                let context = state[0].domain();
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
            .bind(Operation::While(WhileOperation::new()), vec![condition, body], inputs.as_slice())
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

    /// Runs the decode-loop demo's eager reference: the same [`decode_step`] executed over reference [`CoreArray`](ryft_core::Array)
    /// values in a plain Rust loop, with no `While` staging, compilation, or XLA involvement.
    fn reference_decode_loop(
        initial_state: Vec<CoreArray>,
        configuration: &DecodeConfiguration,
        sampling: DecodeSampling,
    ) -> Vec<CoreArray> {
        let context = EagerContext::<CoreArray, ArrayOperation<CoreArray>>::new();
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
                "expected the `{name}` input buffer of the `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call"
            ))),
        };
        let keys = next_buffer("keys")?;
        let values = next_buffer("values")?;
        let query = next_buffer("query")?;
        let position = next_buffer("position")?;
        let mut outputs = call_frame.outputs();
        let Some(Ok(FfiOutput::Buffer { buffer: output })) = outputs.next() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call to have one output buffer"
            )));
        };
        let &[steps, dimension] = keys.dimensions() else {
            return Err(FfiError::invalid_argument(format!(
                "expected the `keys` input of the `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call to have rank 2"
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
                "unexpected `{DECODE_ATTENTION_CUSTOM_CALL_TARGET}` custom call buffer types or shapes"
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
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = single_device_mesh(&client);
        let engine = Session::new(&client).domain();
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
            ArrayType::new_static(data_type, dimensions)
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

        let compiled: CompiledFunction<'_, Vec<ArrayType>, Vec<ArrayType>> = compile(
            |inputs| build_decode_loop(inputs, &configuration, sampling, attention),
            input_types.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let device_input = |index: usize, bytes: Vec<u8>| {
            Array::from_host_buffer(&engine, input_types[index].clone(), mesh.clone(), bytes.as_slice()).unwrap()
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
        let outputs = compiled.executable().execute(&engine, device_inputs).unwrap();
        assert_eq!(outputs.len(), 3);
        let device_tokens = read_i32_array(&client, &outputs[0]);
        let device_cache_keys = read_f32_array(&client, &outputs[1]);
        let device_cache_values = read_f32_array(&client, &outputs[2]);

        let mut reference_state = vec![
            CoreArray::from_elements(ArrayType::new_static(DataType::I32, []), &[0_i32]).unwrap(),
            CoreArray::from_elements(ArrayType::new_static(DataType::I32, []), &[3_i32]).unwrap(),
            CoreArray::from_elements(
                ArrayType::new_static(DataType::F32, [steps, dimension]),
                &vec![0.0_f32; steps * dimension],
            )
            .unwrap(),
            CoreArray::from_elements(
                ArrayType::new_static(DataType::F32, [steps, dimension]),
                &vec![0.0_f32; steps * dimension],
            )
            .unwrap(),
            CoreArray::from_elements(ArrayType::new_static(DataType::I32, [steps]), &vec![0_i32; steps]).unwrap(),
            CoreArray::from_elements(ArrayType::new_static(DataType::U64, [2]), &[42_u64, 0]).unwrap(),
        ];
        reference_state.extend(weight_dimensions.iter().zip(&weights).map(|(dimensions, values)| {
            CoreArray::from_elements(ArrayType::new_static(DataType::F32, *dimensions), values).unwrap()
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
            DecodeSampling::TopK => &[13, 12, 10, 10, 2, 15],
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
