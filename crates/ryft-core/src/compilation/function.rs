//! Backend-agnostic compile-and-execute entry points and the [`CompiledFunction`] handle.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::tracing::Tracer;
use crate::types::Typed;

use super::domain::CompilationDomain;
use super::error::CompilationError;
use super::fingerprint::FunctionFingerprint;
use super::options::CompilationOptions;

/// Handle to a compiled function. Returned by
/// [`compile_and_execute_with_options`] and [`compile_and_execute`].
///
/// Holds the backend's [`CompiledProgram`](CompilationDomain::CompiledProgram) plus the input
/// / output type metadata needed to marshal a [`Parameterized`] tree of runtime
/// [`Value`](crate::tracing::Value)s into the program and reassemble the outputs back into
/// the user's expected output tree shape.
///
/// `In` and `Out` mirror JAX's PyTree pattern: they describe how nested tuples / structs of
/// runtime values are flattened into the program's positional arguments and outputs. For a
/// function `|x: Tracer<E>| -> Tracer<E>`, `In = Out = E::Type`. For
/// `|(a, b): (Tracer<E>, Tracer<E>)| -> (Tracer<E>, Tracer<E>)`, `In = Out = (E::Type, E::Type)`.
///
/// The handle also retains the **user closure** that produced the trace. This is what makes
/// transform composition work: applying an outer `grad` / `jvp` / `vjp` / `vmap` to a
/// [`CompiledFunction`] just wraps the retained closure in a transform-aware closure and
/// returns a new [`CompiledFunction`]; staging a compiled function inside another
/// [`compile_and_execute`] call re-executes the retained closure against the outer trace's
/// tracers via [`Self::call_traced`]. Each call retraces (tracing is cheap), but compilation
/// hits the cache as usual.
pub struct CompiledFunction<'engine, E: CompilationDomain, In, Out>
where
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
{
    /// Backend's compiled artifact. Cached in the [`CompilationContext`]; this handle holds a
    /// `Clone` of the cached entry.
    program: E::CompiledProgram,

    /// Retained user closure. Stored as a refcounted trait object so that clones of this
    /// [`CompiledFunction`] share the same callable, and so that outer transforms / inner
    /// staging can re-execute it against fresh tracers without re-running its captures.
    function: Arc<dyn Fn(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>> + 'engine>,

    /// PyTree shape of the output. Used by [`Self::call`] to reassemble the executor's flat
    /// output buffer list back into the user's expected output tree.
    output_structure: Out::ParameterStructure,

    /// Flat output types in the same order the executor returns its outputs. Exposed via
    /// [`Self::output_types`] for inspection.
    output_types: Vec<E::Type>,

    /// Backend that owns the compiled program. Borrowed for the lifetime of this handle.
    engine: &'engine E,

    /// Holds the `In` type parameter for type-system tracking.
    _input: PhantomData<fn(In)>,
}

impl<'engine, E, In, Out> Clone for CompiledFunction<'engine, E, In, Out>
where
    E: CompilationDomain,
    E::Value: Typed<E::Type>,
    E::CompiledProgram: Clone,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    E::Type: Clone,
    Out::ParameterStructure: Clone,
{
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            function: Arc::clone(&self.function),
            output_structure: self.output_structure.clone(),
            output_types: self.output_types.clone(),
            engine: self.engine,
            _input: PhantomData,
        }
    }
}

impl<'engine, E, In, Out> CompiledFunction<'engine, E, In, Out>
where
    E: CompilationDomain,
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
{
    /// Returns the flat output types in the order the executor produces them. Useful when
    /// callers want to inspect or reuse the abstract result shape without invoking
    /// [`Self::call`].
    #[inline]
    pub fn output_types(&self) -> &[E::Type] {
        &self.output_types
    }

    /// Returns the backend's compiled program.
    #[inline]
    pub fn compiled_program(&self) -> &E::CompiledProgram {
        &self.program
    }

    /// Returns the backend this function was compiled with.
    #[inline]
    pub fn engine(&self) -> &'engine E {
        self.engine
    }

    /// Invokes the compiled program with `inputs`, a [`Parameterized`] tree of runtime values
    /// matching the `In` shape used at compile time. Returns a `Parameterized` tree of values
    /// in the `Out` shape.
    pub fn call(&self, inputs: In::To<E::Value>) -> Result<Out::To<E::Value>, CompilationError<E::Error>>
    where
        In::To<E::Value>: Parameterized<E::Value>,
        Out::To<E::Value>: Parameterized<E::Value, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs_vec: Vec<E::Value> = inputs.into_parameters().collect();
        let outputs_vec = self.engine.execute(&self.program, inputs_vec).map_err(CompilationError::Backend)?;
        Out::To::<E::Value>::from_parameters(self.output_structure.clone(), outputs_vec)
            .map_err(|error| CompilationError::Tracing(error.into()))
    }

    /// Stages this compiled function into an outer trace by re-executing the retained user
    /// closure against `inputs`. Use this to nest one [`CompiledFunction`] inside another
    /// [`compile_and_execute`] body, or to apply an outer transform (`grad`, `jvp`, `vjp`,
    /// `vmap`) by wrapping the call in a transform-aware closure.
    ///
    /// Mirrors how JAX inlines `jit(f)` into outer trace contexts: each call re-traces `f`
    /// against the current trace, so transforms compose naturally without needing a separate
    /// program-walking pass.
    #[inline]
    pub fn call_traced(&self, inputs: In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>> {
        (self.function)(inputs)
    }
}

/// Compiles `function` once, caches the resulting program in [`engine.cache()`](
/// CompilationDomain::cache), and returns a [`CompiledFunction`] handle.
///
/// Mirrors `jax.jit`. The function is traced into a [`Program`](crate::tracing::Program) on
/// every call (the trace cost is small relative to compile), then the engine's [`compile`]
/// step runs only on cache miss. Repeat invocations at the same call site with the same input
/// shapes reuse the cached program and skip the lowering / backend-compilation work entirely.
///
/// If [`CompilationDomain::cache`] returns `None`, every call compiles fresh; the user-visible
/// behavior is otherwise identical.
///
/// [`compile`]: CompilationDomain::compile
#[track_caller]
pub fn compile_and_execute_with_options<'engine, E, F, In, Out>(
    engine: &'engine E,
    function: F,
    input_types: In,
    options: CompilationOptions<E>,
) -> Result<CompiledFunction<'engine, E, In, Out>, CompilationError<E::Error>>
where
    E: 'engine + CompilationDomain,
    E::Value: Typed<E::Type>,
    F: Fn(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>> + 'engine,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // 1. Fingerprint the call site and fold in a hash of the input tree's structure. The
    //    structure (treedef plus any non-`Parameter` fields the user puts in their input
    //    struct — `batch_size: usize`, mode flags, hyperparameters, etc.) partitions the cache
    //    so that repeat invocations at the same source line with structurally-different inputs
    //    still get distinct compiled artifacts.
    let mut structure_hasher = DefaultHasher::new();
    input_types.parameter_structure().hash(&mut structure_hasher);
    let function_fingerprint = FunctionFingerprint::Composite {
        base: Box::new(FunctionFingerprint::from_caller()),
        extra: structure_hasher.finish(),
    };

    // 2. Capture flat input types (before consuming `input_types` in trace).
    let input_types_vec: Vec<E::Type> = input_types.parameters().cloned().collect();

    // 3. Cache key from the engine.
    let cache_key = engine.compilation_key(&function_fingerprint, &input_types_vec, &options.options);

    // 4. Trace the user function via an immutable borrow so `function` remains owned and can
    //    be retained in the resulting handle for transform composition and inner staging.
    let (output_types_tree, program) = engine
        .trace(|tracers| Ok((&function)(tracers)), input_types)
        .map_err(CompilationError::Tracing)?;
    let output_structure = output_types_tree.parameter_structure();
    let output_types_vec: Vec<E::Type> = output_types_tree.parameters().cloned().collect();

    // 5. Compile on miss. If the engine exposes a cache, route through it; otherwise compile
    //    directly without memoization.
    let compiled = match engine.cache() {
        Some(cache) => cache
            .get_or_compile(engine, cache_key, || engine.compile(&program, &options.options))
            .map_err(CompilationError::Backend)?,
        None => engine.compile(&program, &options.options).map_err(CompilationError::Backend)?,
    };

    Ok(CompiledFunction {
        program: compiled,
        function: Arc::new(function),
        output_structure,
        output_types: output_types_vec,
        engine,
        _input: PhantomData,
    })
}

/// Same as [`compile_and_execute_with_options`] but uses [`CompilationOptions::default`].
/// Available only for engines whose [`CompilationDomain::Options`] implement [`Default`].
#[track_caller]
pub fn compile_and_execute<'engine, E, F, In, Out>(
    engine: &'engine E,
    function: F,
    input_types: In,
) -> Result<CompiledFunction<'engine, E, In, Out>, CompilationError<E::Error>>
where
    E: 'engine + CompilationDomain,
    E::Value: Typed<E::Type>,
    E::Options: Default,
    F: Fn(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>> + 'engine,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    compile_and_execute_with_options(engine, function, input_types, CompilationOptions::default())
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
///
/// Useful for inspecting the output shape of a function before paying the trace-and-compile
/// cost — e.g. when sizing buffers or building a higher-level execution graph.
#[track_caller]
pub fn eval_shape<'engine, E, F, In, Out>(
    engine: &'engine E,
    function: F,
    input_types: In,
) -> Result<Out, CompilationError<E::Error>>
where
    E: 'engine + CompilationDomain,
    E::Value: Typed<E::Type>,
    F: FnOnce(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    let (output_types_tree, _program) =
        engine.trace(|tracers| Ok(function(tracers)), input_types).map_err(CompilationError::Tracing)?;
    Ok(output_types_tree)
}
