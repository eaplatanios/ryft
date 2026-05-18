//! Backend-agnostic compile-and-execute entry points and the [`CompiledFunction`] handle.

use std::marker::PhantomData;

use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::tracing::Tracer;
use crate::tracing::programs::Program;
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
/// The handle also retains the **source [`Program`]** that produced the compiled artifact.
/// This is what makes transform-composition possible: an outer `grad` / `jvp` / `vjp` / `vmap`
/// can walk `source_program` and emit a transformed program into its trace context (mirroring
/// the existing
/// [`transpose_nested`](crate::differentiation::transposition::ProgramTracingContext::transpose_nested)
/// pattern), and an outer `compile_and_execute` can stage `self` as a primitive operation that
/// inlines `source_program` into the surrounding MLIR module.
pub struct CompiledFunction<'engine, E: CompilationDomain, In, Out>
where
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
{
    /// Backend's compiled artifact. Cached in the [`CompilationContext`]; this handle holds a
    /// `Clone` of the cached entry.
    program: E::CompiledProgram,

    /// Source [`Program`] that produced [`Self::program`]. Retained so that outer transforms
    /// (re-tracing via `transpose_nested` / a future `linearize_nested` / `jvp_nested`) and
    /// inner staging (`self` as a primitive in another trace) can rebuild a transformed or
    /// nested version of this function without re-running the user's original closure.
    source_program: Program<E::Type, E::Value, E::OperationCarrier, In::To<E::Value>, Out::To<E::Value>>,

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

impl<'engine, E, In, Out> CompiledFunction<'engine, E, In, Out>
where
    E: CompilationDomain,
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
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

    /// Returns the source [`Program`] that produced [`Self::compiled_program`]. This is the
    /// raw, untransformed traced program — the same input the engine's
    /// [`compile`](CompilationDomain::compile) consumed. Outer transforms (`grad` / `jvp` /
    /// `vjp` / `vmap`) and inner staging walk this to derive transformed programs or to inline
    /// the function as a primitive in a larger trace.
    #[inline]
    pub fn source_program(
        &self,
    ) -> &Program<E::Type, E::Value, E::OperationCarrier, In::To<E::Value>, Out::To<E::Value>> {
        &self.source_program
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
        In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
        Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
        In::To<E::Value>: Parameterized<E::Value>,
        Out::To<E::Value>: Parameterized<E::Value, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs_vec: Vec<E::Value> = inputs.into_parameters().collect();
        let outputs_vec = self.engine.execute(&self.program, inputs_vec).map_err(CompilationError::Backend)?;
        Out::To::<E::Value>::from_parameters(self.output_structure.clone(), outputs_vec)
            .map_err(|error| CompilationError::Tracing(error.into()))
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
    F: FnOnce(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    // 1. Fingerprint the call site, mix in static_args_hash.
    let function_fingerprint = if options.static_args_hash == 0 {
        FunctionFingerprint::from_caller()
    } else {
        FunctionFingerprint::Composite {
            base: Box::new(FunctionFingerprint::from_caller()),
            extra: options.static_args_hash,
        }
    };

    // 2. Capture flat input types (before consuming `input_types` in trace).
    let input_types_vec: Vec<E::Type> = input_types.parameters().cloned().collect();

    // 3. Cache key from the engine.
    let cache_key = engine.compilation_key(&function_fingerprint, &input_types_vec, &options.options);

    // 4. Trace the user function. We do this on every call (including cache hits) so the
    //    `Out::ParameterStructure` and flat output types are available to construct the
    //    `CompiledFunction` handle. The compile + serialize work is what's cached, not the
    //    trace.
    let (output_types_tree, program) =
        engine.trace(|tracers| Ok(function(tracers)), input_types).map_err(CompilationError::Tracing)?;
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
        source_program: program,
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
    F: FnOnce(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    compile_and_execute_with_options(engine, function, input_types, CompilationOptions::default())
}

/// Same as [`compile_and_execute_with_options`] but accepts a typed `static_args: S`
/// parameter that the framework auto-hashes into the cache key. Use this when the closure
/// captures values that should partition the cache — the captured state itself flows into
/// the trace via the normal Rust closure-capture mechanism, and the hash of `static_args`
/// ensures repeat invocations at the same source line with different captured state get
/// distinct cache entries.
///
/// The mental model mirrors JAX's `jit(f, static_argnums=[0])`: callers wrap the static
/// values they want to use as compile-time constants in `static_args`, and rely on the
/// closure capturing them by value to make them available inside the body.
#[track_caller]
pub fn compile_and_execute_with_statics<'engine, E, F, S, In, Out>(
    engine: &'engine E,
    function: F,
    static_args: S,
    input_types: In,
    options: CompilationOptions<E>,
) -> Result<CompiledFunction<'engine, E, In, Out>, CompilationError<E::Error>>
where
    E: 'engine + CompilationDomain,
    E::Value: Typed<E::Type>,
    S: std::hash::Hash + 'static,
    F: FnOnce(In::To<Tracer<'engine, E>>) -> Out::To<Tracer<'engine, E>>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Auto-hash the static args. Mix in `TypeId` so that two distinct static-arg types with
    // structurally-identical hash impls (e.g. `f32` `0.0` vs `f64` `0.0`) still partition the
    // cache.
    let mut hasher = DefaultHasher::new();
    std::any::TypeId::of::<S>().hash(&mut hasher);
    static_args.hash(&mut hasher);
    let static_args_hash = hasher.finish();
    // Avoid `0` (which has special "no contribution" meaning) by flipping to `1` on the
    // off chance the user-controlled hash happens to be zero.
    let static_args_hash = if static_args_hash == 0 { 1 } else { static_args_hash };

    let options = CompilationOptions { static_args_hash, ..options };
    compile_and_execute_with_options(engine, function, input_types, options)
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
