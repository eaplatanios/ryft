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
/// [`compile_with_options`] and [`compile`].
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
/// This makes the handle inspectable for diagnostics (see [`Self::source_program`]) and lets
/// outer transforms / inner staging walk the traced IR via [`Self::call_traced`] without
/// re-running the user's original closure. Mirrors JAX's compiled artifact carrying its jaxpr
/// alongside the executable.
pub struct CompiledFunction<'engine, E: CompilationDomain, In, Out>
where
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
{
    /// Backend's compiled artifact. Cached in the [`CompilationContext`]; this handle holds a
    /// `Clone` of the cached entry.
    program: E::CompiledProgram,

    /// Source [`Program`] that produced [`Self::program`]. Retained so callers can inspect the
    /// traced IR (printing, instruction counts, graph rendering) and so outer transforms /
    /// inner staging can walk it via [`Self::call_traced`].
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

impl<'engine, E, In, Out> Clone for CompiledFunction<'engine, E, In, Out>
where
    E: CompilationDomain,
    E::Value: Typed<E::Type> + Clone,
    E::OperationCarrier: Clone,
    E::CompiledProgram: Clone,
    E::Type: Clone,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    In::To<E::Value>: Clone,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    Out::To<E::Value>: Clone,
    Out::ParameterStructure: Clone,
{
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            source_program: self.source_program.clone(),
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
    /// raw, untransformed traced IR — useful for diagnostics (printing, instruction counts,
    /// graph rendering) and as the input to outer-trace inlining via [`Self::call_traced`].
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

    /// Invokes this compiled function on `inputs`.
    ///
    /// Dispatches at the type level via [`ExecutionDispatch`]: pass concrete runtime values
    /// (`In::To<E::Value>`) and the compiled artifact runs, returning `Result<Out::To<E::Value>>`;
    /// pass tracers (`In::To<Tracer<'engine, E>>`) and the retained source program is staged into
    /// the active outer trace, returning `Out::To<Tracer<'engine, E>>`. Mirrors JAX's `f(x)`
    /// behaving identically against concrete arrays and tracers.
    #[inline]
    pub fn call<I, Marker>(&self, inputs: I) -> I::Output
    where
        I: ExecutionDispatch<'engine, E, In, Out, Marker>,
    {
        inputs.invoke(self)
    }
}

/// Dispatch trait that lets [`CompiledFunction::call`] accept either concrete runtime values
/// (executing the compiled artifact) or tracers (staging the source program into the outer
/// trace). Type inference picks the impl from the input type; users write `f.call(x)` and never
/// name the marker.
#[allow(private_bounds)]
pub trait ExecutionDispatch<'engine, E: CompilationDomain, In, Out, Marker>: Sized
where
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
{
    /// Result type of [`CompiledFunction::call`] for this dispatch. Concrete-value execution
    /// returns `Result<Out::To<E::Value>, CompilationError<E::Error>>`; tracer staging returns
    /// `Out::To<Tracer<'engine, E>>` (infallible by construction).
    type Output;

    /// Performs the dispatch: executes the compiled artifact, or stages the source program.
    fn invoke(self, function: &CompiledFunction<'engine, E, In, Out>) -> Self::Output;
}

/// Marker selecting the concrete-value execution path of [`ExecutionDispatch`].
#[doc(hidden)]
pub struct ConcreteValueMarker;

/// Marker selecting the tracer-staging path of [`ExecutionDispatch`].
#[doc(hidden)]
pub struct TracerMarker;

impl<'engine, E, In, Out> ExecutionDispatch<'engine, E, In, Out, ConcreteValueMarker> for In::To<E::Value>
where
    E: CompilationDomain,
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    In::To<E::Value>: Parameterized<E::Value>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value>>,
    Out::To<E::Value>: Parameterized<E::Value, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
{
    type Output = Result<Out::To<E::Value>, CompilationError<E::Error>>;

    fn invoke(self, function: &CompiledFunction<'engine, E, In, Out>) -> Self::Output {
        let inputs_vec: Vec<E::Value> = self.into_parameters().collect();
        let outputs_vec = function.engine.execute(&function.program, inputs_vec).map_err(CompilationError::Backend)?;
        Out::To::<E::Value>::from_parameters(function.output_structure.clone(), outputs_vec)
            .map_err(|error| CompilationError::Tracing(error.into()))
    }
}

impl<'engine, E, In, Out> ExecutionDispatch<'engine, E, In, Out, TracerMarker> for In::To<Tracer<'engine, E>>
where
    E: CompilationDomain,
    E::Value: Typed<E::Type> + Clone,
    E::OperationCarrier: Clone,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    In::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>>,
    In::To<E::Value>: Parameterized<E::Value>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<E::Value>: Parameterized<E::Value>,
    Out::To<Tracer<'engine, E>>:
        Parameterized<Tracer<'engine, E>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
{
    type Output = Out::To<Tracer<'engine, E>>;

    fn invoke(self, function: &CompiledFunction<'engine, E, In, Out>) -> Self::Output {
        let inputs_vec: Vec<Tracer<'engine, E>> = self.into_parameters().collect();
        let context = inputs_vec
            .first()
            .expect("staging a compiled function into a trace requires at least one input tracer")
            .context()
            .clone();
        let outputs_vec = context
            .stage_program(&function.source_program, inputs_vec)
            .expect("staging a well-formed source program into a compatible outer trace should not fail");
        Out::To::<Tracer<'engine, E>>::from_parameters(function.output_structure.clone(), outputs_vec)
            .expect("reassembling outputs from the program's output structure should not fail")
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
pub fn compile_with_options<'engine, E, F, In, Out>(
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

    // 4. Trace the user function. The traced [`Program`] is retained on the resulting handle
    //    so callers can inspect it and so inner-staging / outer-transform paths can walk it.
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

/// Same as [`compile_with_options`] but uses [`CompilationOptions::default`].
/// Available only for engines whose [`CompilationDomain::Options`] implement [`Default`].
#[track_caller]
pub fn compile<'engine, E, F, In, Out>(
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
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>>,
    Out::To<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    compile_with_options(engine, function, input_types, CompilationOptions::default())
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
