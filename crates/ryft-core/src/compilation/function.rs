//! Backend-agnostic compile-and-execute entry points and the [`CompiledFunction`] handle.

use std::marker::PhantomData;

use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::tracing::DomainTracer;
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
/// / output type metadata needed to marshal a [`Parameterized`] tree of runtime values into the
/// program and reassemble the outputs back into the user's expected output tree shape.
///
/// `In` and `Out` mirror JAX's PyTree pattern: they describe how nested tuples / structs of
/// runtime values are flattened into the program's positional arguments and outputs. For a
/// function `|x: DomainTracer<'_, E>| -> DomainTracer<'_, E>`, `In = Out = E::Type`. For
/// `|(a, b): (DomainTracer<'_, E>, DomainTracer<'_, E>)| ->
/// (DomainTracer<'_, E>, DomainTracer<'_, E>)`, `In = Out = (E::Type, E::Type)`.
///
/// The handle also retains the **source [`Program`]** that produced the compiled artifact.
/// This makes the handle inspectable for diagnostics (see [`Self::source_program`]) without
/// re-running the user's original closure.
pub struct CompiledFunction<'engine, E: CompilationDomain, In, Out>
where
    E::Value: Typed<E::Type>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<E::RuntimeValue>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<E::RuntimeValue>>,
{
    /// Backend's compiled artifact. Cached in the [`CompilationContext`]; this handle holds a
    /// `Clone` of the cached entry.
    program: E::CompiledProgram,

    /// Source [`Program`] that produced [`Self::program`]. Retained so callers can inspect the
    /// traced IR (printing, instruction counts, graph rendering).
    source_program: Program<E::Type, E::Value, E::Operation, In::To<E::Value>, Out::To<E::Value>>,

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
    E::Operation: Clone,
    E::CompiledProgram: Clone,
    E::Type: Clone,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<E::RuntimeValue>>,
    In::To<E::Value>: Clone,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<E::RuntimeValue>>,
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
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<E::RuntimeValue>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<E::RuntimeValue>>,
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
    /// raw, untransformed traced IR, useful for diagnostics (printing, instruction counts, graph rendering).
    #[inline]
    pub fn source_program(&self) -> &Program<E::Type, E::Value, E::Operation, In::To<E::Value>, Out::To<E::Value>> {
        &self.source_program
    }

    /// Returns the backend this function was compiled with.
    #[inline]
    pub fn engine(&self) -> &'engine E {
        self.engine
    }

    /// Invokes this compiled function on `inputs`.
    #[inline]
    pub fn call(&self, inputs: In::To<E::RuntimeValue>) -> Result<Out::To<E::RuntimeValue>, CompilationError<E::Error>>
    where
        In::To<E::RuntimeValue>: Parameterized<E::RuntimeValue>,
        Out::To<E::RuntimeValue>:
            Parameterized<E::RuntimeValue, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs_vec: Vec<E::RuntimeValue> = inputs.into_parameters().collect();
        let outputs_vec = self.engine.execute(&self.program, inputs_vec).map_err(CompilationError::Backend)?;
        Out::To::<E::RuntimeValue>::from_parameters(self.output_structure.clone(), outputs_vec)
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
pub fn compile_with_options<'engine, E, F, In, Out>(
    engine: &'engine E,
    function: F,
    input_types: In,
    options: CompilationOptions<E>,
) -> Result<CompiledFunction<'engine, E, In, Out>, CompilationError<E::Error>>
where
    E: 'engine + CompilationDomain,
    E::Value: Typed<E::Type>,
    F: FnOnce(In::To<DomainTracer<'engine, E>>) -> Out::To<DomainTracer<'engine, E>>,
    In: Parameterized<
            E::Type,
            Family: ParameterizedFamily<E::Value>
                        + ParameterizedFamily<E::RuntimeValue>
                        + ParameterizedFamily<DomainTracer<'engine, E>>,
        >,
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<
            E::Type,
            Family: ParameterizedFamily<E::Value>
                        + ParameterizedFamily<E::RuntimeValue>
                        + ParameterizedFamily<DomainTracer<'engine, E>>,
        >,
    Out::To<DomainTracer<'engine, E>>:
        Parameterized<DomainTracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
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
    F: FnOnce(In::To<DomainTracer<'engine, E>>) -> Out::To<DomainTracer<'engine, E>>,
    In: Parameterized<
            E::Type,
            Family: ParameterizedFamily<E::Value>
                        + ParameterizedFamily<E::RuntimeValue>
                        + ParameterizedFamily<DomainTracer<'engine, E>>,
        >,
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<
            E::Type,
            Family: ParameterizedFamily<E::Value>
                        + ParameterizedFamily<E::RuntimeValue>
                        + ParameterizedFamily<DomainTracer<'engine, E>>,
        >,
    Out::To<DomainTracer<'engine, E>>:
        Parameterized<DomainTracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
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
    F: FnOnce(In::To<DomainTracer<'engine, E>>) -> Out::To<DomainTracer<'engine, E>>,
    In: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<DomainTracer<'engine, E>>>,
    Out: Parameterized<E::Type, Family: ParameterizedFamily<E::Value> + ParameterizedFamily<DomainTracer<'engine, E>>>,
    Out::To<DomainTracer<'engine, E>>:
        Parameterized<DomainTracer<'engine, E>, To<E::Type> = Out, To<E::Value> = Out::To<E::Value>>,
{
    let (output_types_tree, _program) =
        engine.trace(|tracers| Ok(function(tracers)), input_types).map_err(CompilationError::Tracing)?;
    Ok(output_types_tree)
}
