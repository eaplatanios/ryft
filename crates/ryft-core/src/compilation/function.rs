use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::Program;
use crate::tracing::DomainTracer;

use super::domain::CompilationDomain;
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
/// function `|x: DomainTracer<D>| -> DomainTracer<D>`, `In = Out = D::Type`. For
/// `|(a, b): (DomainTracer<D>, DomainTracer<D>)| ->
/// (DomainTracer<D>, DomainTracer<D>)`, `In = Out = (D::Type, D::Type)`.
///
/// The handle also retains the **source [`Program`]** that produced the compiled artifact.
/// This makes the handle inspectable for diagnostics (see [`Self::source_program`]) without
/// re-running the user's original closure.
pub struct CompiledFunction<
    'domain,
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<D::Value>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<D::Value>>,
> {
    /// Backend's compiled artifact. Cached in the [`CompilationContext`]; this handle holds a
    /// `Clone` of the cached entry.
    program: D::CompiledProgram,

    /// Source [`Program`] that produced [`Self::program`]. Retained so callers can inspect the
    /// traced IR (printing, instruction counts, graph rendering).
    source_program: Program<D::Constant, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>>,

    /// PyTree shape of the output. Used by [`Self::call`] to reassemble the executor's flat
    /// output buffer list back into the user's expected output tree.
    output_structure: Output::ParameterStructure,

    /// Flat output types in the same order the executor returns its outputs. Exposed via
    /// [`Self::output_types`] for inspection.
    output_types: Vec<D::Type>,

    /// Domain that owns the compiled program. Borrowed for the lifetime of this handle.
    domain: &'domain D,

    /// Holds the `In` type parameter for type-system tracking.
    _input: PhantomData<fn(Input)>,
}

impl<
    'domain,
    D: CompilationDomain<Operation: Clone>,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<D::Value>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<D::Value>>,
> Clone for CompiledFunction<'domain, D, Input, Output>
{
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            source_program: self.source_program.clone(),
            output_structure: self.output_structure.clone(),
            output_types: self.output_types.clone(),
            domain: self.domain,
            _input: PhantomData,
        }
    }
}

impl<
    'domain,
    D: CompilationDomain,
    Input: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<D::Value>>,
    Output: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<D::Value>>,
> CompiledFunction<'domain, D, Input, Output>
{
    /// Returns the flat output types in the order the executor produces them. Useful when
    /// callers want to inspect or reuse the abstract result shape without invoking
    /// [`Self::call`].
    #[inline]
    pub fn output_types(&self) -> &[D::Type] {
        &self.output_types
    }

    /// Returns the backend's compiled program.
    #[inline]
    pub fn compiled_program(&self) -> &D::CompiledProgram {
        &self.program
    }

    /// Returns the source [`Program`] that produced [`Self::compiled_program`]. This is the
    /// raw, untransformed traced IR, useful for diagnostics (printing, instruction counts, graph rendering).
    #[inline]
    pub fn source_program(
        &self,
    ) -> &Program<D::Constant, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>> {
        &self.source_program
    }

    /// Returns the domain this function was compiled with.
    #[inline]
    pub fn domain(&self) -> &'domain D {
        self.domain
    }

    /// Invokes this compiled function on `inputs`.
    #[inline]
    pub fn call(&self, inputs: Input::To<D::Value>) -> Result<Output::To<D::Value>, D::Error>
    where
        Input::To<D::Value>: Parameterized<D::Value>,
        Output::To<D::Value>:
            Parameterized<D::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let inputs_vec: Vec<D::Value> = inputs.into_parameters().collect();
        let outputs_vec = self.domain.execute(&self.program, inputs_vec)?;
        Output::To::<D::Value>::from_parameters(self.output_structure.clone(), outputs_vec)
            .map_err(|error| D::Error::from(error.into()))
    }
}

/// Compiles `function` once, caches the resulting program in [`domain.cache()`](
/// CompilationDomain::cache), and returns a [`CompiledFunction`] handle.
///
/// Mirrors `jax.jit`. The function is traced into a [`Program`] on every call (the trace cost is small relative to
/// compile), then the domain's [`compile`] step runs only on cache miss. Repeat invocations at the same call site with
/// the same input shapes reuse the cached program and skip the lowering / backend-compilation work entirely.
///
/// If [`CompilationDomain::cache`] returns `None`, every call compiles fresh; the user-visible
/// behavior is otherwise identical.
///
/// [`compile`]: CompilationDomain::compile
#[track_caller]
pub fn compile_with_options<
    'domain,
    D: 'domain + CompilationDomain,
    F: FnOnce(Input::To<DomainTracer<D>>) -> Output::To<DomainTracer<D>>,
    Input: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<DomainTracer<D>>,
            ParameterStructure: Hash,
        >,
    Output: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<DomainTracer<D>>,
            To<DomainTracer<D>>: Parameterized<
                DomainTracer<D>,
                To<D::Type> = Output,
                To<D::Constant> = Output::To<D::Constant>,
            >,
        >,
>(
    domain: &'domain D,
    function: F,
    input_types: Input,
    options: CompilationOptions<D>,
) -> Result<CompiledFunction<'domain, D, Input, Output>, D::Error> {
    use std::collections::hash_map::DefaultHasher;

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
    let input_types_vec: Vec<D::Type> = input_types.parameters().cloned().collect();

    // 3. Cache key from the domain.
    let cache_key = domain.compilation_key(&function_fingerprint, &input_types_vec, &options.options);

    // 4. Trace the user function. The traced [`Program`] is retained on the resulting handle
    //    so callers can inspect it and so inner-staging / outer-transform paths can walk it.
    let (output_types_tree, program) = D::trace(|tracers| Ok(function(tracers)), input_types)?;
    let output_structure = output_types_tree.parameter_structure();
    let output_types_vec: Vec<D::Type> = output_types_tree.parameters().cloned().collect();

    // 5. Compile on miss. If the domain exposes a cache, route through it; otherwise compile
    //    directly without memoization.
    let compiled = match domain.cache() {
        Some(cache) => cache.get_or_compile(domain, cache_key, || domain.compile(&program, &options.options))?,
        None => domain.compile(&program, &options.options)?,
    };

    Ok(CompiledFunction {
        program: compiled,
        source_program: program,
        output_structure,
        output_types: output_types_vec,
        domain,
        _input: PhantomData,
    })
}

/// Same as [`compile_with_options`] but uses [`CompilationOptions::default`].
/// Available only for domains whose [`CompilationDomain::Options`] implement [`Default`].
#[track_caller]
pub fn compile<
    'domain,
    D: 'domain + CompilationDomain<Options: Default>,
    F: FnOnce(Input::To<DomainTracer<D>>) -> Output::To<DomainTracer<D>>,
    Input: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<DomainTracer<D>>,
            ParameterStructure: Hash,
        >,
    Output: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<DomainTracer<D>>,
            To<DomainTracer<D>>: Parameterized<
                DomainTracer<D>,
                To<D::Type> = Output,
                To<D::Constant> = Output::To<D::Constant>,
            >,
        >,
>(
    domain: &'domain D,
    function: F,
    input_types: Input,
) -> Result<CompiledFunction<'domain, D, Input, Output>, D::Error> {
    compile_with_options(domain, function, input_types, CompilationOptions::default())
}
