use std::hash::Hash;
use std::rc::Rc;
use std::sync::Arc;

use crate::captures::{CaptureConstant, CapturingContext, ClosedProgram};
use crate::compilation::functions::{
    CompilationTracer, ExecutableFunction, FlatCompilationProgram, ReferenceExecution, StagedFunction,
};
use crate::contexts::{Domain, StagingContext};
use crate::macros::check_builders;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, Typed, validate_reference_boundary};
use crate::tracing::{DomainTracingContext, Tracer};

// TODO(eaplatanios): Review from here onwards.

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four phases:
///
///   1. [`Self::stage`] traces a closure into a [`StagedFunction`] that retains the options for the later phases.
///   2. [`Self::lower`] lowers a staged function into a lowered function.
///   3. [`Self::compile`] compiles a lowered function into a compiled function.
///   4. [`Self::call`] executes a compiled function on structured runtime inputs.
///
/// The domain performs every phase, while staged, lowered, compiled, and executable functions remain passive values.
/// Core owns everything that depends on the structured input and output types of a function. It traces closures,
/// lifts runtime captures into leading program inputs, checks input counts and compiled output types, and reconstructs
/// structured outputs. Backends implement [`Self::lower`], [`Self::compile`], and [`Self::call`] over flat programs,
/// types, and values, including any validation, caching, or runtime conversion that they choose to apply. Staging
/// traces exactly the input types that it receives. Backends apply options that change input types before staging
/// (e.g., in [`Self::dispatch_signature`] for retained dispatch), apply options that change output types during
/// lowering, and reject staged functions that they cannot execute during lowering as well.
pub trait CompilationDomain: Domain<Constant: CaptureConstant> + Clone {
    /// Backend-defined key for one input specialization of a retained Just-In-Time (JIT) dispatcher.
    type DispatchKey: Clone + Eq + Hash;

    /// Backend-specific lowered program representation produced by [`Self::lower`].
    type LoweredProgram;

    /// Backend-specific compiled program representation produced by [`Self::compile`].
    type CompiledProgram;

    /// Backend-specific compilation options type. Meshes, sharding and layout overrides, donation declarations,
    /// compiler flags, etc., are all represented as part of this type.
    type Options;

    /// Backend-specific error type. Staging and call boundary errors flow through it as [`ProgramError`]s,
    /// which is why it requires [`From<ProgramError>`].
    type Error: std::error::Error + From<ProgramError>;

    /// Derives the effective staged input types and their dispatch key from one runtime input signature. Most domains
    /// return the input types unchanged and use them as the key. Backends may instead apply a semantics preserving
    /// dispatch policy such as bounded shape bucketing. The returned key must compare equal exactly when the returned
    /// effective types are interchangeable for tracing, lowering, compilation, and calls.
    ///
    /// The effective types are returned as a shared slice so that key construction on the hot path of retained dispatch
    /// never clones the type vector deeply. Domains whose key embeds the effective types (e.g., exact dispatch) share
    /// one allocation between the key and the returned signature, and the caller clones individual types only on a
    /// specialization miss, when it prepares the staging request.
    fn dispatch_signature(
        &self,
        input_types: Vec<Self::Type>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Self::Type]>), Self::Error>;

    /// Traces the fallible `function` into a staged function. `function` receives the symbolic references of
    /// `captures`, tracers for those captures, and tracers for inputs of `input_types`, and returns its traced outputs.
    /// This is the only staging primitive. It traces at exactly `input_types` and retains `options` in the returned
    /// staged function for lowering and compilation.
    ///
    /// Backends must not override this function (refer to the [`CompilationDomain`] documentation for why).
    #[inline]
    fn stage<F, Input, Output>(
        &self,
        function: F,
        captures: Vec<Self::Value>,
        input_types: Input,
        options: Self::Options,
    ) -> Result<StagedFunction<Self, Input, Output>, Self::Error>
    where
        Self: Sized,
        F: FnOnce(
            Vec<Self::Constant>,
            Vec<CompilationTracer<Self>>,
            Input::To<CompilationTracer<Self>>,
        ) -> Result<Output::To<CompilationTracer<Self>>, Self::Error>,
        Input: Parameterized<Self::Type>,
        Input::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Output: Parameterized<Self::Type>,
        Output::Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<CompilationTracer<Self>>,
        Output::To<CompilationTracer<Self>>: Parameterized<
                CompilationTracer<Self>,
                To<Self::Type> = Output,
                To<Self::Constant> = Output::To<Self::Constant>,
            >,
        DomainTracingContext<Self, Self::Value>:
            StagingContext<Type = Self::Type, Constant = Self::Constant, Operation = Self::Operation>,
    {
        validate_reference_boundary(std::iter::empty(), captures.iter()).map_err(ProgramError::from)?;
        let input_structure = input_types.parameter_structure();
        let input_type_values = input_types.parameters().cloned().collect::<Vec<_>>();
        let context = DomainTracingContext::<Self, Self::Value>::new();
        let capture_table = context.captures().clone();
        let builder = context.builder().clone();
        let capture_references =
            captures.into_iter().map(|capture| context.capture(capture)).collect::<Result<Vec<_>, _>>()?;
        let capture_tracers = capture_references
            .iter()
            .cloned()
            .map(|capture| StagingContext::constant(&context, capture))
            .collect::<Vec<_>>();
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
        let source_program = ClosedProgram::new(program, captures)?.without_unused_captures()?;
        Ok(StagedFunction::new(source_program, input_type_values, output_types, output_structure, options))
    }

    /// Lowers the lifted program of a staged function into the program representation of this backend (i.e.,
    /// [`Self::LoweredProgram`]) and returns it together with its effective flat output types. Callers use
    /// [`StagedFunction::lower`] rather than calling this function directly.
    ///
    /// The lowered program must capture every option, compiler, target, and topology property that affects
    /// compilation, because caching backends derive the identity of compiled programs from the lowered program alone.
    ///
    /// # Parameters
    ///
    ///   - `program`: Source program of the staged function with its runtime captures lifted into leading inputs
    ///     (refer to [`StagedFunction::lifted_program`]).
    ///   - `capture_count`: Number of leading inputs of `program` that are lifted runtime captures.
    ///   - `input_types`: Flat declared public input types of the staged function, excluding captures.
    ///   - `output_types`: Flat output types of the staged function.
    ///   - `options`: Compilation options retained by the staged function.
    fn lower(
        &self,
        program: &FlatCompilationProgram<Self>,
        capture_count: usize,
        input_types: &[Self::Type],
        output_types: &[Self::Type],
        options: &Self::Options,
    ) -> Result<(Self::LoweredProgram, Vec<Self::Type>), Self::Error>;

    /// Compiles a lowered program into an executable program of this backend (i.e., [`Self::CompiledProgram`]) and
    /// returns it together with its effective flat output types. Callers use
    /// [`LoweredFunction::compile`](super::functions::LoweredFunction::compile) rather than calling this function
    /// directly, which also checks that the returned output types refine the lowered ones.
    /// Caching backends usually implement this function via
    /// [`CompilationContext::compile`](super::caching::CompilationContext::compile).
    fn compile(
        &self,
        program: &Self::LoweredProgram,
    ) -> Result<(Arc<Self::CompiledProgram>, Vec<Self::Type>), Self::Error>;

    /// Executes a compiled program on flat runtime `arguments` and returns its flat outputs. Callers use
    /// [`ExecutableFunction::call`] rather than calling this function directly.
    ///
    /// # Parameters
    ///
    ///   - `program`: Compiled program to execute.
    ///   - `input_types`: Flat declared public input types of the compiled function, excluding captures.
    ///   - `arguments`: Runtime captures of the compiled function followed by exactly one public input per entry of
    ///     `input_types`, in the `[captures..., public inputs...]` order of lifted programs.
    fn call(
        &self,
        program: &Self::CompiledProgram,
        input_types: &[Self::Type],
        arguments: Vec<Self::Value>,
    ) -> Result<Vec<Self::Value>, Self::Error>;
}

/// Optional execution capability for compiled functions that bind external mutable references.
///
/// Implementations must keep hidden state results out of the outputs that they return and preserve their reference
/// transaction guarantees even when the public output has no leaves or fails reconstruction.
pub trait StatefulCompilationDomain: CompilationDomain {
    /// Executes a compiled program that binds external mutable references and returns its flat outputs together with
    /// the completion of the whole invocation, without waiting for that completion. The parameters are the same as for
    /// [`CompilationDomain::call`]. Callers use [`ExecutableFunction::call_statefully_async`] rather than calling this
    /// function directly.
    fn call_statefully_async(
        &self,
        program: &Self::CompiledProgram,
        input_types: &[Self::Type],
        arguments: Vec<Self::Value>,
    ) -> ReferenceExecution<Vec<Self::Value>, Self::Error>;
}

/// Optional capability for caching backend-compiled programs in a
/// [`CompilationContext`](super::caching::CompilationContext).
///
/// Cache identity is derived from the lowered program alone. [`CompilationDomain::lower`] folds every option, compiler,
/// target, and topology property that affects compilation into [`CompilationDomain::LoweredProgram`], so key equality
/// always means that compiled programs are interchangeable.
pub trait CompilationCacheDomain: CompilationDomain {
    /// Exact structural cache key for interchangeable compiled programs.
    type CacheKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Constructs the exact cache key for `program`.
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error>;

    /// Returns stable canonical bytes for persistent cache identity, or [`None`] when this domain does not support
    /// persistent executable caching.
    ///
    /// These bytes must remain stable across processes that may share the cache and must cover the complete
    /// computation, options, compiler and backend versions, and target topology. Core hashes them only to obtain safe
    /// filenames and does not add any missing semantic state.
    #[inline]
    fn persistent_cache_key(&self, _key: &Self::CacheKey) -> Option<Vec<u8>> {
        None
    }

    /// Serializes a compiled program for persistent caching and artifact exchange. `Ok(None)` means unsupported.
    #[inline]
    fn serialize_program(&self, _program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(None)
    }

    /// Deserializes a persistent cache or artifact exchange payload. `Ok(None)` means unsupported or incompatible.
    #[inline]
    fn deserialize_program(&self, _bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        Ok(None)
    }
}

/// Optional capability for inspecting a compiled program without recompiling it.
///
/// Analysis is deliberately separate from [`CompilationDomain`] because not every backend or runtime plugin can
/// expose compiler cost and memory information. Implementations should cache immutable backend results when querying
/// them is expensive. Calling this function must never trigger compilation.
pub trait AnalyzableCompilationDomain: CompilationDomain {
    /// Backend-owned typed analysis report.
    type Analysis;

    /// Analyzes the compiled program carried by an [`ExecutableFunction`] without recompiling it.
    fn analyze<Input: Parameterized<Self::Type>, Output: Parameterized<Self::Type>>(
        &self,
        executable_function: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error>
    where
        Self: Sized;
}
