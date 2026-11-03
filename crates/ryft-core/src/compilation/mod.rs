//! Contains backend-neutral machinery for _Just-In-Time (JIT) compilation_.
//!
//! This module supplies the common JIT lifecycle shared by compiler backends. Core stages typed Ryft programs, manages
//! runtime captures and structured signatures, coordinates specialization and executable caches, and validates calls.
//! A backend supplies target-specific options, lowering, executable compilation, and execution through
//! [`CompilationDomain`]s. Backends opt into exact cache identity and serialization through
//! [`CompilationCacheDomain`]s. Backend representations therefore remain outside `ryft-core`
//! while every backend follows the same lifecycle.
//!
//! # Retained JIT Lifecycle
//!
//! A warm call follows the cache-hit edge directly to execution. A cold call stages and lowers once in the producing
//! thread, delegates backend artifact reuse to the shared compilation context, publishes the executable, and then
//! executes it. Refer to [`CompiledFunctionDispatcher`] for a rendered diagram of this dispatch loop.
//!
//! Two transitions exist outside this dispatch loop. A staged function can be embedded as a nested call boundary in
//! an enclosing trace instead of being lowered directly, and a compiled function can shed its staged and lowered
//! transform metadata into the runtime-only executable handle that the cache retains. The secondary transitions are
//! included in the lifecycle diagram on [`CompiledFunctionDispatcher`].
//!
//! # Terminology
//!
//! A bare [`Program`](crate::Program) is the transformable IR value (i.e., atoms, instructions, regions, and boundary
//! types). [`StagedProgram`] owns a flat source program together with its runtime capture table, public types, and
//! compilation options. A _function_ adds the Rust input and output container structures needed for typed calls and
//! carries the resulting backend artifacts through the lifecycle. Every lifecycle handle below is therefore a function,
//! and [`CompiledFunctionDispatcher`] is a retained dispatcher over a Rust closure that produces one program per
//! specialization. In this module, every `*Program` name denotes a flat artifact without container structures.
//! Besides [`StagedProgram`], which core produces by staging, these are the backend-owned
//! [`CompilationDomain::LoweredProgram`] and [`CompilationDomain::CompiledProgram`] that lowering and compilation
//! produce. In between, [`StagedProgram::lifted_program`] provides the source program with its captures lifted into
//! leading inputs as an ordinary [`FlatProgram`](crate::FlatProgram), which backends lower and nested calls embed. The
//! lifecycle handles carry these artifacts from one phase to the next.
//!
//! # Entry Points
//!
//!   - Use [`stage`] for ordinary capture-free staging from runtime input values, recovering the domain from the first
//!     input leaf and deriving the abstract input types. Use [`Compile::stage`] on an explicit domain for abstract
//!     input types or empty inputs, and [`StagedFunction::stage`] when fallibility, runtime captures, or symbolic
//!     capture references are needed. Then continue with [`StagedFunction::lower`] and [`LoweredFunction::compile`].
//!     [`StagedProgram::stage`] exposes the underlying flat staging primitive and returns an owned [`StagedProgram`]
//!     for direct backend use.
//!   - Use [`jit`], [`jit_with_options`], or [`jit_with_options_and_capacity`] for a retained JIT dispatcher that
//!     infers its domain from the first nonempty runtime call. Use [`Compile::jit`] and its variants on an explicit
//!     domain to bind it when the dispatcher is constructed. The closure is fallible during tracing and returns its
//!     outputs wrapped in [`Ok`] when it cannot fail. A [`CompiledFunctionDispatcher`] specializes on explicit static
//!     host parameters, the dynamic input's parameter structure, and the domain's runtime-derived dispatch key. Its
//!     first call for a specialization stages, lowers, and requests compilation; warm calls dispatch directly to the
//!     retained executable.
//!   - Execute an [`ExecutableFunction`] through [`ExecutableFunction::execute`]. Domains with transactional
//!     external state implement [`StatefulCompilationDomain`], and their executable functions are invoked through
//!     [`ExecutableFunction::execute_statefully`] or [`ExecutableFunction::execute_statefully_and_wait`]. Retained
//!     dispatchers expose corresponding invocation functions (e.g.,
//!     [`CompiledFunctionDispatcher::execute_statefully_and_wait`]).
//!
//! Backend-facing crates normally wrap these generic entry points with value- and option-specific APIs.
//! New backend-neutral code can call them directly.
//!
//! # Transactional External State
//!
//! [`StatefulCompilationDomain`] is an opt-in execution capability, not a relaxation of the ordinary pure call
//! contract. A stateful backend snapshots each external reference, chains that reference's cumulative predecessor
//! dependency into the new invocation's joined completion instead of blocking on it, publishes read leases and
//! submitted mutations atomically after successful submission, and installs hidden final state without exposing it as
//! a public function result. Calls involving the same mutated reference allocation therefore serialize by generation
//! without any host wait, read-only calls may overlap, and calls involving independent allocations do not acquire a
//! global lock.
//!
//! [`StatefulExecution`] owns whole-invocation completion. Its blocking `r#await` observes execution failure before
//! returning the reconstructed public output, while dropping it never cancels submitted work. A failure before
//! submission leaves every reference unchanged. After submission, a failure that makes the final state ambiguous
//! poisons the complete affected mutation group; later reference access reports that stored cause. Read-only
//! references are leased rather than donated and remain usable after a failed reader. Concrete backends may support
//! only a subset of shapes, memory placements, shardings, and process topologies and must reject the rest before
//! launch.
//!
//! # Lifecycle Handles
//!
//!   1. [`StagedFunction`] pairs a flat [`StagedProgram`] with the exact input and output container structures. The
//!      artifact owns the source program, captures, public types, and options. It contains no backend lowering or
//!      executable, and its [`StagedFunction::call`] method stages a nested call into an active context rather than
//!      executing at runtime.
//!   2. [`LoweredFunction`] owns the backend's [`CompilationDomain::LoweredProgram`], the source handle, and the
//!      options used to lower it. Compilation computes the domain's exact key and consults its [`CompilationContext`].
//!   3. [`CompiledFunction`] combines the executable with the staged and lowered metadata required for inspection and
//!      transformations. Pure runtime execution uses [`ExecutableFunction::execute`]; transactional external state uses
//!      [`ExecutableFunction::execute_statefully`].
//!   4. [`ExecutableFunction`] retains only the executable, captures, signatures, and output structure. Use
//!      [`CompiledFunction::into_executable`] when transform metadata is no longer needed. This runtime-only
//!      handle gains [`Send`] and [`Sync`] structurally whenever its backend fields do.
//!
//! # Captures and Nested Calls
//!
//! Runtime values closed over by a traced closure are not embedded as literal data.
//! A [`ClosedProgram`](crate::ClosedProgram) keeps them in a side table while the source IR stores typed
//! [`CaptureReference`](crate::CaptureReference) indices. Before lowering, captures are lifted as leading flat inputs,
//! and execution supplies arguments in the same `[captures..., public inputs...]` order. This keeps IR compact,
//! preserves device-resident buffers, and lets compilation depend on capture types rather than data.
//!
//! [`CapturingContext`](crate::CapturingContext) lets ordinary and transform contexts register captures through their
//! parent. To embed a staged function in a larger trace, an operation family implements [`StagedCallOperation`], and
//! that operation then owns how the nested call behaves under lowering, batching, differentiation, partial evaluation,
//! and interpretation.
//!
//! # Specialization and Caching
//!
//! [`CompiledFunctionDispatcher`] keeps one bounded LRU cache of compiled specializations, built on the general
//! [`SpecializationCache`](crate::SpecializationCache) primitive and keyed by the static parameters, the input
//! parameter structure, and the domain's dispatch key. A hit dispatches directly to the retained executable, while a
//! miss stages the Rust closure, lowers the staged source, requests compilation, and inserts the resulting executable.
//! Staging and lowering are cheap relative to backend compilation, so there are no intermediate staging or lowering
//! caches. After an eviction or failure, the miss path simply reruns those phases, and the shared
//! [`CompilationContext`] below still deduplicates the expensive backend compilation.
//!
//! That argument does not carry over to eager (i.e., operation-by-operation) dispatch, where a compiled program is
//! reused for every application of the same operation and the relevant comparison is with _executing_ one small
//! operation rather than with compiling it. Backends that execute eager operations through compiled programs therefore
//! key applications of operations directly to compiled programs in front of staging and lowering, using the operation's
//! identity (refer to the `#[ryft(dispatch(identity))]` attribute of `#[derive(Operation)]`) together with the input
//! types, like JAX's `(primitive, params)` dispatch cache. For example, the XLA backend keeps such a cache in each of
//! its sessions.
//!
//! Cloned [`CompiledFunctionDispatcher`] handles share one cache, and the dispatcher is usable from multiple threads
//! whenever its domain, closure, static parameters, and artifact types are thread-safe. Recursive dispatch on one
//! thread of the specialization currently being produced is rejected with an error, while concurrent cold misses on
//! different threads deliberately duplicate the cheap frontend work with idempotent inserts and leave the coordination
//! of backend compilation to the shared context.
//!
//! [`CompilationContext`] is the backend-artifact cache shared by every compilation using the same domain handle, and
//! it resolves the domain's [`CompilationCacheDomain::CacheKey`] through memory, waiting on in-flight compilations,
//! optional persistent storage, optional distributed exchange, and finally backend compilation. Backends own their
//! contexts, so users configure caching and exchange when they construct a backend handle. Refer to
//! [`CompilationContext`] for the resolution diagram and a configuration example.
//!
//! Misses for the same key are deduplicated, while different keys may compile concurrently. [`DiskCache`] provides
//! optional checksummed persistent storage, and [`CompilationArtifactExchange`] optionally shares serialized artifacts
//! among processes according to [`CompilationArtifactExchangePolicy`]. [`DispatcherStatistics`] reports the frontend
//! activity of retained dispatchers, while [`CompilationCacheStatistics`] and structured [`CompilationEvent`]s report
//! the activity of the shared context across cache lookups, artifact exchange, and backend compilation.
//!
//! # Extending Compilation
//!
//! Implement [`CompilationDomain`] after defining the backend's [`Domain`](crate::Domain). The required contract
//! supplies backend-owned lowered and compiled program types, options, an error, lowering, compilation, output
//! signatures, and flat execution. Lowering must fold every compile-relevant option, compiler, target, and topology
//! property into the lowered artifact, which keeps that artifact self-describing for caching and persistence.
//!
//! Implement [`CompilationCacheDomain`] to opt into the shared [`CompilationContext`]: it derives cache identity
//! from the lowered program alone, and key equality must mean that compiled artifacts are interchangeable.
//! It also supplies the uncached backend compilation, and its [`CompilationDomain::compile`] delegates to
//! [`CompilationContext::compile`], so that every compilation resolves through the cache and every reused program
//! is validated against the requesting lowered program. Optional hooks add that validation, persistent keying,
//! and executable codecs. Implement [`AnalyzableCompilationDomain`] to expose cost or memory
//! analysis without recompilation, and implement [`StagedCallOperation`] on the operation
//! family when staged functions must compose inside other traces.
//!
//! Core owns flat staging through the inherent [`StagedProgram::stage`] constructor. It traces exactly the input types
//! that it receives and invokes the closure exactly once on success. Backends apply options that change input types
//! before staging (e.g., in [`CompilationDomain::dispatch_signature`]) and options that change output types during
//! lowering. [`CompilationDomain`] contains only backend hooks, with no generic functions, over flat programs, types,
//! and values. Core owns everything that depends on structured signatures. [`StagedFunction::stage`] preserves the
//! container structures. [`StagedFunction::lower`], [`LoweredFunction::compile`], and [`ExecutableFunction::execute`]
//! carry the typed metadata through lowering, check input counts and compiled output types, call the backend, and
//! reconstruct structured outputs. [`StagedProgram::lifted_program`] supplies the shared flat capture-lifting step.

pub mod caching;
pub mod domains;
pub mod exchange;
pub mod functions;
pub mod operations;

pub use caching::{
    CompilationCacheStatistics, CompilationCacheTier, CompilationContext, CompilationEvent, CompilationLookupOutcome,
    CompilationMissReason, CompilationSkipReason, CompilationStep, CompilationStepError, CompilationStepOutcome,
    DiskCache,
};
pub use domains::{
    AnalyzableCompilationDomain, CompilationCacheDomain, CompilationDomain, Compile, StatefulCompilationDomain, jit,
    jit_with_options, jit_with_options_and_capacity, stage,
};
pub use exchange::{CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError};
pub use functions::{
    CompilationTracer, CompiledFunction, CompiledFunctionDispatcher, DispatcherStatistics, ExecutableFunction,
    LoweredFunction, StagedFunction, StagedProgram, StatefulExecution,
};
pub use operations::StagedCallOperation;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::{Display, Formatter};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::captures::CaptureReference;
    use crate::compilation::caching::CompilationContext;
    use crate::compilation::domains::{CompilationCacheDomain, CompilationDomain, Compile, StatefulCompilationDomain};
    use crate::compilation::functions::{CompiledFunction, StagedProgram, StatefulExecution};
    use crate::contexts::Domain;
    use crate::macros::check_count;
    use crate::operations::NegOperation;
    use crate::parameters::Parameter;
    use crate::programs::{
        FlatProgram, Operation, ProgramError, Type, Typed, TypedProgram, Value, ValueDirectDispatch,
    };

    /// Lowered program of [`TestDomain`].
    #[derive(Clone)]
    pub(super) struct TestLoweredProgram {
        /// Lifted flat program, whose captures are leading inputs.
        pub(super) program: FlatProgram<TestDomain>,

        /// Number of leading inputs that hold lifted captures.
        pub(super) capture_count: usize,

        /// Lowered output types.
        output_types: Vec<ArrayType>,

        /// Options of the staged program, which compilation consults.
        options: TestOptions,
    }

    /// Compiled program of [`TestDomain`].
    pub(super) struct TestCompiledProgram {
        /// Lifted flat program that execution interprets.
        pub(super) program: FlatProgram<TestDomain>,

        /// Compiled output types.
        pub(super) output_types: Vec<ArrayType>,
    }

    impl TypedProgram for TestLoweredProgram {
        type Type = ArrayType;

        fn input_count(&self) -> usize {
            self.program.input_count()
        }

        fn input_types(&self) -> Cow<'_, [ArrayType]> {
            self.program.input_types()
        }

        fn output_count(&self) -> usize {
            self.output_types.len()
        }

        fn output_types(&self) -> Cow<'_, [ArrayType]> {
            Cow::Borrowed(&self.output_types)
        }
    }

    impl TypedProgram for TestCompiledProgram {
        type Type = ArrayType;

        fn input_count(&self) -> usize {
            self.program.input_count()
        }

        fn input_types(&self) -> Cow<'_, [ArrayType]> {
            self.program.input_types()
        }

        fn output_count(&self) -> usize {
            self.output_types.len()
        }

        fn output_types(&self) -> Cow<'_, [ArrayType]> {
            Cow::Borrowed(&self.output_types)
        }
    }

    /// Options of [`TestDomain`], which inject staged input types, failures, and output types into the lifecycle.
    #[derive(Clone, Debug, Default)]
    pub(super) struct TestOptions {
        /// Type that replaces every input type in dispatch signatures, if any.
        pub(super) staged_input_type: Option<ArrayType>,

        /// Whether lowering fails.
        pub(super) fail_lowering: bool,

        /// Whether compilation fails.
        pub(super) fail_compilation: bool,

        /// Output types that compilation reports instead of the lowered output types, if any.
        pub(super) compiled_output_types: Option<Vec<ArrayType>>,
    }

    /// Compilation domain over [`Array`] values. It compiles through a [`CompilationContext`], so equivalent lowered
    /// programs compile once, and it executes programs by interpreting every instruction as a negation after checking
    /// that the runtime inputs refine the declared input types. Clones share their cache and compilation counter.
    #[derive(Clone)]
    pub(super) struct TestDomain {
        /// Compilation cache shared by every clone of this domain.
        pub(super) cache: Arc<CompilationContext<Self>>,

        /// Number of backend compilations, shared by every clone of this domain.
        compilations: Arc<AtomicUsize>,
    }

    impl TestDomain {
        /// Creates a domain with an empty compilation cache.
        pub(super) fn new() -> Self {
            Self { cache: Arc::new(CompilationContext::new()), compilations: Arc::new(AtomicUsize::new(0)) }
        }

        /// Returns the number of backend compilations performed so far, excluding compilation cache hits.
        pub(super) fn compilation_count(&self) -> usize {
            self.compilations.load(Ordering::Relaxed)
        }
    }

    impl Domain for TestDomain {
        type Type = ArrayType;
        type Value = Array;
        type Constant = CaptureReference<ArrayType>;
        type Operation = ArrayOperation<Array>;
    }

    impl CompilationDomain for TestDomain {
        type DispatchKey = Arc<[ArrayType]>;
        type LoweredProgram = TestLoweredProgram;
        type CompiledProgram = TestCompiledProgram;
        type Options = TestOptions;
        type Error = ProgramError;

        fn dispatch_signature(
            &self,
            input_types: Vec<ArrayType>,
            options: &Self::Options,
        ) -> Result<(Self::DispatchKey, Arc<[ArrayType]>), Self::Error> {
            let input_types: Arc<[ArrayType]> = match &options.staged_input_type {
                Some(input_type) => input_types.iter().map(|_| input_type.clone()).collect(),
                None => input_types.into(),
            };
            Ok((input_types.clone(), input_types))
        }

        fn lower(&self, staged: &StagedProgram<Self>) -> Result<Self::LoweredProgram, Self::Error> {
            let program = staged.lifted_program()?;
            let capture_count = staged.source_program().captures().len();
            let output_types = staged.output_types();
            let options = staged.options();
            if options.fail_lowering {
                return Err(ProgramError::InvalidArgument { message: "expected lowering failure".into() });
            }
            Ok(TestLoweredProgram {
                program: program.as_ref().clone(),
                capture_count,
                output_types: output_types.to_vec(),
                options: options.clone(),
            })
        }

        fn compile(&self, program: &Self::LoweredProgram) -> Result<Arc<Self::CompiledProgram>, Self::Error> {
            self.cache.compile(self, program)
        }

        fn execute(&self, program: &Self::CompiledProgram, inputs: Vec<Array>) -> Result<Vec<Array>, Self::Error> {
            let input_types = program.program.input_types();
            check_count!("input", inputs, input_types.len(), ProgramError);
            for (declared, actual) in input_types.iter().zip(inputs.iter().map(Typed::r#type)) {
                if !declared.is_refined_by(actual.as_ref()) {
                    return Err(ProgramError::InvalidArgument {
                        message: format!("runtime input type {actual} does not refine declared type {declared}"),
                    });
                }
            }
            program.program.interpret_with(
                inputs,
                |_, capture| {
                    Err(ProgramError::MalformedProgram(format!(
                        "lifted test program retained capture {}",
                        capture.index(),
                    )))
                },
                |_, inputs| {
                    if inputs.len() != 1 {
                        return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() });
                    }
                    Ok(vec![-inputs[0].clone()])
                },
            )
        }
    }

    impl CompilationCacheDomain for TestDomain {
        type CacheKey = Vec<String>;

        fn compilation_key(&self, program: &TestLoweredProgram) -> Result<Vec<String>, ProgramError> {
            let mut key = vec![
                format!("captures:{}", program.capture_count),
                format!("atoms:{:?}", program.program.atoms()),
                format!("inputs:{:?}", program.program.input_ids()),
                format!("outputs:{:?}", program.program.output_ids()),
                format!("options:{:?}", program.options),
            ];
            key.extend(program.program.instructions().iter().map(|instruction| {
                format!("{}:{:?}:{:?}", instruction.operation().name(), instruction.inputs(), instruction.outputs(),)
            }));
            Ok(key)
        }

        fn compile_uncached(&self, program: &TestLoweredProgram) -> Result<TestCompiledProgram, ProgramError> {
            self.compilations.fetch_add(1, Ordering::Relaxed);
            if program.options.fail_compilation {
                return Err(ProgramError::InvalidArgument { message: "expected compilation failure".into() });
            }
            let output_types =
                program.options.compiled_output_types.clone().unwrap_or_else(|| program.output_types.clone());
            Ok(TestCompiledProgram { program: program.program.clone(), output_types })
        }
    }

    /// Compilation domain whose runtime values carry it, which lets the global entry points recover it from their
    /// inputs. It lowers and compiles to the lifted program itself, counts compilations, and executes programs by
    /// interpreting every instruction as a negation, which is the only operation that these tests stage.
    #[derive(Clone, Debug)]
    pub(super) struct CarriedDomain {
        /// Number of backend compilations, shared by every clone of this domain.
        compilations: Arc<AtomicUsize>,
    }

    impl CarriedDomain {
        /// Creates a domain with no backend compilations.
        pub(super) fn new() -> Self {
            Self { compilations: Arc::new(AtomicUsize::new(0)) }
        }

        /// Returns the number of backend compilations performed so far.
        pub(super) fn compilation_count(&self) -> usize {
            self.compilations.load(Ordering::Relaxed)
        }

        /// Returns a rank-zero array with `value` that carries this domain.
        pub(super) fn scalar(&self, value: f64) -> CarriedArray {
            CarriedArray { array: Array::scalar(value).unwrap(), domain: self.clone() }
        }
    }

    // Clones of one domain share their compilation counter, which identifies the domain instance.
    impl PartialEq for CarriedDomain {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.compilations, &other.compilations)
        }
    }

    impl Domain for CarriedDomain {
        type Type = ArrayType;
        type Value = CarriedArray;
        type Constant = CaptureReference<ArrayType>;
        type Operation = ArrayOperation<Array>;
    }

    impl CompilationDomain for CarriedDomain {
        type DispatchKey = Arc<[ArrayType]>;
        type LoweredProgram = FlatProgram<Self>;
        type CompiledProgram = FlatProgram<Self>;
        type Options = CarriedOptions;
        type Error = ProgramError;

        fn dispatch_signature(
            &self,
            input_types: Vec<ArrayType>,
            _options: &CarriedOptions,
        ) -> Result<(Self::DispatchKey, Arc<[ArrayType]>), ProgramError> {
            let input_types: Arc<[ArrayType]> = input_types.into();
            Ok((input_types.clone(), input_types))
        }

        fn lower(&self, staged: &StagedProgram<Self>) -> Result<Self::LoweredProgram, ProgramError> {
            Ok(staged.lifted_program()?.as_ref().clone())
        }

        fn compile(&self, program: &Self::LoweredProgram) -> Result<Arc<Self::CompiledProgram>, ProgramError> {
            self.compilations.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(program.clone()))
        }

        fn execute(
            &self,
            program: &Self::CompiledProgram,
            inputs: Vec<CarriedArray>,
        ) -> Result<Vec<CarriedArray>, ProgramError> {
            let outputs = program.interpret_with(
                inputs.into_iter().map(|input| input.array).collect(),
                |_, capture| Err(ProgramError::MalformedProgram(format!("unexpected capture {}", capture.index()))),
                |_, inputs: &[Array]| Ok(vec![-inputs[0].clone()]),
            )?;
            Ok(outputs.into_iter().map(|array| CarriedArray { array, domain: self.clone() }).collect())
        }
    }

    impl StatefulCompilationDomain for CarriedDomain {
        fn execute_statefully(
            &self,
            program: &Self::CompiledProgram,
            inputs: Vec<CarriedArray>,
        ) -> StatefulExecution<Vec<CarriedArray>, ProgramError> {
            StatefulExecution::ready(self.execute(program, inputs))
        }
    }

    /// Opaque [`CarriedDomain`] options that tests compare to verify that entry points retain them.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub(super) struct CarriedOptions {
        /// Arbitrary marker that distinguishes explicit options from the default ones.
        pub(super) marker: usize,
    }

    /// [`Array`] that carries the [`CarriedDomain`] in which it was produced.
    #[derive(Clone, Debug, PartialEq)]
    pub(super) struct CarriedArray {
        /// Array that holds the value.
        array: Array,

        /// Domain recovered by [`Value::domain`].
        domain: CarriedDomain,
    }

    impl Display for CarriedArray {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.array, formatter)
        }
    }

    impl Parameter for CarriedArray {}

    impl Typed for CarriedArray {
        type Type = ArrayType;

        fn r#type(&self) -> Cow<'_, ArrayType> {
            self.array.r#type()
        }
    }

    impl Value for CarriedArray {
        type Dispatch = ValueDirectDispatch;
        type Domain = CarriedDomain;

        fn domain(&self) -> CarriedDomain {
            self.domain.clone()
        }
    }

    /// Compiles a scalar function that negates its input if `negate` is set, so that functions compiled by different
    /// calls share one closure type but compute different results.
    pub(super) fn compile_from_one_call_site(
        domain: &TestDomain,
        negate: bool,
    ) -> CompiledFunction<TestDomain, ArrayType, ArrayType> {
        let staged = domain
            .stage(
                |input| if negate { input.unary(NegOperation::new()) } else { input },
                ArrayType::scalar(DataType::F64),
                TestOptions::default(),
            )
            .unwrap();
        staged.lower(&domain).unwrap().compile(&domain).unwrap()
    }
}
