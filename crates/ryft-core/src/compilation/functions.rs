use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::axes::{MeshContext, NamedAxes};
use crate::captures::{CapturingContext, ClosedProgram, ImportingContext};
use crate::compilation::domains::{CompilationDomain, DispatchInputs, StatefulCompilationDomain};
use crate::compilation::operations::StagedCallOperation;
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{
    AttachedRegions, FlatProgram, ProgramError, ReferenceCompletion, Type, TypeError, TypeRefinements, Typed,
    TypedProgram, Value, validate_reference_boundary,
};
use crate::specialization::{SpecializationCache, SpecializationCacheEntry};
use crate::tracing::{DomainTracingContext, Tracer, TracingScope};

/// Tracer used while staging a compiled function in the operation universe of `C`.
pub type CompilationTracer<C> = Tracer<DomainTracingContext<C, <C as Domain>::Value>>;

/// Owned flat staging artifact constructed by [`Self::stage`] and borrowed by [`CompilationDomain::lower`].
/// This artifact retains the source program, runtime captures, flat public signature, and compilation options.
/// [`StagedFunction`] pairs it with the Rust input and output container structures used by typed callers. Captures
/// remain symbolic constants in the source program and become leading inputs in [`Self::lifted_program`]
pub struct StagedProgram<C: CompilationDomain> {
    /// Flat source program and the runtime values referenced by its captured constants.
    source_program: ClosedProgram<C::Value, C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// Flat declared public input types, excluding captures.
    input_types: Vec<C::Type>,

    /// Flat source-program output types before rewrites during lowering.
    output_types: Vec<C::Type>,

    /// Compilation options retained for lowering and compilation.
    options: C::Options,

    /// Memoized source program with runtime captures lifted into leading flat inputs.
    lifted_program: OnceLock<Arc<FlatProgram<C>>>,
}

impl<C: CompilationDomain> StagedProgram<C> {
    /// Traces the fallible flat `function` into an owned [`StagedProgram`]. `function` receives the symbolic references
    /// of `captures`, tracers for those captures, and one tracer per entry of `input_types`, and returns flat output
    /// tracers. This is the only staging primitive. It traces at exactly `input_types` and retains `options` for
    /// lowering and compilation. On success, this constructor invokes `function` exactly once. Staging depends only
    /// on the types of the domain `C`, not on any domain instance, which is first needed for lowering. Typed callers
    /// typically use [`StagedFunction::stage`] instead.
    ///
    /// # Parameters
    ///
    ///   - `function`: Fallible closure to trace with symbolic captures and flat inputs.
    ///   - `captures`: Runtime values to register in the source program's capture table.
    ///   - `input_types`: Flat abstract public input signature.
    ///   - `options`: Options retained for lowering and compilation.
    ///
    /// # Errors
    ///
    /// Returns errors from capture validation, the closure, or program construction. Foreign output tracers and handles
    /// that escape the trace are rejected during finalization.
    #[inline]
    pub fn stage<F>(
        function: F,
        captures: Vec<C::Value>,
        input_types: Vec<C::Type>,
        options: C::Options,
    ) -> Result<Self, C::Error>
    where
        F: FnOnce(
            Vec<C::Constant>,
            Vec<CompilationTracer<C>>,
            Vec<CompilationTracer<C>>,
        ) -> Result<Vec<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        Self::stage_in_scope(function, captures, input_types, options, TracingScope::default())
    }

    /// Traces like [`Self::stage`] in `scope`, whose named axes and default mesh the traced function resolves as it
    /// would if it were inlined into a context that binds them (e.g., a named batching level or the manual region of
    /// a `shard_map`). Use it to stage a function that is only ever called inside such a scope. A staged program that
    /// reads a named axis cannot be lowered on its own, because nothing outside that scope binds the axis, so it is
    /// meaningful only as the callee of a call in that scope, whose enclosing transforms resolve the axis.
    ///
    /// # Parameters
    ///
    ///   - `function`: Fallible closure to trace with symbolic captures and flat inputs.
    ///   - `captures`: Runtime values to register in the source program's capture table.
    ///   - `input_types`: Flat abstract public input signature.
    ///   - `options`: Options retained for lowering and compilation.
    ///   - `scope`: [`TracingScope`] to trace in.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::stage`].
    pub fn stage_in_scope<F>(
        function: F,
        captures: Vec<C::Value>,
        input_types: Vec<C::Type>,
        options: C::Options,
        scope: TracingScope,
    ) -> Result<Self, C::Error>
    where
        F: FnOnce(
            Vec<C::Constant>,
            Vec<CompilationTracer<C>>,
            Vec<CompilationTracer<C>>,
        ) -> Result<Vec<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        // Captured references must name distinct runtime allocations. Validate them before replacing concrete values
        // with symbolic capture references, while their runtime identities are still available. Public inputs are
        // abstract types here, so only captures can be checked at this boundary.
        validate_reference_boundary(std::iter::empty(), captures.iter()).map_err(ProgramError::from)?;

        // A fresh context isolates this trace. Runtime captures stay in its side table, while the program stores typed
        // references to their slots. The closure receives both those constant payloads and their tracers.
        let context = DomainTracingContext::<C, C::Value>::new().with_scope(scope);
        let capture_table = context.captures().clone();
        let builder = context.builder().clone();
        let capture_constants =
            captures.into_iter().map(|capture| context.capture(capture)).collect::<Result<Vec<_>, _>>()?;
        let capture_tracers = capture_constants
            .iter()
            .cloned()
            .map(|capture| StagingContext::constant(&context, capture))
            .collect::<Vec<_>>();

        // Create one symbolic input per declared public type. Invoking the closure once records the operations it
        // binds through these inputs and the captured tracers, independently of any caller's container structure.
        let inputs = input_types.iter().cloned().map(|input_type| context.input(input_type)).collect();
        let outputs = function(capture_constants, capture_tracers, inputs)?;

        // Atom identifiers are local to a builder. Reject foreign output tracers before reading their identifiers so
        // they cannot accidentally name unrelated atoms in this trace.
        check_builders!(&builder, [outputs.iter().map(|output| output.builder())])?;

        // Convenience operators can record errors in the builder and return poisoned tracers instead of returning an
        // error from the closure. Report the original error even if the closure discarded the affected values.
        if let Some(error) = builder.borrow_mut().error.take() {
            return Err(error.into());
        }

        // Extract the output metadata before releasing the tracers. Atom identifiers finalize the program, while the
        // flat types describe its output signature for lowering and later call validation.
        let output_ids = outputs.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        let output_types = outputs.iter().map(|output| output.r#type().into_owned()).collect::<Vec<_>>();

        // The context and output tracers own `Rc` clones of the capture table and builder. Drop them explicitly because
        // their automatic drops would run at the end of this scope, after the `Rc::try_unwrap` calls below. Those calls
        // require unique ownership, and any remaining clones indicate a handle that escaped the trace.
        drop(outputs);
        drop(context);
        let captures = Rc::try_unwrap(capture_table).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();

        // Remove dead computation before pruning captures so slots referenced only by dead code can be released.
        // Observable effects remain in the program even when no output consumes their results.
        let program = builder
            .build::<Vec<C::Constant>, Vec<C::Constant>>(
                output_ids,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_types.len()],
            )?
            .into_simplified()?;

        // Validate every capture reference against its runtime slot and type, then prune unused captures and renumber
        // the surviving references together with their table entries to keep the closed program consistent.
        let source_program = ClosedProgram::new(program, captures)?.without_unused_captures()?;
        Ok(Self { source_program, input_types, output_types, options, lifted_program: OnceLock::new() })
    }

    /// Returns the flat source program and its runtime captures.
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<C::Value, C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        &self.source_program
    }

    /// Returns the flat declared public input types, excluding captures.
    #[inline]
    pub fn input_types(&self) -> &[C::Type] {
        &self.input_types
    }

    /// Returns the flat source-program output types before rewrites during lowering.
    #[inline]
    pub fn output_types(&self) -> &[C::Type] {
        &self.output_types
    }

    /// Returns the compilation options retained by this artifact.
    #[inline]
    pub fn options(&self) -> &C::Options {
        &self.options
    }

    /// Returns the source program with captures lifted into leading inputs, in `[captures..., public inputs...]`
    /// order. Backends lower this program and use the capture count of [`Self::source_program`] to distinguish the
    /// lifted captures from public inputs. The derived program is cached, so repeated calls return the same [`Arc`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if rebuilding the source program with explicit capture inputs fails.
    pub fn lifted_program(&self) -> Result<Arc<FlatProgram<C>>, ProgramError> {
        if let Some(program) = self.lifted_program.get() {
            return Ok(program.clone());
        }
        let program = Arc::new(self.source_program.to_program_with_lifted_captures()?);
        Ok(self.lifted_program.get_or_init(|| program).clone())
    }
}

/// Staged, unlowered form of one compiled function. A staged function is the first callable package in the lifecycle.
/// It pairs an owned [`StagedProgram`] with the exact public input and output container structures that typed calls
/// need. It can be inspected, lowered by its domain, or embedded as a staged-call boundary in an enclosing trace.
/// No backend lowering or executable compilation has happened yet.
pub struct StagedFunction<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Shared immutable staging metadata. Staged-handle clones are therefore constant-time even for large programs.
    state: Arc<StagedFunctionState<C, Input, Output>>,
}

impl<C, Input, Output> StagedFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Traces a fallible typed `function` for the domain `C`, retaining its exact input and output container
    /// structures. The function receives symbolic capture constants, capture tracers, and structured input tracers.
    /// Runtime captures and compilation options are owned by the resulting [`StagedProgram`]. Like
    /// [`StagedProgram::stage`], staging depends only on the types of `C`, not on any domain instance.
    ///
    /// # Parameters
    ///
    ///   - `function`: Fallible closure to trace once with symbolic captures and inputs.
    ///   - `captures`: Runtime values to register in the source program's capture table.
    ///   - `input_types`: Structured abstract public input signature.
    ///   - `options`: Options retained for lowering and compilation.
    ///
    /// # Errors
    ///
    /// Returns errors from tracing or reconstructing the structured inputs.
    #[inline]
    pub fn stage<F>(
        function: F,
        captures: Vec<C::Value>,
        input_types: Input,
        options: C::Options,
    ) -> Result<Self, C::Error>
    where
        Input::Family: ParameterizedFamily<CompilationTracer<C>>,
        Output::Family: ParameterizedFamily<CompilationTracer<C>>,
        F: FnOnce(
            Vec<C::Constant>,
            Vec<CompilationTracer<C>>,
            Input::To<CompilationTracer<C>>,
        ) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        Self::stage_in_scope(function, captures, input_types, options, TracingScope::default())
    }

    /// Traces like [`Self::stage`] in `scope`. Refer to [`StagedProgram::stage_in_scope`] for when to use it and for
    /// its parameters.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::stage`].
    pub fn stage_in_scope<F>(
        function: F,
        captures: Vec<C::Value>,
        input_types: Input,
        options: C::Options,
        scope: TracingScope,
    ) -> Result<Self, C::Error>
    where
        Input::Family: ParameterizedFamily<CompilationTracer<C>>,
        Output::Family: ParameterizedFamily<CompilationTracer<C>>,
        F: FnOnce(
            Vec<C::Constant>,
            Vec<CompilationTracer<C>>,
            Input::To<CompilationTracer<C>>,
        ) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        // Flat arities cannot recover nested empty containers or other structural metadata. Keep the input structure
        // before flattening, and record the output structure while the closure still owns its structured results.
        let input_structure = input_types.parameter_structure();
        let mut output_structure = None;
        let program = StagedProgram::<C>::stage_in_scope(
            |capture_constants, capture_tracers, inputs| {
                let inputs = Input::To::<CompilationTracer<C>>::from_parameters(input_structure.clone(), inputs)
                    .map_err(ProgramError::from)?;
                let outputs = function(capture_constants, capture_tracers, inputs)?;
                output_structure = Some(outputs.parameter_structure());
                Ok(outputs.into_parameters().collect())
            },
            captures,
            input_types.into_parameters().collect(),
            options,
            scope,
        )?;

        // The inherent flat staging constructor invokes the closure exactly once on success, so it must have recorded
        // this structure. Backends cannot override that constructor or bypass the callback.
        let output_structure = output_structure.unwrap();
        Ok(Self { state: Arc::new(StagedFunctionState { program, input_structure, output_structure }) })
    }

    /// Returns the owned flat staging artifact retained by this function.
    #[inline]
    pub fn staged_program(&self) -> &StagedProgram<C> {
        &self.state.program
    }

    /// Returns the flat source program and its runtime captures. Container structures remain on this function and are
    /// available through [`Self::input_structure`] and [`Self::output_structure`].
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<C::Value, C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        self.state.program.source_program()
    }

    /// Returns the flat declared public input types, excluding hidden captures.
    #[inline]
    pub fn input_types(&self) -> &[C::Type] {
        self.state.program.input_types()
    }

    /// Returns the flat source-program output types.
    #[inline]
    pub fn output_types(&self) -> &[C::Type] {
        self.state.program.output_types()
    }

    /// Returns the exact input parameter structure retained by this function.
    #[inline]
    pub fn input_structure(&self) -> &Input::ParameterStructure {
        &self.state.input_structure
    }

    /// Returns the output parameter structure retained by this staged function.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.state.output_structure
    }

    /// Returns the compilation options bound to this staged function.
    #[inline]
    pub fn options(&self) -> &C::Options {
        self.state.program.options()
    }

    /// Returns the source program with its runtime captures lifted into leading flat inputs, in the `[captures...,
    /// public inputs...]` order that compiled executables are invoked with. Backends lower this program rather than the
    /// [`ClosedProgram`] returned by [`Self::source_program`], and use the capture count of the source program to tell
    /// lifted captures apart from public inputs. The source program itself has a flat boundary, while this function
    /// retains its exact container structures separately.
    ///
    /// The program is derived once and then cached on the shared staged state, so repeated calls, including calls on
    /// clones of this function, return the same [`Arc`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if rebuilding the source program with explicit capture inputs fails.
    #[inline]
    pub fn lifted_program(&self) -> Result<Arc<FlatProgram<C>>, ProgramError> {
        self.state.program.lifted_program()
    }

    /// Lowers this function into a [`LoweredFunction`] through `domain` (refer to [`CompilationDomain::lower`]).
    pub fn lower(&self, domain: &C) -> Result<LoweredFunction<C, Input, Output>, C::Error> {
        let lowered_program = domain.lower(self.staged_program())?;
        Ok(LoweredFunction { program: Arc::new(lowered_program), staged: self.clone() })
    }

    /// Calls this function in the active context recovered from `inputs`. For nullary functions use
    /// [`Self::call_in_context`], because no input value exists from which to recover the active context.
    #[inline]
    pub fn call<V, Path>(&self, inputs: Input::To<V>) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        V::Domain:
            Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation> + ImportingContext<C, Path>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let context = inputs
            .parameters()
            .next()
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
            .domain();
        self.call_in_context(&context, inputs)
    }

    /// Calls this function in the provided `context`, which binds one staged-call operation there. The function's
    /// captured runtime values enter `context` through [`ImportingContext`], so the call works under an eager parent,
    /// under a staging parent, and under any stack of transforms over either. An eager parent executes the call
    /// immediately, a staging parent records it, and a transform parent rewrites it through the call operation's
    /// own rules.
    #[inline]
    pub fn call_in_context<V, CallerContext, Path>(
        &self,
        context: &CallerContext,
        inputs: Input::To<V>,
    ) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        CallerContext: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let outputs = self.call_flat_in_context(context, inputs.into_parameters().collect())?;
        Output::To::<V>::from_parameters(self.state.output_structure.clone(), outputs).map_err(Into::into)
    }

    /// Calls this function in the provided `context` with flat `inputs` and returns its flat outputs, binding one
    /// staged-call operation exactly as [`Self::call_in_context`] does. Use it when the caller holds the flat leaves
    /// of the input structure rather than the structured inputs. Explicit context selection supports empty `inputs`,
    /// including nullary and capture-only calls.
    pub fn call_flat_in_context<V, CallerContext, Path>(
        &self,
        context: &CallerContext,
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        CallerContext: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
    {
        let mut flat_inputs = self
            .state
            .program
            .source_program()
            .captures()
            .iter()
            .cloned()
            .map(|capture| context.import(capture))
            .collect::<Result<Vec<_>, _>>()?;
        let capture_count = flat_inputs.len();
        flat_inputs.extend(inputs);
        let callee = self.callee_program()?;
        let operation =
            C::Operation::staged_call_with_types(capture_count, &callee.input_types(), &callee.output_types())?;
        context.bind(operation, AttachedRegions::Callees(&[callee]), flat_inputs.as_slice())
    }

    /// Returns the lifted program of this function (i.e., [`Self::lifted_program`]) for embedding as the callee of a
    /// nested call. Reference discharge cannot publish a reference that a callee allocates itself, because the caller
    /// has no allocation that could receive it, so this function rejects such outputs here with a diagnostic that
    /// names the offending output. Outputs that forward a reference received as an input or capture remain allowed.
    fn callee_program(&self) -> Result<Arc<FlatProgram<C>>, ProgramError> {
        let callee = self.lifted_program()?;
        if self.output_types().iter().any(Type::is_reference) {
            let capture_count = self.source_program().captures().len();
            let analysis = callee.reference_analysis(capture_count)?;
            for (index, root) in analysis.output_roots().iter().enumerate() {
                if let Some(root) = root
                    && analysis.external_source(*root).is_none()
                {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "output {index} of the called function is a reference that the function allocates itself, \
                             which a nested call cannot return; allocate the reference in the caller and pass it to \
                             the function instead",
                        ),
                    });
                }
            }
        }
        Ok(callee)
    }
}

impl<C, Input, Output> Clone for StagedFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

/// Shared immutable state of a [`StagedFunction`].
struct StagedFunctionState<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Owned flat staging artifact, including runtime captures, signatures, and options.
    program: StagedProgram<C>,

    /// Input parameter structure used to reconstruct typed tracer inputs and abstract signatures.
    input_structure: Input::ParameterStructure,

    /// Output parameter structure used to reconstruct structured values.
    output_structure: Output::ParameterStructure,
}

/// Backend lowering of a [`StagedFunction`], ready for executable compilation. It stays a function rather than a
/// program, because the lowered program of the backend is only one of the artifacts it carries, alongside the captures,
/// signatures, and options inherited from its staged source.
pub struct LoweredFunction<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Backend-owned lowered program.
    program: Arc<C::LoweredProgram>,

    /// Staged source and structured metadata retained across compilation.
    staged: StagedFunction<C, Input, Output>,
}

impl<C, Input, Output> LoweredFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Returns the backend-owned lowered program.
    #[inline]
    pub fn lowered_program(&self) -> &C::LoweredProgram {
        self.program.as_ref()
    }

    /// Returns the staged function from which this lowering was produced.
    #[inline]
    pub fn staged(&self) -> &StagedFunction<C, Input, Output> {
        &self.staged
    }

    /// Returns the compilation options associated with this lowering.
    #[inline]
    pub fn options(&self) -> &C::Options {
        self.staged.options()
    }

    /// Returns the effective flat output types of the lowered program, which include lowering-time rewrites.
    #[inline]
    pub fn output_types(&self) -> Cow<'_, [C::Type]> {
        self.program.output_types()
    }

    /// Compiles this function into a [`CompiledFunction`] through `domain` via [`CompilationDomain::compile`].
    ///
    /// # Errors
    ///
    /// Returns the error of `domain`, or a [`ProgramError`] if the compiled output types differ in count from the
    /// lowered output types of this function or do not refine them.
    #[inline]
    pub fn compile(&self, domain: &C) -> Result<CompiledFunction<C, Input, Output>, C::Error> {
        let (program, lowered_captures) = domain.compile(self.lowered_program())?;
        let declared_output_types = self.output_types();
        let output_types = program.output_types();
        check_count!("output", output_types, declared_output_types.len(), ProgramError);
        for (declared, actual) in declared_output_types.iter().zip(output_types.iter()) {
            if !declared.is_refined_by(actual) {
                return Err(ProgramError::InvalidArgument {
                    message: format!("output type {actual} does not refine declared type {declared}"),
                }
                .into());
            }
        }
        Ok(CompiledFunction::from_parts(self.clone(), program, lowered_captures))
    }

    /// Calls this function in the active context recovered from `inputs` by delegating to [`StagedFunction::call`]
    /// on its staged source. The call stages a call boundary that carries the staged source program into the caller's
    /// trace, so the lowered program itself is not involved. For nullary functions use [`Self::call_in_context`],
    /// because no input value exists from which to recover the active context.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`StagedFunction::call`].
    #[inline]
    pub fn call<V, Path>(&self, inputs: Input::To<V>) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        V::Domain:
            Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation> + ImportingContext<C, Path>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        self.staged.call(inputs)
    }

    /// Calls this function in the provided `context` by delegating to [`StagedFunction::call_in_context`] on its
    /// staged source, which stages a call boundary into the caller's trace exactly as [`Self::call`] does. An explicit
    /// context also serves nullary functions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`StagedFunction::call_in_context`].
    #[inline]
    pub fn call_in_context<V, CallerContext, Path>(
        &self,
        context: &CallerContext,
        inputs: Input::To<V>,
    ) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        CallerContext: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        self.staged.call_in_context(context, inputs)
    }
}

impl<C, Input, Output> Clone for LoweredFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), staged: self.staged.clone() }
    }
}

/// Runtime-only handle for one compiled executable. This handle retains only the state required to validate and
/// execute calls, namely the backend executable, its concrete captures, its flat public input types, and its structured
/// output shape. That call-level state is what still makes it a function rather than a program, even though it has shed
/// every transformable artifact. It deliberately does not retain the staged or lowered programs, so it becomes [`Send`]
/// and [`Sync`] automatically whenever those runtime fields are `Send + Sync`. In contrast, [`CompiledFunction`]
/// remains transformable and may retain `Rc`-backed source programs.
pub struct ExecutableFunction<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>> {
    /// Shared immutable runtime state, which makes clones constant-time.
    state: Arc<ExecutableFunctionState<C, Input, Output>>,
}

impl<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>>
    ExecutableFunction<C, Input, Output>
{
    /// Returns a function that invokes `program` instead of this function's current compiled program, while keeping
    /// this function's runtime captures, flat public input types, and output parameter structure. Backends use it to
    /// swap in an equivalent executable, such as one deserialized from a persistent cache or recompiled with profile
    /// guidance.
    ///
    /// This function does not check compatibility, so the caller must establish it first. `program` must accept the
    /// same `[captures..., public inputs...]` inputs as the current program, and its output types must have one entry
    /// per leaf of the retained output structure. A backend that breaks these invariants produces a function whose
    /// calls fail or return values of unexpected types. The existing function and its clones are unaffected, because
    /// this creates new shared state.
    #[inline]
    pub fn with_compiled_program(&self, program: Arc<C::CompiledProgram>) -> Self {
        Self {
            state: Arc::new(ExecutableFunctionState {
                program,
                captures: self.state.captures.clone(),
                input_types: self.state.input_types.clone(),
                output_structure: self.state.output_structure.clone(),
                input: PhantomData,
            }),
        }
    }

    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &C::CompiledProgram {
        &self.state.program
    }

    /// Returns the concrete runtime captures supplied before public inputs on every call.
    #[inline]
    pub fn captures(&self) -> &[C::Value] {
        &self.state.captures
    }

    /// Returns the flat staged public input types, excluding hidden captures.
    #[inline]
    pub fn input_types(&self) -> &[C::Type] {
        &self.state.input_types
    }

    /// Returns the effective flat output types of the compiled program.
    #[inline]
    pub fn output_types(&self) -> Cow<'_, [C::Type]> {
        self.state.program.output_types()
    }

    /// Returns the output parameter structure used to reconstruct structured runtime values.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.state.output_structure
    }

    /// Executes this function on the structured runtime `inputs` through `domain` and returns its structured outputs.
    /// Backend output values may still represent pending execution. Functions that bind external mutable references
    /// must use [`Self::execute_statefully`] or [`Self::execute_statefully_and_wait`].
    #[inline]
    pub fn execute(&self, domain: &C, inputs: Input::To<C::Value>) -> Result<Output::To<C::Value>, C::Error>
    where
        Input::Family: ParameterizedFamily<C::Value>,
        Input::To<C::Value>: Parameterized<C::Value>,
        Output::Family: ParameterizedFamily<C::Value>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let inputs = inputs.into_parameters().collect::<Vec<_>>();
        self.validate_input_count(inputs.len())?;
        let outputs = domain.execute(self.compiled_program(), self.inputs_with_captures(inputs))?;
        Ok(Output::To::<C::Value>::from_parameters(self.state.output_structure.clone(), outputs)
            .map_err(ProgramError::from)?)
    }

    /// Submits this function on structured runtime `inputs` through the stateful `domain` and returns a
    /// [`StatefulExecution`] tracking completion of the whole invocation. Submission happens before this function
    /// returns. Await the result to observe execution failures. Dropping it does not cancel submitted work. Reference
    /// state publication follows [`StatefulCompilationDomain::execute_statefully`].
    pub fn execute_statefully(
        &self,
        domain: &C,
        inputs: Input::To<C::Value>,
    ) -> StatefulExecution<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
        Input::Family: ParameterizedFamily<C::Value>,
        Input::To<C::Value>: Parameterized<C::Value>,
        Output::Family: ParameterizedFamily<C::Value>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let inputs = inputs.into_parameters().collect::<Vec<_>>();
        if let Err(error) = self.validate_input_count(inputs.len()) {
            return StatefulExecution::ready(Err(error.into()));
        }
        domain
            .execute_statefully(self.compiled_program(), self.inputs_with_captures(inputs))
            .and_then(|outputs| {
                Ok(Output::To::<C::Value>::from_parameters(self.state.output_structure.clone(), outputs)
                    .map_err(ProgramError::from)?)
            })
    }

    /// Executes this function on the structured runtime `inputs` through the stateful `domain` and waits for the whole
    /// invocation to complete. This is the blocking convenience form of [`Self::execute_statefully`].
    #[inline]
    pub fn execute_statefully_and_wait(
        &self,
        domain: &C,
        inputs: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
        Input::Family: ParameterizedFamily<C::Value>,
        Input::To<C::Value>: Parameterized<C::Value>,
        Output::Family: ParameterizedFamily<C::Value>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        self.execute_statefully(domain, inputs).r#await()
    }

    /// Validates that `input_count` matches the number of flat public inputs, excluding hidden captures.
    pub fn validate_input_count(&self, input_count: usize) -> Result<(), ProgramError> {
        if input_count != self.state.input_types.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.state.input_types.len(),
                actual: input_count,
            });
        }
        Ok(())
    }

    /// Prepends the retained runtime captures to the flat public `inputs`, producing the complete flat input
    /// list in the `[captures..., inputs...]` order every execution expects.
    #[inline]
    pub fn inputs_with_captures(&self, inputs: Vec<C::Value>) -> Vec<C::Value> {
        if self.state.captures.is_empty() {
            return inputs;
        }
        let mut inputs_with_captures = self.state.captures.to_vec();
        inputs_with_captures.extend(inputs);
        inputs_with_captures
    }
}

impl<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>> Clone
    for ExecutableFunction<C, Input, Output>
{
    #[inline]
    fn clone(&self) -> Self {
        Self { state: Arc::clone(&self.state) }
    }
}

/// Shared runtime state of an [`ExecutableFunction`].
struct ExecutableFunctionState<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>> {
    /// Shared backend executable.
    program: Arc<C::CompiledProgram>,

    /// Concrete runtime captures supplied before the public inputs on every call.
    captures: Vec<C::Value>,

    /// Flat staged public input types, excluding hidden captures, against which calls validate their inputs. These are
    /// not derived from the compiled program, whose input types are the backend's own (e.g., the XLA backend compiles
    /// reference inputs as their discharged referent types).
    input_types: Vec<C::Type>,

    /// Output parameter structure used to reconstruct structured runtime values.
    output_structure: Output::ParameterStructure,

    /// [`PhantomData`] marker tying this state to its structured input type.
    input: PhantomData<fn(Input)>,
}

/// Compiled executable plus the staged and lowered metadata required to invoke it safely. The compiled program of the
/// backend is the artifact this function carries, while the captures, structured signatures, and options carried with
/// it are what make the handle callable as well as transformable.
pub struct CompiledFunction<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Runtime-only executable function handle.
    executable: ExecutableFunction<C, Input, Output>,

    /// Lowered function retaining source metadata and options.
    lowered: LoweredFunction<C, Input, Output>,
}

impl<C, Input, Output> CompiledFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Assembles a [`CompiledFunction`] from the provided backend-validated parts. Execution supplies the
    /// `lowered_captures` that the backend returned with the program (refer to [`CompilationDomain::compile`])
    /// after the source captures.
    fn from_parts(
        lowered: LoweredFunction<C, Input, Output>,
        program: Arc<C::CompiledProgram>,
        lowered_captures: Vec<C::Value>,
    ) -> Self {
        let mut captures = lowered.staged.source_program().captures().to_vec();
        captures.extend(lowered_captures);
        let executable_function = ExecutableFunction {
            state: Arc::new(ExecutableFunctionState {
                program,
                captures,
                input_types: lowered.staged.input_types().to_vec(),
                output_structure: lowered.staged.output_structure().clone(),
                input: PhantomData,
            }),
        };
        Self { executable: executable_function, lowered }
    }

    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &C::CompiledProgram {
        self.executable.compiled_program()
    }

    /// Returns the runtime-only executable function handle, which omits staged and lowered transform metadata.
    #[inline]
    pub fn executable(&self) -> &ExecutableFunction<C, Input, Output> {
        &self.executable
    }

    /// Consumes this transformable handle and returns its underlying runtime-only executable function.
    #[inline]
    pub fn into_executable(self) -> ExecutableFunction<C, Input, Output> {
        self.executable
    }

    /// Returns the lowering from which this executable was compiled.
    #[inline]
    pub fn lowered(&self) -> &LoweredFunction<C, Input, Output> {
        &self.lowered
    }

    /// Returns the staged source function.
    #[inline]
    pub fn staged(&self) -> &StagedFunction<C, Input, Output> {
        self.lowered.staged()
    }

    /// Returns the source program and runtime captures.
    #[inline]
    pub fn source_program(
        &self,
    ) -> &ClosedProgram<C::Value, C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        self.staged().source_program()
    }

    /// Returns the effective flat output types of the compiled program.
    #[inline]
    pub fn output_types(&self) -> Cow<'_, [C::Type]> {
        self.executable.output_types()
    }

    /// Calls this function in the active context recovered from `inputs` by delegating to [`StagedFunction::call`]
    /// on its staged source. The call stages a call boundary that carries the staged source program into the
    /// caller's trace, so the retained executable is not involved, even under an eager context. Use
    /// [`ExecutableFunction::execute`] to run the retained executable directly on concrete values. For nullary
    /// functions use [`Self::call_in_context`], because no input value exists from which to recover the active
    /// context.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`StagedFunction::call`].
    #[inline]
    pub fn call<V, Path>(&self, inputs: Input::To<V>) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        V::Domain:
            Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation> + ImportingContext<C, Path>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        self.staged().call(inputs)
    }

    /// Calls this function in the provided `context` by delegating to [`StagedFunction::call_in_context`] on its
    /// staged source, which stages a call boundary into the caller's trace exactly as [`Self::call`] does. An explicit
    /// context also serves nullary functions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`StagedFunction::call_in_context`].
    #[inline]
    pub fn call_in_context<V, CallerContext, Path>(
        &self,
        context: &CallerContext,
        inputs: Input::To<V>,
    ) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        CallerContext: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        self.staged().call_in_context(context, inputs)
    }
}

impl<C, Input, Output> Clone for CompiledFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { executable: self.executable.clone(), lowered: self.lowered.clone() }
    }
}

/// Key of one retained specialization of a [`CompiledFunctionDispatcher`]. The dispatcher reuses a specialization only
/// when all three of the following components agree:
///
///   - **Static Parameters:** The host values the traced function may branch on, read, or embed as literals. Unequal
///     static parameters can stage arbitrarily different programs, so they must separate specializations. Runtime
///     arrays and other backend values should be represented as dynamic inputs rather than static parameters.
///   - **Input Structure:** The [`Parameterized::ParameterStructure`] shape of the dynamic function input. Tracing
///     rebuilds the function's argument from this structure, so a function may legitimately branch on container arity.
///     Keying on the structure rather than on the flattened leaves also distinguishes inputs that differ only in
///     _empty_ substructure, which flat leaf paths and flat leaf types cannot see.
///   - **Dispatch Key:** The [`CompilationDomain::DispatchKey`] that [`CompilationDomain::dispatch_signature`] derives
///     from the runtime inputs and their types. It may be an exact abstract input signature (possibly extended with
///     runtime facts that types do not record, such as the placement of arrays over auto mesh axes) or a normalized
///     equivalence-class key such as a shape bucket. Equal dispatch keys must guarantee that the retained executable
///     can safely serve either call, while unequal keys conservatively separate specializations.
///
/// Equal keys must identify calls that can safely share the same staged [`Program`](crate::Program). Unequal keys
/// remain separate even if they happen to stage structurally equivalent programs. Everything else that affects staging
/// (e.g., the closure itself, its captures, the domain, and the compilation options, except for what a domain derives
/// from the dispatch key through [`CompilationDomain::dispatch_signature`]) is fixed by the dispatcher, whose
/// specialization cache serves exactly one retained closure. That is why this key carries no fragile function-pointer
/// identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExecutableSpecializationKey<Static, Structure, Key> {
    /// Host values declared static for the keyed specialization, such as an axis number, mode enum, or boolean option.
    static_parameters: Static,

    /// Parameter structure of the dynamic input, such as the shape of a nested tuple or named parameter container.
    input_structure: Structure,

    /// Domain-derived dispatch key of the runtime inputs (refer to [`CompilationDomain::dispatch_signature`]).
    dispatch_key: Key,
}

impl<Static, Structure, Key> ExecutableSpecializationKey<Static, Structure, Key> {
    /// Creates a new [`ExecutableSpecializationKey`].
    #[inline]
    fn new(static_parameters: Static, input_structure: Structure, dispatch_key: Key) -> Self {
        Self { static_parameters, input_structure, dispatch_key }
    }

    /// Returns the static parameters of the keyed specialization.
    #[inline]
    fn static_parameters(&self) -> &Static {
        &self.static_parameters
    }

    /// Returns the parameter structure of the dynamic input of the keyed specialization.
    #[inline]
    fn input_structure(&self) -> &Structure {
        &self.input_structure
    }
}

/// Key of one retained staged specialization of a [`CompiledFunctionDispatcher`] (refer to
/// [`CompiledFunctionDispatcher::call_in_context`]). Like [`ExecutableSpecializationKey`], it includes static
/// parameters and input structure. It uses the effective staged input types and calling scope, because the closure
/// resolves named axes and the default mesh against that scope. These types come from a [`DispatchInputs::Types`]
/// signature (refer to [`CompilationDomain::dispatch_signature`]). The callee is lowered and dispatched as part of its
/// enclosing computation, so executable dispatch policies such as shape bucketing do not apply to it.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StagedSpecializationKey<Static, Structure, T> {
    /// Host values declared static for the keyed specialization.
    static_parameters: Static,

    /// Parameter structure of the dynamic input.
    input_structure: Structure,

    /// Effective staged input types.
    input_types: Vec<T>,

    /// Scope of the calling context (i.e., its named axes and default mesh).
    scope: TracingScope,
}

/// Retained Just-In-Time (JIT) compilation dispatcher that caches compiled specializations of one Rust closure.
///
/// Unlike [`CompiledFunction`], which represents one executable that is already specialized, a
/// [`CompiledFunctionDispatcher`] holds no program of its own. It is a retained dispatcher over a Rust closure that
/// produces one program, and one function around that program, per specialization from explicit static host parameters
/// and dynamic runtime inputs. Its first call for a specialization stages, lowers, and requests compilation, and later
/// calls with the same static values, input parameter structure, and dispatch key dispatch directly to the retained
/// executable. Domain dispatch signatures may normalize distinct runtime signatures to the same staged signature,
/// which can produce harmless duplicate dispatch specializations.
///
/// [`Compile::jit`](crate::Compile::jit) selects the domain explicitly. The global [`jit`](crate::jit) instead
/// recovers it from the first leaf of the first non-empty call. That first attempt binds the domain even if staging or
/// compilation later fails. All clones, later calls, and specializations use the selected domain, including after cache
/// clearing. Concurrent first calls atomically select one domain and all use that same selection. The dispatcher does
/// not recover a domain from the inputs of later calls, so inputs that originate in another domain still execute on the
/// selected one. Before selection, a call with no input leaves fails because there is no value from which to recover a
/// domain. Use an explicit domain for nullary functions.
///
/// Tracing executes Rust host code only on a specialization miss. Host side effects inside `function` therefore run
/// once per retained specialization rather than once per runtime call, and observable work for every call must be
/// represented by staged effectful operations. Static values with high cardinality can cause repeated tracing and
/// Least Recently Used (LRU) eviction, so arrays and frequently changing data should remain dynamic inputs.
///
/// Every runtime input is specialized at its own type by default, and eager values are typed exactly (e.g., an eager
/// array by its static shape and an eager first-class dimension by its extent), so each distinct extent gets its own
/// specialization. [`Self::with_staged_input_types`] instead declares the types at which selected inputs are staged
/// (e.g., `dimension<n ∈ [1, 9)>` for a dimension input), so that one specialization serves every runtime input whose
/// type refines them.
///
/// The dispatcher identity only namespaces its specialization cache, which is local to the process.
/// For domains that cache compiled programs, executable correctness and reuse still depend on
/// [`CompilationCacheDomain::compilation_key`](crate::CompilationCacheDomain::compilation_key),
/// which receives the complete lowering.
///
/// Cloned handles share one specialization cache, and the dispatcher is usable from multiple threads whenever the
/// domain, closure, static parameters, and domain artifact types are thread-safe, because [`Send`] and [`Sync`] derive
/// structurally from those field types. Recursive dispatch on one thread of the specialization currently being produced
/// is rejected with an error. Concurrent cold misses for one specialization on different threads instead deliberately
/// duplicate the frontend work, because staging and lowering are cheap and inserts are idempotent, while the shared
/// [`CompilationContext`](crate::CompilationContext) of the domain still coordinates the expensive backend compilation.
///
/// # Retained JIT Lifecycle
///
/// ```text
/// ┌────────────────────────────────────────┐
/// │ `CompiledFunctionDispatcher::execute`  │
/// └──────────────┬─────────────────────────┘
///                │ static parameters and dynamic inputs
///                ▼
/// ┌────────────────────────────┐                                ┌─────────────────────────────┐
/// │ Derive Specialization Key  │                                │ Structured Runtime Outputs  │
/// └──────────────┬─────────────┘  insert after cold production  └─────────────────────────────┘
///                │        ┌─────────────────────────────────────────┐          ▲
///                ▼        ▼                                         │          │ execute
/// ┌────────────────────────────┐              hit               ┌───┴──────────┴──────────────┐
/// │ Specialization Cache       ├───────────────────────────────▶│ Shared `ExecutableFunction` │
/// └──────────────┬─────────────┘                                └─────────────────────────────┘
///                │ miss                                             ▲
///                ▼                                                  │ `into_executable`
/// ┌────────────────────────────┐ stage ┌────────────────────┐   ┌───┴─────────────────────────┐
/// │ Invoke Rust Closure        ├──────▶│ `StagedFunction`   │   │ `CompiledFunction`          │
/// └────────────────────────────┘       └──┬──────┬──────────┘   └─────────────────────────────┘
///     ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘      │ lower            ▲
///     ▼ `call` in an active context              ▼                  │ restore or compile
/// ┌────────────────────────────┐       ┌────────────────────┐   ┌───┴─────────────────────────┐
/// │ Nested Call Operation      │       │ `LoweredFunction`  ├──▶│ Shared `CompilationContext` │
/// └────────────────────────────┘       └────────────────────┘   └─────────────────────────────┘
/// ```
///
/// The cache hit edge skips staging, lowering, and compilation. The dotted edge is the alternative staging path, along
/// which a staged function becomes a nested call boundary instead of continuing immediately toward an executable.
///
/// Calls (i.e., [`Self::call`] and [`Self::call_in_context`]) never touch executables. They retain the staged
/// functions themselves in a separate staged specialization cache, keyed by the abstract input types and the scope
/// of the calling context, in which a miss traces the closure and inserts the staged function:
///
/// ```text
/// ┌────────────────────────────────────────┐
/// │ `CompiledFunctionDispatcher::call`     │
/// └──────────────┬─────────────────────────┘
///                │ static parameters, abstract input types, and calling scope
///                ▼
/// ┌──────────────────────────────┐           hit            ┌──────────────────────────────┐
/// │ Staged Specialization Cache  ├─────────────────────────▶│ `StagedFunction`             │
/// └──────────────┬───────────────┘                          └──────────────┬───────────────┘
///                │ miss                                         ▲          │ `call_in_context`
///                ▼                                              │ stage    ▼
/// ┌──────────────────────────────┐                              │     ┌──────────────────────────────┐
/// │ Invoke Rust Closure in Scope ├──────────────────────────────┘     │ Nested Call Operation        │
/// └──────────────────────────────┘                                    └──────────────────────────────┘
/// ```
pub struct CompiledFunctionDispatcher<
    C: CompilationDomain<Type: Eq + Hash>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type>,
    Output: Parameterized<C::Type>,
    F,
> where
    Input::Family: ParameterizedFamily<C::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// State shared by every clone of this dispatcher.
    state: Arc<CompiledFunctionDispatcherState<C, Static, Input, Output, F>>,
}

impl<C, Static: Clone + Debug + Eq + Hash, Input, Output, F> CompiledFunctionDispatcher<C, Static, Input, Output, F>
where
    C: CompilationDomain,
    C::Type: Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Creates a [`CompiledFunctionDispatcher`] for `function` that retains at most `capacity` specializations.
    #[inline]
    pub(crate) fn new(domain: &C, function: F, options: C::Options, capacity: usize) -> Self {
        Self::from_domain(DispatcherDomain::Explicit(domain.clone()), function, options, capacity)
    }

    /// Creates a [`CompiledFunctionDispatcher`] that selects its domain from its first non-empty invocation.
    #[inline]
    pub(crate) fn inferred(function: F, options: C::Options, capacity: usize) -> Self
    where
        C::Value: Value<Domain = C>,
    {
        Self::from_domain(
            DispatcherDomain::Inferred { domain: OnceLock::new(), from_input: Value::domain },
            function,
            options,
            capacity,
        )
    }

    /// Creates a [`CompiledFunctionDispatcher`] with an explicit or deferred domain selection.
    #[inline]
    fn from_domain(domain: DispatcherDomain<C>, function: F, options: C::Options, capacity: usize) -> Self {
        Self {
            state: Arc::new(CompiledFunctionDispatcherState {
                domain,
                function,
                options,
                staged_input_types: None,
                executable_specializations: SpecializationCache::new(capacity),
                staged_specializations: SpecializationCache::new(capacity),
                statistics: DispatcherStatisticsAccumulator::default(),
            }),
        }
    }

    /// Returns this dispatcher with the types at which its runtime inputs are staged (which also determine their
    /// dispatch keys and compiled signatures) declared in the structure of its inputs, as JAX's `abstracted_axes` does.
    /// A [`Some`] leaf stages its input at that type instead of at the input's own type, so that one specialization
    /// serves every runtime input whose type refines it, and a [`None`] leaf keeps its input exact. For example,
    /// declaring `Some(dimension<n ∈ [1, 9)>)` for a first-class dimension input reuses one specialization for every
    /// extent from `1` to `8`. Inputs declared with one shared identity (e.g., two `dimension<n>` inputs, or a
    /// `dimension<n>` input and an `f32[n]` input) must agree at runtime. Every call validates its inputs against the
    /// declared types before dispatching, and rejects a violation (including inputs whose structure differs from the
    /// declaration) rather than producing another specialization. The declared types are created once by the caller,
    /// so their identities, and thus the dispatch keys, remain unchanged across calls.
    ///
    /// # Errors
    ///
    /// Returns an error when this dispatcher was already cloned, since its clones share one specialization cache
    /// that must not mix specializations of different declarations.
    pub fn with_staged_input_types(
        mut self,
        staged_input_types: Input::To<Option<C::Type>>,
    ) -> Result<Self, ProgramError>
    where
        Input::Family: ParameterizedFamily<Option<C::Type>>,
    {
        let state = Arc::get_mut(&mut self.state).ok_or_else(|| ProgramError::InvalidArgument {
            message: "staged input types must be declared before a compilation dispatcher is shared".to_string(),
        })?;
        state.staged_input_types = Some(StagedInputTypes {
            structure: staged_input_types.parameter_structure(),
            types: staged_input_types.into_parameters().collect(),
        });
        Ok(self)
    }

    /// Returns the compilation domain used by this dispatcher, or [`None`] while a global dispatcher's domain has
    /// not yet been recovered from a non-empty call.
    #[inline]
    pub fn domain(&self) -> Option<&C> {
        match &self.state.domain {
            DispatcherDomain::Explicit(domain) => Some(domain),
            DispatcherDomain::Inferred { domain, .. } => domain.get(),
        }
    }

    /// Returns the declared staged input types of this dispatcher, or [`None`] when every input is staged at its own
    /// type. Refer to the documentation of [`Self::with_staged_input_types`] for more information.
    #[inline]
    pub fn staged_input_types(&self) -> Option<Input::To<Option<C::Type>>>
    where
        Input::Family: ParameterizedFamily<Option<C::Type>>,
    {
        let StagedInputTypes { structure, types } = self.state.staged_input_types.as_ref()?;
        Some(Input::To::<Option<C::Type>>::from_parameters(structure.clone(), types.iter().cloned()).unwrap())
    }

    /// Returns the fixed compilation options used for every specialization.
    #[inline]
    pub fn options(&self) -> &C::Options {
        &self.state.options
    }

    /// Returns the number of executable specializations currently retained by this dispatcher.
    #[inline]
    pub fn executable_specialization_count(&self) -> usize {
        self.state.executable_specializations.len()
    }

    /// Returns the maximum number of specializations retained in each of this dispatcher's two caches.
    #[inline]
    pub fn cache_capacity(&self) -> usize {
        self.state.executable_specializations.capacity()
    }

    /// Returns a snapshot of this dispatcher's frontend activity.
    #[inline]
    pub fn statistics(&self) -> DispatcherStatistics {
        self.state.statistics.snapshot()
    }

    /// Resets the statistics of this dispatcher and its clones without clearing either specialization cache.
    #[inline]
    pub fn clear_statistics(&self) {
        self.state.statistics.clear();
    }

    /// Clears every retained executable and staged specialization without changing statistics.
    #[inline]
    pub fn clear_cache(&self) {
        self.state.executable_specializations.clear();
        self.state.staged_specializations.clear();
    }

    /// Executes a specialization with explicit host `static_parameters` and dynamic runtime `inputs`, compiling it
    /// on a specialization miss.
    #[inline]
    pub fn execute(
        &self,
        static_parameters: Static,
        inputs: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, C::Error>
    where
        C::Options: Clone,
        Input::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Input::To<C::Value>: Parameterized<
                C::Value,
                Family = Input::Family,
                ParameterStructure = Input::ParameterStructure,
                To<C::Type> = Input,
            >,
        Output::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        Output::To<CompilationTracer<C>>:
            Parameterized<CompilationTracer<C>, To<C::Type> = Output, To<C::Constant> = Output::To<C::Constant>>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (domain, executable) = self.executable_specialization(static_parameters, &inputs)?;
        executable.execute(domain, inputs)
    }

    /// Executes a specialization through its domain's stateful execution capability and returns a [`StatefulExecution`]
    /// tracking completion of the whole invocation. Submission publishes the domain's reference dependencies before
    /// this function returns. The returned completion must be awaited to observe asynchronous failure, and dropping
    /// it does not cancel the invocation.
    #[inline]
    pub fn execute_statefully(
        &self,
        static_parameters: Static,
        inputs: Input::To<C::Value>,
    ) -> StatefulExecution<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
        C::Options: Clone,
        Input::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Input::To<C::Value>: Parameterized<
                C::Value,
                Family = Input::Family,
                ParameterStructure = Input::ParameterStructure,
                To<C::Type> = Input,
            >,
        Output::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        Output::To<CompilationTracer<C>>:
            Parameterized<CompilationTracer<C>, To<C::Type> = Output, To<C::Constant> = Output::To<C::Constant>>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (domain, executable) = match self.executable_specialization(static_parameters, &inputs) {
            Ok(specialization) => specialization,
            Err(error) => return StatefulExecution::ready(Err(error)),
        };
        executable.execute_statefully(domain, inputs)
    }

    /// Executes a specialization statefully and waits for the whole invocation to complete. This is the blocking
    /// convenience form of [`Self::execute_statefully`]. This routes through
    /// [`ExecutableFunction::execute_statefully_and_wait`], preserving the domain's reference
    /// sequencing and reporting asynchronous execution errors before returning.
    #[inline]
    pub fn execute_statefully_and_wait(
        &self,
        static_parameters: Static,
        inputs: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
        C::Options: Clone,
        Input::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Input::To<C::Value>: Parameterized<
                C::Value,
                Family = Input::Family,
                ParameterStructure = Input::ParameterStructure,
                To<C::Type> = Input,
            >,
        Output::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        Output::To<CompilationTracer<C>>:
            Parameterized<CompilationTracer<C>, To<C::Type> = Output, To<C::Constant> = Output::To<C::Constant>>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (domain, executable) = self.executable_specialization(static_parameters, &inputs)?;
        executable.execute_statefully_and_wait(domain, inputs)
    }

    /// Calls the specialization of this dispatcher for `static_parameters` and the abstract types of `inputs` in the
    /// active context recovered from `inputs`. Refer to [`Self::call_in_context`] for more information. For nullary
    /// functions, use [`Self::call_in_context`], because no input value exists from which to recover the active
    /// context.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::call_in_context`], and [`ProgramError::InvalidInputCount`] when `inputs` has no
    /// leaves.
    #[inline]
    pub fn call<V, Path>(&self, static_parameters: Static, inputs: Input::To<V>) -> Result<Output::To<V>, C::Error>
    where
        C::Options: Clone,
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        V::Domain: Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>
            + NamedAxes
            + MeshContext,
        Input::Family: ParameterizedFamily<V> + ParameterizedFamily<CompilationTracer<C>>,
        Input::To<V>: Parameterized<V, Family = Input::Family, ParameterStructure = Input::ParameterStructure>,
        Output::Family: ParameterizedFamily<V> + ParameterizedFamily<CompilationTracer<C>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let context = inputs
            .parameters()
            .next()
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
            .domain();
        self.call_in_context(&context, static_parameters, inputs)
    }

    /// Calls the specialization of this dispatcher for `static_parameters` and the abstract types of `inputs` in the
    /// provided `context`, which binds one staged-call operation there, exactly as [`StagedFunction::call_in_context`]
    /// does. Unlike [`Self::execute`], it works under any context, including staging and transform contexts, so the
    /// dispatcher's closure composes with enclosing transforms and compilations. Its executable specializations are not
    /// involved. The call neither lowers nor compiles anything itself, and an eager `context` executes the call as an
    /// ordinary eager operation.
    ///
    /// On the first call for one static value, input structure, effective staged input type list, and calling scope,
    /// the dispatcher traces its closure into a staged specialization and retains it in a cache separate from its
    /// executables. The calling scope is the [`TracingScope`] of `context` (i.e., its named axes and default mesh),
    /// and the closure is traced in that scope so that it resolves them as it would if it were inlined. The effective
    /// staged input types are the abstract input types after the declarations of [`Self::with_staged_input_types`] and
    /// the [`DispatchInputs::Types`] signature of [`CompilationDomain::dispatch_signature`], which need no domain
    /// instance, so a dispatcher that infers its domain need not have selected one. Host side effects of the closure
    /// therefore run once per staged specialization. [`Self::clear_cache`] and [`Self::invalidate_specializations`]
    /// remove staged specializations together with executable ones.
    ///
    /// # Errors
    ///
    /// Returns an error when the inputs do not satisfy the declared staged input types, when the domain rejects the
    /// staged input types, when tracing the closure fails, when the same thread recursively requests a staged
    /// specialization that it is still producing, or when binding the call fails in `context`.
    pub fn call_in_context<V, CallerContext, Path>(
        &self,
        context: &CallerContext,
        static_parameters: Static,
        inputs: Input::To<V>,
    ) -> Result<Output::To<V>, C::Error>
    where
        C::Options: Clone,
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        CallerContext: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>
            + NamedAxes
            + MeshContext,
        Input::Family: ParameterizedFamily<V> + ParameterizedFamily<CompilationTracer<C>>,
        Input::To<V>: Parameterized<V, Family = Input::Family, ParameterStructure = Input::ParameterStructure>,
        Output::Family: ParameterizedFamily<V> + ParameterizedFamily<CompilationTracer<C>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let input_structure = inputs.parameter_structure();
        let input_types = inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let staged = self.staged_specialization(
            static_parameters,
            input_structure,
            input_types,
            TracingScope::from_context(context),
        )?;
        staged.call_in_context(context, inputs).map_err(Into::into)
    }

    /// Invalidates every retained executable and staged specialization for `static_parameters`, across all input
    /// signatures and calling scopes, and returns the total number of specializations removed from both. Backend
    /// compilation caches remain available for reuse. A specialization that is still being produced when this
    /// function runs may be retained once its production completes.
    #[inline]
    pub fn invalidate_specializations(&self, static_parameters: &Static) -> usize {
        self.state
            .executable_specializations
            .invalidate_entries_if(|key| key.static_parameters() == static_parameters)
            + self
                .state
                .staged_specializations
                .invalidate_entries_if(|key| &key.static_parameters == static_parameters)
    }

    /// Acquires or produces the executable specialization for one runtime input signature and returns it together with
    /// the resolved domain that executes it.
    fn executable_specialization(
        &self,
        static_parameters: Static,
        inputs: &Input::To<C::Value>,
    ) -> Result<(&C, ExecutableFunction<C, Input, Output>), C::Error>
    where
        C::Options: Clone,
        Input::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Input::To<C::Value>: Parameterized<
                C::Value,
                Family = Input::Family,
                ParameterStructure = Input::ParameterStructure,
                To<C::Type> = Input,
            >,
        Output::Family: ParameterizedFamily<C::Value> + ParameterizedFamily<CompilationTracer<C>>,
        Output::To<C::Value>:
            Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
        Output::To<CompilationTracer<C>>:
            Parameterized<CompilationTracer<C>, To<C::Type> = Output, To<C::Constant> = Output::To<C::Constant>>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        // Static values and input structure can change the traced program, while the domain's dispatch key groups
        // input signatures that can safely share it. Dynamic values remain runtime inputs, and the closure, captures,
        // selected domain, and options are fixed for this dispatcher. Time all dispatch preparation, including domain
        // selection and input validation.
        let dispatch_preparation_start = Instant::now();

        // Every specialization and clone shares one domain, which remains selected after failures or cache eviction.
        let domain = match &self.state.domain {
            DispatcherDomain::Explicit(domain) => domain,
            DispatcherDomain::Inferred { domain, from_input } => {
                // Reuse an existing selection before inspecting inputs so later calls may have no input leaves.
                if let Some(domain) = domain.get() {
                    domain
                } else {
                    let input = inputs.parameters().next().ok_or_else(|| ProgramError::InvalidArgument {
                        message: "compilation requires an input leaf to infer a domain; use an explicit domain for \
                                  empty inputs"
                            .to_string(),
                    })?;

                    // Recover inside initialization so racing clones all use the stored winner's domain.
                    domain.get_or_init(|| from_input(input))
                }
            }
        };

        let input_structure = inputs.parameter_structure();
        let mut runtime_input_types = inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.apply_staged_input_types(&input_structure, &mut runtime_input_types)?;

        let mut runtime_inputs = inputs.parameters();
        let (dispatch_key, effective_input_types, specialized_options) = C::dispatch_signature(
            DispatchInputs::Values { values: &mut runtime_inputs, types: runtime_input_types },
            &self.state.options,
        )?;

        // Backend normalization can share dispatch keys, but static values and container structures remain distinct.
        let key = ExecutableSpecializationKey::new(static_parameters.clone(), input_structure, dispatch_key);

        // One clock read both ends dispatch preparation and starts the specialization lookup, which keeps the timing
        // of every cached call to three clock reads.
        let statistics = &self.state.statistics;
        let dispatch_lookup_start = Instant::now();
        accumulate_duration(
            dispatch_lookup_start.duration_since(dispatch_preparation_start),
            &statistics.dispatch_preparation_nanoseconds,
        );
        let entry = self.state.executable_specializations.try_entry(key);
        accumulate_duration(dispatch_lookup_start.elapsed(), &statistics.dispatch_lookup_nanoseconds);

        // A hit skips all frontend production. A vacant producer clears its in-flight marker if any later phase fails,
        // allowing retries. Only recursive production of the same key on this thread is rejected.
        let producer = match entry {
            Ok(SpecializationCacheEntry::Occupied(executable)) => {
                statistics.dispatch_hits.fetch_add(1, Ordering::Relaxed);
                return Ok((domain, executable));
            }
            Ok(SpecializationCacheEntry::Vacant(producer)) => {
                statistics.dispatch_misses.fetch_add(1, Ordering::Relaxed);
                producer
            }
            Err(_) => {
                statistics.dispatch_misses.fetch_add(1, Ordering::Relaxed);
                return Err(ProgramError::InvalidArgument {
                    message: "recursive compilation dispatch requested a specialization that is already being produced"
                        .into(),
                }
                .into());
            }
        };

        // Rebuild the structured signature from the backend's effective types, retaining empty container structure.
        let input_types =
            Input::from_parameters(producer.key().input_structure().clone(), effective_input_types.iter().cloned())
                .map_err(ProgramError::from)?;

        // Record tracing and lowering durations before propagating errors so failed attempts remain in statistics.
        statistics.traces.fetch_add(1, Ordering::Relaxed);
        let tracing_start = Instant::now();
        let staged = StagedFunction::<C, Input, Output>::stage(
            |_, _, traced_inputs| (self.state.function)(static_parameters, traced_inputs),
            Vec::new(),
            input_types,
            specialized_options.unwrap_or_else(|| self.state.options.clone()),
        );
        accumulate_duration(tracing_start.elapsed(), &statistics.tracing_nanoseconds);
        let staged = staged?;
        statistics.lowerings.fetch_add(1, Ordering::Relaxed);
        let lowering_start = Instant::now();
        let lowered = staged.lower(domain);
        accumulate_duration(lowering_start.elapsed(), &statistics.lowering_nanoseconds);
        let lowered = lowered?;
        statistics.compilation_requests.fetch_add(1, Ordering::Relaxed);
        let compiled = lowered.compile(domain)?;

        // Retain only execution state, allowing the staged and lowered program artifacts to be released.
        Ok((domain, producer.insert(compiled.into_executable())))
    }

    /// Acquires or produces the staged specialization of [`Self::call_in_context`] for `static_parameters`, the
    /// abstract `input_types` of inputs with `input_structure`, and the calling `scope`. It is independent of the
    /// caller's context and value types, so every caller of this dispatcher shares one instantiation of its tracing
    /// and caching code.
    fn staged_specialization(
        &self,
        static_parameters: Static,
        input_structure: Input::ParameterStructure,
        mut input_types: Vec<C::Type>,
        scope: TracingScope,
    ) -> Result<StagedFunction<C, Input, Output>, C::Error>
    where
        C::Options: Clone,
        Input::Family: ParameterizedFamily<CompilationTracer<C>>,
        Output::Family: ParameterizedFamily<CompilationTracer<C>>,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        self.apply_staged_input_types(&input_structure, &mut input_types)?;
        let (_, input_types, specialized_options) =
            C::dispatch_signature(DispatchInputs::Types(input_types), &self.state.options)?;
        let input_types = input_types.to_vec();
        let key = StagedSpecializationKey {
            static_parameters: static_parameters.clone(),
            input_structure,
            input_types,
            scope,
        };

        // A hit reuses the retained staged function. A vacant producer clears its in-flight marker if tracing fails,
        // allowing retries. Only recursive production of the same key on this thread is rejected.
        let statistics = &self.state.statistics;
        let staged = match self.state.staged_specializations.try_entry(key) {
            Ok(SpecializationCacheEntry::Occupied(staged)) => {
                statistics.staged_cache_hits.fetch_add(1, Ordering::Relaxed);
                staged
            }
            Ok(SpecializationCacheEntry::Vacant(producer)) => {
                statistics.staged_cache_misses.fetch_add(1, Ordering::Relaxed);
                let StagedSpecializationKey { input_structure, input_types, scope, .. } = producer.key().clone();
                let input_types = Input::from_parameters(input_structure, input_types).map_err(ProgramError::from)?;
                statistics.traces.fetch_add(1, Ordering::Relaxed);
                let tracing_start = Instant::now();
                let staged = StagedFunction::<C, Input, Output>::stage_in_scope(
                    |_, _, traced_inputs| (self.state.function)(static_parameters, traced_inputs),
                    Vec::new(),
                    input_types,
                    specialized_options.unwrap_or_else(|| self.state.options.clone()),
                    scope,
                );
                accumulate_duration(tracing_start.elapsed(), &statistics.tracing_nanoseconds);
                producer.insert(staged?)
            }
            Err(_) => {
                statistics.staged_cache_misses.fetch_add(1, Ordering::Relaxed);
                return Err(ProgramError::InvalidArgument {
                    message: "recursive staged call requested a staged specialization that is already being produced"
                        .into(),
                }
                .into());
            }
        };

        Ok(staged)
    }

    /// Replaces the types of `input_types` that [`Self::with_staged_input_types`] declares with their declared staged
    /// types, after validating that each replaced type refines its declaration. Execution and calls share this
    /// contract, so that a declaration means the same thing for both. Inputs without declarations keep their types.
    fn apply_staged_input_types(
        &self,
        input_structure: &Input::ParameterStructure,
        input_types: &mut [C::Type],
    ) -> Result<(), ProgramError> {
        let Some(StagedInputTypes { structure, types }) = &self.state.staged_input_types else {
            return Ok(());
        };

        // Equal structures guarantee that declared and runtime leaves align, including empty substructures.
        if structure != input_structure {
            return Err(ProgramError::InvalidArgument {
                message: "compilation inputs do not have the structure of their declared staged input types"
                    .to_string(),
            });
        }

        let declared = types
            .iter()
            .enumerate()
            .filter_map(|(index, r#type)| Some((index, r#type.as_ref()?)))
            .collect::<Vec<_>>();
        for &(index, staged_type) in &declared {
            if !staged_type.is_refined_by(&input_types[index]) {
                let (path, _) = structure.named_parameters().nth(index).unwrap();
                return Err(ProgramError::from(TypeError::invalid(format!(
                    "compilation input `{}` has type `{}`, which does not refine its declared staged type `{}`",
                    path, input_types[index], staged_type,
                ))));
            }
        }

        // Check the declarations together before replacing any types. Individual refinement checks cannot detect
        // conflicting runtime extents for an identity shared by multiple inputs, such as a dimension and an array.
        <C::Type as Type>::Refinements::new(
            declared.iter().map(|(_, staged_type)| *staged_type),
            declared.iter().map(|(index, _)| &input_types[*index]),
        )
        .map_err(ProgramError::from)?;

        // Declared leaves specialize at their staged types so compatible runtime values reuse a trace.
        // Undeclared leaves keep their runtime types. Validate before lookup so cache hits obey the same contract.
        for (index, staged_type) in declared {
            input_types[index] = staged_type.clone();
        }

        Ok(())
    }
}

impl<C, Static: Clone + Debug + Eq + Hash, Input, Output, F> Clone
    for CompiledFunctionDispatcher<C, Static, Input, Output, F>
where
    C: CompilationDomain,
    C::Type: Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

/// Shared state of a [`CompiledFunctionDispatcher`].
struct CompiledFunctionDispatcherState<
    C: CompilationDomain<Type: Eq + Hash>,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type>,
    Output: Parameterized<C::Type>,
    F,
> where
    Input::Family: ParameterizedFamily<C::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Compilation domain that stages, lowers, compiles, and executes every specialization.
    domain: DispatcherDomain<C>,

    /// Rust closure traced once per specialization.
    function: F,

    /// Compilation options fixed for every specialization.
    options: C::Options,

    /// Declared staged input types, or [`None`] when every input is staged at its own type.
    staged_input_types: Option<StagedInputTypes<C::Type, Input::ParameterStructure>>,

    /// Bounded Least Recently Used (LRU) cache of compiled specializations.
    executable_specializations: SpecializationCache<
        ExecutableSpecializationKey<Static, Input::ParameterStructure, C::DispatchKey>,
        ExecutableFunction<C, Input, Output>,
    >,

    /// Bounded Least Recently Used (LRU) cache of staged specializations that calls bind into their calling contexts.
    /// It is separate from [`Self::executable_specializations`], so that calls never evict executables and execution
    /// never reads or retains staged programs. Every invalidation applies to both caches.
    staged_specializations: SpecializationCache<
        StagedSpecializationKey<Static, Input::ParameterStructure, C::Type>,
        StagedFunction<C, Input, Output>,
    >,

    /// Dispatch activity counters.
    statistics: DispatcherStatisticsAccumulator,
}

/// Explicit or lazily recovered [`CompilationDomain`] shared by one dispatcher's specializations and clones.
enum DispatcherDomain<C: CompilationDomain> {
    /// Domain selected when the dispatcher was constructed.
    Explicit(C),

    /// Domain recovered once from a runtime input, with the recovery function available only for compatible values.
    Inferred {
        /// Domain selected atomically by the first call with an input leaf.
        domain: OnceLock<C>,

        /// Recovers the compilation domain carried by a runtime input.
        from_input: fn(&C::Value) -> C,
    },
}

/// Staged input types declared for a [`CompiledFunctionDispatcher`], stored flattened so that the dispatcher needs no
/// `Option`-leaved instantiation of its input family outside of the functions that declare and return them. Refer to
/// the documentation of [`CompiledFunctionDispatcher::with_staged_input_types`] for more information.
struct StagedInputTypes<T, Structure> {
    /// Parameter structure of the declared inputs.
    structure: Structure,

    /// Declared staged type of each flattened input, or [`None`] for an input that is staged at its own type.
    types: Vec<Option<T>>,
}

/// Completion-bearing result of one externally stateful [`CompiledFunction`] invocation. The public result may be
/// reconstructed immediately from pending backend values, but [`Self::await`] always observes the whole invocation
/// completion before exposing that result. Completion includes every predecessor dependency of the references used by
/// the invocation, not only the most recently submitted backend event. Dropping this wrapper never cancels submitted
/// work or rolls reference state back, so callers that need to observe asynchronous failure must await it. A backend
/// may also record a failure that happens after submission on the affected mutated references, so that later accesses
/// report the same cause for their poisoned state.
#[must_use = "stateful execution errors are observed by awaiting this completion"]
pub struct StatefulExecution<Output, Error> {
    /// Public output or reconstruction failure prepared after submission.
    result: Result<Output, Error>,

    /// Whole-invocation completion and its backend error conversion, absent only for already-completed results.
    completion: Option<(ReferenceCompletion, fn(Arc<str>) -> Error)>,
}

impl<Output, Error> StatefulExecution<Output, Error> {
    /// Creates an already-completed stateful invocation result, which returns `result` without waiting when awaited.
    /// Backends use it when a stateful call finishes synchronously or fails before submitting any work. A backend must
    /// not use this function once work that reads or replaces a [`Reference`](crate::Reference) has been submitted,
    /// because the result would then hide the completion of that work and any asynchronous failure that it reports.
    /// Use [`Self::pending`] for submitted work instead.
    #[inline]
    pub fn ready(result: Result<Output, Error>) -> Self {
        Self { result, completion: None }
    }

    /// Creates a submitted invocation whose public result is already reconstructed but whose execution remains pending.
    /// Completion failure takes precedence over `result` because it may invalidate every pending output. Backends must
    /// pass a `completion` that covers the whole invocation, which usually means [`ReferenceCompletion::joined`] over
    /// the submitted work and the completions that the invocation's references already depended on. A completion that
    /// only tracks the latest backend event would let awaiting the result return before an earlier reference write
    /// has finished or reported its failure.
    ///
    /// # Parameters
    ///
    ///   - `result`: Public output or reconstruction failure prepared right after submission.
    ///   - `completion`: Whole-invocation completion, including every cumulative reference predecessor dependency.
    ///   - `completion_error`: Converts a backend completion failure message into the domain error type. It is applied
    ///     only when the completion itself fails, and its result then replaces `result`.
    #[inline]
    pub fn pending(
        result: Result<Output, Error>,
        completion: ReferenceCompletion,
        completion_error: fn(Arc<str>) -> Error,
    ) -> Self {
        Self { result, completion: Some((completion, completion_error)) }
    }

    /// Waits for the whole execution and its cumulative dependencies, and returns the final output.
    #[inline]
    pub fn r#await(self) -> Result<Output, Error> {
        if let Some((completion, completion_error)) = self.completion
            && let Err(error) = completion.r#await()
        {
            return Err(completion_error(error));
        }
        self.result
    }

    /// Applies `function` to the output of this invocation without changing its completion status.
    #[inline]
    pub fn and_then<T>(self, function: impl FnOnce(Output) -> Result<T, Error>) -> StatefulExecution<T, Error> {
        StatefulExecution { result: self.result.and_then(function), completion: self.completion }
    }
}

/// Snapshot of the frontend activity of one retained [`CompiledFunctionDispatcher`], covering dispatch preparation,
/// specialization lookup, tracing, lowering, and compilation requests. The statistics accumulate from construction or
/// from the latest [`CompiledFunctionDispatcher::clear_statistics`] call, are shared by every clone of the dispatcher,
/// and are unaffected by [`CompiledFunctionDispatcher::clear_cache`]. Each field is read independently, so a snapshot
/// taken during concurrent calls need not correspond to a single point in time.
///
/// Every execution that reaches executable specialization lookup counts as exactly one of
/// [`Self::dispatch_hits`] and [`Self::dispatch_misses`]. Invocations through [`CompiledFunctionDispatcher::call`]
/// or [`CompiledFunctionDispatcher::call_in_context`] instead count toward [`Self::staged_cache_hits`] and
/// [`Self::staged_cache_misses`]. Requests that fail before lookup (e.g., because an inferred domain cannot be
/// recovered, an input does not refine its declared staged type, or [`CompilationDomain::dispatch_signature`] fails)
/// count as neither a hit nor a miss. Tracing counts and durations cover both paths. Production counters record
/// attempts, including failed ones, and concurrent cold misses for one executable specialization may each trace, lower,
/// and request compilation. Durations are host durations summed over all invocations, so concurrency can make them
/// exceed elapsed wall-clock time. Backend compilation time is not included, because a compilation request may be
/// served by a compilation cache. Refer to
/// [`CompilationCacheStatistics::compilation_duration`](crate::CompilationCacheStatistics::compilation_duration)
/// for the backend compilation activity of a shared [`CompilationContext`](crate::CompilationContext).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DispatcherStatistics {
    /// Executions served by a retained executable specialization.
    pub dispatch_hits: u64,

    /// Executions that found no retained executable specialization, including recursive execution requests for
    /// a specialization that was still being produced and were therefore rejected.
    pub dispatch_misses: u64,

    /// Traces of the dispatcher's closure started after executable or staged cache misses, including failed traces.
    pub traces: u64,

    /// Lowerings started after successful traces, including lowerings that failed.
    pub lowerings: u64,

    /// Compilation requests made to the domain after successful lowerings, including requests that failed or that the
    /// domain served from a compilation cache.
    pub compilation_requests: u64,

    /// Total host duration spent preparing dispatch, including recovering an inferred domain, reading input structure
    /// and types, validating declared staged input types, calling [`CompilationDomain::dispatch_signature`], and
    /// constructing the specialization key.
    pub dispatch_preparation_duration: Duration,

    /// Total host duration spent looking up retained specializations by their specialization keys.
    pub dispatch_lookup_duration: Duration,

    /// Total host duration spent tracing the dispatcher's closure after executable or staged cache misses.
    pub tracing_duration: Duration,

    /// Total host duration spent lowering newly traced specializations.
    pub lowering_duration: Duration,

    /// Invocations through [`CompiledFunctionDispatcher::call`] or [`CompiledFunctionDispatcher::call_in_context`]
    /// served by a retained staged specialization.
    pub staged_cache_hits: u64,

    /// Calls that found no retained staged specialization, including recursive calls that requested a staged
    /// specialization that was still being produced and were therefore rejected.
    pub staged_cache_misses: u64,
}

/// Lock-free counters underlying [`DispatcherStatistics`]. Durations accumulate in nanoseconds.
#[derive(Default)]
struct DispatcherStatisticsAccumulator {
    /// Counter underlying [`DispatcherStatistics::dispatch_hits`].
    dispatch_hits: AtomicU64,

    /// Counter underlying [`DispatcherStatistics::dispatch_misses`].
    dispatch_misses: AtomicU64,

    /// Counter underlying [`DispatcherStatistics::traces`].
    traces: AtomicU64,

    /// Counter underlying [`DispatcherStatistics::lowerings`].
    lowerings: AtomicU64,

    /// Counter underlying [`DispatcherStatistics::compilation_requests`].
    compilation_requests: AtomicU64,

    /// Nanoseconds underlying [`DispatcherStatistics::dispatch_preparation_duration`].
    dispatch_preparation_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`DispatcherStatistics::dispatch_lookup_duration`].
    dispatch_lookup_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`DispatcherStatistics::tracing_duration`].
    tracing_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`DispatcherStatistics::lowering_duration`].
    lowering_nanoseconds: AtomicU64,

    /// Counter underlying [`DispatcherStatistics::staged_cache_hits`].
    staged_cache_hits: AtomicU64,

    /// Counter underlying [`DispatcherStatistics::staged_cache_misses`].
    staged_cache_misses: AtomicU64,
}

impl DispatcherStatisticsAccumulator {
    /// Returns a snapshot of the counters (refer to [`DispatcherStatistics`] for its consistency guarantees).
    fn snapshot(&self) -> DispatcherStatistics {
        DispatcherStatistics {
            dispatch_hits: self.dispatch_hits.load(Ordering::Relaxed),
            dispatch_misses: self.dispatch_misses.load(Ordering::Relaxed),
            traces: self.traces.load(Ordering::Relaxed),
            lowerings: self.lowerings.load(Ordering::Relaxed),
            compilation_requests: self.compilation_requests.load(Ordering::Relaxed),
            dispatch_preparation_duration: Duration::from_nanos(
                self.dispatch_preparation_nanoseconds.load(Ordering::Relaxed),
            ),
            dispatch_lookup_duration: Duration::from_nanos(self.dispatch_lookup_nanoseconds.load(Ordering::Relaxed)),
            tracing_duration: Duration::from_nanos(self.tracing_nanoseconds.load(Ordering::Relaxed)),
            lowering_duration: Duration::from_nanos(self.lowering_nanoseconds.load(Ordering::Relaxed)),
            staged_cache_hits: self.staged_cache_hits.load(Ordering::Relaxed),
            staged_cache_misses: self.staged_cache_misses.load(Ordering::Relaxed),
        }
    }

    /// Resets every counter to zero.
    fn clear(&self) {
        self.dispatch_hits.store(0, Ordering::Relaxed);
        self.dispatch_misses.store(0, Ordering::Relaxed);
        self.traces.store(0, Ordering::Relaxed);
        self.lowerings.store(0, Ordering::Relaxed);
        self.compilation_requests.store(0, Ordering::Relaxed);
        self.dispatch_preparation_nanoseconds.store(0, Ordering::Relaxed);
        self.dispatch_lookup_nanoseconds.store(0, Ordering::Relaxed);
        self.tracing_nanoseconds.store(0, Ordering::Relaxed);
        self.lowering_nanoseconds.store(0, Ordering::Relaxed);
        self.staged_cache_hits.store(0, Ordering::Relaxed);
        self.staged_cache_misses.store(0, Ordering::Relaxed);
    }
}

/// Adds `duration` to `counter`, which accumulates nanoseconds and saturates at [`u64::MAX`].
pub(super) fn accumulate_duration(duration: Duration, counter: &AtomicU64) {
    let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
    counter
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |total| Some(total.saturating_add(nanoseconds)))
        .unwrap();
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::hash::{Hash, Hasher};
    use std::rc::Rc;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Operation;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, DimensionBounds, DimensionError,
        DimensionType, DimensionValue, LogicalMesh, MeshAxis, MeshAxisType, Shape,
    };
    use crate::axes::NamedAxis;
    use crate::captures::CaptureReference;
    use crate::compilation::domains::{Compile, jit};
    use crate::compilation::tests::{
        CarriedDomain, CarriedOptions, TestCompiledProgram, TestDomain, TestOptions, compile_from_one_call_site,
    };
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::{NegOperation, ZeroOperation};
    use crate::programs::{EmptyRegionDriver, Operation, RegionInterface, RegionSlot, TypeError};

    use super::*;

    /// Compilation domain over array IR values, whose dimension inputs exercise declared staged input types. It lowers
    /// and compiles to the lifted program itself, counts compilations, and executes programs by interpreting them
    /// eagerly.
    #[derive(Clone)]
    struct TestArrayIrDomain {
        /// Number of backend compilations performed by this domain.
        compilations: Arc<AtomicUsize>,
    }

    /// Compilation options of [`TestArrayIrDomain`], which requires none.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct TestArrayIrOptions;

    impl TestArrayIrDomain {
        /// Creates a domain with no backend compilations.
        fn new() -> Self {
            Self { compilations: Arc::new(AtomicUsize::new(0)) }
        }

        /// Returns the number of backend compilations performed so far.
        fn compilation_count(&self) -> usize {
            self.compilations.load(Ordering::Relaxed)
        }
    }

    impl Domain for TestArrayIrDomain {
        type Type = ArrayIrType;
        type Value = ArrayIrValue<Array>;
        type Constant = CaptureReference<ArrayIrType>;
        type Operation = ArrayIrOperation<Array>;
    }

    impl CompilationDomain for TestArrayIrDomain {
        type DispatchKey = Arc<[ArrayIrType]>;
        type LoweredProgram = FlatProgram<Self>;
        type CompiledProgram = FlatProgram<Self>;
        type Options = TestArrayIrOptions;
        type Error = ProgramError;

        fn dispatch_signature(
            inputs: DispatchInputs<'_, '_, Self::Value>,
            _options: &Self::Options,
        ) -> Result<(Self::DispatchKey, Arc<[ArrayIrType]>, Option<Self::Options>), Self::Error> {
            let input_types = inputs.into_types();
            let input_types: Arc<[ArrayIrType]> = input_types.into();
            Ok((input_types.clone(), input_types, None))
        }

        fn lower(&self, staged: &StagedProgram<Self>) -> Result<Self::LoweredProgram, ProgramError> {
            Ok(staged.lifted_program()?.as_ref().clone())
        }

        fn compile(
            &self,
            program: &Self::LoweredProgram,
        ) -> Result<(Arc<Self::CompiledProgram>, Vec<Self::Value>), ProgramError> {
            self.compilations.fetch_add(1, Ordering::Relaxed);
            Ok((Arc::new(program.clone()), Vec::new()))
        }

        fn execute(
            &self,
            program: &Self::CompiledProgram,
            inputs: Vec<ArrayIrValue<Array>>,
        ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
            program.interpret_with(
                inputs,
                |_, capture| {
                    Err(ProgramError::MalformedProgram(format!(
                        "compiled test program retained capture {}",
                        capture.index(),
                    )))
                },
                |instruction, inputs| {
                    instruction.operation().interpret(
                        &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        inputs,
                    )
                },
            )
        }
    }

    /// Call to a staged function, whose results are the results of its callee.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestCallOperation {
        /// Number of leading inputs that hold the lifted captures of the callee.
        capture_count: usize,
    }

    impl Operation for TestCallOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            "call"
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            const { &[RegionSlot::computation("callee")] }
        }

        fn infer_output_types(
            &self,
            _input_types: &[ArrayType],
            region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            Ok(region_interfaces[0].output_types().to_vec())
        }
    }

    /// Operation family of [`TestCallDomain`], which negates values and calls staged functions.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayType), constant(CaptureReference<ArrayType>))]
    enum TestCallFamily {
        Neg(NegOperation<ArrayType>),
        Call(TestCallOperation),
    }

    impl StagedCallOperation<CaptureReference<ArrayType>> for TestCallFamily {
        fn staged_call(capture_count: usize) -> Self {
            Self::Call(TestCallOperation { capture_count })
        }
    }

    /// Compilation domain over [`Array`] values whose operation family can stage calls to staged functions. The tests
    /// that use it only stage programs, so it rejects lowering.
    #[derive(Clone)]
    struct TestCallDomain;

    impl Domain for TestCallDomain {
        type Type = ArrayType;
        type Value = Array;
        type Constant = CaptureReference<ArrayType>;
        type Operation = TestCallFamily;
    }

    impl CompilationDomain for TestCallDomain {
        type DispatchKey = Arc<[ArrayType]>;
        type LoweredProgram = FlatProgram<Self>;
        type CompiledProgram = FlatProgram<Self>;
        type Options = ();
        type Error = ProgramError;

        fn dispatch_signature(
            inputs: DispatchInputs<'_, '_, Self::Value>,
            _options: &(),
        ) -> Result<(Self::DispatchKey, Arc<[ArrayType]>, Option<Self::Options>), ProgramError> {
            let input_types = inputs.into_types();
            let input_types: Arc<[ArrayType]> = input_types.into();
            Ok((input_types.clone(), input_types, None))
        }

        fn lower(&self, _staged: &StagedProgram<Self>) -> Result<FlatProgram<Self>, ProgramError> {
            Err(ProgramError::UnsupportedOperation { message: "the test call domain does not lower programs".into() })
        }

        fn compile(&self, _program: &FlatProgram<Self>) -> Result<(Arc<FlatProgram<Self>>, Vec<Array>), ProgramError> {
            unreachable!("the test call domain never lowers programs")
        }

        fn execute(&self, _program: &FlatProgram<Self>, _inputs: Vec<Array>) -> Result<Vec<Array>, ProgramError> {
            unreachable!("the test call domain never lowers programs")
        }
    }

    /// Compiles a function without inputs that returns its only capture, which is `7.0`.
    fn compile_capture_function(domain: &TestDomain) -> CompiledFunction<TestDomain, (), ArrayType> {
        let staged: StagedFunction<TestDomain, (), ArrayType> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            TestOptions::default(),
        )
        .unwrap();
        staged.lower(domain).unwrap().compile(domain).unwrap()
    }

    /// Assembles a [`CompiledFunction`] of [`TestCallDomain`] from `staged`, whose lowered and compiled programs are
    /// both its lifted program. The test call domain rejects lowering, so this assembles the handle from its parts,
    /// which lets the call tests show that calls on lowered and compiled functions use only their staged source.
    fn assemble_compiled_call_function<Input, Output>(
        staged: StagedFunction<TestCallDomain, Input, Output>,
    ) -> CompiledFunction<TestCallDomain, Input, Output>
    where
        Input: Parameterized<ArrayType, Family: ParameterizedFamily<CaptureReference<ArrayType>>>,
        Output: Parameterized<ArrayType, Family: ParameterizedFamily<CaptureReference<ArrayType>>>,
    {
        let program = Arc::new(staged.lifted_program().unwrap().as_ref().clone());
        CompiledFunction::from_parts(LoweredFunction { program: program.clone(), staged }, program, Vec::new())
    }

    /// Converts a reference completion failure into the error type of the [`StatefulExecution`] tests.
    fn reference_completion_error(reason: Arc<str>) -> ProgramError {
        ProgramError::InvalidArgument { message: reason.to_string() }
    }

    #[test]
    fn test_staged_program_stage() {
        let input_type = ArrayType::scalar(DataType::F64);
        let capture = Array::scalar(7.0).unwrap();
        let invocations = Cell::new(0);
        let staged = StagedProgram::<TestDomain>::stage(
            |_, mut captures, mut inputs| {
                invocations.set(invocations.get() + 1);
                Ok(vec![captures.remove(0), inputs.remove(0)])
            },
            vec![capture.clone()],
            vec![input_type.clone()],
            TestOptions::default(),
        )
        .unwrap();

        assert_eq!(invocations.get(), 1);

        // The flat artifact owns its captures, so any domain instance can lower it.
        assert_eq!(staged.source_program().captures(), &[capture]);
        assert_eq!(staged.input_types(), &[input_type.clone()]);
        assert_eq!(staged.output_types(), &[input_type.clone(), input_type]);
        let lowered = TestDomain::new().lower(&staged).unwrap();
        assert_eq!(lowered.capture_count, 1);
        assert_eq!(lowered.output_types(), staged.output_types());
        assert_eq!(
            lowered.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                in (%0, %1)"},
        );
    }

    #[test]
    fn test_staged_program_stage_propagates_closure_error() {
        let result = StagedProgram::<TestDomain>::stage(
            |_, _, _| Err(ProgramError::InvalidArgument { message: "staging failed".into() }),
            Vec::new(),
            vec![ArrayType::scalar(DataType::F64)],
            TestOptions::default(),
        );

        assert!(matches!(result, Err(ProgramError::InvalidArgument { message }) if message == "staging failed"));
    }

    #[test]
    fn test_staged_program_stage_prunes_unused_captures() {
        let staged = StagedProgram::<TestDomain>::stage(
            |_, _, inputs| Ok(inputs),
            vec![Array::scalar(7.0).unwrap()],
            vec![ArrayType::scalar(DataType::F64)],
            TestOptions::default(),
        )
        .unwrap();

        assert!(staged.source_program().captures().is_empty());
    }

    #[test]
    fn test_staged_program_stage_rejects_escaped_tracers() {
        let retained = RefCell::new(None);
        let result = StagedProgram::<TestDomain>::stage(
            |_, _, mut inputs| {
                *retained.borrow_mut() = Some(inputs.remove(0));
                Ok(Vec::new())
            },
            Vec::new(),
            vec![ArrayType::scalar(DataType::F64)],
            TestOptions::default(),
        );
        assert!(matches!(result, Err(ProgramError::EscapedProgramBuilder)));
    }

    #[test]
    fn test_staged_program_stage_in_scope() {
        // The traced function observes the seeded scope exactly as a function inlined into that scope would.
        let named_axes = vec![("items".to_string(), NamedAxis::Batched { size: Some(3) })];
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let observed = RefCell::new(None);
        let staged = StagedProgram::<TestDomain>::stage_in_scope(
            |_, _, inputs| {
                let context = inputs[0].context();
                *observed.borrow_mut() = Some((context.named_axes(), context.default_mesh()));
                Ok(inputs)
            },
            Vec::new(),
            vec![ArrayType::scalar(DataType::F64)],
            TestOptions::default(),
            TracingScope::new(named_axes.clone(), Some(mesh.clone())),
        )
        .unwrap();
        assert_eq!(observed.into_inner(), Some((named_axes, Some(mesh))));
        assert_eq!(staged.output_types(), &[ArrayType::scalar(DataType::F64)]);
    }

    #[test]
    fn test_staged_function_stage() {
        let domain = TestDomain::new();
        let input_types = (vec![Vec::new(), Vec::new()], ArrayType::scalar(DataType::F64));
        let staged: StagedFunction<TestDomain, (Vec<Vec<ArrayType>>, ArrayType), (Vec<Vec<ArrayType>>, ArrayType)> =
            StagedFunction::stage(
                |_, _, (_, input)| Ok((vec![Vec::new(), Vec::new(), Vec::new()], input)),
                Vec::new(),
                input_types.clone(),
                TestOptions::default(),
            )
            .unwrap();

        assert_eq!(staged.input_structure(), &input_types.parameter_structure());
        assert_eq!(staged.input_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                in (%0)"},
        );
        let executable = staged.lower(&domain).unwrap().compile(&domain).unwrap().into_executable();
        assert_eq!(
            executable.execute(&domain, (vec![Vec::new(), Vec::new()], Array::scalar(3.0).unwrap())).unwrap(),
            (vec![Vec::new(), Vec::new(), Vec::new()], Array::scalar(3.0).unwrap()),
        );
    }

    #[test]
    fn test_staged_function_call() {
        let callee: StagedFunction<TestCallDomain, ArrayType, (ArrayType, ArrayType)> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, input: CompilationTracer<TestCallDomain>| {
                Ok((captures.remove(0), input.unary(NegOperation::new())))
            },
            vec![Array::scalar(7.0).unwrap()],
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();

        // The call registers the captures of the callee in the capture table of the caller and passes them as the
        // leading inputs of the call, before the public inputs.
        let caller: StagedFunction<TestCallDomain, ArrayType, (ArrayType, ArrayType)> =
            StagedFunction::stage(|_, _, input| callee.call(input), Vec::new(), ArrayType::scalar(DataType::F64), ())
                .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[], %3:f64[] = call %1 %0 [
                        callee={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = neg %1
                            in (%0, %2)
                        },
                    ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_staged_function_call_rejects_empty_inputs() {
        let callee: StagedFunction<TestCallDomain, (), ArrayType> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            (),
        )
        .unwrap();
        let caller = StagedFunction::<TestCallDomain, (), ArrayType>::stage(
            |_, _, ()| callee.call::<CompilationTracer<TestCallDomain>, _>(()),
            Vec::new(),
            (),
            (),
        );
        assert!(matches!(caller, Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })));
    }

    #[test]
    fn test_staged_function_call_in_context() {
        // An explicit context supports calls to functions without inputs, from which no context can be recovered.
        let callee: StagedFunction<TestCallDomain, (), ArrayType> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            (),
        )
        .unwrap();
        let caller: StagedFunction<TestCallDomain, ArrayType, ArrayType> = StagedFunction::stage(
            |_, _, input: CompilationTracer<TestCallDomain>| callee.call_in_context(input.context(), ()),
            Vec::new(),
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[] = call %1 [
                        callee={
                            lambda %0:f64[] .
                            in (%0)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_staged_function_call_flat_in_context() {
        // The captures of the callee enter the caller through its context, so the caller captures them and passes them
        // to the call ahead of the flat inputs.
        let callee: StagedFunction<TestCallDomain, ArrayType, ArrayType> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, _input| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();
        let caller = StagedProgram::<TestCallDomain>::stage(
            |_, _, inputs| {
                let context = inputs[0].context().clone();
                callee.call_flat_in_context(&context, inputs)
            },
            Vec::new(),
            vec![ArrayType::scalar(DataType::F64)],
            (),
        )
        .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[] = call %1 %0 [
                        callee={
                            lambda %0:f64[], %1:f64[] .
                            in (%0)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_staged_function_clone() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, ArrayType, ArrayType> = domain
            .stage(
                |input: CompilationTracer<TestDomain>| input.unary(NegOperation::new()),
                ArrayType::scalar(DataType::F64),
                TestOptions::default(),
            )
            .unwrap();
        let staged_clone = staged.clone();
        assert!(std::ptr::eq(staged.options(), staged_clone.options()));
        assert!(Arc::ptr_eq(&staged.lifted_program().unwrap(), &staged_clone.lifted_program().unwrap()));

        let first = staged_clone.lower(&domain).unwrap().compile(&domain).unwrap();
        let second = staged.lower(&domain).unwrap().compile(&domain).unwrap();

        assert_eq!(
            first.executable().execute(&domain, Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(-2.0).unwrap()
        );
        assert_eq!(
            second.executable().execute(&domain, Array::scalar(4.0).unwrap()).unwrap(),
            Array::scalar(-4.0).unwrap()
        );
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(domain.cache.statistics().memory_hits, 1);
    }

    #[test]
    fn test_lowered_function_compile() {
        let domain = TestDomain::new();
        let compile = |compiled_output_types: Vec<ArrayType>| {
            let options = TestOptions { compiled_output_types: Some(compiled_output_types), ..TestOptions::default() };
            let staged: StagedFunction<TestDomain, ArrayType, ArrayType> =
                domain.stage(|input| input, ArrayType::scalar(DataType::F64), options).unwrap();
            staged.lower(&domain).unwrap().compile(&domain).map(|compiled| compiled.output_types().to_vec())
        };

        // The compiled output types must match the lowered output types in count and refine them.
        assert_eq!(compile(vec![ArrayType::scalar(DataType::F64)]).unwrap(), vec![ArrayType::scalar(DataType::F64)]);
        assert!(matches!(compile(Vec::new()), Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })));
        assert!(matches!(
            compile(vec![ArrayType::scalar(DataType::I64)]),
            Err(ProgramError::InvalidArgument { message })
                if message == "output type i64[] does not refine declared type f64[]",
        ));
    }

    #[test]
    fn test_lowered_function_call() {
        let callee: StagedFunction<TestCallDomain, ArrayType, (ArrayType, ArrayType)> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, input: CompilationTracer<TestCallDomain>| {
                Ok((captures.remove(0), input.unary(NegOperation::new())))
            },
            vec![Array::scalar(7.0).unwrap()],
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();
        let callee = assemble_compiled_call_function(callee).lowered().clone();

        // The call stages the same boundary as a call to the staged source, including its capture inputs.
        let caller: StagedFunction<TestCallDomain, ArrayType, (ArrayType, ArrayType)> =
            StagedFunction::stage(|_, _, input| callee.call(input), Vec::new(), ArrayType::scalar(DataType::F64), ())
                .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[], %3:f64[] = call %1 %0 [
                        callee={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = neg %1
                            in (%0, %2)
                        },
                    ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_lowered_function_call_in_context() {
        let callee: StagedFunction<TestCallDomain, (), ArrayType> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            (),
        )
        .unwrap();
        let callee = assemble_compiled_call_function(callee).lowered().clone();

        // An explicit context supports calls to functions without inputs, from which no context can be recovered.
        let caller: StagedFunction<TestCallDomain, ArrayType, ArrayType> = StagedFunction::stage(
            |_, _, input: CompilationTracer<TestCallDomain>| callee.call_in_context(input.context(), ()),
            Vec::new(),
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[] = call %1 [
                        callee={
                            lambda %0:f64[] .
                            in (%0)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_executable_function_with_compiled_program() {
        let domain = TestDomain::new();
        let identity = compile_from_one_call_site(&domain, false).into_executable();
        let negate = compile_from_one_call_site(&domain, true).into_executable();
        let replaced = identity.with_compiled_program(Arc::new(TestCompiledProgram {
            program: negate.compiled_program().program.clone(),
            output_types: negate.output_types().to_vec(),
        }));

        // The replacement runs the new program with the retained signature, while the original function is unaffected.
        assert_eq!(replaced.input_types(), identity.input_types());
        assert_eq!(replaced.execute(&domain, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(-3.0).unwrap());
        assert_eq!(identity.execute(&domain, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(3.0).unwrap());
    }

    #[test]
    fn test_executable_function_execute_statefully() {
        let domain = CarriedDomain::new();
        let staged: StagedFunction<CarriedDomain, ArrayType, ArrayType> = domain
            .stage(
                |input: CompilationTracer<CarriedDomain>| input.unary(NegOperation::new()),
                ArrayType::scalar(DataType::F64),
                CarriedOptions::default(),
            )
            .unwrap();
        let executable = staged.lower(&domain).unwrap().compile(&domain).unwrap().into_executable();
        assert_eq!(executable.execute_statefully(&domain, domain.scalar(2.0)).r#await(), Ok(domain.scalar(-2.0)));
    }

    #[test]
    fn test_executable_function_execute_statefully_and_wait() {
        let domain = CarriedDomain::new();
        let staged: StagedFunction<CarriedDomain, ArrayType, ArrayType> = domain
            .stage(
                |input: CompilationTracer<CarriedDomain>| input.unary(NegOperation::new()),
                ArrayType::scalar(DataType::F64),
                CarriedOptions::default(),
            )
            .unwrap();
        let executable = staged.lower(&domain).unwrap().compile(&domain).unwrap().into_executable();
        assert_eq!(executable.execute_statefully_and_wait(&domain, domain.scalar(2.0)), Ok(domain.scalar(-2.0)));
    }

    #[test]
    fn test_executable_function_validate_input_count() {
        // Hidden captures are not public inputs, so a capture-only function accepts no inputs.
        let executable = compile_capture_function(&TestDomain::new()).into_executable();
        assert_eq!(executable.validate_input_count(0), Ok(()));
        assert_eq!(executable.validate_input_count(1), Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }));
    }

    #[test]
    fn test_executable_function_inputs_with_captures() {
        let executable = compile_capture_function(&TestDomain::new()).into_executable();
        assert_eq!(
            executable.inputs_with_captures(vec![Array::scalar(1.0).unwrap()]),
            vec![Array::scalar(7.0).unwrap(), Array::scalar(1.0).unwrap()],
        );
    }

    #[test]
    fn test_executable_function_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        assert_send_and_sync::<ExecutableFunction<TestDomain, ArrayType, ArrayType>>();

        let domain = TestDomain::new();
        let executable = compile_from_one_call_site(&domain, true).into_executable();
        let second = executable.clone();
        let first_domain = domain.clone();
        let second_domain = domain.clone();
        let first_thread =
            thread::spawn(move || executable.execute(&first_domain, Array::scalar(3.0).unwrap()).unwrap());
        let second_thread = thread::spawn(move || second.execute(&second_domain, Array::scalar(4.0).unwrap()).unwrap());

        assert_eq!(first_thread.join().unwrap(), Array::scalar(-3.0).unwrap());
        assert_eq!(second_thread.join().unwrap(), Array::scalar(-4.0).unwrap());
    }

    #[test]
    fn test_compiled_function() {
        let domain = TestDomain::new();
        let compiled = compile_capture_function(&domain);
        assert_eq!(compiled.executable().execute(&domain, ()).unwrap(), Array::scalar(7.0).unwrap());
        assert_eq!(compiled.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert!(compiled.staged().input_types().is_empty());
        assert_eq!(compiled.lowered().output_types().as_ref(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(compiled.output_types().as_ref(), &[ArrayType::scalar(DataType::F64)]);
    }

    #[test]
    fn test_compiled_function_into_executable() {
        // The runtime handle outlives the transform metadata and still supplies its captures on every call.
        let domain = TestDomain::new();
        let executable = compile_capture_function(&domain).into_executable();
        assert_eq!(executable.captures(), &[Array::scalar(7.0).unwrap()]);
        assert!(executable.input_types().is_empty());
        assert_eq!(executable.output_types().as_ref(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(executable.execute(&domain, ()).unwrap(), Array::scalar(7.0).unwrap());
    }

    #[test]
    fn test_compiled_function_call() {
        let callee: StagedFunction<TestCallDomain, ArrayType, (ArrayType, ArrayType)> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, input: CompilationTracer<TestCallDomain>| {
                Ok((captures.remove(0), input.unary(NegOperation::new())))
            },
            vec![Array::scalar(7.0).unwrap()],
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();
        let callee = assemble_compiled_call_function(callee);

        // The call stages the same boundary as a call to the staged source, including its capture inputs, rather than
        // invoking the retained executable.
        let caller: StagedFunction<TestCallDomain, ArrayType, (ArrayType, ArrayType)> =
            StagedFunction::stage(|_, _, input| callee.call(input), Vec::new(), ArrayType::scalar(DataType::F64), ())
                .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[], %3:f64[] = call %1 %0 [
                        callee={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = neg %1
                            in (%0, %2)
                        },
                    ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_compiled_function_call_in_context() {
        let callee: StagedFunction<TestCallDomain, (), ArrayType> = StagedFunction::stage(
            |_, mut captures: Vec<CompilationTracer<TestCallDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            (),
        )
        .unwrap();
        let callee = assemble_compiled_call_function(callee);

        // An explicit context supports calls to functions without inputs, from which no context can be recovered.
        let caller: StagedFunction<TestCallDomain, ArrayType, ArrayType> = StagedFunction::stage(
            |_, _, input: CompilationTracer<TestCallDomain>| callee.call_in_context(input.context(), ()),
            Vec::new(),
            ArrayType::scalar(DataType::F64),
            (),
        )
        .unwrap();
        assert_eq!(caller.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[] = call %1 [
                        callee={
                            lambda %0:f64[] .
                            in (%0)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_compiled_function_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        // A compiled function owns its lowered function, which owns its staged function, so they are thread-safe too.
        assert_send_and_sync::<CompiledFunction<TestDomain, ArrayType, ArrayType>>();
    }

    #[test]
    fn test_executable_specialization_key() {
        let key: ExecutableSpecializationKey<(&str, usize), Vec<()>, &str> =
            ExecutableSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]");
        assert_eq!(key.static_parameters(), &("training", 4));
        assert_eq!(key.input_structure(), &vec![(), ()]);

        // Every component participates in equality.
        assert_eq!(key, ExecutableSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, ExecutableSpecializationKey::new(("inference", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, ExecutableSpecializationKey::new(("training", 4), vec![()], "f32[2,3]"));
        assert_ne!(key, ExecutableSpecializationKey::new(("training", 4), vec![(), ()], "f32[4,3]"));
    }

    #[test]
    fn test_compiled_function_dispatcher_with_staged_input_types() {
        let domain = TestArrayIrDomain::new();
        let zeros = |(), extent: CompilationTracer<TestArrayIrDomain>| {
            let ArrayIrType::Dimension(extent_type) = extent.r#type().into_owned() else {
                return Err(ProgramError::InvalidArgument { message: "expected a dimension input".to_string() });
            };
            // A static (e.g., exact eager) extent is part of the output type and needs no dimension input.
            let dimension = extent_type.dimension().clone();
            let inputs = if dimension.variable().is_some() { vec![extent.clone()] } else { Vec::new() };
            Ok(extent
                .context()
                .bind(
                    ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![dimension]))),
                    Vec::new(),
                    &inputs,
                )?
                .remove(0))
        };
        let extent_type = DimensionType::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let four = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap());
        let three_zeros = ArrayIrValue::Array(Array::vector(vec![0f32; 3]).unwrap());
        let four_zeros = ArrayIrValue::Array(Array::vector(vec![0f32; 4]).unwrap());

        // Eager dimension values have static types, so by default each runtime extent gets its own specialization,
        // which separately created values of that extent share because static types have no identity.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), ArrayIrType, ArrayIrType, _> =
            domain.jit(zeros);
        assert_eq!(function.execute((), three.clone()), Ok(three_zeros.clone()));
        assert_eq!(function.execute((), four.clone()), Ok(four_zeros.clone()));
        assert_eq!(
            function.execute((), ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
            Ok(three_zeros.clone()),
        );
        assert_eq!(function.executable_specialization_count(), 2);
        assert_eq!(function.statistics().dispatch_hits, 1);
        assert_eq!(domain.compilation_count(), 2);

        // Declaring a symbolic staged type for the input makes calls with different runtime extents share one staged
        // input type, and therefore one retained trace, lowering, and compiled specialization, while still producing
        // outputs with different logical shapes.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), ArrayIrType, ArrayIrType, _> =
            domain.jit(zeros).with_staged_input_types(Some(extent_type.clone().into())).unwrap();
        assert_eq!(function.staged_input_types(), Some(Some(extent_type.clone().into())));
        assert_eq!(function.execute((), three), Ok(three_zeros));
        assert_eq!(function.execute((), four), Ok(four_zeros));
        assert_eq!(function.executable_specialization_count(), 1);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);
        assert_eq!(domain.compilation_count(), 3);

        // An input whose type does not refine its declared staged type is rejected rather than specialized.
        let wide_type = DimensionType::new("wide", DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(
            function.execute((), ArrayIrValue::Dimension(DimensionValue::new(wide_type, 7).unwrap())),
            Err(TypeError::invalid(
                "compilation input `$` has type `dimension<7>`, which does not refine its declared staged type \
                 `dimension<extent ∈ [1, 5)>`",
            )
            .into()),
        );
        assert_eq!(function.executable_specialization_count(), 1);

        // Staged input types must be declared before the dispatcher is shared.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), ArrayIrType, ArrayIrType, _> =
            domain.jit(zeros);
        let _clone = function.clone();
        assert!(matches!(
            function.with_staged_input_types(Some(extent_type.into())),
            Err(ProgramError::InvalidArgument { message })
                if message == "staged input types must be declared before a compilation dispatcher is shared",
        ));
    }

    #[test]
    fn test_compiled_function_dispatcher_with_staged_input_types_shared_identities() {
        let domain = TestArrayIrDomain::new();
        let zeros = |(), extents: Vec<CompilationTracer<TestArrayIrDomain>>| {
            let dimensions = extents
                .iter()
                .map(|extent| match extent.r#type().into_owned() {
                    ArrayIrType::Dimension(extent_type) => Ok(extent_type.dimension().clone()),
                    _ => Err(ProgramError::InvalidArgument { message: "expected a dimension input".to_string() }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Static (e.g., exact eager) extents are part of the output type and need no dimension inputs.
            let inputs = extents
                .iter()
                .zip(&dimensions)
                .filter(|(_, dimension)| dimension.variable().is_some())
                .map(|(extent, _)| extent.clone())
                .collect::<Vec<_>>();
            Ok(extents[0]
                .context()
                .bind(ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(dimensions))), Vec::new(), &inputs)?
                .remove(0))
        };
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let rows = DimensionType::new("rows", bounds);
        let columns = DimensionType::new("columns", bounds);
        let extents = |rows_extent, columns_extent| {
            vec![
                ArrayIrValue::Dimension(DimensionValue::new(rows.clone(), rows_extent).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(columns.clone(), columns_extent).unwrap()),
            ]
        };

        // Distinct declared identities abstract each input independently, so calls that differ only in their runtime
        // extents share one specialization.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), Vec<ArrayIrType>, ArrayIrType, _> = domain
            .jit(zeros)
            .with_staged_input_types(vec![Some(rows.clone().into()), Some(columns.clone().into())])
            .unwrap();
        assert_eq!(
            function.execute((), extents(2, 3)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![0f32; 6]).unwrap())),
        );
        assert_eq!(
            function.execute((), extents(3, 2)),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![0f32; 6]).unwrap())),
        );
        assert_eq!(function.executable_specialization_count(), 1);

        // One declared identity shared by both inputs requires their runtime extents to agree.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), Vec<ArrayIrType>, ArrayIrType, _> = domain
            .jit(zeros)
            .with_staged_input_types(vec![Some(rows.clone().into()), Some(rows.clone().into())])
            .unwrap();
        assert_eq!(
            function.execute((), extents(2, 2)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![0f32; 4]).unwrap())),
        );
        assert_eq!(
            function.execute((), extents(2, 3)),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "rows".to_string(),
                expected: 2,
                actual: 3,
            })
            .into()),
        );
        assert_eq!(function.executable_specialization_count(), 1);

        // A `None` leaf keeps its input exact, so only calls that also agree on that input's extent share one
        // specialization.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), Vec<ArrayIrType>, ArrayIrType, _> =
            domain.jit(zeros).with_staged_input_types(vec![Some(rows.clone().into()), None]).unwrap();
        assert_eq!(function.staged_input_types(), Some(vec![Some(rows.clone().into()), None]));
        for (rows_extent, columns_extent) in [(2, 3), (3, 3), (2, 4)] {
            assert_eq!(
                function.execute((), extents(rows_extent, columns_extent)),
                Ok(ArrayIrValue::Array(
                    Array::matrix(rows_extent, columns_extent, vec![0f32; rows_extent * columns_extent]).unwrap(),
                )),
            );
        }
        assert_eq!(function.executable_specialization_count(), 2);

        // Inputs must have the structure of the declaration, which otherwise would apply its leaves to other inputs.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, (), Vec<ArrayIrType>, ArrayIrType, _> =
            domain.jit(zeros).with_staged_input_types(vec![Some(rows.clone().into())]).unwrap();
        assert!(matches!(
            function.execute((), extents(2, 3)),
            Err(ProgramError::InvalidArgument { message })
                if message == "compilation inputs do not have the structure of their declared staged input types",
        ));
        assert_eq!(function.executable_specialization_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_domain() {
        // An explicit dispatcher reports its domain from construction.
        let domain = CarriedDomain::new();
        let explicit: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            domain.jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input));
        assert_eq!(explicit.domain(), Some(&domain));

        // An inferred dispatcher reports no domain until a call through any clone selects one, which all clones then
        // share and which clearing the cache does not reset.
        let inferred: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input));
        let clone = inferred.clone();
        assert_eq!(inferred.domain(), None);
        assert_eq!(clone.execute((), domain.scalar(1.0)), Ok(domain.scalar(1.0)));
        assert_eq!(inferred.domain(), Some(&domain));
        inferred.clear_cache();
        assert_eq!(clone.domain(), Some(&domain));
    }

    #[test]
    fn test_compiled_function_dispatcher_clear_statistics() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            domain.jit(|(), input: CompilationTracer<TestDomain>| Ok(input));
        assert_eq!(function.execute((), Array::scalar(1.0).unwrap()), Ok(Array::scalar(1.0).unwrap()));

        // Clearing the statistics through any clone resets them for every clone but retains the specializations.
        function.clone().clear_statistics();
        assert_eq!(function.statistics(), DispatcherStatistics::default());
        assert_eq!(function.execute((), Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0).unwrap()));
        let statistics = function.statistics();
        assert_eq!((statistics.dispatch_hits, statistics.dispatch_misses, statistics.traces), (1, 0, 0));
    }

    #[test]
    fn test_compiled_function_dispatcher_clear_cache() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            domain.jit(|(), input: CompilationTracer<TestDomain>| Ok(input));
        assert_eq!(function.execute((), Array::scalar(1.0).unwrap()), Ok(Array::scalar(1.0).unwrap()));

        // Clearing the cache drops the specializations but keeps the statistics, so the next call produces its
        // specialization again, and the compilation cache of the domain serves the repeated compilation request.
        function.clear_cache();
        assert_eq!(function.executable_specialization_count(), 0);
        assert_eq!(function.statistics().traces, 1);
        assert_eq!(function.execute((), Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0).unwrap()));
        let statistics = function.statistics();
        assert_eq!((statistics.dispatch_misses, statistics.traces, statistics.compilation_requests), (2, 2, 2));
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });

        assert_eq!(function.execute(true, Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(-2.0).unwrap());
        assert_eq!(function.execute(true, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(-3.0).unwrap());
        assert_eq!(function.execute(false, Array::scalar(4.0).unwrap()).unwrap(), Array::scalar(4.0).unwrap());
        assert_eq!(function.executable_specialization_count(), 2);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_with_empty_inputs() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, (), (), (), _> = domain.jit(|(), ()| Ok(()));

        assert!(function.domain().is_some());
        assert_eq!(domain.compilation_count(), 0);
        assert_eq!(function.execute((), ()), Ok(()));
        assert_eq!(function.execute((), ()), Ok(()));
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_with_inferred_domain() {
        // Later calls never select another domain, so inputs from another domain execute on the selected one, both on
        // warm calls and on the cold calls that follow cache clearing.
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.execute((), first_domain.scalar(1.0)), Ok(first_domain.scalar(-1.0)));
        assert_eq!(function.execute((), second_domain.scalar(2.0)), Ok(first_domain.scalar(-2.0)));
        function.clear_cache();
        assert_eq!(function.execute((), second_domain.scalar(3.0)), Ok(first_domain.scalar(-3.0)));
        assert_eq!(first_domain.compilation_count(), 2);
        assert_eq!(second_domain.compilation_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_with_inferred_domain_and_empty_inputs() {
        // Empty inputs carry no domain, so they fail before tracing until a non-empty call selects one.
        let domain = CarriedDomain::new();
        let traces = AtomicUsize::new(0);
        let function: CompiledFunctionDispatcher<CarriedDomain, (), Vec<ArrayType>, Vec<ArrayType>, _> =
            jit(|(), inputs: Vec<CompilationTracer<CarriedDomain>>| {
                traces.fetch_add(1, Ordering::Relaxed);
                Ok(inputs)
            });
        assert!(matches!(
            function.execute((), Vec::new()),
            Err(ProgramError::InvalidArgument { message })
                if message == "compilation requires an input leaf to infer a domain; use an explicit domain for \
                               empty inputs",
        ));
        assert_eq!(function.domain(), None);
        assert_eq!(traces.load(Ordering::Relaxed), 0);
        assert_eq!(function.execute((), vec![domain.scalar(1.0)]), Ok(vec![domain.scalar(1.0)]));
        assert_eq!(function.execute((), Vec::new()), Ok(Vec::new()));
        assert_eq!(traces.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_applies_staging_options_before_tracing() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let function: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            domain.jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), options);

        assert!(matches!(
            function.execute((), Array::scalar(2.0).unwrap()),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type f64[] does not refine declared type i64[]",
        ));
        assert_eq!(function.statistics().traces, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_distinguishes_hash_colliding_static_values() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct CollidingStatic(bool);

        impl Hash for CollidingStatic {
            fn hash<H: Hasher>(&self, state: &mut H) {
                0u8.hash(state);
            }
        }

        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, CollidingStatic, ArrayType, ArrayType, _> =
            domain.jit(|static_parameters: CollidingStatic, input: CompilationTracer<TestDomain>| {
                Ok(if static_parameters.0 { input.unary(NegOperation::new()) } else { input })
            });

        assert_eq!(
            function.execute(CollidingStatic(false), Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(2.0).unwrap()
        );
        assert_eq!(
            function.execute(CollidingStatic(true), Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(-2.0).unwrap()
        );
        assert_eq!(function.executable_specialization_count(), 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_partitions_input_structures() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, (), Vec<ArrayType>, ArrayType, _> =
            domain.jit(|(), mut inputs: Vec<CompilationTracer<TestDomain>>| {
                inputs.drain(..1).next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        assert_eq!(function.execute((), vec![Array::scalar(2.0).unwrap()]).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(
            function.execute((), vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]).unwrap(),
            Array::scalar(2.0).unwrap(),
        );
        assert_eq!(function.executable_specialization_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_distinguishes_empty_substructure() {
        // Both calls flatten to zero leaves, so they share static parameters and an empty dispatch signature. Only
        // the input parameter structure distinguishes them, while the traced closure branches on the observed
        // container arity. Key reuse across these calls would therefore silently return the first call's staged
        // behavior for the second call's structurally different input.
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, (), Vec<Vec<ArrayType>>, Vec<ArrayType>, _> =
            domain.jit(|(), inputs: Vec<Vec<CompilationTracer<TestDomain>>>| {
                if inputs.len() == 1 {
                    Ok(inputs.into_iter().next().unwrap())
                } else {
                    Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() })
                }
            });

        assert_eq!(function.execute((), vec![Vec::new()]).unwrap(), Vec::new());
        assert!(matches!(
            function.execute((), Vec::new()),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().traces, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_flattens_map_inputs_deterministically() {
        // Map-like containers flatten in key order, so two runtime maps with the same keys must produce one
        // specialization regardless of construction order, and replay must bind flat inputs to the same leaves.
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, (), BTreeMap<&'static str, ArrayType>, ArrayType, _> =
            domain.jit(|(), inputs: BTreeMap<&'static str, CompilationTracer<TestDomain>>| {
                inputs.into_values().next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        let first = BTreeMap::from([("a", Array::scalar(1.0).unwrap()), ("b", Array::scalar(2.0).unwrap())]);
        let second = BTreeMap::from([("b", Array::scalar(4.0).unwrap()), ("a", Array::scalar(3.0).unwrap())]);
        assert_eq!(function.execute((), first).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(function.execute((), second).unwrap(), Array::scalar(3.0).unwrap());
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_retries_tracing_failures() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(|fail, input: CompilationTracer<TestDomain>| {
                if fail {
                    Err(ProgramError::InvalidArgument { message: "expected trace failure".into() })
                } else {
                    Ok(input)
                }
            });

        for _ in 0..2 {
            assert!(matches!(
                function.execute(true, Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message }) if message == "expected trace failure",
            ));
        }
        assert_eq!(function.execute(false, Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().traces, 3);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_retains_inferred_domain_after_tracing_failure() {
        // The first non-empty call selects the domain before tracing, so a failed trace still fixes it for retries.
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let attempts = AtomicUsize::new(0);
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit(|(), input: CompilationTracer<CarriedDomain>| {
                if attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                    Err(ProgramError::InvalidArgument { message: "expected trace failure".into() })
                } else {
                    Ok(input)
                }
            });
        assert!(matches!(
            function.execute((), first_domain.scalar(1.0)),
            Err(ProgramError::InvalidArgument { message }) if message == "expected trace failure",
        ));
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(function.executable_specialization_count(), 0);
        assert_eq!(function.execute((), second_domain.scalar(2.0)), Ok(first_domain.scalar(2.0)));
        assert_eq!(first_domain.compilation_count(), 1);
        assert_eq!(second_domain.compilation_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_retries_lowering_and_compilation_failures() {
        let lowering_domain = TestDomain::new();
        let lowering_options = TestOptions { fail_lowering: true, ..TestOptions::default() };
        let lowering_function: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            lowering_domain.jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), lowering_options);
        for _ in 0..2 {
            assert!(matches!(
                lowering_function.execute((), Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message }) if message == "expected lowering failure",
            ));
        }
        assert_eq!(lowering_function.executable_specialization_count(), 0);
        let statistics = lowering_function.statistics();
        assert_eq!((statistics.traces, statistics.lowerings, statistics.compilation_requests), (2, 2, 0));

        let compilation_domain = TestDomain::new();
        let compilation_options = TestOptions { fail_compilation: true, ..TestOptions::default() };
        let compilation_function: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            compilation_domain
                .jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), compilation_options);
        for _ in 0..2 {
            assert!(matches!(
                compilation_function.execute((), Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message }) if message == "expected compilation failure",
            ));
        }
        assert_eq!(compilation_function.executable_specialization_count(), 0);
        let statistics = compilation_function.statistics();
        assert_eq!((statistics.traces, statistics.lowerings, statistics.compilation_requests), (2, 2, 2));
        assert_eq!(compilation_domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_retains_specialization_after_execution_failure() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let function: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            domain.jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), options);

        for _ in 0..2 {
            assert!(matches!(
                function.execute((), Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message })
                    if message == "runtime input type f64[] does not refine declared type i64[]",
            ));
        }

        // The compiled entry is inserted before runtime execution, so an execution failure does not evict it and the
        // second call dispatches directly to the retained specialization.
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().traces, 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_rejects_same_key_recursive_dispatch() {
        let domain = TestDomain::new();
        let recursive: Rc<RefCell<Option<Box<dyn Fn() -> Result<Array, ProgramError>>>>> = Rc::new(RefCell::new(None));
        let closure_recursive = recursive.clone();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(move |_, input: CompilationTracer<TestDomain>| {
                if let Some(call) = closure_recursive.borrow().as_ref() {
                    call()?;
                }
                Ok(input)
            });
        let function_clone = function.clone();
        *recursive.borrow_mut() = Some(Box::new(move || function_clone.execute(true, Array::scalar(1.0).unwrap())));

        assert!(matches!(
            function.execute(true, Array::scalar(1.0).unwrap()),
            Err(ProgramError::InvalidArgument { message })
                if message == "recursive compilation dispatch requested a specialization that is already being \
                               produced",
        ));
        assert_eq!(function.executable_specialization_count(), 0);
        assert_eq!(function.statistics().dispatch_misses, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_allows_different_key_recursive_dispatch() {
        let domain = TestDomain::new();
        let recursive: Rc<RefCell<Option<Box<dyn Fn() -> Result<Array, ProgramError>>>>> = Rc::new(RefCell::new(None));
        let closure_recursive = recursive.clone();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(move |nested, input: CompilationTracer<TestDomain>| {
                if nested && let Some(call) = closure_recursive.borrow().as_ref() {
                    call()?;
                }
                Ok(input)
            });
        let function_clone = function.clone();
        *recursive.borrow_mut() = Some(Box::new(move || function_clone.execute(false, Array::scalar(1.0).unwrap())));

        assert_eq!(function.execute(true, Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(function.executable_specialization_count(), 2);
        assert_eq!(function.statistics().traces, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_isolates_dispatchers() {
        let domain = TestDomain::new();
        let first: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            domain.jit(|(), input: CompilationTracer<TestDomain>| Ok(input));
        let second: CompiledFunctionDispatcher<TestDomain, (), ArrayType, ArrayType, _> =
            domain.jit(|(), input: CompilationTracer<TestDomain>| Ok(input));

        assert_eq!(first.execute((), Array::scalar(1.0).unwrap()).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(second.execute((), Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(first.executable_specialization_count(), 1);
        assert_eq!(second.executable_specialization_count(), 1);
        assert_eq!(first.statistics().traces, 1);
        assert_eq!(second.statistics().traces, 1);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_concurrent_warm_calls() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });
        function.execute(true, Array::scalar(1.0).unwrap()).unwrap();

        thread::scope(|scope| {
            for index in 0..4 {
                let function = function.clone();
                scope.spawn(move || {
                    let value = f64::from(index);
                    assert_eq!(
                        function.execute(true, Array::scalar(value).unwrap()).unwrap(),
                        Array::scalar(-value).unwrap()
                    );
                });
            }
        });

        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 4);
        assert_eq!(function.statistics().traces, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_concurrent_cold_misses() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });

        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            for _ in 0..2 {
                let function = function.clone();
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    assert_eq!(
                        function.execute(true, Array::scalar(3.0).unwrap()).unwrap(),
                        Array::scalar(-3.0).unwrap()
                    );
                });
            }
        });

        // Cross-thread same-key cold misses deliberately race and may duplicate frontend production; both inserts are
        // idempotent and exactly one retained entry remains, while the shared compilation context deduplicates the
        // backend compilation itself.
        assert_eq!(function.executable_specialization_count(), 1);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits + statistics.dispatch_misses, 2);
        assert!(statistics.traces >= 1 && statistics.traces <= 2);
        assert_eq!(statistics.traces, statistics.dispatch_misses);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_concurrent_domain_inference() {
        // Racing first calls through clones with inputs from different domains select one domain, which then executes
        // both calls.
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input));

        let barrier = Barrier::new(2);
        let (first_output, second_output) = thread::scope(|scope| {
            let first_call = scope.spawn(|| {
                barrier.wait();
                function.clone().execute((), first_domain.scalar(1.0))
            });
            let second_call = scope.spawn(|| {
                barrier.wait();
                function.clone().execute((), second_domain.scalar(2.0))
            });
            (first_call.join().unwrap(), second_call.join().unwrap())
        });

        // Same-key cold misses may still duplicate production (refer to the concurrent cold misses test), but only on
        // the selected domain.
        let selected_domain = function.domain().unwrap();
        let unselected_domain = if selected_domain == &first_domain { &second_domain } else { &first_domain };
        assert!(selected_domain == &first_domain || selected_domain == &second_domain);
        assert_eq!(first_output, Ok(selected_domain.scalar(1.0)));
        assert_eq!(second_output, Ok(selected_domain.scalar(2.0)));
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(selected_domain.compilation_count() as u64, function.statistics().traces);
        assert_eq!(unselected_domain.compilation_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_statefully() {
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.execute_statefully((), first_domain.scalar(1.0)).r#await(), Ok(first_domain.scalar(-1.0)));
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(function.execute_statefully((), second_domain.scalar(2.0)).r#await(), Ok(first_domain.scalar(-2.0)));
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_execute_statefully_and_wait() {
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, (), ArrayType, ArrayType, _> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.execute_statefully_and_wait((), first_domain.scalar(1.0)), Ok(first_domain.scalar(-1.0)));
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(function.execute_statefully_and_wait((), second_domain.scalar(2.0)), Ok(first_domain.scalar(-2.0)));
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call() {
        let function: CompiledFunctionDispatcher<TestCallDomain, (), ArrayType, ArrayType, _> =
            TestCallDomain.jit(|(), input: CompilationTracer<TestCallDomain>| Ok(input.unary(NegOperation::new())));
        let stage_caller = || -> StagedFunction<TestCallDomain, ArrayType, ArrayType> {
            StagedFunction::stage(
                |_, _, input: CompilationTracer<TestCallDomain>| function.call((), input),
                Vec::new(),
                ArrayType::scalar(DataType::F64),
                (),
            )
            .unwrap()
        };

        // A call binds the traced closure as a callee. The test call domain rejects lowering, so the call
        // succeeding also shows that calls neither lower nor compile anything.
        let caller = stage_caller();
        assert_eq!(
            caller.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = call %0 [
                    callee={
                        lambda %0:f64[] .
                        let %1:f64[] = neg %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (0, 1, 1));
        assert_eq!((statistics.dispatch_misses, statistics.lowerings, statistics.compilation_requests), (0, 0, 0));

        // Another call with the same types and scope reuses the staged specialization.
        stage_caller();
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (1, 1, 1));

        // Clearing the cache removes staged specializations too.
        function.clear_cache();
        stage_caller();
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (1, 2, 2));
    }

    #[test]
    fn test_compiled_function_dispatcher_call_in_context() {
        // The closure records the named axes that each trace sees, which come from the context that calls it.
        let observed = Rc::new(RefCell::new(Vec::new()));
        let function: CompiledFunctionDispatcher<TestCallDomain, (), ArrayType, ArrayType, _> = TestCallDomain.jit({
            let observed = observed.clone();
            move |(), input: CompilationTracer<TestCallDomain>| {
                observed.borrow_mut().push(input.context().named_axes());
                Ok(input.unary(NegOperation::new()))
            }
        });
        let named_axes = vec![("items".to_string(), NamedAxis::Batched { size: Some(3) })];
        let stage_caller = |named_axes: Vec<(String, NamedAxis)>| {
            StagedFunction::<TestCallDomain, ArrayType, ArrayType>::stage_in_scope(
                |_, _, input: CompilationTracer<TestCallDomain>| {
                    let context = input.context().clone();
                    function.call_in_context(&context, (), input)
                },
                Vec::new(),
                ArrayType::scalar(DataType::F64),
                (),
                TracingScope::new(named_axes, None),
            )
            .unwrap()
        };

        // Each calling scope gets its own staged specialization, traced in that scope, and repeated scopes reuse it.
        stage_caller(named_axes.clone());
        stage_caller(Vec::new());
        stage_caller(named_axes.clone());
        assert_eq!(observed.borrow().as_slice(), &[named_axes, Vec::new()]);
        let statistics = function.statistics();
        assert_eq!((statistics.staged_cache_hits, statistics.staged_cache_misses, statistics.traces), (1, 2, 2));

        // Invalidation removes the staged specializations of every scope and counts them.
        assert_eq!(function.invalidate_specializations(&()), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_invalidate_specializations() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });
        function.execute(true, Array::scalar(2.0).unwrap()).unwrap();
        function.execute(false, Array::scalar(2.0).unwrap()).unwrap();

        assert_eq!(function.invalidate_specializations(&true), 1);
        assert_eq!(function.executable_specialization_count(), 1);
        function.execute(true, Array::scalar(2.0).unwrap()).unwrap();
        assert_eq!(function.statistics().traces, 3);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_send_and_sync() {
        // The dispatcher type names its closure type, which only a value can supply.
        fn assert_send_and_sync<T: Send + Sync>(_value: &T) {}

        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, bool, ArrayType, ArrayType, _> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });
        assert_send_and_sync(&function);
    }

    #[test]
    fn test_stateful_execution_ready() {
        // A ready execution carries no completion at all, so awaiting it just yields the stored result.
        let execution: StatefulExecution<u32, ProgramError> = StatefulExecution::ready(Ok(7));
        assert_eq!(execution.r#await(), Ok(7));

        let execution: StatefulExecution<u32, ProgramError> =
            StatefulExecution::ready(Err(ProgramError::MalformedProgram("specialization failed".to_string())));
        let expected = Err(ProgramError::MalformedProgram("specialization failed".to_string()));
        assert_eq!(execution.r#await(), expected);
    }

    #[test]
    fn test_stateful_execution_pending() {
        // A successful completion never rewrites the reconstructed public output.
        let execution: StatefulExecution<u32, ProgramError> =
            StatefulExecution::pending(Ok(11), ReferenceCompletion::ready(Ok(())), reference_completion_error);
        assert_eq!(execution.r#await(), Ok(11));
    }

    #[test]
    fn test_stateful_execution_await_prioritizes_completion_failure() {
        // A completion failure may invalidate every pending output, so it replaces the reconstruction error.
        let execution: StatefulExecution<(), ProgramError> = StatefulExecution::pending(
            Err(ProgramError::MalformedProgram("public reconstruction failed".to_string())),
            ReferenceCompletion::ready(Err("execution failed".into())),
            reference_completion_error,
        );
        let expected = Err(ProgramError::InvalidArgument { message: "execution failed".to_string() });
        assert_eq!(execution.r#await(), expected);
    }

    #[test]
    fn test_stateful_execution_and_then() {
        let execution: StatefulExecution<u32, ProgramError> = StatefulExecution::ready(Ok(7));
        assert_eq!(execution.and_then(|output| Ok(output + 1)).r#await(), Ok(8));

        // The mapped execution keeps the completion, whose failure still replaces the mapped output.
        let execution: StatefulExecution<u32, ProgramError> = StatefulExecution::pending(
            Ok(7),
            ReferenceCompletion::ready(Err("execution failed".into())),
            reference_completion_error,
        );
        assert_eq!(
            execution.and_then(|output| Ok(output + 1)).r#await(),
            Err(ProgramError::InvalidArgument { message: "execution failed".to_string() }),
        );
    }
}
