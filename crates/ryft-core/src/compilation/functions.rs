use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::captures::{CapturingContext, ClosedProgram};
use crate::compilation::caching::accumulate_duration;
use crate::compilation::contexts::{CompilationDomain, StatefulCompilationDomain};
use crate::compilation::operations::StagedCallOperation;
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::operations::Constant;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::transforms::{Transform, TransformCache};
use crate::programs::{
    AttachedRegions, FlatProgram, ProgramError, ReferenceCompletion, Type, TypeError, TypeRefinements, Typed, Value,
    validate_reference_boundary,
};
use crate::specialization::SpecializationCacheEntry;
use crate::tracing::{DomainTracingContext, Tracer};

/// Tracer used while staging a compiled function in the operation universe of `C`.
pub type CompilationTracer<C> = Tracer<DomainTracingContext<C, <C as Domain>::Value>>;

/// Cache key identifying one specialization of a retained function. Cache reuse is authorized only when all
/// three of the following components agree:
///
///   - **Static Parameters:** The host values the traced function may branch on, read, or embed as literals. Unequal
///     static parameters can stage arbitrarily different programs, so they must separate specializations. Static
///     parameters must be `Clone + Debug + Eq + Hash`. Runtime arrays and other backend values should be represented
///     as dynamic inputs rather than static parameters.
///   - **Input Structure:** The [`Parameterized::ParameterStructure`] shape of the dynamic function input. Tracing
///     rebuilds the function's argument from this structure, so a function may legitimately branch on container arity.
///     Keying on the structure rather than on the flattened leaves also distinguishes inputs that differ only in
///     _empty_ substructure, which flat leaf paths and flat leaf types cannot see.
///   - **Dispatch Key:** An owner-defined key used to select an interchangeable retained specialization for the dynamic
///     inputs. It may be an exact abstract input signature or a normalized equivalence-class key such as a shape
///     bucket. Equal dispatch keys must guarantee that the retained artifact can safely serve either call, while
///     unequal keys conservatively separate specializations. Retained Just-In-Time (JIT) compilation dispatch uses
///     [`CompilationDomain::DispatchKey`], for example.
///
/// Equal keys must identify calls that can safely share the same staged [`Program`](crate::Program). Unequal keys
/// remain separate even if they happen to stage structurally equivalent programs.
///
/// Everything else that affects staging (e.g., the closure itself, its captures, the domain, any fixed options, etc.)
/// is implicit in the cache's owner, because a cache is scoped to exactly one retained callable. That is why this key
/// carries no fragile function-pointer identity. All three components are call-level (i.e., function-level) concepts,
/// since a bare program has neither static parameters nor a structured input. That is what separates function-level
/// specialization caching from structural region transformation through [`Transform`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionSpecializationKey<P, I, K> {
    /// Host values declared static for the keyed specialization, such as an axis number, mode enum, or boolean option.
    static_parameters: P,

    /// Parameter structure of the dynamic input, such as the shape of a nested tuple or named parameter container.
    input_structure: I,

    /// Owner-defined cache key for the dynamic input, such as an exact abstract signature or normalized shape bucket.
    dispatch_key: K,
}

impl<P, I, K> FunctionSpecializationKey<P, I, K> {
    /// Creates a new [`FunctionSpecializationKey`].
    #[inline]
    pub fn new(static_parameters: P, input_structure: I, dispatch_key: K) -> Self {
        Self { static_parameters, input_structure, dispatch_key }
    }

    /// Returns the static parameters (i.e., the host values declared static) for the keyed specialization.
    /// Refer to the documentation of [`FunctionSpecializationKey`] for more information.
    #[inline]
    pub fn static_parameters(&self) -> &P {
        &self.static_parameters
    }

    /// Returns the parameter structure of the dynamic input for the keyed specialization.
    /// Refer to the documentation of [`FunctionSpecializationKey`] for more information.
    #[inline]
    pub fn input_structure(&self) -> &I {
        &self.input_structure
    }

    /// Returns the owner-defined dispatch key for the dynamic inputs.
    /// Refer to the documentation of [`FunctionSpecializationKey`] for more information.
    #[inline]
    pub fn dispatch_key(&self) -> &K {
        &self.dispatch_key
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Owned flat staging artifact constructed by [`Self::stage`] and borrowed by [`CompilationDomain::lower`].
///
/// This artifact retains the source program, runtime captures, flat public signature, and compilation options.
/// [`StagedFunction`] pairs it with the Rust input and output container structures used by typed callers. Captures
/// remain symbolic constants in the source program and become leading inputs in [`Self::lifted_program`].
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
    /// lowering and compilation. On success, this constructor invokes `function` exactly once. Typed callers typically
    /// use [`StagedFunction::stage`] instead.
    ///
    /// # Parameters
    ///
    ///   - `_domain`: Domain whose type, constant, operation, and runtime value families the trace uses.
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
        _domain: &C,
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
        // Captured references must name distinct runtime allocations. Validate them before replacing concrete values
        // with symbolic capture references, while their runtime identities are still available. Public inputs are
        // abstract types here, so only captures can be checked at this boundary.
        validate_reference_boundary(std::iter::empty(), captures.iter()).map_err(ProgramError::from)?;

        // A fresh context isolates this trace. Runtime captures stay in its side table, while the program stores typed
        // references to their slots. The closure receives both those constant payloads and their tracers.
        let context = DomainTracingContext::<C, C::Value>::new();
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

/// Staged, unlowered form of one compiled function.
///
/// A staged function is the first callable package in the lifecycle. It pairs an owned [`StagedProgram`] with the
/// exact public input and output container structures that typed calls need. It can be inspected, lowered by its
/// domain, or embedded as a staged-call boundary in an enclosing trace. No backend lowering or executable compilation
/// has happened yet.
pub struct StagedFunction<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Shared immutable staging metadata. Staged-handle clones are therefore constant-time even for large programs.
    state: Arc<StagedFunctionState<C, Input, Output>>,
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

impl<C, Input, Output> Clone for StagedFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

impl<C, Input, Output> StagedFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Traces a fallible typed `function` through `domain`, retaining its exact input and output container structures.
    /// The function receives symbolic capture constants, capture tracers, and structured input tracers. Runtime
    /// captures and compilation options are owned by the resulting [`StagedProgram`].
    ///
    /// # Parameters
    ///
    ///   - `domain`: Domain whose flat staging primitive records the function.
    ///   - `function`: Fallible closure to trace once with symbolic captures and inputs.
    ///   - `captures`: Runtime values to register in the source program's capture table.
    ///   - `input_types`: Structured abstract public input signature.
    ///   - `options`: Options retained for lowering and compilation.
    ///
    /// # Errors
    ///
    /// Returns errors from tracing or reconstructing the structured inputs.
    pub fn stage<F>(
        domain: &C,
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
        // Flat arities cannot recover nested empty containers or other structural metadata. Keep the input structure
        // before flattening, and record the output structure while the closure still owns its structured results.
        let input_structure = input_types.parameter_structure();
        let mut output_structure = None;
        let program = StagedProgram::stage(
            domain,
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

    /// Reconstructs the structured abstract input signature retained by this function.
    pub fn input_signature(&self) -> Result<Input, ParameterError>
    where
        Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Type, To = Input>, To<C::Type> = Input>,
    {
        Input::from_parameters(self.state.input_structure.clone(), self.input_types().iter().cloned())
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

    /// Stages a call to this function into the active context recovered from `inputs`.
    ///
    /// For nullary functions use [`Self::call_in_context`], because no input value exists from which to recover the
    /// active context.
    pub fn call<V>(&self, inputs: Input::To<V>) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        V::Domain: Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
            + CapturingContext<Capture = C::Value>
            + Constant<V, C::Constant>,
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

    /// Stages a call to this function through an explicitly supplied active `context`.
    pub fn call_in_context<Caller, V>(
        &self,
        context: &Caller,
        inputs: Input::To<V>,
    ) -> Result<Output::To<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        Caller: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + CapturingContext<Capture = C::Value>
            + Constant<V, C::Constant>,
        Input: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Input::To<V>: Parameterized<V>,
        Output: Parameterized<C::Type, Family: ParameterizedFamily<V>>,
        Output::To<V>: Parameterized<V, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
    {
        let capture_references = self
            .state
            .program
            .source_program()
            .captures()
            .iter()
            .cloned()
            .map(|capture| context.capture(capture))
            .collect::<Result<Vec<_>, _>>()?;
        let capture_count = capture_references.len();
        let mut flat_inputs = capture_references
            .into_iter()
            .map(|capture| context.constant(capture))
            .collect::<Result<Vec<_>, _>>()?;
        flat_inputs.extend(inputs.into_parameters());
        let callee = self.callee_program()?;
        let operation =
            C::Operation::staged_call_with_types(capture_count, &callee.input_types(), &callee.output_types())?;
        let outputs = context.bind(operation, AttachedRegions::Callees(&[callee]), flat_inputs.as_slice())?;
        Output::To::<V>::from_parameters(self.state.output_structure.clone(), outputs).map_err(Into::into)
    }

    /// Binds this staged function using capture references already registered in an enclosing capture table.
    pub fn call_with_flat_capture_references<V>(
        &self,
        capture_references: &[C::Constant],
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        V::Domain: Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation> + Constant<V, C::Constant>,
    {
        let context = inputs
            .first()
            .map(|input| input.domain())
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        self.call_with_flat_capture_references_in_context(&context, capture_references, inputs)
    }

    /// Binds this staged function through `context` using already-registered flat capture references.
    ///
    /// Unlike [`Self::call_with_flat_capture_references`], this form also supports nullary and capture-only calls,
    /// because it does not need a public input from which to recover the active context.
    pub fn call_with_flat_capture_references_in_context<Caller, V>(
        &self,
        context: &Caller,
        capture_references: &[C::Constant],
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        C::Operation: StagedCallOperation<C::Constant>,
        V: Value<Type = C::Type>,
        Caller: Context<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>
            + Constant<V, C::Constant>,
    {
        // Capture-reference indices belong to the caller's capture table and may differ from the source program's
        // local indices, and so only positional arity and types can be validated against the source program's
        // encapsulated capture table.
        let captures = self.source_program().captures();
        check_count!("input", capture_references, captures.len(), ProgramError);
        for (index, (expected, actual)) in captures.iter().zip(capture_references).enumerate() {
            let expected_type = expected.r#type();
            let actual_type = actual.r#type();
            if expected_type.as_ref() != actual_type.as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "capture input #{index} has type {actual_type}, but capture #{index} has type {expected_type}",
                )));
            }
        }
        let mut flat_inputs = capture_references
            .iter()
            .cloned()
            .map(|capture| context.constant(capture))
            .collect::<Result<Vec<_>, _>>()?;
        flat_inputs.extend(inputs);
        let callee = self.callee_program()?;
        let operation =
            C::Operation::staged_call_with_types(captures.len(), &callee.input_types(), &callee.output_types())?;
        context.bind(operation, AttachedRegions::Callees(&[callee]), flat_inputs.as_slice())
    }

    /// Returns the lifted program of this function (refer to [`Self::lifted_program`]) for embedding as the callee of
    /// a nested call. Reference discharge cannot publish a reference that a callee allocates itself, because the caller
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

    /// Returns the source program with its runtime captures lifted into leading flat inputs, in the
    /// `[captures..., public inputs...]` order that compiled executables are invoked with. Backends lower this program
    /// rather than the [`ClosedProgram`] returned by [`Self::source_program`], and use the capture count of the source
    /// program to tell lifted captures apart from public inputs. The source program itself has a flat boundary, while
    /// this function retains its exact container structures separately.
    ///
    /// The program is derived once and then cached on the shared staged state, so repeated calls, including calls on
    /// clones of this function, return the same [`Arc`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if rebuilding the source program with explicit capture inputs fails.
    pub fn lifted_program(&self) -> Result<Arc<FlatProgram<C>>, ProgramError> {
        self.state.program.lifted_program()
    }

    /// Lowers this function into a [`LoweredFunction`] through `domain` (refer to [`CompilationDomain::lower`]).
    pub fn lower(&self, domain: &C) -> Result<LoweredFunction<C, Input, Output>, C::Error> {
        let (lowered_program, output_types) = domain.lower(self.staged_program())?;
        Ok(LoweredFunction { program: Arc::new(lowered_program), staged: self.clone(), output_types })
    }
}

/// Backend lowering of a [`StagedFunction`], ready for executable compilation.
///
/// It stays a function rather than a program, because the lowered program of the backend is only one of the artifacts
/// it carries, alongside the captures, signatures, and options inherited from its staged source.
pub struct LoweredFunction<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Backend-owned lowered program.
    program: Arc<C::LoweredProgram>,

    /// Staged source and structured metadata retained across compilation.
    staged: StagedFunction<C, Input, Output>,

    /// Effective flat output types after lowering-time rewrites.
    output_types: Vec<C::Type>,
}

impl<C, Input, Output> Clone for LoweredFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), staged: self.staged.clone(), output_types: self.output_types.clone() }
    }
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

    /// Returns the effective flat output types after lowering-time rewrites.
    #[inline]
    pub fn output_types(&self) -> &[C::Type] {
        &self.output_types
    }

    /// Compiles this function into a [`CompiledFunction`] through `domain` (refer to [`CompilationDomain::compile`]).
    ///
    /// # Errors
    ///
    /// Returns the error of `domain`, or a [`ProgramError`] if the compiled output types differ in count from the
    /// lowered output types of this function or do not refine them.
    pub fn compile(&self, domain: &C) -> Result<CompiledFunction<C, Input, Output>, C::Error> {
        let (program, output_types) = domain.compile(self.lowered_program())?;
        check_count!("output", output_types, self.output_types.len(), ProgramError);
        for (declared, actual) in self.output_types.iter().zip(&output_types) {
            if !declared.is_refined_by(actual) {
                return Err(ProgramError::InvalidArgument {
                    message: format!("output type {actual} does not refine declared type {declared}"),
                }
                .into());
            }
        }
        Ok(CompiledFunction::from_parts(self.clone(), program, output_types))
    }
}

/// Runtime-only handle for one compiled executable.
///
/// This handle retains only the state required to validate and execute calls, namely the backend executable, concrete
/// captures, flat signatures, and structured output shape. That call-level state is what still makes it a function
/// rather than a program, even though it has shed every transformable artifact. It deliberately does not retain the
/// staged or lowered programs, so it becomes [`Send`] and [`Sync`] automatically whenever those runtime fields are
/// `Send + Sync`. In contrast, [`CompiledFunction`] remains transformable and may retain `Rc`-backed source programs.
pub struct ExecutableFunction<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>> {
    /// Shared immutable runtime state, which makes clones constant-time.
    state: Arc<ExecutableFunctionState<C, Input, Output>>,
}

/// Shared runtime state of an [`ExecutableFunction`].
struct ExecutableFunctionState<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>> {
    /// Shared backend executable.
    program: Arc<C::CompiledProgram>,

    /// Concrete runtime captures supplied before the public inputs on every call.
    captures: Vec<C::Value>,

    /// Flat declared public input types, excluding hidden captures.
    input_types: Vec<C::Type>,

    /// Effective flat output types produced by the executable.
    output_types: Vec<C::Type>,

    /// Output parameter structure used to reconstruct structured runtime values.
    output_structure: Output::ParameterStructure,

    /// Structured input type marker.
    input: PhantomData<fn(Input)>,
}

impl<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>> Clone
    for ExecutableFunction<C, Input, Output>
{
    #[inline]
    fn clone(&self) -> Self {
        Self { state: Arc::clone(&self.state) }
    }
}

impl<C: CompilationDomain, Input: Parameterized<C::Type>, Output: Parameterized<C::Type>>
    ExecutableFunction<C, Input, Output>
{
    /// Returns a function that invokes `program` instead of this function's current compiled program, while keeping
    /// this function's runtime captures, flat input types, and output parameter structure. Backends use it to swap in
    /// an equivalent executable, such as one deserialized from a persistent cache or recompiled with profile guidance.
    ///
    /// This function does not check compatibility, so the caller must establish it first. `program` must accept the
    /// same `[captures..., public inputs...]` inputs as the current program, and `output_types` must be its
    /// effective flat output types, with one entry per leaf of the retained output structure. A backend that breaks
    /// these invariants produces a function whose calls fail or return values of unexpected types. The existing
    /// function and its clones are unaffected, because this creates new shared state.
    ///
    /// # Parameters
    ///
    ///   - `program`: Replacement backend executable.
    ///   - `output_types`: Effective flat output types of `program`.
    pub fn with_compiled_program(&self, program: Arc<C::CompiledProgram>, output_types: Vec<C::Type>) -> Self {
        Self {
            state: Arc::new(ExecutableFunctionState {
                program,
                captures: self.state.captures.clone(),
                input_types: self.state.input_types.clone(),
                output_types,
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

    /// Returns the flat declared public input types, excluding hidden captures.
    #[inline]
    pub fn input_types(&self) -> &[C::Type] {
        &self.state.input_types
    }

    /// Returns the effective flat output types produced by the executable.
    #[inline]
    pub fn output_types(&self) -> &[C::Type] {
        &self.state.output_types
    }

    /// Returns the output parameter structure used to reconstruct structured runtime values.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.state.output_structure
    }

    /// Validates that `inputs` matches the declared flat public input arity, excluding hidden captures.
    pub fn validate_flat_input_count(&self, inputs: &[C::Value]) -> Result<(), ProgramError> {
        if inputs.len() != self.state.input_types.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.state.input_types.len(),
                actual: inputs.len(),
            });
        }
        Ok(())
    }

    /// Prepends the retained runtime captures to the flat public `inputs`, producing the complete flat input
    /// list in the `[captures..., public inputs...]` order every execution expects.
    pub fn inputs_with_captures(&self, inputs: Vec<C::Value>) -> Vec<C::Value> {
        if self.state.captures.is_empty() {
            return inputs;
        }
        let mut runtime_inputs = self.state.captures.to_vec();
        runtime_inputs.extend(inputs);
        runtime_inputs
    }
}

impl<C, Input, Output> ExecutableFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Value>>,
    Output::To<C::Value>:
        Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
{
    /// Reconstructs structured runtime outputs from the executable's flat `outputs` using the retained output
    /// parameter structure.
    pub fn reconstruct_outputs(&self, outputs: Vec<C::Value>) -> Result<Output::To<C::Value>, ProgramError> {
        Output::To::<C::Value>::from_parameters(self.state.output_structure.clone(), outputs).map_err(Into::into)
    }
}

impl<C, Input, Output> ExecutableFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Value>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Value>>,
    Input::To<C::Value>: Parameterized<C::Value>,
    Output::To<C::Value>:
        Parameterized<C::Value, Family = Output::Family, ParameterStructure = Output::ParameterStructure>,
{
    /// Executes this function on the structured runtime `inputs` through `domain` and returns its structured outputs.
    ///
    /// Functions that bind external mutable references must be executed through [`Self::call_statefully`] or
    /// [`Self::call_statefully_async`] instead.
    pub fn call(&self, domain: &C, inputs: Input::To<C::Value>) -> Result<Output::To<C::Value>, C::Error> {
        let inputs = self.prepare_inputs(inputs)?;
        let outputs = domain.execute(self.compiled_program(), inputs)?;
        self.reconstruct_outputs(outputs).map_err(C::Error::from)
    }

    /// Executes this function on the structured runtime `inputs` through the stateful `domain` and returns a result
    /// that carries the completion of the whole invocation.
    ///
    /// The returned completion owns error observation but not cancellation. The semantics of submission and of the
    /// publication of reference state are defined by [`StatefulCompilationDomain::execute_statefully`].
    pub fn call_statefully_async(
        &self,
        domain: &C,
        inputs: Input::To<C::Value>,
    ) -> StatefulExecution<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
    {
        let inputs = match self.prepare_inputs(inputs) {
            Ok(inputs) => inputs,
            Err(error) => return StatefulExecution::ready(Err(error.into())),
        };
        domain
            .execute_statefully(self.compiled_program(), inputs)
            .and_then(|outputs| self.reconstruct_outputs(outputs).map_err(C::Error::from))
    }

    /// Executes this function on the structured runtime `inputs` through the stateful `domain` and waits for the whole
    /// invocation to complete, which makes it the blocking form of [`Self::call_statefully_async`].
    pub fn call_statefully(&self, domain: &C, inputs: Input::To<C::Value>) -> Result<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
    {
        self.call_statefully_async(domain, inputs).r#await()
    }

    /// Flattens the structured runtime `inputs`, validates their count, and prepends the runtime captures to them.
    fn prepare_inputs(&self, inputs: Input::To<C::Value>) -> Result<Vec<C::Value>, ProgramError> {
        let inputs = inputs.into_parameters().collect::<Vec<_>>();
        self.validate_flat_input_count(&inputs)?;
        Ok(self.inputs_with_captures(inputs))
    }
}

/// Compiled executable plus the staged and lowered metadata required to invoke it safely.
///
/// The compiled program of the backend is the artifact this function carries, while the captures, structured
/// signatures, and options carried with it are what make the handle callable as well as transformable.
pub struct CompiledFunction<
    C: CompilationDomain,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> {
    /// Runtime-only executable handle.
    executable_function: ExecutableFunction<C, Input, Output>,

    /// Lowered function retaining source metadata and options.
    lowered: LoweredFunction<C, Input, Output>,
}

impl<C, Input, Output> Clone for CompiledFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    fn clone(&self) -> Self {
        Self { executable_function: self.executable_function.clone(), lowered: self.lowered.clone() }
    }
}

impl<C, Input, Output> CompiledFunction<C, Input, Output>
where
    C: CompilationDomain,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Assembles a compiled function from backend-validated parts.
    fn from_parts(
        lowered: LoweredFunction<C, Input, Output>,
        program: Arc<C::CompiledProgram>,
        output_types: Vec<C::Type>,
    ) -> Self {
        let executable_function = ExecutableFunction {
            state: Arc::new(ExecutableFunctionState {
                program,
                captures: lowered.staged.source_program().captures().to_vec(),
                input_types: lowered.staged.input_types().to_vec(),
                output_types,
                output_structure: lowered.staged.output_structure().clone(),
                input: PhantomData,
            }),
        };
        Self { executable_function, lowered }
    }

    /// Returns the shared backend executable.
    #[inline]
    pub fn compiled_program(&self) -> &C::CompiledProgram {
        self.executable_function.compiled_program()
    }

    /// Returns the runtime-only handle, which omits staged and lowered transform metadata.
    #[inline]
    pub fn executable_function(&self) -> &ExecutableFunction<C, Input, Output> {
        &self.executable_function
    }

    /// Consumes this transformable handle and returns its runtime-only executable state.
    #[inline]
    pub fn into_executable_function(self) -> ExecutableFunction<C, Input, Output> {
        self.executable_function
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

    /// Returns the effective flat output types produced by the executable.
    #[inline]
    pub fn output_types(&self) -> &[C::Type] {
        self.executable_function.output_types()
    }
}

/// Retained JIT dispatcher that caches compiled specializations of one Rust closure.
///
/// Unlike [`CompiledFunction`], which represents one executable that is already specialized, a
/// `CompiledFunctionDispatcher` holds no program of its own. It is a retained dispatcher over a Rust closure that
/// produces one program, and one function around that program, per specialization from explicit static host
/// parameters and dynamic runtime inputs. Its first call for a specialization stages, lowers, and requests
/// compilation, and later calls with the same static values, input parameter structure, and dispatch key dispatch
/// directly to the retained executable. Domain dispatch signatures may normalize distinct runtime signatures to the
/// same staged signature, which can produce harmless duplicate dispatch specializations.
///
/// [`Compile::jit`](crate::Compile::jit) selects the domain explicitly. The global [`jit`](crate::jit) instead recovers
/// it from the first leaf of the first nonempty call. That first attempt binds the domain even if staging or
/// compilation later fails. All clones, later calls, and specializations use the selected domain, including after cache
/// clearing. Concurrent first calls atomically select one domain and all use that same selection. The dispatcher does
/// not recover a domain from the inputs of later calls, so inputs that originate in another domain still execute on the
/// selected one. Before selection, a call with no input leaves fails because there is no value from which to recover a
/// domain. Use an explicit domain for nullary functions.
///
/// Tracing executes Rust host code only on a specialization miss. Host side effects inside `function` therefore run
/// once per retained specialization rather than once per runtime call, and observable work for every call must be
/// represented by staged effectful operations. Static values with high cardinality can cause repeated tracing and LRU
/// eviction, so arrays and frequently changing data should remain dynamic inputs.
///
/// Every runtime input is specialized at its own type by default, and eager values are typed exactly (e.g., an eager
/// array by its static shape and an eager first-class dimension by its extent), so each distinct extent gets its own
/// specialization. [`Self::with_staged_input_types`] instead declares the types at which selected inputs are staged
/// (e.g., `dimension<n ∈ [1, 9)>` for a dimension input), so that one specialization serves every runtime input whose
/// type refines them.
///
/// The dispatcher identity only namespaces its specialization cache, which is local to the process. For domains that
/// cache compiled programs, executable correctness and reuse still depend on
/// [`CompilationCacheDomain::compilation_key`](super::contexts::CompilationCacheDomain::compilation_key), which
/// receives the complete lowering.
///
/// Cloned handles share one specialization cache, and the dispatcher is usable from multiple threads whenever the
/// domain, closure, static parameters, and domain artifact types are thread-safe, because [`Send`] and [`Sync`] derive
/// structurally from those field types. Recursive dispatch on one thread of the specialization currently being
/// produced is rejected with an error. Concurrent cold misses for one specialization on different threads instead
/// deliberately duplicate the frontend work, because staging and lowering are cheap and inserts are idempotent, while
/// the shared [`CompilationContext`](super::caching::CompilationContext) of the domain still coordinates the expensive
/// backend compilation.
///
/// # Retained JIT Lifecycle
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   call["&lt;code&gt;CompiledFunctionDispatcher::call&lt;/code&gt; with Static Parameters and Dynamic Inputs"]
///   call --> key["Derive Function Specialization Key"]
///   key --> frontend_cache["Per-Function Bounded Specialization Cache"]
///   frontend_cache -->|"hit"| executable["Shared &lt;code&gt;ExecutableFunction&lt;/code&gt;"]
///   frontend_cache -->|"miss"| closure["Invoke Rust Closure"]
///   closure -->|"stage"| staged["&lt;code&gt;StagedFunction&lt;/code&gt; and Closed Source Program"]
///   staged -->|"lower"| lowered["&lt;code&gt;LoweredFunction&lt;/code&gt; and Backend Lowered Program"]
///   staged -.->|"&lt;code&gt;call&lt;/code&gt; inside an outer trace"| nested["Nested Call Operation"]
///   lowered --> compilation_context["Shared &lt;code&gt;CompilationContext&lt;/code&gt;"]
///   compilation_context -->|"restore or compile"| compiled["&lt;code&gt;CompiledFunction&lt;/code&gt;"]
///   compiled -->|"&lt;code&gt;into_executable_function&lt;/code&gt;"| executable
///   executable -->|"insert after cold production"| frontend_cache
///   executable -->|"call"| outputs["Structured Runtime Outputs"]
/// ```
///
/// The cache hit edge skips staging, lowering, and compilation. The dotted edge is the alternative staging path, along
/// which a staged function becomes a nested call boundary instead of continuing immediately toward an executable.
#[cfg_attr(doc, ryft_macros::mermaid)]
pub struct CompiledFunctionDispatcher<
    C: CompilationDomain<Type: Eq + Hash>,
    F,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> where
    Input::ParameterStructure: Eq + Hash,
{
    /// State shared by every clone of this dispatcher.
    state: Arc<CompiledFunctionDispatcherState<C, F, Static, Input, Output>>,
}

impl<C, F, Static: Clone + Debug + Eq + Hash, Input, Output> Clone
    for CompiledFunctionDispatcher<C, F, Static, Input, Output>
where
    C: CompilationDomain,
    C::Type: Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

impl<C, F, Static: Clone + Debug + Eq + Hash, Input, Output> CompiledFunctionDispatcher<C, F, Static, Input, Output>
where
    C: CompilationDomain,
    C::Type: Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::Family: ParameterizedFamily<C::Constant>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
    Output::Family: ParameterizedFamily<C::Constant>,
{
    /// Creates a dispatcher for `function` that retains at most `capacity` specializations.
    pub(crate) fn new(domain: &C, function: F, options: C::Options, capacity: usize) -> Self {
        Self::from_domain(DispatcherDomain::Explicit(domain.clone()), function, options, capacity)
    }

    /// Creates a dispatcher that selects its domain from its first nonempty invocation.
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

    /// Creates the shared dispatcher state with an explicit or deferred domain selection.
    fn from_domain(domain: DispatcherDomain<C>, function: F, options: C::Options, capacity: usize) -> Self {
        Self {
            state: Arc::new(CompiledFunctionDispatcherState {
                domain,
                function,
                options,
                staged_input_types: None,
                specializations: TransformCache::<
                    JitCompilationTransform,
                    JitCompilationSource<C, Static, Input, Output>,
                >::new(capacity),
                statistics: JitCacheStatisticsAccumulator::default(),
            }),
        }
    }

    /// Returns this dispatcher with the types at which its runtime inputs are staged (which also determine their
    /// dispatch keys and compiled signatures) declared in the structure of its inputs, as JAX's `abstracted_axes`
    /// does. A `Some` leaf stages its input at that type instead of at the input's own type, so that one
    /// specialization serves every runtime input whose type refines it, and a `None` leaf keeps its input exact (as in
    /// [`Parameterized::replace_parameters`]).
    /// For example, declaring `Some(dimension<n ∈ [1, 9)>)` for a first-class dimension input reuses one
    /// specialization for every extent from 1 to 8. Inputs declared with one shared identity (e.g., two `dimension<n>`
    /// inputs, or a `dimension<n>` input and an `f32[n]` input) must agree at runtime. Every call validates its inputs
    /// against the declared types before dispatching, and rejects a violation (including inputs whose structure
    /// differs from the declaration) rather than producing another specialization. The declared types are created
    /// once by the caller, so their identities, and thus the dispatch keys, are stable across calls.
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
            message: "staged input types must be declared before a jit dispatcher is shared".to_string(),
        })?;
        state.staged_input_types = Some(StagedInputTypes {
            structure: staged_input_types.parameter_structure(),
            types: staged_input_types.into_parameters().collect(),
        });
        Ok(self)
    }

    /// Returns the compilation domain used by this dispatcher, or [`None`] while a global dispatcher's domain has
    /// not yet been recovered from a nonempty call.
    #[inline]
    pub fn domain(&self) -> Option<&C> {
        match &self.state.domain {
            DispatcherDomain::Explicit(domain) => Some(domain),
            DispatcherDomain::Inferred { domain, .. } => domain.get(),
        }
    }

    /// Returns the declared staged input types of this dispatcher (refer to [`Self::with_staged_input_types`]), or
    /// [`None`] when every input is staged at its own type.
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

    /// Returns the number of compiled specializations currently retained by this dispatcher.
    #[inline]
    pub fn specialization_count(&self) -> usize {
        self.state.specializations.len()
    }

    /// Returns the maximum number of compiled specializations retained by this dispatcher.
    #[inline]
    pub fn cache_capacity(&self) -> usize {
        self.state.specializations.capacity()
    }

    /// Returns a snapshot of this dispatcher's cache activity.
    #[inline]
    pub fn statistics(&self) -> JitCacheStatistics {
        self.state.statistics.snapshot()
    }

    /// Resets dispatch statistics without clearing compiled specializations.
    #[inline]
    pub fn clear_statistics(&self) {
        self.state.statistics.clear();
    }

    /// Clears every retained specialization without changing statistics.
    #[inline]
    pub fn clear_cache(&self) {
        self.state.specializations.clear();
    }

    /// Invalidates every retained specialization for `static_parameters` and returns the number removed.
    pub fn invalidate_static(&self, static_parameters: &Static) -> usize {
        self.state.specializations.invalidate_entries_if(|key| key.static_parameters() == static_parameters)
    }

    /// Calls this dispatcher with explicit host-side `static_parameters` and dynamic runtime `inputs`.
    pub fn call(&self, static_parameters: Static, inputs: Input::To<C::Value>) -> Result<Output::To<C::Value>, C::Error>
    where
        C::Options: Clone,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
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
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (domain, executable) = self.specialization(static_parameters, &inputs)?;
        executable.call(domain, inputs)
    }

    /// Calls this dispatcher through its domain's completion-bearing stateful execution capability.
    ///
    /// Submission publishes the domain's reference dependencies before this function returns. The returned completion
    /// must be awaited to observe asynchronous failure, and dropping it does not cancel the invocation.
    pub fn call_statefully_async(
        &self,
        static_parameters: Static,
        inputs: Input::To<C::Value>,
    ) -> StatefulExecution<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
        C::Options: Clone,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
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
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (domain, executable) = match self.specialization(static_parameters, &inputs) {
            Ok(specialization) => specialization,
            Err(error) => return StatefulExecution::ready(Err(error)),
        };
        executable.call_statefully_async(domain, inputs)
    }

    /// Calls this dispatcher statefully and waits for whole-invocation completion.
    ///
    /// This routes through [`ExecutableFunction::call_statefully`], so it reports asynchronous execution errors
    /// before returning and does not weaken the domain's reference sequencing contract.
    pub fn call_statefully(
        &self,
        static_parameters: Static,
        inputs: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, C::Error>
    where
        C: StatefulCompilationDomain,
        C::Options: Clone,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
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
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (domain, executable) = self.specialization(static_parameters, &inputs)?;
        executable.call_statefully(domain, inputs)
    }

    /// Acquires or produces the executable specialization for one runtime input signature and returns it together with
    /// the resolved domain that executes it.
    fn specialization(
        &self,
        static_parameters: Static,
        inputs: &Input::To<C::Value>,
    ) -> Result<(&C, ExecutableFunction<C, Input, Output>), C::Error>
    where
        C::Options: Clone,
        F: Fn(Static, Input::To<CompilationTracer<C>>) -> Result<Output::To<CompilationTracer<C>>, C::Error>,
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
        DomainTracingContext<C, C::Value>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let abstractification_start = Instant::now();
        let domain = self.resolve_domain(inputs)?;
        let input_structure = inputs.parameter_structure();
        let runtime_input_types = inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let runtime_input_types = self.staged_runtime_input_types(&input_structure, runtime_input_types)?;
        let (dispatch_key, effective_input_types) =
            domain.dispatch_signature(runtime_input_types, &self.state.options)?;
        let key = FunctionSpecializationKey::new(static_parameters.clone(), input_structure, dispatch_key);

        // One clock read both ends input abstractification (including key construction) and starts the dispatch
        // lookup, which keeps the timing of every cached call to three clock reads.
        let statistics = &self.state.statistics;
        let dispatch_start = Instant::now();
        accumulate_duration(
            &statistics.input_abstractification_nanoseconds,
            dispatch_start.duration_since(abstractification_start),
        );
        let entry = self.state.specializations.try_entry(key);
        accumulate_duration(&statistics.dispatch_nanoseconds, dispatch_start.elapsed());
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
                    message: "recursive JIT dispatch requested a specialization that is already being produced".into(),
                }
                .into());
            }
        };

        let input_types =
            Input::from_parameters(producer.key().input_structure().clone(), effective_input_types.iter().cloned())
                .map_err(ProgramError::from)?;
        statistics.traces.fetch_add(1, Ordering::Relaxed);
        let tracing_start = Instant::now();
        let staged = StagedFunction::stage(
            domain,
            |_, _, traced_inputs| (self.state.function)(static_parameters, traced_inputs),
            Vec::new(),
            input_types,
            self.state.options.clone(),
        );
        accumulate_duration(&statistics.tracing_nanoseconds, tracing_start.elapsed());
        let staged = staged?;
        statistics.lowerings.fetch_add(1, Ordering::Relaxed);
        let lowering_start = Instant::now();
        let lowered = staged.lower(domain);
        accumulate_duration(&statistics.lowering_nanoseconds, lowering_start.elapsed());
        let lowered = lowered?;
        statistics.compilation_requests.fetch_add(1, Ordering::Relaxed);
        let compiled = lowered.compile(domain)?;
        Ok((domain, producer.insert(compiled.into_executable_function())))
    }

    /// Resolves this dispatcher's domain before producing or executing any specialization.
    fn resolve_domain(&self, inputs: &Input::To<C::Value>) -> Result<&C, C::Error>
    where
        Input::Family: ParameterizedFamily<C::Value>,
    {
        let (domain, from_input) = match &self.state.domain {
            DispatcherDomain::Explicit(domain) => return Ok(domain),
            DispatcherDomain::Inferred { domain, from_input } => (domain, from_input),
        };
        if let Some(domain) = domain.get() {
            return Ok(domain);
        }
        let input = inputs.parameters().next().ok_or_else(|| ProgramError::InvalidArgument {
            message: "compilation requires an input leaf to infer a domain; use an explicit domain for empty inputs"
                .to_string(),
        })?;
        // Recovery runs inside initialization so racing clones cannot produce or execute against a losing domain.
        // The binding remains fixed after failures and cache eviction because it owns every retained specialization.
        Ok(domain.get_or_init(|| from_input(input)))
    }

    /// Validates `input_types`, the flattened runtime input types of inputs with structure `input_structure`, against
    /// the declared staged input types (refer to [`Self::with_staged_input_types`]) and replaces every declared one by
    /// its staged type.
    fn staged_runtime_input_types(
        &self,
        input_structure: &Input::ParameterStructure,
        mut input_types: Vec<C::Type>,
    ) -> Result<Vec<C::Type>, C::Error> {
        let Some(StagedInputTypes { structure, types }) = &self.state.staged_input_types else {
            return Ok(input_types);
        };
        if structure != input_structure {
            return Err(ProgramError::InvalidArgument {
                message: "jit inputs do not have the structure of their declared staged input types".to_string(),
            }
            .into());
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
                    "jit input `{path}` has type `{}`, which does not refine its declared staged type `{staged_type}`",
                    input_types[index],
                )))
                .into());
            }
        }
        // Inputs that share an identity in their declared types must agree, which establishing the facts of all of
        // them together checks.
        <C::Type as Type>::Refinements::new(
            declared.iter().map(|(_, staged_type)| *staged_type),
            declared.iter().map(|(index, _)| &input_types[*index]),
        )
        .map_err(ProgramError::from)?;
        for (index, staged_type) in declared {
            input_types[index] = staged_type.clone();
        }
        Ok(input_types)
    }
}

/// Shared state of a [`CompiledFunctionDispatcher`].
struct CompiledFunctionDispatcherState<
    C: CompilationDomain<Type: Eq + Hash>,
    F,
    Static: Clone + Debug + Eq + Hash,
    Input: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
    Output: Parameterized<C::Type, Family: ParameterizedFamily<C::Constant>>,
> where
    Input::ParameterStructure: Eq + Hash,
{
    /// Compilation domain that stages, lowers, compiles, and executes every specialization.
    domain: DispatcherDomain<C>,

    /// Rust closure traced once per specialization.
    function: F,

    /// Compilation options fixed for every specialization.
    options: C::Options,

    /// Declared staged input types, or [`None`] when every input is staged at its own type.
    staged_input_types: Option<StagedInputTypes<C::Type, Input::ParameterStructure>>,

    /// Bounded LRU cache of compiled specializations.
    specializations: TransformCache<JitCompilationTransform, JitCompilationSource<C, Static, Input, Output>>,

    /// Dispatch activity counters.
    statistics: JitCacheStatisticsAccumulator,
}

/// Explicit or lazily recovered domain shared by one dispatcher's specializations and clones.
enum DispatcherDomain<C: CompilationDomain> {
    /// Domain selected when the dispatcher was constructed.
    Explicit(C),

    /// Domain recovered once from a runtime input, with the recovery function available only for compatible values.
    Inferred {
        /// Domain selected atomically by the first call with an input leaf.
        domain: OnceLock<C>,

        /// Recovers the compilation domain carried by one runtime input.
        from_input: fn(&C::Value) -> C,
    },
}

/// Staged input types declared for a [`CompiledFunctionDispatcher`] (refer to
/// [`CompiledFunctionDispatcher::with_staged_input_types`]), stored flattened so that the dispatcher needs no
/// `Option`-leaved instantiation of its input family outside of the functions that declare and return them.
struct StagedInputTypes<T, Structure> {
    /// Parameter structure of the declared inputs.
    structure: Structure,

    /// Declared staged type of each flattened input, or [`None`] for an input that is staged at its own type.
    types: Vec<Option<T>>,
}

/// Completion-bearing result of one externally stateful compiled invocation.
///
/// The public result may be reconstructed immediately from pending backend values, but [`Self::await`] always observes
/// the whole invocation completion before exposing that result. Completion includes every predecessor dependency of
/// the references used by the invocation, not only the most recently submitted backend event. Dropping this wrapper
/// never cancels submitted work or rolls reference state back, so callers that need to observe asynchronous failure
/// must await it. A backend may also record a failure that happens after submission on the affected mutated
/// references, so that later accesses report the same cause for their poisoned state.
#[must_use = "stateful execution errors are observed by awaiting this completion"]
pub struct StatefulExecution<Output, Error> {
    /// Public output or reconstruction failure prepared after submission.
    result: Result<Output, Error>,

    /// Whole-invocation completion and its backend error conversion, absent only for already-completed results.
    completion: Option<(ReferenceCompletion, fn(Arc<str>) -> Error)>,
}

impl<Output, Error> StatefulExecution<Output, Error> {
    /// Creates an already-completed stateful invocation result, which returns `result` without waiting when awaited.
    /// Backends use it when a stateful call finishes synchronously or fails before submitting any work.
    ///
    /// A backend must not use this function once work that reads or replaces a
    /// [`Reference`](crate::programs::Reference) has been submitted, because the result would then hide the completion
    /// of that work and any asynchronous failure that it reports. Use [`Self::pending`] for submitted work instead.
    #[inline]
    pub fn ready(result: Result<Output, Error>) -> Self {
        Self { result, completion: None }
    }

    /// Creates a submitted invocation whose public result is already reconstructed but whose execution remains
    /// pending. Completion failure takes precedence over `result` because it may invalidate every pending output.
    ///
    /// Backends must pass a `completion` that covers the whole invocation, which usually means
    /// [`ReferenceCompletion::joined`] over the submitted work and the completions that the invocation's references
    /// already depended on. A completion that only tracks the latest backend event would let awaiting the result
    /// return before an earlier reference write has finished or reported its failure.
    ///
    /// # Parameters
    ///
    ///   - `result`: Public output or reconstruction failure prepared right after submission.
    ///   - `completion`: Whole-invocation completion, including every cumulative reference predecessor dependency.
    ///   - `completion_error`: Converts a backend completion failure message into the domain error type. It is applied
    ///     only when the completion itself fails, and its result then replaces `result`.
    pub fn pending(
        result: Result<Output, Error>,
        completion: ReferenceCompletion,
        completion_error: fn(Arc<str>) -> Error,
    ) -> Self {
        Self { result, completion: Some((completion, completion_error)) }
    }

    /// Waits for the whole execution and its cumulative dependencies, then returns the public output.
    pub fn r#await(self) -> Result<Output, Error> {
        if let Some((completion, completion_error)) = self.completion
            && let Err(error) = completion.r#await()
        {
            return Err(completion_error(error));
        }
        self.result
    }

    /// Applies `function` to the public output of this invocation without changing its completion.
    fn and_then<T>(self, function: impl FnOnce(Output) -> Result<T, Error>) -> StatefulExecution<T, Error> {
        StatefulExecution { result: self.result.and_then(function), completion: self.completion }
    }
}

/// Snapshot of the dispatch activity of one retained [`CompiledFunctionDispatcher`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct JitCacheStatistics {
    /// Calls served by an already compiled specialization.
    pub dispatch_hits: u64,

    /// Calls for which no compiled specialization was retained.
    pub dispatch_misses: u64,

    /// Closure traces performed after dispatch misses.
    pub traces: u64,

    /// Backend lowerings performed after new traces.
    pub lowerings: u64,

    /// Compilation requests made to the domain after lowering.
    pub compilation_requests: u64,

    /// Total host duration spent flattening inputs and deriving their dispatch key and effective signature.
    pub input_abstractification_duration: Duration,

    /// Total host duration spent looking up retained specializations.
    pub dispatch_duration: Duration,

    /// Total host duration spent tracing specialization misses.
    pub tracing_duration: Duration,

    /// Total host duration spent lowering newly traced specializations.
    pub lowering_duration: Duration,
}

/// Lock-free counters underlying [`JitCacheStatistics`]. Durations accumulate in nanoseconds.
#[derive(Default)]
struct JitCacheStatisticsAccumulator {
    /// Counter underlying [`JitCacheStatistics::dispatch_hits`].
    dispatch_hits: AtomicU64,

    /// Counter underlying [`JitCacheStatistics::dispatch_misses`].
    dispatch_misses: AtomicU64,

    /// Counter underlying [`JitCacheStatistics::traces`].
    traces: AtomicU64,

    /// Counter underlying [`JitCacheStatistics::lowerings`].
    lowerings: AtomicU64,

    /// Counter underlying [`JitCacheStatistics::compilation_requests`].
    compilation_requests: AtomicU64,

    /// Nanoseconds underlying [`JitCacheStatistics::input_abstractification_duration`].
    input_abstractification_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`JitCacheStatistics::dispatch_duration`].
    dispatch_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`JitCacheStatistics::tracing_duration`].
    tracing_nanoseconds: AtomicU64,

    /// Nanoseconds underlying [`JitCacheStatistics::lowering_duration`].
    lowering_nanoseconds: AtomicU64,
}

impl JitCacheStatisticsAccumulator {
    /// Returns a snapshot of the counters. The counters are read independently, so a snapshot taken during concurrent
    /// calls need not correspond to a single point in time.
    fn snapshot(&self) -> JitCacheStatistics {
        JitCacheStatistics {
            dispatch_hits: self.dispatch_hits.load(Ordering::Relaxed),
            dispatch_misses: self.dispatch_misses.load(Ordering::Relaxed),
            traces: self.traces.load(Ordering::Relaxed),
            lowerings: self.lowerings.load(Ordering::Relaxed),
            compilation_requests: self.compilation_requests.load(Ordering::Relaxed),
            input_abstractification_duration: Duration::from_nanos(
                self.input_abstractification_nanoseconds.load(Ordering::Relaxed),
            ),
            dispatch_duration: Duration::from_nanos(self.dispatch_nanoseconds.load(Ordering::Relaxed)),
            tracing_duration: Duration::from_nanos(self.tracing_nanoseconds.load(Ordering::Relaxed)),
            lowering_duration: Duration::from_nanos(self.lowering_nanoseconds.load(Ordering::Relaxed)),
        }
    }

    /// Resets every counter to zero.
    fn clear(&self) {
        self.dispatch_hits.store(0, Ordering::Relaxed);
        self.dispatch_misses.store(0, Ordering::Relaxed);
        self.traces.store(0, Ordering::Relaxed);
        self.lowerings.store(0, Ordering::Relaxed);
        self.compilation_requests.store(0, Ordering::Relaxed);
        self.input_abstractification_nanoseconds.store(0, Ordering::Relaxed);
        self.dispatch_nanoseconds.store(0, Ordering::Relaxed);
        self.tracing_nanoseconds.store(0, Ordering::Relaxed);
        self.lowering_nanoseconds.store(0, Ordering::Relaxed);
    }
}

/// Default number of compiled specializations retained by one [`CompiledFunctionDispatcher`].
pub(crate) const DEFAULT_JIT_CACHE_CAPACITY: usize = 256;

/// Type-level source universe for retained JIT specialization.
///
/// The owning [`CompiledFunctionDispatcherState`] fixes the closure identity, lexical captures, domain instance, and
/// immutable compilation options. This token stores none of them and only makes the key and artifact families of the
/// descriptor coherent at the type level. If any of those owner semantics becomes variable per call, it must enter the
/// transform inputs instead.
struct JitCompilationSource<C, StaticParameters, Input, Output>(
    PhantomData<fn() -> (C, StaticParameters, Input, Output)>,
);

/// Transform descriptor for the executable specializations retained by one [`CompiledFunctionDispatcher`].
struct JitCompilationTransform;

impl<C, StaticParameters, Input, Output> Transform<JitCompilationSource<C, StaticParameters, Input, Output>>
    for JitCompilationTransform
where
    C: CompilationDomain,
    StaticParameters: Clone + Eq + Hash,
    Input: Parameterized<C::Type>,
    Input::ParameterStructure: Eq + Hash,
    Output: Parameterized<C::Type>,
{
    type Arguments = FunctionSpecializationKey<StaticParameters, Input::ParameterStructure, C::DispatchKey>;
    type Artifact = ExecutableFunction<C, Input, Output>;

    const DEFAULT_CACHE_CAPACITY: usize = DEFAULT_JIT_CACHE_CAPACITY;
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::fmt::{Display, Formatter};
    use std::hash::{Hash, Hasher};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionError, DimensionType, DimensionValue, Shape,
    };
    use crate::captures::CaptureReference;
    use crate::compilation::caching::CompilationContext;
    use crate::compilation::contexts::{CompilationCacheDomain, Compile, jit};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::{NegOperation, ZeroOperation};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{EmptyRegionDriver, Operation, Type, TypeError, ValueDirectDispatch};

    use super::*;

    #[derive(Clone)]
    pub(crate) struct TestLoweredProgram {
        program: FlatProgram<TestDomain>,
        capture_count: usize,
        output_types: Vec<ArrayType>,
        options: TestOptions,
    }

    pub(crate) struct TestCompiledProgram {
        program: FlatProgram<TestDomain>,
        output_types: Vec<ArrayType>,
    }

    #[derive(Clone, Debug, Default)]
    pub(crate) struct TestOptions {
        staged_input_type: Option<ArrayType>,
        fail_lowering: bool,
        fail_compilation: bool,
        compiled_output_types: Option<Vec<ArrayType>>,
    }

    #[derive(Clone)]
    pub(crate) struct TestDomain {
        cache: Arc<CompilationContext<Self>>,
        compilations: Arc<AtomicUsize>,
    }

    impl TestDomain {
        pub(crate) fn new() -> Self {
            Self { cache: Arc::new(CompilationContext::new()), compilations: Arc::new(AtomicUsize::new(0)) }
        }

        pub(crate) fn compilation_count(&self) -> usize {
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

        fn lower(&self, staged: &StagedProgram<Self>) -> Result<(Self::LoweredProgram, Vec<ArrayType>), Self::Error> {
            let program = staged.lifted_program()?;
            let capture_count = staged.source_program().captures().len();
            let output_types = staged.output_types();
            let options = staged.options();
            if options.fail_lowering {
                return Err(ProgramError::InvalidArgument { message: "expected lowering failure".into() });
            }
            let program = TestLoweredProgram {
                program: program.as_ref().clone(),
                capture_count,
                output_types: output_types.to_vec(),
                options: options.clone(),
            };
            Ok((program, output_types.to_vec()))
        }

        fn compile(
            &self,
            program: &Self::LoweredProgram,
        ) -> Result<(Arc<Self::CompiledProgram>, Vec<ArrayType>), Self::Error> {
            let program = self.cache.compile(self, program, |program| {
                self.compilations.fetch_add(1, Ordering::Relaxed);
                if program.options.fail_compilation {
                    return Err(ProgramError::InvalidArgument { message: "expected compilation failure".into() });
                }
                let output_types =
                    program.options.compiled_output_types.clone().unwrap_or_else(|| program.output_types.clone());
                Ok(TestCompiledProgram { program: program.program.clone(), output_types })
            })?;
            let output_types = program.output_types.clone();
            Ok((program, output_types))
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
    }

    /// Minimal array IR compilation domain used to prove the retained-JIT contract over dimension inputs: it
    /// stages through the ordinary tracing path, "lowers" and "compiles" to the lifted flat program itself, counts
    /// backend compilations, and executes calls by eager interpretation of the compiled program.
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
            &self,
            input_types: Vec<ArrayIrType>,
            _options: &Self::Options,
        ) -> Result<(Self::DispatchKey, Arc<[ArrayIrType]>), Self::Error> {
            let input_types: Arc<[ArrayIrType]> = input_types.into();
            Ok((input_types.clone(), input_types))
        }

        fn lower(
            &self,
            staged: &StagedProgram<Self>,
        ) -> Result<(Self::LoweredProgram, Vec<ArrayIrType>), ProgramError> {
            Ok((staged.lifted_program()?.as_ref().clone(), staged.output_types().to_vec()))
        }

        fn compile(
            &self,
            program: &Self::LoweredProgram,
        ) -> Result<(Arc<Self::CompiledProgram>, Vec<ArrayIrType>), ProgramError> {
            self.compilations.fetch_add(1, Ordering::Relaxed);
            let output_types = program.output_types();
            Ok((Arc::new(program.clone()), output_types))
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
                        "retained-JIT test program retained capture {}",
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

    /// Compilation domain whose runtime values carry it, which lets the global entry points recover it from their
    /// inputs. It lowers and compiles to the lifted program itself, counts compilations, and executes programs by
    /// interpreting every instruction as a negation, which is the only operation that these tests stage.
    #[derive(Clone, Debug)]
    pub(crate) struct CarriedDomain {
        /// Number of backend compilations, shared by every clone of this domain.
        compilations: Arc<AtomicUsize>,
    }

    impl CarriedDomain {
        pub(crate) fn new() -> Self {
            Self { compilations: Arc::new(AtomicUsize::new(0)) }
        }

        pub(crate) fn compilation_count(&self) -> usize {
            self.compilations.load(Ordering::Relaxed)
        }

        pub(crate) fn scalar(&self, value: f64) -> CarriedArray {
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

        fn lower(&self, staged: &StagedProgram<Self>) -> Result<(Self::LoweredProgram, Vec<ArrayType>), ProgramError> {
            Ok((staged.lifted_program()?.as_ref().clone(), staged.output_types().to_vec()))
        }

        fn compile(
            &self,
            program: &Self::LoweredProgram,
        ) -> Result<(Arc<Self::CompiledProgram>, Vec<ArrayType>), ProgramError> {
            self.compilations.fetch_add(1, Ordering::Relaxed);
            Ok((Arc::new(program.clone()), program.output_types()))
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
    pub(crate) struct CarriedOptions {
        /// Arbitrary marker that distinguishes explicit options from the default ones.
        pub(crate) marker: usize,
    }

    /// [`Array`] that carries the [`CarriedDomain`] in which it was produced.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct CarriedArray {
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

    pub(crate) fn compile_from_one_call_site(
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

    fn reference_completion_error(reason: Arc<str>) -> ProgramError {
        ProgramError::InvalidArgument { message: reason.to_string() }
    }

    #[test]
    fn test_function_specialization_key() {
        let key: FunctionSpecializationKey<(&str, usize), Vec<()>, &str> =
            FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]");
        assert_eq!(key.static_parameters(), &("training", 4));
        assert_eq!(key.input_structure(), &vec![(), ()]);
        assert_eq!(key.dispatch_key(), &"f32[2,3]");

        // Every component participates in equality.
        assert_eq!(key, FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("inference", 4), vec![(), ()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("training", 4), vec![()], "f32[2,3]"));
        assert_ne!(key, FunctionSpecializationKey::new(("training", 4), vec![(), ()], "f32[4,3]"));
    }

    #[test]
    fn test_staged_program_stage() {
        let input_type = ArrayType::scalar(DataType::F64);
        let capture = Array::scalar(7.0).unwrap();
        let invocations = Cell::new(0);
        let staged = {
            let domain = TestDomain::new();
            StagedProgram::stage(
                &domain,
                |_, mut captures, mut inputs| {
                    invocations.set(invocations.get() + 1);
                    Ok(vec![captures.remove(0), inputs.remove(0)])
                },
                vec![capture.clone()],
                vec![input_type.clone()],
                TestOptions::default(),
            )
            .unwrap()
        };

        assert_eq!(invocations.get(), 1);

        // The flat artifact owns its captures and can be lowered after the domain that staged it is gone.
        assert_eq!(staged.source_program().captures(), &[capture]);
        assert_eq!(staged.input_types(), &[input_type.clone()]);
        assert_eq!(staged.output_types(), &[input_type.clone(), input_type]);
        let (lowered, output_types) = TestDomain::new().lower(&staged).unwrap();
        assert_eq!(lowered.capture_count, 1);
        assert_eq!(output_types, staged.output_types());
        assert_eq!(
            lowered.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                in (%0, %1)"},
        );
    }

    #[test]
    fn test_staged_program_stage_propagates_closure_error() {
        let domain = TestDomain::new();
        let result = StagedProgram::stage(
            &domain,
            |_, _, _| Err(ProgramError::InvalidArgument { message: "staging failed".into() }),
            Vec::new(),
            vec![ArrayType::scalar(DataType::F64)],
            TestOptions::default(),
        );

        assert!(matches!(result, Err(ProgramError::InvalidArgument { message }) if message == "staging failed"));
    }

    #[test]
    fn test_staged_program_stage_prunes_unused_captures() {
        let domain = TestDomain::new();
        let staged = StagedProgram::stage(
            &domain,
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
        let result = StagedProgram::stage(
            &TestDomain::new(),
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
    fn test_staged_function_stage_preserves_empty_structures() {
        let domain = TestDomain::new();
        let input_types = (vec![Vec::new(), Vec::new()], ArrayType::scalar(DataType::F64));
        let staged: StagedFunction<TestDomain, (Vec<Vec<ArrayType>>, ArrayType), (Vec<Vec<ArrayType>>, ArrayType)> =
            StagedFunction::stage(
                &domain,
                |_, _, (_, input)| Ok((vec![Vec::new(), Vec::new(), Vec::new()], input)),
                Vec::new(),
                input_types.clone(),
                TestOptions::default(),
            )
            .unwrap();

        assert_eq!(staged.input_signature().unwrap(), input_types);
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                in (%0)"},
        );
        let executable = staged.lower(&domain).unwrap().compile(&domain).unwrap().into_executable_function();
        assert_eq!(
            executable.call(&domain, (vec![Vec::new(), Vec::new()], Array::scalar(3.0).unwrap())).unwrap(),
            (vec![Vec::new(), Vec::new(), Vec::new()], Array::scalar(3.0).unwrap()),
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
            first.executable_function().call(&domain, Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(-2.0).unwrap()
        );
        assert_eq!(
            second.executable_function().call(&domain, Array::scalar(4.0).unwrap()).unwrap(),
            Array::scalar(-4.0).unwrap()
        );
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(domain.cache.statistics().memory_hits, 1);
    }

    #[test]
    fn test_executable_function_with_compiled_program() {
        let domain = TestDomain::new();
        let identity = compile_from_one_call_site(&domain, false).into_executable_function();
        let negate = compile_from_one_call_site(&domain, true).into_executable_function();
        let replaced = identity.with_compiled_program(
            Arc::new(TestCompiledProgram {
                program: negate.compiled_program().program.clone(),
                output_types: negate.output_types().to_vec(),
            }),
            negate.output_types().to_vec(),
        );

        // The replacement runs the new program with the retained signature, while the original function is unaffected.
        assert_eq!(replaced.input_types(), identity.input_types());
        assert_eq!(replaced.call(&domain, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(-3.0).unwrap());
        assert_eq!(identity.call(&domain, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(3.0).unwrap());
    }

    #[test]
    fn test_executable_function_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        assert_send_and_sync::<ExecutableFunction<TestDomain, ArrayType, ArrayType>>();

        let domain = TestDomain::new();
        let executable = compile_from_one_call_site(&domain, true).into_executable_function();
        let second = executable.clone();
        let first_domain = domain.clone();
        let second_domain = domain.clone();
        let first_thread =
            std::thread::spawn(move || executable.call(&first_domain, Array::scalar(3.0).unwrap()).unwrap());
        let second_thread =
            std::thread::spawn(move || second.call(&second_domain, Array::scalar(4.0).unwrap()).unwrap());

        assert_eq!(first_thread.join().unwrap(), Array::scalar(-3.0).unwrap());
        assert_eq!(second_thread.join().unwrap(), Array::scalar(-4.0).unwrap());
    }

    #[test]
    fn test_compiled_function() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), ArrayType> = StagedFunction::stage(
            &domain,
            |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            TestOptions::default(),
        )
        .unwrap();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();

        assert_eq!(compiled.executable_function().call(&domain, ()).unwrap(), Array::scalar(7.0).unwrap());
        assert_eq!(compiled.source_program().captures(), &[Array::scalar(7.0).unwrap()]);
        assert!(compiled.staged().input_types().is_empty());
        assert_eq!(compiled.lowered().output_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(compiled.output_types(), &[ArrayType::scalar(DataType::F64)]);
    }

    #[test]
    fn test_lowered_function_compile_validates_output_types() {
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
    fn test_compiled_function_into_executable_function() {
        let domain = TestDomain::new();
        let staged: StagedFunction<TestDomain, (), ArrayType> = StagedFunction::stage(
            &domain,
            |_, mut captures: Vec<CompilationTracer<TestDomain>>, ()| Ok(captures.remove(0)),
            vec![Array::scalar(7.0).unwrap()],
            (),
            TestOptions::default(),
        )
        .unwrap();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let executable = compiled.into_executable_function();

        // The runtime handle outlives the transform metadata and still supplies its captures on every call.
        assert_eq!(executable.captures(), &[Array::scalar(7.0).unwrap()]);
        assert!(executable.input_types().is_empty());
        assert_eq!(executable.output_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(executable.call(&domain, ()).unwrap(), Array::scalar(7.0).unwrap());
        assert_eq!(executable.validate_flat_input_count(&[]), Ok(()));
        assert_eq!(
            executable.validate_flat_input_count(&[Array::scalar(1.0).unwrap()]),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );
        assert_eq!(
            executable.inputs_with_captures(vec![Array::scalar(1.0).unwrap()]),
            vec![Array::scalar(7.0).unwrap(), Array::scalar(1.0).unwrap()],
        );
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
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), ArrayIrType, ArrayIrType> =
            domain.jit(zeros);
        assert_eq!(function.call((), three.clone()), Ok(three_zeros.clone()));
        assert_eq!(function.call((), four.clone()), Ok(four_zeros.clone()));
        assert_eq!(
            function.call((), ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
            Ok(three_zeros.clone()),
        );
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(function.statistics().dispatch_hits, 1);
        assert_eq!(domain.compilation_count(), 2);

        // Declaring a symbolic staged type for the input makes calls with different runtime extents share one staged
        // input type, and therefore one retained trace, lowering, and compiled specialization, while still producing
        // outputs with different logical shapes.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), ArrayIrType, ArrayIrType> =
            domain.jit(zeros).with_staged_input_types(Some(extent_type.clone().into())).unwrap();
        assert_eq!(function.staged_input_types(), Some(Some(extent_type.clone().into())));
        assert_eq!(function.call((), three), Ok(three_zeros));
        assert_eq!(function.call((), four), Ok(four_zeros));
        assert_eq!(function.specialization_count(), 1);
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
            function.call((), ArrayIrValue::Dimension(DimensionValue::new(wide_type, 7).unwrap())),
            Err(TypeError::invalid(
                "jit input `$` has type `dimension<7>`, which does not refine its declared staged type \
                 `dimension<extent ∈ [1, 5)>`",
            )
            .into()),
        );
        assert_eq!(function.specialization_count(), 1);

        // Staged input types must be declared before the dispatcher is shared.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), ArrayIrType, ArrayIrType> =
            domain.jit(zeros);
        let _clone = function.clone();
        assert!(matches!(
            function.with_staged_input_types(Some(extent_type.into())),
            Err(ProgramError::InvalidArgument { message })
                if message == "staged input types must be declared before a jit dispatcher is shared",
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
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), Vec<ArrayIrType>, ArrayIrType> = domain
            .jit(zeros)
            .with_staged_input_types(vec![Some(rows.clone().into()), Some(columns.clone().into())])
            .unwrap();
        assert_eq!(
            function.call((), extents(2, 3)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![0f32; 6]).unwrap())),
        );
        assert_eq!(
            function.call((), extents(3, 2)),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![0f32; 6]).unwrap())),
        );
        assert_eq!(function.specialization_count(), 1);

        // One declared identity shared by both inputs requires their runtime extents to agree.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), Vec<ArrayIrType>, ArrayIrType> = domain
            .jit(zeros)
            .with_staged_input_types(vec![Some(rows.clone().into()), Some(rows.clone().into())])
            .unwrap();
        assert_eq!(
            function.call((), extents(2, 2)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![0f32; 4]).unwrap())),
        );
        assert_eq!(
            function.call((), extents(2, 3)),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "rows".to_string(),
                expected: 2,
                actual: 3,
            })
            .into()),
        );
        assert_eq!(function.specialization_count(), 1);

        // A `None` leaf keeps its input exact, so only calls that also agree on that input's extent share one
        // specialization.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), Vec<ArrayIrType>, ArrayIrType> =
            domain.jit(zeros).with_staged_input_types(vec![Some(rows.clone().into()), None]).unwrap();
        assert_eq!(function.staged_input_types(), Some(vec![Some(rows.clone().into()), None]));
        for (rows_extent, columns_extent) in [(2, 3), (3, 3), (2, 4)] {
            assert_eq!(
                function.call((), extents(rows_extent, columns_extent)),
                Ok(ArrayIrValue::Array(
                    Array::matrix(rows_extent, columns_extent, vec![0f32; rows_extent * columns_extent]).unwrap(),
                )),
            );
        }
        assert_eq!(function.specialization_count(), 2);

        // Inputs must have the structure of the declaration, which otherwise would apply its leaves to other inputs.
        let function: CompiledFunctionDispatcher<TestArrayIrDomain, _, (), Vec<ArrayIrType>, ArrayIrType> =
            domain.jit(zeros).with_staged_input_types(vec![Some(rows.clone().into())]).unwrap();
        assert!(matches!(
            function.call((), extents(2, 3)),
            Err(ProgramError::InvalidArgument { message })
                if message == "jit inputs do not have the structure of their declared staged input types",
        ));
        assert_eq!(function.specialization_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_domain() {
        // An explicit dispatcher reports its domain from construction.
        let domain = CarriedDomain::new();
        let explicit: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            domain.jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input));
        assert_eq!(explicit.domain(), Some(&domain));

        // An inferred dispatcher reports no domain until a call through any clone selects one, which all clones then
        // share and which clearing the cache does not reset.
        let inferred: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input));
        let clone = inferred.clone();
        assert_eq!(inferred.domain(), None);
        assert_eq!(clone.call((), domain.scalar(1.0)), Ok(domain.scalar(1.0)));
        assert_eq!(inferred.domain(), Some(&domain));
        inferred.clear_cache();
        assert_eq!(clone.domain(), Some(&domain));
    }

    #[test]
    fn test_compiled_function_dispatcher_invalidate_static() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });
        function.call(true, Array::scalar(2.0).unwrap()).unwrap();
        function.call(false, Array::scalar(2.0).unwrap()).unwrap();

        assert_eq!(function.cache_capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert_eq!(function.invalidate_static(&true), 1);
        assert_eq!(function.specialization_count(), 1);
        function.call(true, Array::scalar(2.0).unwrap()).unwrap();
        assert_eq!(function.statistics().traces, 3);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });

        assert_eq!(function.call(true, Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(-2.0).unwrap());
        assert_eq!(function.call(true, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(-3.0).unwrap());
        assert_eq!(function.call(false, Array::scalar(4.0).unwrap()).unwrap(), Array::scalar(4.0).unwrap());
        assert_eq!(function.specialization_count(), 2);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_with_empty_inputs() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), (), ()> = domain.jit(|(), ()| Ok(()));

        assert!(function.domain().is_some());
        assert_eq!(domain.compilation_count(), 0);
        assert_eq!(function.call((), ()), Ok(()));
        assert_eq!(function.call((), ()), Ok(()));
        assert_eq!(domain.compilation_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_with_inferred_domain() {
        // Later calls never select another domain, so inputs from another domain execute on the selected one, both on
        // warm calls and on the cold calls that follow cache clearing.
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.call((), first_domain.scalar(1.0)), Ok(first_domain.scalar(-1.0)));
        assert_eq!(function.call((), second_domain.scalar(2.0)), Ok(first_domain.scalar(-2.0)));
        function.clear_cache();
        assert_eq!(function.call((), second_domain.scalar(3.0)), Ok(first_domain.scalar(-3.0)));
        assert_eq!(first_domain.compilation_count(), 2);
        assert_eq!(second_domain.compilation_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_with_inferred_domain_and_empty_inputs() {
        // Empty inputs carry no domain, so they fail before tracing until a nonempty call selects one.
        let domain = CarriedDomain::new();
        let traces = AtomicUsize::new(0);
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), Vec<ArrayType>, Vec<ArrayType>> =
            jit(|(), inputs: Vec<CompilationTracer<CarriedDomain>>| {
                traces.fetch_add(1, Ordering::Relaxed);
                Ok(inputs)
            });
        assert!(matches!(
            function.call((), Vec::new()),
            Err(ProgramError::InvalidArgument { message })
                if message == "compilation requires an input leaf to infer a domain; use an explicit domain for \
                               empty inputs",
        ));
        assert_eq!(function.domain(), None);
        assert_eq!(traces.load(Ordering::Relaxed), 0);
        assert_eq!(function.call((), vec![domain.scalar(1.0)]), Ok(vec![domain.scalar(1.0)]));
        assert_eq!(function.call((), Vec::new()), Ok(Vec::new()));
        assert_eq!(traces.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_applies_staging_options_before_tracing() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            domain.jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), options);

        assert!(matches!(
            function.call((), Array::scalar(2.0).unwrap()),
            Err(ProgramError::InvalidArgument { message })
                if message == "runtime input type f64[] does not refine declared type i64[]",
        ));
        assert_eq!(function.statistics().traces, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_distinguishes_hash_colliding_static_values() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct CollidingStatic(bool);

        impl Hash for CollidingStatic {
            fn hash<H: Hasher>(&self, state: &mut H) {
                0u8.hash(state);
            }
        }

        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, CollidingStatic, ArrayType, ArrayType> =
            domain.jit(|static_parameters: CollidingStatic, input: CompilationTracer<TestDomain>| {
                Ok(if static_parameters.0 { input.unary(NegOperation::new()) } else { input })
            });

        assert_eq!(
            function.call(CollidingStatic(false), Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(2.0).unwrap()
        );
        assert_eq!(
            function.call(CollidingStatic(true), Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(-2.0).unwrap()
        );
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_partitions_input_structures() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), Vec<ArrayType>, ArrayType> =
            domain.jit(|(), mut inputs: Vec<CompilationTracer<TestDomain>>| {
                inputs.drain(..1).next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        assert_eq!(function.call((), vec![Array::scalar(2.0).unwrap()]).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(
            function.call((), vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]).unwrap(),
            Array::scalar(2.0).unwrap(),
        );
        assert_eq!(function.specialization_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_distinguishes_empty_substructure() {
        // Both calls flatten to zero leaves, so they share static parameters and an empty dispatch signature. Only
        // the input parameter structure distinguishes them, while the traced closure branches on the observed
        // container arity. Key reuse across these calls would therefore silently return the first call's staged
        // behavior for the second call's structurally different input.
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), Vec<Vec<ArrayType>>, Vec<ArrayType>> =
            domain.jit(|(), inputs: Vec<Vec<CompilationTracer<TestDomain>>>| {
                if inputs.len() == 1 {
                    Ok(inputs.into_iter().next().unwrap())
                } else {
                    Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() })
                }
            });

        assert_eq!(function.call((), vec![Vec::new()]).unwrap(), Vec::new());
        assert!(matches!(
            function.call((), Vec::new()),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_flattens_map_inputs_deterministically() {
        // Map-like containers flatten in key order, so two runtime maps with the same keys must produce one
        // specialization regardless of construction order, and replay must bind flat inputs to the same leaves.
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, (), BTreeMap<&'static str, ArrayType>, ArrayType> =
            domain.jit(|(), inputs: BTreeMap<&'static str, CompilationTracer<TestDomain>>| {
                inputs.into_values().next().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
            });

        let first = BTreeMap::from([("a", Array::scalar(1.0).unwrap()), ("b", Array::scalar(2.0).unwrap())]);
        let second = BTreeMap::from([("b", Array::scalar(4.0).unwrap()), ("a", Array::scalar(3.0).unwrap())]);
        assert_eq!(function.call((), first).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(function.call((), second).unwrap(), Array::scalar(3.0).unwrap());
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_retries_tracing_failures() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(|fail, input: CompilationTracer<TestDomain>| {
                if fail {
                    Err(ProgramError::InvalidArgument { message: "expected trace failure".into() })
                } else {
                    Ok(input)
                }
            });

        for _ in 0..2 {
            assert!(matches!(
                function.call(true, Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message }) if message == "expected trace failure",
            ));
        }
        assert_eq!(function.call(false, Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 3);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_retains_inferred_domain_after_tracing_failure() {
        // The first nonempty call selects the domain before tracing, so a failed trace still fixes it for retries.
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let attempts = AtomicUsize::new(0);
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit(|(), input: CompilationTracer<CarriedDomain>| {
                if attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                    Err(ProgramError::InvalidArgument { message: "expected trace failure".into() })
                } else {
                    Ok(input)
                }
            });
        assert!(matches!(
            function.call((), first_domain.scalar(1.0)),
            Err(ProgramError::InvalidArgument { message }) if message == "expected trace failure",
        ));
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(function.specialization_count(), 0);
        assert_eq!(function.call((), second_domain.scalar(2.0)), Ok(first_domain.scalar(2.0)));
        assert_eq!(first_domain.compilation_count(), 1);
        assert_eq!(second_domain.compilation_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_retries_lowering_and_compilation_failures() {
        let lowering_domain = TestDomain::new();
        let lowering_options = TestOptions { fail_lowering: true, ..TestOptions::default() };
        let lowering_function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            lowering_domain.jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), lowering_options);
        for _ in 0..2 {
            assert!(matches!(
                lowering_function.call((), Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message }) if message == "expected lowering failure",
            ));
        }
        assert_eq!(lowering_function.specialization_count(), 0);
        let statistics = lowering_function.statistics();
        assert_eq!((statistics.traces, statistics.lowerings, statistics.compilation_requests), (2, 2, 0));

        let compilation_domain = TestDomain::new();
        let compilation_options = TestOptions { fail_compilation: true, ..TestOptions::default() };
        let compilation_function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            compilation_domain
                .jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), compilation_options);
        for _ in 0..2 {
            assert!(matches!(
                compilation_function.call((), Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message }) if message == "expected compilation failure",
            ));
        }
        assert_eq!(compilation_function.specialization_count(), 0);
        let statistics = compilation_function.statistics();
        assert_eq!((statistics.traces, statistics.lowerings, statistics.compilation_requests), (2, 2, 2));
        assert_eq!(compilation_domain.compilation_count(), 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_retains_specialization_after_execution_failure() {
        let domain = TestDomain::new();
        let options =
            TestOptions { staged_input_type: Some(ArrayType::scalar(DataType::I64)), ..TestOptions::default() };
        let function: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            domain.jit_with_options(|(), input: CompilationTracer<TestDomain>| Ok(input), options);

        for _ in 0..2 {
            assert!(matches!(
                function.call((), Array::scalar(2.0).unwrap()),
                Err(ProgramError::InvalidArgument { message })
                    if message == "runtime input type f64[] does not refine declared type i64[]",
            ));
        }

        // The compiled entry is inserted before runtime execution, so an execution failure does not evict it and the
        // second call dispatches directly to the retained specialization.
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().traces, 1);
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_rejects_same_key_recursive_dispatch() {
        let domain = TestDomain::new();
        let recursive: Rc<RefCell<Option<Box<dyn Fn() -> Result<Array, ProgramError>>>>> = Rc::new(RefCell::new(None));
        let closure_recursive = recursive.clone();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(move |_, input: CompilationTracer<TestDomain>| {
                if let Some(call) = closure_recursive.borrow().as_ref() {
                    call()?;
                }
                Ok(input)
            });
        let function_clone = function.clone();
        *recursive.borrow_mut() = Some(Box::new(move || function_clone.call(true, Array::scalar(1.0).unwrap())));

        assert!(matches!(
            function.call(true, Array::scalar(1.0).unwrap()),
            Err(ProgramError::InvalidArgument { message })
                if message == "recursive JIT dispatch requested a specialization that is already being produced",
        ));
        assert_eq!(function.specialization_count(), 0);
        assert_eq!(function.statistics().dispatch_misses, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_allows_different_key_recursive_dispatch() {
        let domain = TestDomain::new();
        let recursive: Rc<RefCell<Option<Box<dyn Fn() -> Result<Array, ProgramError>>>>> = Rc::new(RefCell::new(None));
        let closure_recursive = recursive.clone();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(move |nested, input: CompilationTracer<TestDomain>| {
                if nested {
                    if let Some(call) = closure_recursive.borrow().as_ref() {
                        call()?;
                    }
                }
                Ok(input)
            });
        let function_clone = function.clone();
        *recursive.borrow_mut() = Some(Box::new(move || function_clone.call(false, Array::scalar(1.0).unwrap())));

        assert_eq!(function.call(true, Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(function.statistics().traces, 2);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_isolates_dispatchers() {
        let domain = TestDomain::new();
        let first: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            domain.jit(|(), input: CompilationTracer<TestDomain>| Ok(input));
        let second: CompiledFunctionDispatcher<TestDomain, _, (), ArrayType, ArrayType> =
            domain.jit(|(), input: CompilationTracer<TestDomain>| Ok(input));

        assert_eq!(first.call((), Array::scalar(1.0).unwrap()).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(second.call((), Array::scalar(2.0).unwrap()).unwrap(), Array::scalar(2.0).unwrap());
        assert_eq!(first.specialization_count(), 1);
        assert_eq!(second.specialization_count(), 1);
        assert_eq!(first.statistics().traces, 1);
        assert_eq!(second.statistics().traces, 1);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_concurrent_warm_calls() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });
        function.call(true, Array::scalar(1.0).unwrap()).unwrap();

        std::thread::scope(|scope| {
            for index in 0..4 {
                let function = function.clone();
                scope.spawn(move || {
                    let value = f64::from(index);
                    assert_eq!(
                        function.call(true, Array::scalar(value).unwrap()).unwrap(),
                        Array::scalar(-value).unwrap()
                    );
                });
            }
        });

        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_hits, 4);
        assert_eq!(function.statistics().traces, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_concurrent_cold_misses() {
        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });

        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let function = function.clone();
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    assert_eq!(function.call(true, Array::scalar(3.0).unwrap()).unwrap(), Array::scalar(-3.0).unwrap());
                });
            }
        });

        // Cross-thread same-key cold misses deliberately race and may duplicate frontend production; both inserts are
        // idempotent and exactly one retained entry remains, while the shared compilation context deduplicates the
        // backend compilation itself.
        assert_eq!(function.specialization_count(), 1);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits + statistics.dispatch_misses, 2);
        assert!(statistics.traces >= 1 && statistics.traces <= 2);
        assert_eq!(statistics.traces, statistics.dispatch_misses);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_concurrent_domain_inference() {
        // Racing first calls through clones with inputs from different domains select one domain, which then executes
        // both calls.
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input));

        let barrier = std::sync::Barrier::new(2);
        let (first_output, second_output) = std::thread::scope(|scope| {
            let first_call = scope.spawn(|| {
                barrier.wait();
                function.clone().call((), first_domain.scalar(1.0))
            });
            let second_call = scope.spawn(|| {
                barrier.wait();
                function.clone().call((), second_domain.scalar(2.0))
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
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(selected_domain.compilation_count() as u64, function.statistics().traces);
        assert_eq!(unselected_domain.compilation_count(), 0);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_statefully_async() {
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(
            function.call_statefully_async((), first_domain.scalar(1.0)).r#await(),
            Ok(first_domain.scalar(-1.0))
        );
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(
            function.call_statefully_async((), second_domain.scalar(2.0)).r#await(),
            Ok(first_domain.scalar(-2.0)),
        );
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_call_statefully() {
        let first_domain = CarriedDomain::new();
        let second_domain = CarriedDomain::new();
        let function: CompiledFunctionDispatcher<CarriedDomain, _, (), ArrayType, ArrayType> =
            jit(|(), input: CompilationTracer<CarriedDomain>| Ok(input.unary(NegOperation::new())));
        assert_eq!(function.call_statefully((), first_domain.scalar(1.0)), Ok(first_domain.scalar(-1.0)));
        assert_eq!(function.domain(), Some(&first_domain));
        assert_eq!(function.call_statefully((), second_domain.scalar(2.0)), Ok(first_domain.scalar(-2.0)));
        assert_eq!(function.statistics().dispatch_hits, 1);
    }

    #[test]
    fn test_compiled_function_dispatcher_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        fn assert_dispatcher_send_and_sync<F: Send + Sync>(
            _function: &CompiledFunctionDispatcher<TestDomain, F, bool, ArrayType, ArrayType>,
        ) {
            assert_send_and_sync::<CompiledFunctionDispatcher<TestDomain, F, bool, ArrayType, ArrayType>>();
        }

        assert_send_and_sync::<StagedFunction<TestDomain, ArrayType, ArrayType>>();
        assert_send_and_sync::<LoweredFunction<TestDomain, ArrayType, ArrayType>>();
        assert_send_and_sync::<CompiledFunction<TestDomain, ArrayType, ArrayType>>();

        let domain = TestDomain::new();
        let function: CompiledFunctionDispatcher<TestDomain, _, bool, ArrayType, ArrayType> =
            domain.jit(|negate, input: CompilationTracer<TestDomain>| {
                Ok(if negate { input.unary(NegOperation::new()) } else { input })
            });
        assert_dispatcher_send_and_sync(&function);
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
    fn test_jit_compilation_transform() {
        type Source = JitCompilationSource<TestDomain, usize, Vec<ArrayType>, Vec<ArrayType>>;
        type Arguments = FunctionSpecializationKey<usize, Vec<Placeholder>, Arc<[ArrayType]>>;
        type Artifact = ExecutableFunction<TestDomain, Vec<ArrayType>, Vec<ArrayType>>;

        fn assert_contract<T, S>()
        where
            T: Transform<S, Arguments = Arguments, Artifact = Artifact>,
        {
        }

        assert_contract::<JitCompilationTransform, Source>();
        assert_eq!(<JitCompilationTransform as Transform<Source>>::DEFAULT_CACHE_CAPACITY, DEFAULT_JIT_CACHE_CAPACITY,);
    }
}
